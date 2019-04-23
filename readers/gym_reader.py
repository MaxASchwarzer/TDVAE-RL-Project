from collections import deque
from multiprocessing import Process, Pipe
import pickle

import gym
import gym_vecenv as vecenv
import numpy as np
import torch
import torch.nn.functional as F

from pylego import misc
from pylego.reader import Reader


def make_env(env_name, frameskip=4, steps=1000000, secs=100000):
    if "minigrid" in env_name.lower():
        env = gym.make(env_name)
    else:
        env = gym.make(env_name)
        env.env.frameskip = frameskip
    env._max_episode_steps = steps
    env._max_episode_seconds = secs
    return env


class ActionConditionalBatch:  # TODO move generalized form of this to pylego

    def __init__(self, env, seq_len, batch_size, threads, downsample=True, inner_frameskip=4, raw=False,
                 data_dir='data'):
        self.env_name = env
        if "minigrid" in self.env_name.lower():
            self.minigrid = True
        elif "mujoco" in self.env_name.lower():
            self.mujoco = True
        else:
            self.minigrid = False
            self.mujoco = False
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.raw = raw  # return raw images

        fn = lambda: make_env(env, frameskip=inner_frameskip)
        env_fcns = [fn for i in range(batch_size)]
        self.env_name = env
        self.env = SubprocVecEnv(env_fcns, n_workers=threads)
        self.env.reset()
        self.buffers = [deque(maxlen=seq_len) for i in range(7)]
        self.downsample = downsample

        sample_env = gym.make(env)
        self.action_space = sample_env.action_space.n
        sample_env.close()
        self.frame_count = 0

        if not raw:
            with open(data_dir + '/' + env + '/img_stats.pk', 'rb') as f:
                (self.img_mean, self.img_std, self.img_min, self.img_max, self.img_true_min, self.img_true_max,
                 self.img_hcrop_top, self.img_hcrop_bottom, self.img_vcrop_left, self.img_vcrop_right) = pickle.load(f)
                self.img_mean = torch.tensor(self.img_mean)
                self.img_std = torch.tensor(self.img_std)

    def get_next(self, actions=None, fill_buffer=True):
        if actions is None:
            # We have to assume a random policy if nobody gives us actions
            in_actions = np.random.randint(0, self.action_space, size=self.batch_size)
        else:
            in_actions = actions

        self.env.step_async(in_actions)
        orig_obs, rewards, done, meta = self.env.step_wait()

        if self.minigrid:
            im = np.array([obs["image"] for obs in orig_obs])
            dir = np.array([obs["direction"] for obs in orig_obs])
            dir_flat = np.zeros((im.shape[0], im.shape[1], im.shape[2], 1)) + dir[:, None, None, None]
            obs = np.concatenate([im, dir_flat], axis=-1)
            obs = np.transpose(obs, axes=(0, 3, 1, 2))
            obs = torch.tensor(obs)

        elif self.mujoco:
            obs = np.stack(orig_obs, 0)
            obs = torch.tensor(obs)

        else:
            orig_obs = np.stack(orig_obs, 0)
            orig_obs = np.transpose(orig_obs, axes=(0, 3, 1, 2))  # original unnormalized images
            obs = torch.tensor(orig_obs).float() / 255

            if not self.raw:
                obs = ((obs - self.img_mean) / self.img_std).clamp_(self.img_min, self.img_max)
                obs = (obs - self.img_min) / (self.img_max - self.img_min)

                h, w = obs.size()[2:]
                obs = obs[:, :, self.img_hcrop_top:h-self.img_hcrop_bottom, self.img_vcrop_left:w-self.img_vcrop_right]
                h, w = obs.size()[2:]
                pad_h, pad_w = 160 - h, 160 - w
                if self.downsample:
                    obs = F.avg_pool2d(obs, (2, 2,), stride=2)
                    pad_h = np.ceil(pad_h / 2.0)
                    pad_w = np.ceil(pad_w / 2.0)
                pad_h /= 2.0
                pad_w /= 2.0

                obs = F.pad(obs, (int(np.ceil(pad_w)), int(np.floor(pad_w)), int(np.ceil(pad_h)), int(np.floor(pad_h))),
                            mode="constant", value=0.5)

        next_frame_count = np.uint32(self.frame_count + obs.size(0))
        if next_frame_count < self.frame_count:  # overflow
            self.frame_count = 0
            next_frame_count = self.frame_count + obs.size(0)
        indices = np.arange(self.frame_count, next_frame_count, dtype=np.uint32)
        self.frame_count = next_frame_count

        for i, val in enumerate([obs, in_actions, rewards, done, indices, meta, orig_obs]):
            self.buffers[i].append(val)

        if fill_buffer and len(self.buffers[0]) < self.seq_len:
            return self.get_next(actions=actions, fill_buffer=True)

        output = [torch.stack(list(self.buffers[0]), 1)] + [np.array(buffer).swapaxes(0, 1)
                                                            for buffer in self.buffers[1:]]

        output = [o[:self.batch_size] for o in output]
        return output

    def close(self):
        self.env.close()


class GymReader(Reader):  # TODO move generalized form of this to pylego

    def __init__(self, env, seq_len, batch_size, threads, iters_per_epoch, raw=False):
        self.env = env
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.threads = threads
        super().__init__({'train': iters_per_epoch})
        self.action_conditional_batch = ActionConditionalBatch(env, seq_len, batch_size, threads, raw=raw)

    def action_space(self):
        return self.action_conditional_batch.action_space

    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        assert split_name == 'train'
        assert batch_size == self.batch_size
        assert threads == self.threads

        epoch_size = self.splits[split_name]
        if max_batches > 0:
            epoch_size = min(max_batches, epoch_size)
        if epoch_size is np.inf:
            generator = iter(int, 1)
        else:
            generator = range(epochs * epoch_size)
        for _ in generator:
            yield self.action_conditional_batch

    def close(self):
        self.action_conditional_batch.close()


class ReplayBuffer(Reader):
    '''Replay buffer implementing prioritized experience replay.'''

    def __init__(self, emulator, buffer_size, iters_per_epoch, t_diff_min, t_diff_max, gamma, initial_len=-1,
                 clip_errors=2.0, skip_init=False,):
        self.emulator = emulator  # only to save and load frame_count
        self.t_diff_min = t_diff_min
        self.t_diff_max = t_diff_max
        self.clip_errors = clip_errors  # default value ideal for Atari
        self.buffer = misc.SumTree(buffer_size)
        self.cache = {}
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.e = 0.01
        self.a = 0.6
        self.gammas = np.array([gamma ** i for i in range(emulator.seq_len)], dtype=np.float32)

        if skip_init:
            print('* Skipping replay buffer initialization')
        else:
            print('* Initializing replay buffer')
            while self.buffer.count < buffer_size:
                for conditional_batch in emulator.iter_batches('train', emulator.batch_size, threads=emulator.threads,
                                                               max_batches=int(np.ceil(buffer_size /
                                                                                       emulator.batch_size))):
                    self.add(conditional_batch.get_next()[:5], truncated_len=initial_len)
                    if self.buffer.count >= buffer_size:
                        break
            print('* Replay buffer initialized')
        super().__init__({'train': iters_per_epoch})

    def add_cached(self, priority, idx_data, ob):
        ob = (ob * 255.0).numpy().astype(np.uint8)
        for i, o in zip(idx_data[0], ob):
            if i not in self.cache:
                self.cache[i] = [o, 1]
            else:
                self.cache[i][1] += 1
        old_idx_data = self.buffer.add(priority, idx_data)
        if isinstance(old_idx_data, tuple):
            for i in old_idx_data[0]:
                ref_count = self.cache[i][1]
                if ref_count == 1:
                    del self.cache[i]
                else:
                    self.cache[i][1] -= 1

    def get_cached(self, s):
        (idx, p, idx_data) = self.buffer.get(s)
        ob = np.stack([self.cache[i][0] for i in idx_data[0]])
        data = (torch.from_numpy((ob / 255.0).astype(np.float32)),) + idx_data[1:]
        return idx, p, data

    def calc_priority(self, error):
        return np.minimum(self.clip_errors, error + self.e) ** self.a

    def add(self, trajs, t_diff_min=None, t_diff_max=None, truncated_len=-1):
        t_diff_min = t_diff_min or self.t_diff_min
        t_diff_max = t_diff_max or self.t_diff_max

        priority = self.calc_priority(np.inf)
        for ob, action, reward, done, ob_ind in zip(*trajs):
            if truncated_len <= 0:
                truncated_len = ob.size(0)

            # generate random (t1, t2) combination such that there is no done between them
            bool_done = done[:truncated_len - 1] > 1e-6
            if np.any(bool_done):
                # first, get segments that don't have dones within them (except for final state)
                done_idx = np.nonzero(bool_done)[0]
                min_t1s = np.concatenate([[-1], done_idx]) + 1
                max_t2s = np.concatenate([done_idx, [truncated_len - 1]]) - 1  # -1 to leave room for next reward
                max_t1s = max_t2s - t_diff_min
                t1s_possible = max_t1s - min_t1s + 1
                if np.all(t1s_possible < 1):
                    continue  # skip this trajectory
                possible_combs = np.zeros_like(t1s_possible)
                for i, t1s_p in enumerate(t1s_possible):
                    for j in range(t1s_p):
                        possible_combs[i] += min(j, t_diff_max - t_diff_min) + 1
                segment_probs = possible_combs / possible_combs.sum()
                segment = np.random.choice(segment_probs.shape[0], p=segment_probs)
                min_t1, max_t1, max_t2 = min_t1s[segment], max_t1s[segment], max_t2s[segment]
            else:
                min_t1 = 0
                max_t2 = truncated_len - 2  # -1 (of 2) to leave room for next reward
                max_t1 = max_t2 - t_diff_min

            t1 = np.random.randint(min_t1, max_t1 + 1)
            t2 = np.random.randint(t1 + t_diff_min, min(t1 + t_diff_max, max_t2) + 1)
            if t1 + 1 <= t2:
                clipped_reward = np.clip(reward[t1 + 1:t2 + 1], -1.0, 1.0)
                returns = (self.gammas[:t2 - t1] * clipped_reward).sum()
            else:
                returns = 0.0
            self.add_cached(priority, (ob_ind, action, reward, done, t1, t2, returns), ob)

    def update(self, indices, errors):
        priorities = self.calc_priority(errors)
        for idx, priority in zip(indices, priorities):
            self.buffer.update(idx, priority)

    def get_buffer(self):
        return self.buffer, self.cache, self.beta, self.emulator.action_conditional_batch.frame_count

    def load_buffer(self, buffer):
        print('* Loading external replay buffer')
        # this has to be done before the first get_next() is called in the emulator
        self.buffer, self.cache, self.beta, self.emulator.action_conditional_batch.frame_count = buffer

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.buffer.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.get_cached(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.buffer.total()
        is_weight = np.power(self.buffer.count * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        assert split_name == 'train'
        assert shuffle

        epoch_size = self.splits[split_name]
        if max_batches > 0:
            epoch_size = min(max_batches, epoch_size)
        for _ in range(epochs * epoch_size):
            batch, idxs, is_weight = self.sample(batch_size)
            obs, actions, rewards, done, t1, t2, returns = list(zip(*batch))
            yield (torch.stack(obs), np.array(actions), np.array(rewards), np.array(done), np.array(t1), np.array(t2),
                   np.array(returns), np.array(is_weight), idxs)


# The following code is largely adapted from https://github.com/agakshat/gym_vecenv,
# and carries the following license:
# MIT License
#
# Copyright (c) 2018 Akshat Agarwal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def worker(remote, parent_remote, env_fn_wrappers):
    parent_remote.close()
    envs = [env_fn_wrapper.x() for env_fn_wrapper in env_fn_wrappers]
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, rewards, dones, infos = [], [], [], []
            for env, action in zip(envs, data):
                ob, reward, done, info = env.step(action)
                if np.any(done):
                    ob = env.reset()
                obs.append(ob)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
            remote.send((obs, np.array(rewards, dtype=np.float32),
                         np.array(dones, dtype=np.float32), np.array(infos)))
        elif cmd == 'reset':
            ob = [env.reset() for env in envs]
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = [env.reset_task() for env in envs]
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(([(env.observation_space, env.action_space) for env in envs]))
        else:
            raise NotImplementedError


class SubprocVecEnv(vecenv.vec_env.VecEnv):
    def __init__(self, env_fns, n_workers=-1, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.n_workers = n_workers
        if n_workers == -1:
            self.n_workes = nenvs
        env_fns = np.array_split(env_fns, n_workers)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_workers)])
        self.ps = [Process(target=worker,
                           args=(work_remote, remote, [vecenv.vec_env.CloudpickleWrapper(fn) for fn in env_fns]))
                   for (work_remote, remote, env_fns) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        spaces = self.remotes[0].recv()
        vecenv.vec_env.VecEnv.__init__(self, len(env_fns), spaces[0][0], spaces[0][1])

    def step_async(self, actions):
        actions = np.array_split(actions, self.n_workers)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs_lists, rews, dones, infos = zip(*results)
        obs = [obs for ls in obs_lists for obs in ls]
        return obs, np.concatenate(rews, 0), np.concatenate(dones, 0), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [obs for ls in [remote.recv() for remote in self.remotes] for obs in ls]

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return [obs for ls in [remote.recv() for remote in self.remotes] for obs in ls]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
