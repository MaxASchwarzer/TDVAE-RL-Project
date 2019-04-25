import importlib
import time

import numpy as np
import torch

from readers.gym_reader import GymReader, ReplayBuffer
from pylego import misc, runner


def trim_batch(seq_len, batch):
    if seq_len <= 0:
        return batch
    else:
        obs, actions, rewards, done, t1, t2, returns, is_weight, idxs = batch
        return (obs[:, :seq_len], actions[:, :seq_len], rewards[:, :seq_len], done[:, :seq_len], t1, t2, returns,
                is_weight, idxs)


class BaseRLRunner(runner.Runner):

    def __init__(self, flags, model_class, log_keys, *args, **kwargs):
        self.flags = flags
        if flags.samples_per_seq != 1:
            raise ValueError('! ERROR: samples_per_seq != 1 is not supported for RL')

        # MiniGrid is an optional dependency, so only import if the user wants it
        # MiniGrid can be installed by cloning and installing https://github.com/maximecb/gym-minigrid.git
        if "minigrid" in flags.env.lower():
            print("Attempting to import MiniGrid environments.")
            # import gym_minigrid  # XXX why importing this here?
            # Image normalization will crash minigrid, since observations aren't pixels, so force this
            flags.raw = True

        if "minigrid" in flags.env.lower() or "mujoco" in flags.env.lower():
            self.visualize = False
            flags.visualize_every = -1
        else:
            self.visualize = True

        self.mpc = flags.mpc
        self.boltzmann_mpc = flags.boltzmann_mpc
        self.emulator = GymReader(flags.env, flags.seq_len, int(np.ceil(flags.batch_size / flags.add_replay_every)),
                                  flags.threads, np.inf, raw=flags.raw)
        self.action_space = self.emulator.action_space()
        self.seq_len_upper = flags.seq_len
        self.eps_decay = misc.LinearDecay(flags.eps_decay_start, flags.eps_decay_end, 1.0, flags.eps_final)

        if flags.seq_len_initial < 0:
            flags.seq_len_initial = flags.seq_len
        self.seq_len_decay = misc.LinearDecay(flags.seq_len_decay_start, flags.seq_len_decay_end,
                                              flags.seq_len_initial, flags.seq_len)
        reader = ReplayBuffer(self.emulator, flags.replay_size, flags.initial_replay_size, flags.iters_per_epoch,
                              flags.t_diff_min, flags.t_diff_max, flags.discount_factor,
                              initial_len=int(self.seq_len_decay.get_y(0)), skip_init=bool(flags.load_file))

        self.discount_factor = flags.discount_factor

        summary_dir = flags.log_dir + '/summary'
        log_keys.extend(['rewards_per_ep_mean', 'rewards_per_ep_std'])
        super().__init__(reader, flags.batch_size, flags.epochs, summary_dir, log_keys=log_keys,
                         threads=flags.threads, print_every=flags.print_every, visualize_every=flags.visualize_every,
                         max_batches=flags.max_batches, *args, **kwargs)
        model_class = misc.get_subclass(importlib.import_module('models.' + self.flags.model), model_class)
        self.model = model_class(self.flags, action_space=self.action_space, rl=True, replay_buffer=reader,
                                 optimizer=flags.optimizer, learning_rate=flags.learning_rate, cuda=flags.cuda,
                                 load_file=flags.load_file, save_every=flags.save_every, save_file=flags.save_file,
                                 max_save_files=2, debug=flags.debug)

        # consider history length for simulation to be the expected t seen during TDQVAE training
        self.history_length = int(np.ceil(0.5 * (int(self.seq_len_decay.get_y(0)) + flags.t_diff_min))) - 1
        print('* Initial simulation history length:', self.history_length)

        self.emulator_iter = self.emulator.iter_batches('train', self.emulator.batch_size, threads=flags.threads)
        self.emulator_state = next(self.emulator_iter).get_next()[:5]  # init emulator state after model loading

        self.eval_episodes = self.flags.eval_episodes
        if self.eval_episodes <= 0:
            self.eval_episodes = self.flags.batch_size
        self.eval_emulator = GymReader(flags.env, flags.seq_len, self.eval_episodes, flags.threads,
                                       flags.eval_episode_length, raw=flags.raw)

    def run_epoch(self, epoch, split, train=False, log=True):
        """Iterates the epoch data for a specific split."""
        print('\n* Starting epoch %d, split %s' % (epoch, split), '(train: ' + str(train) + ')')
        self.reset_epoch_reports()
        self.model.set_train(train)

        timestamp = time.time()
        reader_iter = self.reader.iter_batches(split, self.batch_size, shuffle=train, partial_batching=not train,
                                               threads=self.threads, max_batches=self.max_batches)
        for i in range(self.reader.get_size(split)):
            try:
                train_batch = next(reader_iter)
            except StopIteration:
                break

            seq_len = int(self.seq_len_decay.get_y(self.model.get_train_steps()))
            self.history_length = int(np.ceil(0.5 * (seq_len + self.flags.t_diff_min))) - 1
            simulation_start = seq_len - self.history_length
            obs, actions, rewards, done = self.emulator_state[:4]
            obs = obs[:, simulation_start:]
            actions = actions[:, simulation_start:]
            rewards = rewards[:, simulation_start:]
            done = done[:, simulation_start:]
            obs, actions, rewards, done = self.model.prepare_batch([obs, actions, rewards, done])

            self.model.set_train(False)
            with torch.no_grad():
                if self.mpc:
                    selected_actions = self.model.model.predictive_control(obs, actions, rewards, done,
                                                                           num_rollouts=50, rollout_length=1,
                                                                           jump_length=5,
                                                                           gamma=self.discount_factor,
                                                                           boltzmann=self.boltzmann_mpc)
                else:
                    q = self.model.model.compute_q(obs, actions, rewards, done)
                    selected_actions = torch.argmax(q, dim=1).cpu().numpy()

            self.model.set_train(True)
            random_actions = np.random.randint(0, self.action_space, size=selected_actions.shape)
            eps = self.eps_decay.get_y(self.model.get_train_steps())
            do_random = np.random.choice(2, size=selected_actions.shape, p=[1. - eps, eps])
            actions = np.where(do_random, random_actions, selected_actions)

            self.emulator_state = next(self.emulator_iter).get_next(actions)[:5]

            # add trajectory to replay buffer
            self.reader.add(self.emulator_state, truncated_len=seq_len)

            train_batch = trim_batch(seq_len, train_batch)
            report = self.clean_report(self.run_batch(train_batch, train=train))
            if self.model.get_train_steps() % self.flags.freeze_every == 0:
                print('* Updating target Q network')
                self.model.update_target_net()

            report['time_'] = time.time() - timestamp
            if train:
                self.log_train_report(report, self.model.get_train_steps())
            self.epoch_reports.append(report)
            if self.print_every > 0 and i % self.print_every == 0:
                self.print_report(epoch, report, step=i)
            if train and self.visualize_every > 0 and self.model.get_train_steps() % self.visualize_every == 0:
                self.model.set_train(False)
                self.train_visualize()
                self.model.set_train(True)

            timestamp = time.time()

        # Evaluate
        print('* Evaluating greedy policy')
        actions = None
        sum_of_rewards = np.zeros([self.eval_episodes])
        all_dones = np.zeros([self.eval_episodes])
        self.eval_emulator.reset()
        for cond_batch in self.eval_emulator.iter_batches('train', self.eval_episodes, threads=self.flags.threads):
            obs, actions, rewards, done = cond_batch.get_next(actions)[:4]
            if actions is not None:
                sum_of_rewards += rewards[:, -1] * np.maximum(1.0 - all_dones, 0.0)
                all_dones += done[:, -1]
                if np.all(all_dones > 1e-6):
                    break

            obs = obs[:, simulation_start:]
            actions = actions[:, simulation_start:]
            rewards = rewards[:, simulation_start:]
            done = done[:, simulation_start:]
            obs, actions, rewards, done = self.model.prepare_batch([obs, actions, rewards, done])
            self.model.set_train(False)
            with torch.no_grad():
                if self.mpc:
                    actions = self.model.model.predictive_control(obs, actions, rewards, done,
                                                                  num_rollouts=50, rollout_length=1,
                                                                  jump_length=5,
                                                                  gamma=self.discount_factor,
                                                                  boltzmann=False)
                else:
                    q = self.model.model.compute_q(obs, actions, rewards, done)
                    actions = torch.argmax(q, dim=1).cpu().numpy()
            self.model.set_train(True)

        report = {'rewards_per_ep_mean': sum_of_rewards.mean(), 'rewards_per_ep_std': sum_of_rewards.std()}
        self.log_report(report, self.model.get_train_steps())

        # Finish epoch
        epoch_report = self.get_epoch_report()
        epoch_report.update(report)
        self.print_report(epoch, epoch_report)
        if train:
            self.log_epoch_train_report(epoch_report, self.model.get_train_steps())
        elif log:
            self.log_epoch_val_report(epoch_report, self.model.get_train_steps())

        self.model.set_train(False)
        if self.visualize:
            self.post_epoch_visualize(epoch, split)
