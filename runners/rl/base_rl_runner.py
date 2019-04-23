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
            import gym_minigrid

            # Image normalization will crash minigrid, since observations aren't pixels, so force this
            flags.raw = True

        if "minigrid" in flags.env.lower() or "mujoco" in flags.env.lower():
            self.visualize = False
            flags.visualize_every = -1
        else:
            self.visualize = True

        self.emulator = GymReader(flags.env, flags.seq_len, flags.batch_size, flags.threads, np.inf, raw=flags.raw)
        self.action_space = self.emulator.action_space()
        self.seq_len_upper = flags.seq_len
        self.eps_decay = misc.LinearDecay(flags.eps_decay_start, flags.eps_decay_end, 1.0, flags.eps_final)
        if flags.add_every_initial < 0:
            flags.add_every_initial = flags.add_replay_every
        self.replay_ratio_decay = misc.LinearDecay(flags.add_every_start, flags.add_every_end,
                                                   flags.add_every_initial, flags.add_replay_every)

        if flags.seq_len_initial < 0:
            flags.seq_len_initial = flags.seq_len

        self.seq_len_decay = misc.LinearDecay(flags.seq_len_decay_start, flags.seq_len_decay_end,
                                              flags.seq_len_initial, flags.seq_len)
        reader = ReplayBuffer(self.emulator, flags.replay_size, flags.iters_per_epoch, flags.t_diff_min,
                              flags.t_diff_max, flags.discount_factor, initial_len=int(self.seq_len_decay.get_y(0)),
                              skip_init=bool(flags.load_file))

        summary_dir = flags.log_dir + '/summary'
        log_keys.append('rewards_per_ep')
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

        self.emulator_iter = self.emulator.iter_batches('train', flags.batch_size, threads=flags.threads)
        self.emulator_state = next(self.emulator_iter).get_next()[:5]  # init emulator state after model loading
        self.rewards = np.zeros([self.emulator_state[0].size(0)])

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

            ratio = int(self.replay_ratio_decay.get_y(self.model.get_train_steps()))
            seq_len = int(self.seq_len_decay.get_y(self.model.get_train_steps()))
            self.history_length = int(np.ceil(0.5 * (seq_len + self.flags.t_diff_min))) - 1
            simulation_start = seq_len - self.history_length
            if self.model.get_train_steps() % ratio == 0:
                obs, actions, rewards, done = self.emulator_state[:4]
                obs = obs[:, simulation_start:]
                actions = actions[:, simulation_start:]
                rewards = rewards[:, simulation_start:]
                done = done[:, simulation_start:]
                obs, actions, rewards, done = self.model.prepare_batch([obs, actions, rewards, done])

                self.model.set_train(False)
                with torch.no_grad():
                    q = self.model.model.compute_q(obs, actions, rewards, done)
                self.model.set_train(True)

                selected_actions = torch.argmax(q, dim=1).cpu().numpy()
                random_actions = np.random.randint(0, self.action_space, size=selected_actions.shape)
                eps = self.eps_decay.get_y(self.model.get_train_steps())
                do_random = np.random.choice(2, size=selected_actions.shape, p=[1. - eps, eps])
                actions = np.where(do_random, random_actions, selected_actions)

                self.emulator_state = next(self.emulator_iter).get_next(actions)[:5]

                # add trajectory to replay buffer
                self.reader.add(self.emulator_state, truncated_len=seq_len)

                rewards = self.emulator_state[2][:, -1]
                self.rewards += rewards
                dones = self.emulator_state[3][:, -1] > 1e-6
                if np.any(dones):
                    rewards_per_ep = self.rewards[dones].mean()
                    self.rewards[dones] = 0.0
                    self.log_train_report({'rewards_per_ep': rewards_per_ep}, self.model.get_train_steps())

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

        epoch_report = self.get_epoch_report()
        self.print_report(epoch, epoch_report)
        if train:
            self.log_epoch_train_report(epoch_report, self.model.get_train_steps())
        elif log:
            self.log_epoch_val_report(epoch_report, self.model.get_train_steps())

        self.model.set_train(False)
        if self.visualize:
            self.post_epoch_visualize(epoch, split)
