import importlib
import time

import numpy as np
import torch

from readers.gym_reader import GymReader, ReplayBuffer
from pylego import misc, runner


class BaseRLRunner(runner.Runner):

    def __init__(self, flags, model_class, log_keys, *args, **kwargs):
        self.flags = flags

        emulator = GymReader(flags.env, flags.seq_len, flags.batch_size, flags.threads, np.inf)
        self.emulator_iter = emulator.iter_batches('train', flags.batch_size, threads=flags.threads)
        self.emulator_state = next(self.emulator_iter).get_next()[:3]

        reader = ReplayBuffer(emulator, flags.replay_size, flags.iters_per_epoch)

        summary_dir = flags.log_dir + '/summary'
        super().__init__(reader, flags.batch_size, flags.epochs, summary_dir, log_keys=log_keys,
                         threads=flags.threads, print_every=flags.print_every, visualize_every=flags.visualize_every,
                         max_batches=flags.max_batches, *args, **kwargs)
        model_class = misc.get_subclass(importlib.import_module('models.' + self.flags.model), model_class)
        self.model = model_class(self.flags, rl=True, optimizer=flags.optimizer, learning_rate=flags.learning_rate,
                                 cuda=flags.cuda, load_file=flags.load_file, save_every=flags.save_every,
                                 save_file=flags.save_file, debug=flags.debug)

        # consider history length for simulation to be the expected t seen during TDQVAE training
        self.simulation_start = flags.seq_len - int(np.ceil(0.5 * (flags.seq_len + flags.t_diff_min))) + 1

    def run_epoch(self, epoch, split, train=False, log=True):
        """Iterates the epoch data for a specific split."""
        print('\n* Starting epoch %d, split %s' % (epoch, split), '(train: ' + str(train) + ')')
        self.reset_epoch_reports()
        self.model.set_train(train)

        timestamp = time.time()
        reader_iter = self.reader.iter_batches(split, self.batch_size, shuffle=train, partial_batching=not train,
                                               threads=self.threads, max_batches=self.max_batches)
        for i in range(self.reader.get_size(split)):
            # Sequence length for deciding actions: int(np.ceil((seq_len - t_diff_max) / 2))
            obs, actions, rewards = self.emulator_state
            obs = obs[:, self.simulation_start:]
            actions = actions[:, self.simulation_start:]
            rewards = rewards[:, self.simulation_start:]
            obs, actions, rewards = self.model.prepare_batch([obs, actions, rewards])

            self.model.set_train(False)
            with torch.no_grad():
                q = self.model.model.compute_q(obs, actions)
            actions = torch.argmax(q, dim=1).cpu().numpy()
            self.model.set_train(True)

            self.emulator_state = next(self.emulator_iter).get_next(actions)[:3]
            self.reader.add(self.emulator_state)  # add trajectory to replay buffer

            report = self.clean_report(self.run_batch(next(reader_iter), train=train))
            if self.model.get_train_steps() % self.flags.freeze_every == 0:
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
        self.post_epoch_visualize(epoch, split)
