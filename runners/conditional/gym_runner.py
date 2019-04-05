import collections
import time
from abc import ABC, abstractmethod
import numpy as np

from tensorboardX import SummaryWriter

from pylego import misc, runner

from models.baseconditional import BaseGymTDVAE
from .base_runner import BaseRunner


class GymRunner(BaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseGymTDVAE, ['loss', 'bce_diff', 'kl_div_qs_pb', 'kl_shift_qb_pt'])
        self.maxlen = flags.seq_len
        self.adv_start = flags.d_start
        self.d_weight = flags.d_weight

    def run_batch(self, batch, train=False):
        images, actions = self.model.prepare_batch(batch[0:2])
        images = images.contiguous()
        loss, bce_diff, kl_div_qs_pb, kl_shift_qb_pt, bce_optimal = self.model.run_loss((images, actions))
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        return collections.OrderedDict([('loss', loss.item()),
                                        ('bce_diff', bce_diff.item()),
                                        ('kl_div_qs_pb', kl_div_qs_pb.item()),
                                        ('kl_shift_qb_pt', kl_shift_qb_pt.item()),
                                        ('bce_optimal', bce_optimal.item())])

    def _visualize_split(self, split, t, n):
        bs = min(self.batch_size, 15)
        batch = self.reader.iter_batches(split, bs, shuffle=True, partial_batching=True, threads=self.threads,
                                              max_batches=1)
        images = batch[0]
        actions = batch[1]
        images, actions = self.model.prepare_batch([images[:, :t + 2], actions[:, :t+2]])
        out = self.model.run_batch([images, t, n, actions], visualize=True)

        batch = images.cpu().numpy()
        out = out.cpu().numpy().reshape(out.shape[0], out.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        vis_data = np.concatenate([batch, out], axis=1)
        bs, seq_len = vis_data.shape[:2]
        vis_data = vis_data.reshape(bs*seq_len, batch.shape[2], batch.shape[3], batch.shape[4])
        return vis_data, (bs, seq_len)

    def post_epoch_visualize(self, epoch, split):
        print('* Visualizing', split)
        vis_data, aspect = self._visualize_split(split, min(10, self.maxlen - 1), 5)
        if split == 'test':
            fname = self.flags.log_dir + '/test.png'
        else:
            fname = self.flags.log_dir + '/{}'.format(split) + '%03d.png' % epoch
        misc.save_comparison_grid(fname, vis_data, rows_cols=aspect, border_shade=1.0, retain_sequence=True)
        print('* Visualizations saved to', fname)

        if split == 'test':
            print('* Generating more visualizations for', split)
            vis_data, aspect = self._visualize_split(split, min(10, self.maxlen - 1), 5)
            fname = self.flags.log_dir + '/test_more.png'
            misc.save_comparison_grid(fname, vis_data, rows_cols=aspect, border_shade=1.0, retain_sequence=True)
            print('* More visualizations saved to', fname)

    def run_epoch(self, epoch, split, train=False, log=True):
        """Iterates the epoch data for a specific split."""
        print('\n* Starting epoch %d, split %s' % (epoch, split), '(train: ' + str(train) + ')')
        self.reset_epoch_reports()
        self.model.set_train(train)

        timestamp = time.time()
        i = 0
        if epoch > self.adv_start:
            self.model.d_weight = self.d_weight
        else:
            self.model.d_weight = 0
        while not self.reader.done:
            i += 1
            batch = self.reader.iter_batches(split, self.batch_size, shuffle=train,
                                             partial_batching=not train, threads=self.threads)
            ret_report = self.run_batch(batch, train=train)
            if not ret_report:
                report = collections.OrderedDict()
            elif not isinstance(ret_report, collections.OrderedDict):
                report = collections.OrderedDict()
                for k in sorted(ret_report.keys()):
                    report[k] = ret_report[k]
            else:
                report = ret_report

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

        self.reader.reset()
