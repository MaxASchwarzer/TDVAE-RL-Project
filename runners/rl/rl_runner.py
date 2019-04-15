import numpy as np

from pylego import misc

from models.baseconditional import BaseGymTDVAE
from .base_rl_runner import BaseRLRunner


class GymRLRunner(BaseRLRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseGymTDVAE, ['loss', 'rl_loss', 'bce_diff', 'kl_div_qs_pb', 'kl_shift_qb_pt'])

    def run_batch(self, batch, train=False):
        images, actions, rewards, done, t1, t2, returns, is_weight = self.model.prepare_batch(batch[:8])
        replay_indices = batch[8]
        report = self.model.run_loss([images, actions, rewards, done, t1, t2, returns], labels=is_weight)
        rl_errors = report.pop('rl_errors', None).cpu().numpy()
        self.reader.update(replay_indices, rl_errors)
        if train:
            self.model.train(report['loss'], clip_grad_norm=self.flags.grad_norm)
        return report

    def _visualize_split(self, split, t, n):
        bs = min(self.batch_size, 15)
        batch = next(self.reader.iter_batches(split, bs, shuffle=True, partial_batching=True, threads=self.threads,
                                              max_batches=1))
        cond_batch = self.emulator.action_conditional_batch
        mean, std, imin, imax = (cond_batch.img_mean.numpy(), cond_batch.img_std.numpy(), cond_batch.img_true_min,
                                 cond_batch.img_true_max)
        images, actions, rewards, done = batch[:4]
        images, actions, rewards, done = self.model.prepare_batch([images[:, :t + 2], actions[:, :t + 2],
                                                                   rewards[:, :t + 2], done[:, :t + 2]])
        out = self.model.run_batch([images, t, n, actions, rewards, done], visualize=True)

        batch = images.cpu().numpy()
        out = out.cpu().numpy().reshape(out.shape[0], out.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        vis_data = np.concatenate([batch, out], axis=1)
        bs, seq_len = vis_data.shape[:2]
        vis_data = vis_data.reshape(bs*seq_len, batch.shape[2], batch.shape[3], batch.shape[4])
        vis_data = np.clip((vis_data * (imax - imin) + imin) * std + mean, 0.0, 1.0)
        return vis_data, (bs, seq_len)

    def post_epoch_visualize(self, epoch, split):
        print('* Visualizing', split)
        vis_data, aspect = self._visualize_split(split, self.history_length, 1)  # FIXME n is 1
        fname = self.flags.log_dir + '/{}'.format(split) + '%03d.png' % epoch
        misc.save_comparison_grid(fname, vis_data, rows_cols=aspect, border_shade=1.0, retain_sequence=True)
        print('* Visualizations saved to', fname)
