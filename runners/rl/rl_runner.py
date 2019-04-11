from models.baseconditional import BaseGymTDVAE
from .base_rl_runner import BaseRLRunner


class GymRLRunner(BaseRLRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseGymTDVAE, ['loss', 'rl_loss', 'bce_diff', 'kl_div_qs_pb', 'kl_shift_qb_pt'])

    def run_batch(self, batch, train=False):
        images, actions, rewards, done, t1, t2, is_weight = self.model.prepare_batch(batch[:7])
        replay_indices = batch[7]
        report = self.model.run_loss([images, actions, t1, t2], labels=(rewards, is_weight, done))
        rl_errors = report.pop('rl_errors', None).cpu().numpy()
        self.reader.update(replay_indices, rl_errors)
        if train:
            self.model.train(report['loss'], clip_grad_norm=self.flags.grad_norm)
        return report
