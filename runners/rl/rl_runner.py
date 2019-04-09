from models.baseconditional import BaseGymTDVAE
from ..conditional.base_runner import BaseRunner


class GymRLRunner(BaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseGymTDVAE, ['loss', 'rl_loss', 'bce_diff', 'kl_div_qs_pb', 'kl_shift_qb_pt'])

    def run_batch(self, batch, train=False):
        data = batch.get_next()
        images, actions, rewards = self.model.prepare_batch(data[:3])
        report = self.model.run_loss([images, actions], labels=rewards)
        if train:
            self.model.train(report['loss'], clip_grad_norm=self.flags.grad_norm)
        return report
