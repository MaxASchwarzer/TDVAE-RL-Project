import importlib

from readers.gym_reader import GymReader
from pylego import misc, runner


class BaseRunner(runner.Runner):

    def __init__(self, flags, model_class, log_keys, *args, **kwargs):
        self.flags = flags
        reader = GymReader(flags.env, flags.seq_len, flags.batch_size, flags.threads, flags.iters_per_epoch)

        summary_dir = flags.log_dir + '/summary'
        super().__init__(reader, flags.batch_size, flags.epochs, summary_dir, log_keys=log_keys,
                         threads=flags.threads, print_every=flags.print_every, visualize_every=flags.visualize_every,
                         max_batches=flags.max_batches, *args, **kwargs)
        model_class = misc.get_subclass(importlib.import_module('models.' + self.flags.model), model_class)
        self.model = model_class(self.flags, optimizer=flags.optimizer, learning_rate=flags.learning_rate,
                                 cuda=flags.cuda, load_file=flags.load_file, save_every=flags.save_every,
                                 save_file=flags.save_file, debug=flags.debug)
