import argparse
import os

from pylego.misc import add_argument as arg

from runners.imgtdvae.tdvaerunner import TDVAERunner
from runners.conditional.gym_runner import GymRunner
from runners.rl.rl_runner import GymRLRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg(parser, 'name', type=str, required=True, help='name of the experiment')
    arg(parser, 'model', type=str, default='conditional.gymtdvae', help='model to use')
    arg(parser, 'cuda', type=bool, default=True, help='enable CUDA')
    arg(parser, 'load_file', type=str, default='', help='file to load model from')
    arg(parser, 'save_file', type=str, default='model.dat', help='model save file')
    arg(parser, 'save_every', type=int, default=2500, help='save every these many global steps (-1 to disable saving)')
    arg(parser, 'data_path', type=str, default='data/MNIST')
    arg(parser, 'data', type=str, default='gym', help="Data source to use.  Set to gym and set env flag for gym.")
    arg(parser, 'env', type=str, default='Seaquest-v0', help="Gym environment to use (if data=gym)")
    arg(parser, 'iters_per_epoch', type=int, default=100, help="Number of batches per epoch if in Gym.")
    arg(parser, 'logs_path', type=str, default='logs')
    arg(parser, 'force_logs', type=bool, default=False)
    arg(parser, 'optimizer', type=str, default='adam', help='one of: adam')
    arg(parser, 'learning_rate', type=float, default=5e-4, help='-1 to use model default')
    arg(parser, 'd_lr', type=float, default=1e-3, help='LR for discriminator, if in use')
    arg(parser, 'd_steps', type=int, default=1, help='Disc steps per Gen step.')
    arg(parser, 'd_start', type=int, default=5, help='Epochs before generator starts to train on disc. loss.')
    arg(parser, 'beta', type=float, default=1, help='Parameter controlling KL loss scale')
    arg(parser, 'd_weight', type=float, default=10, help='Parameter for discriminator loss scale')
    arg(parser, 'tdvae_weight', type=float, default=1.0, help='Parameter for TDVAE loss scale')
    arg(parser, 'rl_weight', type=float, default=1.0, help='Parameter for DQN loss scale')
    arg(parser, 'grad_norm', type=float, default=5.0, help='gradient norm clipping (-1 to disable)')
    arg(parser, 'adversarial', type=bool, default=False, help='Use an auxiliary adversarial loss on reconstructions')
    arg(parser, 'rl', type=bool, default=False, help='Do RL')
    arg(parser, 'seq_len', type=int, default=20, help='sequence length')
    arg(parser, 'batch_size', type=int, default=64, help='batch size')
    arg(parser, 'replay_size', type=int, default=10000, help='Experience replay buffer size')
    arg(parser, 'freeze_every', type=int, default=5000, help='Freeze a Q network every this many steps')
    arg(parser, 'samples_per_seq', type=int, default=1, help='(t1, t2) samples per input sequence')
    arg(parser, 'discount_factor', type=float, default=0.99, help='RL discount factor (aka gamma)')
    arg(parser, 'eps_decay_start', type=int, default=2500, help='Iteration to start decaying epsilon at')
    arg(parser, 'eps_decay_end', type=int, default=100000, help='Iteration to stop decaying epsilon at')
    arg(parser, 'eps_final', type=float, default=0.01, help='Final epsilon for epsilon-greedy')
    arg(parser, 'h_size', type=int, default=32, help='Base #channels for resnets before downscaling.')
    arg(parser, 'd_size', type=int, default=16, help='Base #channels for discriminator before downscaling.')
    arg(parser, 'b_size', type=int, default=128, help='belief size')
    arg(parser, 'z_size', type=int, default=32, help='state size')
    arg(parser, 'layers', type=int, default=2, help='number of layers')
    arg(parser, 't_diff_min', type=int, default=1, help='minimum time difference t2-t1')
    arg(parser, 't_diff_max', type=int, default=4, help='maximum time difference t2-t1')
    arg(parser, 't_diff_max_poss', type=int, default=10, help="""Maximum time difference across
                                                                entire curriculum, not just current run.
                                                                Needed to set weight dims correctly.""")
    arg(parser, 'epochs', type=int, default=50000, help='no. of training epochs')
    arg(parser, 'max_batches', type=int, default=-1, help='max batches per split (if not -1, for debugging)')
    arg(parser, 'print_every', type=int, default=10, help='print losses every these many steps')
    arg(parser, 'gpus', type=str, default='0')
    arg(parser, 'threads', type=int, default=-1, help='data processing threads (-1 to determine from CPUs)')
    arg(parser, 'debug', type=bool, default=False, help='run model in debug mode')
    arg(parser, 'visualize_every', type=int, default=-1,
        help='visualize during training every these many steps (-1 to disable)')
    arg(parser, 'visualize_only', type=bool, default=False, help='epoch visualize the loaded model and exit')
    arg(parser, 'visualize_split', type=str, default='test', help='split to visualize with visualize_only')
    flags = parser.parse_args()
    if flags.threads < 0:
        flags.threads = max(1, len(os.sched_getaffinity(0)) - 1)
    if flags.grad_norm < 0:
        flags.grad_norm = None

    iters = 0
    while True:
        if iters == 4:
            raise IOError("Too many retries, choose a different name.")
        flags.log_dir = '{}/{}'.format(flags.logs_path, flags.name)
        try:
            print('* Creating log dir', flags.log_dir)
            os.makedirs(flags.log_dir)
            break
        except IOError as e:
            if flags.force_logs:
                print('*', flags.log_dir, 'not recreated')
                break
            else:
                print('*', flags.log_dir, 'already exists')
                flags.name = flags.name + "_"
        iters += 1

    print('Arguments:', flags)
    if flags.visualize_only and not flags.load_file:
        print('! WARNING: visualize_only without load_file!')

    if flags.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpus

    flags.save_file = flags.log_dir + '/' + flags.save_file

    if flags.model.startswith('conditional.'):
        if flags.rl:
            runner = GymRLRunner
        else:
            runner = GymRunner
        val_split = None
        test_split = None
    elif flags.model.startswith('tdvae.'):
        runner = TDVAERunner
        val_split = 'val'
        test_split = 'test'
    runner(flags).run(val_split=val_split, test_split=test_split, visualize_only=flags.visualize_only,
                      visualize_split=flags.visualize_split)
