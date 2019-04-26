import collections
import glob
from pathlib import Path
import pickle

import tensorflow as tf


def read(*log_files):
    plots = collections.defaultdict(list)
    for log_file in log_files:
        log_file = glob.glob('../logs/' + log_file + '/summary/*')[0]
        for e in tf.train.summary_iterator(log_file):
            step = e.step
            for v in e.summary.value:
                key, value = v.tag, v.simple_value
                keyed_list = plots[key]
                if len(keyed_list) > 0:
                    while step <= keyed_list[-1][0]:
                        keyed_list.pop()
                plots[key].append((step, value))
    return plots


def plot_compare(*plots):
    pass


if __name__ == '__main__':
    dump_file = 'tempdump.dat'
    if Path(dump_file).is_file():
        print('Loading from dump (delete', dump_file, 'to reread event files)')
        with open(dump_file, 'rb') as f:
            with_tdvae, without_tdvae, drqn = pickle.load(f)
        print('Loaded')
    else:
        print('Reading event files')
        with_tdvae = ('with_tdvae', read('with_tdvae', 'with_tdvae_cont'))
        without_tdvae = ('without_tdvae', read('without_tdvae', 'without_tdvae_cont'))
        drqn = ('drqn', read('drqn'))
        print('Dumping to file')
        with open(dump_file, 'wb') as f:
            pickle.dump((with_tdvae, without_tdvae, drqn), f)
        print('Dumpted and loaded')

    plot_compare(with_tdvae, without_tdvae, drqn)
