import collections
import glob
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


colors = ['#d11414', '#4fa51a', '#1765b7', '#3421dd', '#8221dd', '#c91cbd', '#cebc31', '#18a56f', '#18a8af']


def running_mean(x, N):
    after = N // 2
    before = N - after
    cumsum = np.cumsum(np.concatenate([np.zeros([N], dtype=x.dtype), x, np.zeros([N], dtype=x.dtype)], axis=0))
    ret = (cumsum[N:] - cumsum[:-N]) / float(N)
    return ret[before:-after]


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


def plot_compare(*plots, smoothing_window=50):
    keys = set()
    for plot in plots:
        keys.update(plot[1].keys())
    for key in keys:
        if not key.endswith('_mean'):
            continue
        fig = plt.figure(figsize=(6.75, 5.0))
        ax = fig.gca()
        xlim = np.inf
        for i_plot, plot in enumerate(plots):
            name, data = plot
            key_data = data[key]
            x, y = zip(*key_data)
            x = np.array(x) / 1000
            means = running_mean(np.array(y), smoothing_window)
            xlim = min(xlim, x[-1])
            if key.endswith('_mean'):
                std_key = key[:-len('_mean')] + '_std'
                stds = running_mean(np.array([p[1] for p in data[std_key]]), smoothing_window)
            else:
                stds = None
            plt.plot(x, means, color=colors[i_plot], label='%s' % name)
            if stds is not None:
                plt.fill_between(x, means - stds, means + stds, color=colors[i_plot], alpha=0.2, linewidth=0)

        ax.set_xlim([0, xlim])
        plt.legend(loc='lower right', fontsize='small')
        plt.xlabel('Iteration (x1000)')
        plt.ylabel('Sum of rewards')
        plt.title('%s' % key)
        plt.show()
        # plt.savefig('figures/logs_%s_%s_%d.pdf' % (best_using, set_str, i_gamma), bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    dump_file = 'tempdump.dat'
    if Path(dump_file).is_file():
        print('Loading from dump (delete', dump_file, 'to re-read event files)')
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
    print()

    print('Plotting')
    plot_compare(with_tdvae, without_tdvae, drqn)
    print('All done')
