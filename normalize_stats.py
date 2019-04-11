import os
from pathlib import Path
import pickle

import numpy as np

from pylego import misc

from readers.gym_reader import GymReader, ReplayBuffer


GAME = 'Seaquest-v0'
DATA_DIR = 'data/' + GAME

def get_batches(batches_fname):
    if not Path(batches_fname).is_file():
        emulator = GymReader(GAME, 2, 16, 6, np.inf, raw=True)
        reader = ReplayBuffer(emulator, 5000, 512, 0, 0)

        batches = []
        print('* Collecting batches')
        for batch in reader.iter_batches('train', 16, max_batches=256):
            batch = batch[0].numpy()
            batch = batch.reshape(-1, *batch.shape[2:])
            batches.append(batch)

        print('* Dumping batches')
        batches = np.concatenate(batches, 0)
        with open(batches_fname, 'wb') as f:
            pickle.dump(batches, f)
    else:
        with open(batches_fname, 'rb') as f:
            batches = pickle.load(f)
    return batches


if __name__ == '__main__':
    try:
        os.makedirs(DATA_DIR)
    except IOError as e:
        pass

    batches = get_batches(DATA_DIR + '/norm_batches.pk')
    misc.save_comparison_grid('example1.png', batches[:16], border_shade=0.8)
    h, w = batches.shape[2:]
    crop_top, crop_bottom, crop_left, crop_right = 46, 35, 8, 0
    batches = batches[:, :, crop_top:h-crop_bottom, crop_left:w-crop_right]
    flat_batches = batches.transpose(1, 0, 2, 3).reshape(3, -1)
    mean = flat_batches.mean(axis=1)[None, :, None, None]
    std = flat_batches.std(axis=1)[None, :, None, None]
    batches -= mean
    batches /= std
    # batches = batches.mean(axis=1, keepdims=True)  # greyscale
    true_min = batches.min()
    true_max = batches.max()
    print(true_min, true_max)
    bmin = np.percentile(batches, 2)
    bmax = np.percentile(batches, 98)
    print(bmin, bmax)
    batches = np.clip(batches, bmin, bmax)
    batches -= batches.min()
    batches /= batches.max()
    misc.save_comparison_grid('example2.png', batches[:16], border_shade=0.8)

    with open(DATA_DIR + '/img_stats.pk', 'wb') as f:
        pickle.dump([mean, std, bmin, bmax, true_min, true_max, crop_top, crop_bottom, crop_left, crop_right], f)
    print('* Stats dumped!')
