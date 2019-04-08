from pylego.misc import save_comparison_grid

from readers.gym_reader import GymReader, ReplayBuffer


if __name__ == '__main__':
    emulator = GymReader('Seaquest-v0', 6, 4, 2, 100)
    reader = ReplayBuffer(emulator, 50)

    for i, batch in enumerate(reader.iter_batches('train', 4, max_batches=5)):
        print(len(batch))
        obs, actions, rewards = batch
        print('obs', obs.size())
        print('actions', actions.shape)
        print('rewards', rewards.shape)
        print()
        if i < 3:
            batch = batch[0].numpy().reshape(4 * 6, 3, 112, 80)
            save_comparison_grid('seq%d.png' % i, batch)
