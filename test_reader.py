from pylego.misc import save_comparison_grid

from readers.gym_reader import GymReader, ReplayBuffer


if __name__ == '__main__':
    emulator = GymReader('Seaquest-v0', 6, 4, 2, 100)
    reader = ReplayBuffer(emulator, 5000, 100)

    print('EMULATOR:')
    for i, batch in enumerate(emulator.iter_batches('train', 4, max_batches=5, threads=2)):
        obs, actions, rewards = batch.get_next()[:3]
        print('obs', obs.size())
        print('actions', actions.shape)
        print('rewards', rewards.shape)
        print()
        if i < 3:
            batch = obs.numpy().reshape(obs.shape[0] * obs.shape[1], 3, 80, 80)
            save_comparison_grid('eseq%d.png' % i, batch, rows_cols=obs.shape[:2], retain_sequence=True)
            print(actions)
            print()

    print('REPLAY BUFFER READER:')
    for i, batch in enumerate(reader.iter_batches('train', 4, max_batches=5)):
        obs, actions, rewards = batch
        print('obs', obs.size())
        print('actions', actions.shape)
        print('rewards', rewards.shape)
        print()
        if i < 3:
            batch = obs.numpy().reshape(4 * 6, 3, 80, 80)
            save_comparison_grid('rseq%d.png' % i, batch, rows_cols=(4, 6), retain_sequence=True)
            print(actions)
            print()
