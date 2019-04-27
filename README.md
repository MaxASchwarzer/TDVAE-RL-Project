# TD-VAE

TD-VAE implementation in PyTorch 1.0.

This code implements the ideas presented in the paper [Temporal Difference Variational Auto-Encoder (Gregor et al)][2]. This implementation includes configurable number of stochastic layers as well as the specific multilayer RNN design proposed in the paper.

**NOTE**: This implementation also makes use of [`pylego`][1], which is a minimal library to write easily extendable experimental machine learning code.

[1]: https://github.com/ankitkv/pylego
[2]: https://arxiv.org/abs/1806.03107

## Replication
To replicate our results:
1.  For model-free, run `python main.py --model conditional.tdvae --name tdqvae`
2.  For the DRQN baseline, run `python main.py --model conditional.drqn --name dqn`
2.  For model-based, run `python main.py --model conditional.modeltdvae --tdvae_weight 1 --rl_weight 10 --mpc --eps_decay_end 1 --name mpc`
