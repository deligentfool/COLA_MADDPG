# COLA-MADDPG
PyTorch Implementation of COLA-MADDPG based on [MADDPG-pytorch](https://github.com/shariqiqbal2810/maddpg-pytorch).

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/)
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard)

## How to Run

All training code is contained within `main.py`. To view options simply run:

```shell
python main.py --help
```

For vanilla MADDPG:

```shell
python main.py simple_tag_coop examplemodel --n_episodes 40000
```

For COLA-MADDPG:

```shell
python main.py simple_tag_coop examplemodel --n_episodes 40000 --consensus
```

## Multi-agent Particle Env

- To install, `cd` into `multiagent-particle-envs` directory and type `pip install -e .`

- To interactively view moving to landmark scenario (see others in ./scenarios/):
  `bin/interactive.py --scenario simple.py`

- Known dependencies: OpenAI gym, numpy

- To use the environments, look at the code for importing them in `make_env.py`.

## Scenarios

The three scenarios we used in the paper are `simple_tag_coop`, `simple_spread`, and `simple_reference_no_comm`. They correspond to "Cooperative Predator-Prey", "Cooperative Navigation" and "Cooperative Pantomime" in the text, respectively.

## Acknowledgements

The OpenAI baselines [Tensorflow implementation](https://github.com/openai/baselines/tree/master/baselines/ddpg) and Ilya Kostrikov's [Pytorch implementation](https://github.com/ikostrikov/pytorch-ddpg-naf) of DDPG were used as references. After the majority of this codebase was complete, OpenAI released their [code](https://github.com/openai/maddpg) for MADDPG, and I made some tweaks to this repo to reflect some of the details in their implementation (e.g. gradient norm clipping and policy regularization).
