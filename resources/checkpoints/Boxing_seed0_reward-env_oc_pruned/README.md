---
language: en
tags:
- stable-baselines3
- reinforcement-learning
- boxing
- object centric
model:
  seed: 0
  reward_function: env
  observation_type: object centric
  pruned: pruned_boxing.yaml
  normalized: Yes
  num_timesteps: 20004864
  total_timesteps: 20000000
  sde_sample_freq: -1
  n_envs: 8
  n_epochs: 8
  gae_lambda: 0.95
  n_steps: 2048
  batch_size: 256
  gamma: 0.99
  ent_coef: 0.01
license: mit
---
# Model Card for Boxing

## Overview
This model is trained for the **Boxing** game. It was trained using the following configurations:

- **Seed**: 0
- **Reward Function**: env
- **Observation Type**: object centric
- **Pruned**: pruned_boxing.yaml

## Training Details
- **Framework**: Stable-Baselines3 (SB3)
- **Policy**: ActorCriticPolicy
- **Number of Timesteps**: 20004864
- **Total Timesteps**: 20000000
- **SDE Sample Frequency**: -1
- **Number of Environments**: 8
- **Number of Epochs**: 8
- **GAE Lambda**: 0.95
- **Number of Steps**: 2048
- **Batch Size**: 256
- **Gamma**: 0.99
- **Ent_coef**: 0.01
- **Reward Details**: env

## Usage
For more detailed usage, visit https://github.com/k4ntz/SCoBots and its README which explains in detail how to use SCoBots

## License
This model is released under the MIT License. See the LICENSE file for more details.
