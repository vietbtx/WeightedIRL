# WeightedIRL
The official code of the paper "Enhancing Inverse Reinforcement Learning with Weighted Causal Entropy"
# Abstract
We study inverse reinforcement learning (IRL), the problem of recovering a reward function from expert's demonstrated trajectories. We propose a way to enhance IRL by adding a weight function to the maximum causal entropy framework, with the motivation of having the ability to control and learn the stochasticity of the modelled policy. Our IRL framework and algorithms allow to learn both a reward function and the structure of the entropy terms added to the Markov Decision Processes, thus enhancing the IRL procedure. Our numerical experiments using human and simulated demonstrations and with discrete and continuous IRL tasks show that our approach outperforms prior methods.

## Setup
This repo tested on MuJoCo v1.3.1, please make sure install MuJoCo first. Note that you need a MuJoCo license. Please follow the instruction in [mujoco-py](https://github.com/openai/mujoco-py) for help.
Run `pip install -r requirements.txt` for installing Python libraries.

## Train Expert
We use Soft Actor-Critic(SAC)[^sac] for training experts and collecting demonstrations:
- Run `python train_expert.py --env_id <env> --num_steps <numb-of-steps>` 
- Run `python collect_demo.py --env_id <env> --weight <pretrained-expert-path>`

## Train Imitation
Each expert tested with 6 imitation algorithms based on 2 existed studies: GAIL[^gail] and AIRL[^airl]
- `python train_imitation.py --env_id <env> --test_env_id <eval-env> --seed <seed> <arguments>`

| Algorithms | Arguments |
| --- | --- |
| GAIL | `--algo gail` |
| Weighted GAIL | `--algo gail --weighted` |
| AIRL | `--algo airl` |
| Weighted AIRL | `--algo airl --weighted` |
| AIRL state-only | `--algo airl --state_only` |
| Weighted AIRL state-only | `--algo airl --weighted --state_only` |

[^sac]: Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML.

[^gail]: Ho, Jonathan and Stefano Ermon. “Generative Adversarial Imitation Learning.” NIPS (2016).

[^airl]: Fu, J., Luo, K., & Levine, S. (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning. ArXiv, abs/1710.11248.

## Examples
Checking *run_mujoco.cmd*, *run_disabled_ant.cmd*, *run_point_maze.cmd* for more details.

## Experimental Settings
We evaluate on Mujoco tasks and transfer learning tasks with 8 different seeds without tunning hyperparameters.

## Visualization

- Run `python benchmark_score.py` for generating improvement graphs over 1M steps
<p align="center">
<img width="420" alt="Screenshot 2022-08-12 at 7 49 03 AM" src="https://user-images.githubusercontent.com/13542863/184260871-bc751159-ab77-4d26-bd59-afc7103150d1.png">
</p>

- Run `python create_heatmap.py` for showing the heat map of Point Mass-Maze environment

<p align="center">
<img width="360" alt="Screenshot 2022-08-12 at 7 56 28 AM" src="https://user-images.githubusercontent.com/13542863/184261347-0c509fc6-3b38-4b58-9d9b-fbf3fa3ea766.png">
</p>

## Notes:
- [x] This code is based on gail-airl-ppo.pytorch repository [^ku2482].
- [x] The transfer environments are pulled from the official AIRL repository [^justinjfu]

[^ku2482]: PyTorch implementation of GAIL and AIRL based on PPO. https://github.com/ku2482/gail-airl-ppo.pytorch
[^justinjfu]: Implementations for imitation learning/IRL algorithms in RLLAB. https://github.com/justinjfu/inverse_rl/tree/master/inverse_rl/envs


