## Reinforcement Learning - Training using Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

### Background

Train two agents to control rackets to bounce a tennis ball over a net using Multi-Agent Deep Deterministic Policy Gradient Reinforcement Learning.

![Tennis Environment](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

 This Tennis environment has been kindly provided by Unity is provided by [Unity Machine Learning agents]([https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)). In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

#### State Space

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. **Each agent receives its own, local observation**.  The observation vector for each agent corresponds to a vector of size 24 potentially corresponding to 3 stacked observations to express temporal and spatial directional aspects. 

#### Continuous Action Space
Two continuous actions for each agent, corresponding to movement toward (or away from) the net, and jumping.

#### When Solved

 After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. We take the maximum of these 2 scores to get a single score for each episode.
 
In order to solve this environment we must get an average score of +0.5 over 100 consecutive episodes. 

### Training Environment

We were provided with an Udacity version of the Unity Tennis environment.

### Getting Started


#### Install Dependencies and needed files

The project was finally completed using the Udacity's workspace environment with GPU disabled (torch device set to CPU). The workspace environment uses the Unity ML-Agent v0.4 interface. 

We would strongly recommend using the ML-Agent v0.4 interface as attempts to download the provided environment , grpcio, Pytorch modules with the latest version of Python installed either on mac-os or Google Colab were not successful.
##### Dependencies
- Python 3
- Pytorch
- Unity ML-Agents v0.4
- Jupyter 

### Instructions

#### Files
- `Tennis.ipynb` : The main Jupyter notebook to execute the training invoking the MADDPG trainer.
-  `maddpg.py`: The multi-agent training orchestrator implementation of the Actor Critic Reinforcement Learning as as outlined in the paper [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- `ddpg.py`: The individual agent implementation of the Actor Critic Reinforcement Learning as as outlined in the paper - essentially an adaptation of actor-critic methods that considers action policies of other agents.
- `model.py` : The neural networks for the Actor and Critic.
- `buffer.py`: Implementation of the ReplayBuffer to store experiences (s, a, r, s', done). Agents sample batches of these experiences to learn from.
- `OUNoise.py`: Implementation of the [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process). Added to continuous action result to facilitate exploration (in addition to exploitation).
- `actor_agent_{0,1}.pth`:  The trained checkpoints for the Actor model corresponding to each agent. Load the checkpoints to avoid training the model from scratch. 
- `critic_agent_{0,1}.pth`: The trained checkpoints for the Critic model corresponding to each agent. Load the checkpoints to avoid training the model from scratch.

#### Training Multiple Agents

Follow the steps in `Tennis.ipynb` to train your agents using MADDPG.
