# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
from buffer import ReplayBuffer
import torch
import random
import numpy as np

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
device = torch.device("cpu")               # hard‑code CPU

BUFFER_SIZE = int(1e6) # replay buffer size
BATCH_SIZE =  256      # mini batch size
GAMMA = 0.99           # discount factor
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
HIDDEN_DIM_ACTOR = [256, 256]
HIDDEN_DIM_CRITIC = [256, 256]
MAX_NOISE_STEPS = 30000

class MADDPG:
    def __init__(self, state_size, action_size, num_agents, seed, noise_weight, noise_decay, update_every) :
        """
        Initialize a MADDPG trainer to manage multiple agents.
        :param state_size: Observation space size for each agent
        :param action_size: Action space size for each agent
        :param num_agents: Total number of agents in the environment
        :param seed: Random seed
        """
        self.agents = [DDPGAgent(state_size, action_size, i, num_agents, HIDDEN_DIM_ACTOR, HIDDEN_DIM_CRITIC, seed, LR_ACTOR, LR_CRITIC) for i in range(num_agents)]
        self.num_agents = num_agents
        self.seed = random.seed(seed)
        self.step_count = 0
        self.update_every = update_every

        self.noise_on = True
        self.noise_weight = noise_weight
        self.noise_decay = noise_decay

        self.gamma = GAMMA
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        
    def reset(self):
        """
        Reset all agents noise processes
        """
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        """ Get actions from all agents based on current states."""
        actions = [agent.act(state, noise_weight=self.noise_weight, add_noise=self.noise_on) for agent, state in zip(self.agents, states)]
        self.noise_weight *= self.noise_decay
        return np.stack(actions)

    def step(self, states, actions, rewards, next_states, dones):
        """
        Save experience in replay buffer and trigger learning if there are enough samples
        """
        self.step_count += 1
        if self.step_count > MAX_NOISE_STEPS:
            self.noise_on = False;
        
        self.memory.add(states, actions, rewards, next_states, dones)

        if self.step_count % self.update_every == 0:
            if len(self.memory) > self.memory.batch_size:
                for agent in self.agents:
                    experiences = self.memory.sample()
                    agent.learn(experiences, self.agents, self.gamma)
               
            




