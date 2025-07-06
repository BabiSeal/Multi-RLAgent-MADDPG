import numpy as np
import random
import torch
import copy

device = torch.device("cpu")               # hard‑code CPU

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:

    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        """
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        Update internal state and return it as a noise sample
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
 