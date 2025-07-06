from collections import namedtuple, deque
import random
import numpy as np
import torch  

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
device = torch.device("cpu")               # hard‑code CPU

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
          
           :param buffer_size (int): maximum size of buffer
           :param batch_size (int): size of each training batch
           :param seed: random seed 
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory.
           :param  states: Current states for all agents
           :param  actions: Actions taken by all agents
           :param  rewards: Rewards received by all agents
           :param  next_states: Next states observed by all agents
           :param  dones: Episode done flags for all agents
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.as_tensor(np.stack([e.state for e in experiences if e is not None]), device=device, dtype=torch.float32)
        actions = torch.as_tensor(np.stack([e.action for e in experiences if e is not None]), device=device, dtype=torch.float32)
        rewards = torch.as_tensor(np.stack([e.reward for e in experiences if e is not None]), device=device, dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([e.next_state for e in experiences if e is not None]),  device=device, dtype=torch.float32)
        dones = torch.as_tensor(np.stack([e.done for e in experiences if e is not None]), device=device, dtype=torch.float32)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


