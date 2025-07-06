import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")               # hard‑code CPU

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_dim, action_dim, hidden_dim, seed=0, norm_type='batch'):
        """Initialize parameters and build model.
        Params
        ======
            input_dim (int): Input, State sizee
            action_dim (int): Output, Action size 
            hidden_dim (list): List of integers representing the number of nodes in each hidden layer
            norm_type (str): Type of normalization to apply ('batch' or 'layer')
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.norm_type = norm_type

        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.norm1 = self._get_norm(hidden_dim[0])

        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        
        self.fc_out = nn.Linear(hidden_dim[1], action_dim)
        self.activation = F.leaky_relu
    
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc_out.weight.data.uniform_(-3e-3, 3e-3)

    def _get_norm(self, num_features):
        """Get normalization layer based on the specified type."""
        if self.norm_type == 'batch':
            return nn.BatchNorm1d(num_features)
        elif self.norm_type == 'layer':
            return nn.LayerNorm(num_features)
        else:
            raise ValueError("Unsupported normalization type: {}".format(self.norm_type))


    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.norm1(self.activation(self.fc1(state)))
        x = self.activation(self.fc2(x))
        a = F.tanh(self.fc_out(x))
        return a


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_all_dim, action_all_dim, hidden_dim, output_dim, seed=0, norm_type='batch'):
        """Initialize parameters and build model.
        Params
        ======
            state_all_dim (int): Dimension of state space, input
            action_all_dim (int): Dimension of action space, input
            hidden_dim (list): List of integers representing the number of nodes in each hidden layer            
            output_dim (int): Size of output, should be 1
            norm_type (str): Type of normalization to apply ('batch' or 'layer')
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.norm_type = norm_type
        
        self.fc1 = nn.Linear(state_all_dim+action_all_dim, hidden_dim[0])
        self.norm1 = self._get_norm(hidden_dim[0])

        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.q_out = nn.Linear(hidden_dim[1], output_dim)
        self.activation = F.leaky_relu

        self.reset_parameters()

    def _get_norm(self, num_features):
        """Get normalization layer based on the specified type."""
        if self.norm_type == 'batch':
            return nn.BatchNorm1d(num_features)
        elif self.norm_type == 'layer':
            return nn.LayerNorm(num_features)
        else:
            raise ValueError("Unsupported normalization type: {}".format(self.norm_type))

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.q_out.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_all, action_all):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state_all, action_all), dim=1)
        x = self.norm1(self.activation(self.fc1(x)))
        x = self.activation(self.fc2(x))
        return (self.q_out(x))
