import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import Actor, Critic

WEIGHT_DECAY = 0.0  # weight decay for critic optimizer

# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")               # hard‑code CPU

class DDPGAgent:
    def __init__(self, state_dim, action_dim, agent_id, num_agents, hidden_dim_actor, hidden_dim_critic, seed, lr_actor, lr_critic):
        """
        Initialize a DDPG agent.
        :param state_dim: Dimension of each state
        :param action_dim: Dimension of each action
        :param agent_id: Unique ID for the agent (used for indexing)
        :param seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.seed = seed

        super(DDPGAgent, self).__init__()

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_dim, action_dim, hidden_dim_actor, seed, norm_type='batch').to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim_actor, seed, norm_type='batch').to(device)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network (w/ Target Network)
        # Each critic in MADPDG env observes state for all agents.
        self.critic_local = Critic(state_dim*num_agents, action_dim*num_agents, hidden_dim_critic, 1,
                             seed, norm_type='batch').to(device)
        self.critic_target = Critic(state_dim*num_agents, action_dim*num_agents, hidden_dim_critic, 1,
                                    seed, norm_type='batch').to(device)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=WEIGHT_DECAY)

        self.agent_id = agent_id

         # Noise process
        self.noise = OUNoise(action_dim, seed)

        # Initialize targets same as original networks
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
    
        
    def act(self, state, noise_weight, add_noise=True):
        """
        Returns actions for given state as per current policy.
        :param state: Current state (numpy array)
        :param noise_weight: Weight for the noise to be added to the action
        :param add_noise: Whether to add noise to the action for exploration
        :return: Action to be taken (numpy array)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        
        self.actor_local.train()
        
        if add_noise:
            action += (self.noise.sample() * noise_weight)

        return np.clip(action.squeeze(0), -1, 1)


    def reset(self):
        """
        Resets the internal state of the noise process to its initial state.

        This method is typically called at the beginning of each episode to ensure that the noise added to actions is uncorrelated across episodes.
        """
        self.noise.reset()


    def learn(self, experiences, all_agents, gamma=0.99):
        """
        Update this agent's actor and critic using a random sample of experience.
        experiences  : tuple of (states, actions, rewards, next_states, dones)
        all_agents   : list of *all* DDPGAgent objects (needed for joint tensors)
        gamma        : discount factor
        """
        states, actions, rewards, next_states, dones = experiences    # unpack

        B = states.size(0)                                            # batch size
        # The ReplayBuffer.sample() mini-batch has the following dimensions
        # B  - batch_size
        # N  - number of agents
        # S  - per agent state size (state_dim - 24)
        # A  - per agent action size (action_dim - 2)
        # 
        # states      : (B, N, S)
        # actions     : (B, N, A)
        # rewards     : (B, N)
        # next_states : (B, N, S)
        # dones       : (B, N)

        # ******************** Update Critic ************************ #

        # -----------------------------------------------------------
        # 1. Critic target:   y_i = r_i + γ * Q_i'(s', mu_1'(o'_1),…,mu_N'(o'_N))
        # -----------------------------------------------------------
        # Next actions from target actors (no gradients needed)
        # Result list of length N with each tensor shape  (B, A)
        next_actions = [ag.actor_target(next_states[:, j, :])          # (B,A)
                        for j, ag in enumerate(all_agents)]

        # Concatenate along the action dimesion (B, N.A) 
        next_actions_flat = torch.cat(next_actions, dim=1)                

        # Flatten state from (B,N,S) -> (B,N·S) using Pytorch view
        next_states_flat = next_states.view(B, -1)

        with torch.no_grad():                                          # no grad on target
            Q_targets_next = self.critic_target(next_states_flat, next_actions_flat)  # (B,1)

        # Select this agent's reward / done columns, shape (B,1)
        r_i    = rewards[:, self.agent_id].unsqueeze(1)
        done_i = dones[:,   self.agent_id].unsqueeze(1).float()

        Q_targets = r_i + gamma * Q_targets_next * (1 - done_i)              # (B,1)

        # -----------------------------------------------------------
        # 2. Critic loss:  MSE(Q_i(s, a) , Q_targets)
        # ---------------------------------------------------------
        states_flat  = states.view(B, -1)                              # (B,N·S)
        actions_flat = actions.view(B, -1)                             # (B,N·A)

        Q_expected = self.critic_local(states_flat, actions_flat)          # (B,1)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # ********************** Update Actor ************************ #
        # -----------------------------------------------------------
        # 3. Actor loss:  -E[ Q_i(s, mu_i(o_i), a_{-i}) ]
        #    Only this agent's action is differentiable.
        # -----------------------------------------------------------
        actions_pred = [] 
        for j, ag in enumerate(all_agents):
            a_j = ag.actor_local(states[:, j, :])                            # (B,A)
            if j != self.agent_id:
                a_j = a_j.detach()                                     # stop grad
            actions_pred.append(a_j)
        actions_pred_flat = torch.cat(actions_pred, dim=1)                  # (B,N·A)

        actor_loss = -self.critic_local(states_flat, actions_pred_flat).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # ******************** Update Target Networks **************** #
        # -----------------------------------------------------------
        # 4. Soft‑update the target networks
        # ---------------------------------------------------------
        self.soft_update(self.critic_local, self.critic_target, tau=1e-3)
        self.soft_update(self.actor_local, self.actor_target,   tau=1e-3)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = tau*theta_local + (1 - tau)*theta_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


