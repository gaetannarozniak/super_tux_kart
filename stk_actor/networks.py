import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ELU())
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = x
        out = self.fc(x)
        if identity.size(1) != out.size(1):
            identity = self.projection(identity)
        out += identity
        return out


class CriticNetwork(nn.Module):

    def __init__(
        self,
        beta,
        input_dims,
        n_actions,
        fc1_dims=256,
        fc2_dims=256,
        name="critic",
        chkpt_dir="tmp/sac",
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        self.fc1 = ResidualBlock(input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = ResidualBlock(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):

    def __init__(
        self,
        beta,
        input_dims,
        fc1_dims=256,
        fc2_dims=256,
        name="value",
        chkpt_dir="tmp/sac",
    ):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")

        self.fc1 = ResidualBlock(input_dims[0], self.fc1_dims)
        self.fc2 = ResidualBlock(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):

    def __init__(
        self,
        alpha,
        input_dims,
        max_action,
        fc1_dims=256,
        fc2_dims=256,
        n_actions=2,
        name="actor",
        chkpt_dir="tmp/sac",
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + "_sac")
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = ResidualBlock(input_dims[0], self.fc1_dims)
        self.fc2 = ResidualBlock(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        action = action * torch.tensor(self.max_action, dtype=action.dtype).to(
            self.device
        )

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
