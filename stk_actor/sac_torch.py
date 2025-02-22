import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .buffer import ReplayBuffer
from .networks import ActorNetwork, CriticNetwork, ValueNetwork


class AgentSac(nn.Module):

    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        action_space=None,
        observation_space=None,
        gamma=0.99,
        max_size=1000000,
        tau=0.005,
        batch_size=256,
        reward_scale=2,
    ):
        super().__init__()
        self.input_dims = [
            np.sum(np.prod(space.shape) for space in observation_space.values())
        ]
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = action_space.shape[0]
        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions)

        self.actor = ActorNetwork(
            alpha,
            self.input_dims,
            n_actions=self.n_actions,
            name="actor",
            max_action=action_space.high,
        )
        self.critic_1 = CriticNetwork(
            beta, self.input_dims, n_actions=self.n_actions, name="critic1"
        )
        self.critic_2 = CriticNetwork(
            beta, self.input_dims, n_actions=self.n_actions, name="critic2"
        )
        self.value = ValueNetwork(beta, self.input_dims, name="value")
        self.target_value = ValueNetwork(beta, self.input_dims, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def flatten_observation(self, observation: dict) -> np.ndarray:
        flattened_components = [
            np.array(value).flatten() for value in observation.values()
        ]
        return np.concatenate(flattened_components)

    def choose_action(self, observation):
        observation = self.flatten_observation(observation)
        state = torch.Tensor(np.array([observation])).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(
            self.flatten_observation(state),
            action,
            reward,
            self.flatten_observation(new_state),
            done,
        )

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = (
                tau * value_state_dict[name].clone()
                + (1 - tau) * target_value_state_dict[name].clone()
            )

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print(".... saving models ....")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print(".... loading models ....")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, new_state, action, reward, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward()
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
