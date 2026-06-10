import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Actor and Critic Networks (Unchanged) ---
# These are generic and will adapt to the state_dim and action_dim
# provided during instantiation.

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 800)
        self.l2 = nn.Linear(800, 600)
        self.l3 = nn.Linear(600, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return self.max_action * a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 800)
        self.l2 = nn.Linear(800, 600)
        self.l3 = nn.Linear(600, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 800)
        self.l5 = nn.Linear(800, 600)
        self.l6 = nn.Linear(600, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1


class TD3(object):
    """
    Standard TD3 (Twin Delayed Deep Deterministic Policy Gradient)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        critic_lr=3e-4,
        actor_lr=3e-4
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state_np):
        state = torch.FloatTensor(state_np.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        """
        One training iteration (sample batch, update critic and occasionally actor/targets).
        """
        self.total_it += 1

        # sample
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        # Move to device
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        reward = reward.to(device)
        not_done = not_done.to(device)


        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # critic step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'Q_value': current_Q1.mean().item()
        }


    def save(self, prefix):
        torch.save(self.critic.state_dict(), prefix + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), prefix + "_critic_optimizer.pth")
        torch.save(self.actor.state_dict(), prefix + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), prefix + "_actor_optimizer.pth")


    def load(self, prefix):
        self.critic.load_state_dict(torch.load(prefix + "_critic.pth", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(prefix + "_critic_optimizer.pth", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(prefix + "_actor.pth", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(prefix + "_actor_optimizer.pth", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)

