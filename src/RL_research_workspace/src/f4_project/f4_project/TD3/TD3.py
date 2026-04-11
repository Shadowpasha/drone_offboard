import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
dimension_mem = 5

# class SelfAttention(nn.Module):
#   def __init__(self, input_dim):
#     super(SelfAttention, self).__init__()
#     self.input_dim = input_dim
#     self.query = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
#     self.key = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
#     self.value = nn.Linear(input_dim, input_dim)
#     self.softmax = nn.Softmax(dim=2)
   
#   def forward(self, x): # x.shape (batch_size, seq_length, input_dim)
#     queries = self.query(x)
#     keys = self.key(x)
#     values = self.value(x)

#     score = torch.bmm(queries, keys.transpose(1, 2))/(self.input_dim**0.5)
#     attention = self.softmax(score)
#     weighted = torch.bmm(attention, values)
#     return weighted

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()
		self.l1 = nn.Linear((state_dim), 1024)
		self.l2 = nn.Linear(1024, 512)
		self.l3 = nn.Linear(512, action_dim)


		self.max_action = max_action
		

	def forward(self, state ):

		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))

		a = torch.tanh(self.l3(a))

		return self.max_action * a


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		# self.flatten = nn.Flatten()
		# Q1 architecture
		self.l1 = nn.Linear((state_dim ) + action_dim, 800)
		self.l2 = nn.Linear(800, 600)
		self.l3 = nn.Linear(600, 1)
		# self.g3 = SelfAttention(600)

		# Q2 architecture
		self.l4 = nn.Linear((state_dim ) + action_dim, 800)
		self.l5 = nn.Linear(800, 600)
		self.l6 = nn.Linear(600, 1)
		# self.g7 = SelfAttention(600)


	def forward(self, state, action):

		# state_laser = self.flatten(state_laser)
		# state_goal = self.flatten(state_goal)
		# combined = torch.cat((state_laser,state_goal),dim=1)

		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)


		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):

		# state_laser = self.flatten(state_laser)
		# state_goal = self.flatten(state_goal)
		# combined = torch.cat((state_laser,state_goal),dim=1)
		
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
	
	
	def select_action(self,state):
		state = torch.FloatTensor(
                state.reshape(1, -1)).to(device)
		action = self.actor(state)
		return action.cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			Q_value = torch.mean(target_Q)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()   
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		return Q_value, critic_loss


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		# Enhanced load to handle .zip and non-.zip files
		suffix_list = ["_critic", "_critic_optimizer", "_actor", "_actor_optimizer"]
		paths = {}
		for suffix in suffix_list:
			path = filename + suffix
			if not os.path.exists(path) and os.path.exists(path + ".zip"):
				path += ".zip"
			paths[suffix] = path

		self.critic.load_state_dict(torch.load(paths["_critic"], weights_only=True))
		self.critic_optimizer.load_state_dict(torch.load(paths["_critic_optimizer"], weights_only=True))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(paths["_actor"], weights_only=True))
		self.actor_optimizer.load_state_dict(torch.load(paths["_actor_optimizer"], weights_only=True))
		self.actor_target = copy.deepcopy(self.actor)
		