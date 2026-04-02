import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.state_array = np.zeros((max_size, 5, state_dim ))
		self.next_state_array = np.zeros((max_size, 5, state_dim))
		self.action_array = np.zeros((max_size, 5, action_dim ))
		self.next_action_array = np.zeros((max_size, 5, action_dim))


		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done, state_array, next_state_array, action_array, next_action_array):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.state_array[self.ptr] = state_array
		self.next_state_array[self.ptr] = next_state_array
		self.action_array[self.ptr] = action_array
		self.next_action_array[self.ptr] = next_action_array

		

		# Detach the hidden state so that BPTT only goes through 1 timestep
	
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		s = torch.FloatTensor(
            np.expand_dims(self.state[ind],1)).to(self.device)
		
		a = torch.FloatTensor(
			np.expand_dims(self.action[ind],1)).to(self.device)
		ns = torch.FloatTensor(
			np.expand_dims(self.next_state[ind],1)).to(self.device)

		r = torch.FloatTensor(
			np.expand_dims(self.reward[ind],1)).to(self.device)
		d = torch.FloatTensor(
			np.expand_dims(self.not_done[ind],1)).to(self.device)
		
		s_a =  torch.FloatTensor(
			self.state_array[ind]).to(self.device)

		n_s_a =  torch.FloatTensor(
			self.next_state_array[ind]).to(self.device)
		
		a_a =  torch.FloatTensor(
			self.action_array[ind]).to(self.device)

		n_a_a =  torch.FloatTensor(
			self.next_action_array[ind]).to(self.device)

		return (
			s,a,ns,r,d,s_a,n_s_a,a_a, n_a_a
		)
	
	def save(self,filename):
		np.save(filename + "_state",self.state)
		np.save(filename + "_action",self.action)
		np.save(filename + "_next_state",self.next_state)
		np.save(filename + "_reward",self.reward)
		np.save(filename + "_not_done",self.not_done)

	
	def load(self,filename):
			self.state = np.load(filename + "_state.npy")
			self.action = np.load(filename + "_action.npy")
			self.next_state = np.load(filename + "_next_state.npy")
			self.reward = np.load(filename + "_reward.npy")
			self.not_done = np.load(filename + "_not_done.npy")
			self.ptr = np.max(np.nonzero(self.state_laser))
			print(self.ptr)
			