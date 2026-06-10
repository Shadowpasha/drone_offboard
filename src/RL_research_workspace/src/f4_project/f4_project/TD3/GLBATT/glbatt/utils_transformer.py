import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, seq_len=20, max_size=int(2e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.seq_len = seq_len

        # Standard components (Using float32 to save 50% memory)
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        
        # Sequence components (last N steps)
        self.state_seq = np.zeros((max_size, seq_len, state_dim), dtype=np.float32)
        self.next_state_seq = np.zeros((max_size, seq_len, state_dim), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, state_seq, next_state_seq):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        self.state_seq[self.ptr] = state_seq
        self.next_state_seq[self.ptr] = next_state_seq

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.state_seq[ind]).to(self.device),
            torch.FloatTensor(self.next_state_seq[ind]).to(self.device)
        )

    def save(self, folder, filename):
        np.save(f"{folder}/{filename}_state.npy", self.state[:self.size])
        np.save(f"{folder}/{filename}_action.npy", self.action[:self.size])
        np.save(f"{folder}/{filename}_next_state.npy", self.next_state[:self.size])
        np.save(f"{folder}/{filename}_reward.npy", self.reward[:self.size])
        np.save(f"{folder}/{filename}_not_done.npy", self.not_done[:self.size])
        np.save(f"{folder}/{filename}_state_seq.npy", self.state_seq[:self.size])
        np.save(f"{folder}/{filename}_next_state_seq.npy", self.next_state_seq[:self.size])

    def load(self, folder, filename):
        self.state = np.load(f"{folder}/{filename}_state.npy")
        self.action = np.load(f"{folder}/{filename}_action.npy")
        self.next_state = np.load(f"{folder}/{filename}_next_state.npy")
        self.reward = np.load(f"{folder}/{filename}_reward.npy")
        self.not_done = np.load(f"{folder}/{filename}_not_done.npy")
        self.state_seq = np.load(f"{folder}/{filename}_state_seq.npy")
        self.next_state_seq = np.load(f"{folder}/{filename}_next_state_seq.npy")
        
        self.size = self.state.shape[0]
        self.ptr = self.size % self.max_size
        print(f"Replay buffer loaded with {self.size} samples.")
