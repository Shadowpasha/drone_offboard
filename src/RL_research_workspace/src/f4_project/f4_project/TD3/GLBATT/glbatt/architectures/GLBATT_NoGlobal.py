import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.utils import spectral_norm
from glbatt.gtrxl_torch.gtrxl_torch import GTrXL

class GLBATT_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, latent_dim=128):
        super().__init__()
        
        self.d_model = 128
        self.nhead = 8
        self.state_embed = nn.Linear(state_dim, self.d_model)
        self.embed_norm = nn.LayerNorm(self.d_model)
        
        # Backbone remains the same scale for fair comparison
        self.backbone = GTrXL(self.d_model, self.nhead, 2, hidden_dims=latent_dim, batch_first=True, dropout=0.0)
        self.backbone_norm = nn.LayerNorm(self.d_model)
        
        self.l1 = nn.Linear(self.d_model * 3, 512)
        self.fusion_norm = nn.LayerNorm(self.d_model * 3)
        self.l2 = nn.Linear(512, action_dim)
        
        self.belief_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.LayerNorm(4),
            nn.Tanh()
        )
        
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_seq):
        # ABLATION: Use most recent 30 tokens ONLY (Contiguous)
        # No more global striding. Window is still 100 in buffer, but we only "see" 30.
        strided_seq = state_seq[:, -30:, :]
        token_indices = torch.arange(70, 100).to(device) # Aligned to the end of 100-step buffer
        
        embedded_seq = self.embed_norm(self.state_embed(strided_seq))
        features, attn_weights = self.backbone(embedded_seq, indices=token_indices) 
        features = self.backbone_norm(features)
        
        f_t = features[:, -1, :]
        f_mean = torch.tanh(torch.mean(features, dim=1))
        f_max = torch.tanh(torch.max(features, dim=1)[0])
        
        f_t_bounded = torch.tanh(f_t)
        fused = torch.cat([f_t_bounded, f_mean, f_max], dim=1)
        fused = self.fusion_norm(fused)
        
        a = F.leaky_relu(self.l1(fused), 0.01)
        belief = self.belief_head(f_mean)
        
        return self.max_action * torch.tanh(self.l2(a)), belief, attn_weights

class GLBATT_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, belief_dim=4):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim + belief_dim, 800)
        self.l2 = nn.Linear(800, 600)
        self.l3 = nn.Linear(600, 1)

        self.l4 = nn.Linear(state_dim + action_dim + belief_dim, 800)
        self.l5 = nn.Linear(800, 600)
        self.l6 = nn.Linear(600, 1)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action, belief):
        sa = torch.cat([state, action, belief], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, belief):
        sa = torch.cat([state, action, belief], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)

class GLBATT(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.3, noise_clip=0.6, policy_freq=3):
        self.actor = GLBATT_Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        belief_params = list(self.actor.belief_head.parameters())
        belief_ids = list(map(id, belief_params))
        base_params = [p for p in self.actor.parameters() if id(p) not in belief_ids]
        
        self.actor_optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 3e-4, 'weight_decay': 0.0},
            {'params': belief_params, 'lr': 5e-4, 'weight_decay': 1e-2}
        ])

        self.critic = GLBATT_Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state_seq):
        state_seq = torch.FloatTensor(state_seq.reshape(1, state_seq.shape[0], -1)).to(device)
        action, _, _ = self.actor(state_seq)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done, state_seq, next_state_seq = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, next_belief, next_attn = self.actor_target(next_state_seq)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            intrinsic_penalty = -0.5 * torch.sigmoid(next_belief[:, 0:1])
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_belief)
            target_Q = torch.min(target_Q1, target_Q2)
            Q_value = torch.mean(target_Q)
            target_Q = (reward + intrinsic_penalty) + not_done * self.discount * target_Q
            target_Q = target_Q.clamp(-500.0, 500.0)
            collision_signal = (reward < -50.0).float()
            target_B = collision_signal + 0.95 * torch.sigmoid(next_belief[:, 0:1]) * not_done
            target_B = target_B.clamp(0.0, 1.0)

        with torch.no_grad():
            _, belief, _ = self.actor(state_seq)
        current_Q1, current_Q2 = self.critic(state, action, belief)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            curr_action, curr_belief, curr_attn = self.actor(state_seq)
            
            # Actor loss: now informed by the belief bottleneck
            self.critic.eval()
            actor_loss = -self.critic.Q1(state, curr_action, curr_belief).mean()
            self.critic.train()

            safety_loss = F.mse_loss(torch.sigmoid(curr_belief[:, 0:1]), target_B)
            self.actor_optimizer.zero_grad()
            (actor_loss + 1.0 * safety_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        with torch.no_grad():
             _, belief_test, attn_test = self.actor(state_seq)
             z_var = torch.var(belief_test, dim=0).mean()
             intrinsic_avg = -0.5 * torch.sigmoid(belief_test[:, 0]).mean()
             safety_acc = F.mse_loss(torch.sigmoid(belief_test[:, 0:1]), target_B).item()
             final_attn = attn_test[:, -1, :].mean(dim=0)

        return Q_value, critic_loss, z_var, final_attn, intrinsic_avg, safety_acc

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", weights_only=True))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", weights_only=True))
        self.actor_target = copy.deepcopy(self.actor)
