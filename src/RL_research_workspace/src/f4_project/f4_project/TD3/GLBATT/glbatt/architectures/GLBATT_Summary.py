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
    def __init__(self, state_dim, action_dim, max_action, latent_dim=128, belief_state_indices=None):
        super().__init__()
        
        self.d_model = 128
        self.nhead = 8
        self.belief_state_indices = belief_state_indices
        self.belief_dim = len(belief_state_indices) if belief_state_indices is not None else 4
        
        # State embedding
        self.state_embed = nn.Linear(state_dim, self.d_model)
        self.embed_norm = nn.LayerNorm(self.d_model)
        
        # The Backbone: GTrXL processes the sequence (now explicitly overriding dropout to 0.0)
        # Because we never call .eval() in the unified train script, dropout=0.1 would constantly add noise and cap accuracy
        self.backbone = GTrXL(self.d_model, self.nhead, 2, hidden_dims=latent_dim, batch_first=True, dropout=0.0)
        self.backbone_norm = nn.LayerNorm(self.d_model)
        
        # Decision Fusion: Strictly matching CrossQ architecture
        # Concatenated dimension: 128 * 3 = 384
        self.fusion_norm = nn.LayerNorm(self.d_model * 3)
        self.l1 = nn.Linear(self.d_model * 3, 800)
        self.ln1 = nn.LayerNorm(800)
        self.l2 = nn.Linear(800, 600)
        self.ln2 = nn.LayerNorm(600)
        self.dropout = nn.Dropout(0.0)
        self.l_out = nn.Linear(600, action_dim)
        
        # BELIEF BOTTLENECK: Small summary for the Critic
        # Added LayerNorm for numerical stability
        self.belief_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.belief_dim),
            nn.LayerNorm(self.belief_dim)
        )
        
        self.max_action = max_action
        self.latent_dim = latent_dim # for compatibility
        
        # Apply Orthogonal Initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2)) # ReLU/LeakyReLU gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state_seq):
        # TIERED STRIDING: We keep the number of processed tokens small (30) for O(N^2) speed
        # but sample them dynamically based on the input sequence length.
        seq_len = state_seq.shape[1]
        local_size = min(20, seq_len)
        global_size = 10
        
        # 1. Local Memory: Most recent steps
        local_seq = state_seq[:, -local_size:, :]
        
        # 2. Global Memory: Distant steps sampled from the remainder
        remaining_len = seq_len - local_size
        if remaining_len >= global_size:
            stride = remaining_len // global_size
            global_indices = torch.arange(0, global_size * stride, stride)[:global_size]
        else:
            global_indices = torch.arange(0, remaining_len)
        
        global_seq = state_seq[:, global_indices, :]
        
        # 3. Combined Memory
        strided_seq = torch.cat([global_seq, local_seq], dim=1) # (B, tokens, state_dim)
        
        # 4. Correct Temporal Indices for Positional Encoding
        token_indices = torch.cat([global_indices, torch.arange(seq_len - local_size, seq_len)]).to(device)
        
        embedded_seq = self.embed_norm(self.state_embed(strided_seq))
        features, attn_weights = self.backbone(embedded_seq, indices=token_indices) 
        features = self.backbone_norm(features) # (B, 30, D)
        
        # 1. Local View: Current timestep (last token)
        f_t = features[:, -1, :] # (B, D)
        
        # 2. Global View A: Average history (Deterministic Summary)
        # Reverting to Global View A: Average history (Deterministic Summary)
        # Apply Tanh to bound the summary features, preventing variance explosion
        f_mean = torch.tanh(torch.mean(features, dim=1)) # (B, D)
        
        # 3. Global View B: Peak history (Catches sudden events)
        # Apply Tanh to bound the peak features, preventing variance explosion
        f_max = torch.tanh(torch.max(features, dim=1)[0]) # (B, D)
        
        # Decision Fusion: Simply stack them (Deterministic "Top-Level View")
        f_t_bounded = torch.tanh(f_t)
        fused = torch.cat([f_t_bounded, f_mean, f_max], dim=1) # (B, 3*D)
        fused = self.fusion_norm(fused)
        a = self.l1(fused)
        a = self.ln1(a)
        a = F.relu(a)
        
        a = self.l2(a)
        a = self.ln2(a)
        a = F.relu(a)
        
        a = self.dropout(a)
        
        # Belief bottleneck: Inform the Critic about hidden context
        belief = self.belief_head(f_mean)
        
        return self.max_action * torch.tanh(self.l_out(a)), belief, attn_weights

class GLBATT_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, belief_dim):
        super(GLBATT_Critic, self).__init__()
        
        # Q1 architecture (Matching TD3 scale)
        input_dim = state_dim + action_dim + belief_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(800, 600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(600, 1)
        )
        self.q2 = copy.deepcopy(self.q1)
        
        # Apply initialization
        self.q1.apply(self.init_weights)
        self.q2.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action, belief):
        sa = torch.cat([state, action, belief], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action, belief):
        sa = torch.cat([state, action, belief], 1)
        return self.q1(sa)

class GLBATT(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.3, noise_clip=0.6, policy_freq=3, belief_state_indices=None):
        self.belief_state_indices = belief_state_indices
        self.belief_dim = len(belief_state_indices) if belief_state_indices is not None else 4
        self.actor = GLBATT_Actor(state_dim, action_dim, max_action, belief_state_indices=belief_state_indices).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        # Differentiated Learning Rate & Weight Decay (MAIS stability)
        belief_params = list(self.actor.belief_head.parameters())
        belief_ids = list(map(id, belief_params))
        base_params = [p for p in self.actor.parameters() if id(p) not in belief_ids]
        
        # Apply weight_decay=1e-2 to beliefs to prevent weight explosion (MAIS patch)
        # Apply weight_decay=1e-2 to beliefs to prevent weight explosion (MAIS patch)
        self.actor_optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 3e-4, 'weight_decay': 0.0, 'betas': (0.5, 0.999)},
            {'params': belief_params, 'lr': 5e-4, 'weight_decay': 1e-2, 'betas': (0.5, 0.999)}
        ])

        self.critic = GLBATT_Critic(state_dim, action_dim, self.belief_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # Removed weight decay and restored aggressive TD3 3e-4 learning rate for Peak Capacity
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, betas=(0.5, 0.999))

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
        sample = replay_buffer.sample(batch_size)
        
        # Support extended replay buffers that provide explicit ground-truth targets (e.g. masked velocities)
        if len(sample) == 8:
            state, action, next_state, reward, not_done, state_seq, next_state_seq, true_belief = sample
        else:
            state, action, next_state, reward, not_done, state_seq, next_state_seq = sample
            true_belief = None

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, next_belief, next_attn = self.actor_target(next_state_seq)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # INTRINSIC PENALTY: Removed for aggressive performance
            intrinsic_penalty = 0.0
            
            # Critic target uses state, action, and belief bottleneck (MAIS)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_belief)
            target_Q = torch.min(target_Q1, target_Q2)
            Q_value = torch.mean(target_Q)
            
            # Robust Target Scaling (Including Intrinsic Motivation)
            target_Q = (reward + intrinsic_penalty) + not_done * self.discount * target_Q
            target_Q = target_Q.clamp(-500.0, 500.0) # Prevent Q-value explosion
            
            # ORACLE SAFETY GROUNDING: Ground belief[0] to predict collisions (-100 reward)
            # We use TD-learning to train the belief bit as a "Safe-Value" head
            if true_belief is not None:
                # Dynamic Hazards: node 0 predicts imminent termination (e.g. falling over)
                hazard_signal = 1.0 - not_done
                target_B = hazard_signal + 0.95 * torch.sigmoid(next_belief[:, 0:1]) * not_done
                target_B = target_B.clamp(0.0, 1.0)
            elif self.belief_state_indices is not None:
                target_B = None  # Supervised target comes from state directly
            else:
                collision_signal = (reward < -50.0).float() # Env gives -100 for crashes
                # target_B = 1.0 if crash, else discounted future crash probability
                target_B = collision_signal + 0.95 * torch.sigmoid(next_belief[:, 0:1]) * not_done
                target_B = target_B.clamp(0.0, 1.0)

        # Current Q estimates
        with torch.no_grad():
            _, belief, _ = self.actor(state_seq)

        current_Q1, current_Q2 = self.critic(state, action, belief)
        
        # Exact MSE Loss matching TD3 for peak sharp Q-value estimation instead of smooth Huber
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

            # GROUNDING LOSS: Supervised auxiliary task to fix the belief semantics
            if true_belief is not None:
                # Joint formulation: Node 0 predicts Hazard Horizon, Node 1..N predict missing kinematics
                safety_loss_hazard = F.mse_loss(torch.sigmoid(curr_belief[:, 0:1]), target_B)
                safety_loss_kinematics = F.mse_loss(curr_belief[:, 1:], true_belief)
                safety_loss = safety_loss_hazard + safety_loss_kinematics
            elif self.belief_state_indices is not None:
                safety_loss = F.mse_loss(curr_belief, state_seq[:, -1, self.belief_state_indices])
            else:
                # Forces belief[0] to follow the TD-target for collisions
                safety_loss = F.mse_loss(torch.sigmoid(curr_belief[:, 0:1]), target_B)
            
            self.actor_optimizer.zero_grad()
            (actor_loss + 1.0 * safety_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log intention diversity and Attention distribution
        with torch.no_grad():
             _, belief_test, attn_test = self.actor(state_seq)
             z_var = torch.var(belief_test, dim=0).mean()
             if true_belief is not None:
                 intrinsic_avg = -0.5 * torch.sigmoid(belief_test[:, 0]).mean().item()
                 safety_acc = F.mse_loss(belief_test[:, 1:], true_belief).item()
             elif self.belief_state_indices is not None:
                 intrinsic_avg = 0.0
                 safety_acc = F.mse_loss(belief_test, state_seq[:, -1, self.belief_state_indices]).item()
             else:
                 intrinsic_avg = -0.5 * torch.sigmoid(belief_test[:, 0]).mean().item()
                 safety_acc = F.mse_loss(torch.sigmoid(belief_test[:, 0:1]), target_B).item()
             # attn_test is (B, 30, 30) - PyTorch averages over heads by default
             # We care about what the LAST token (current time) attends to
             # Averaging over batch dimension only
             final_attn = attn_test[:, -1, :].mean(dim=0) # (30,)

        return Q_value, critic_loss, z_var, final_attn, intrinsic_avg, safety_acc

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.actor.state_dict(), filename + "_actor")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", weights_only=True))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", weights_only=True))
        self.actor_target = copy.deepcopy(self.actor)
