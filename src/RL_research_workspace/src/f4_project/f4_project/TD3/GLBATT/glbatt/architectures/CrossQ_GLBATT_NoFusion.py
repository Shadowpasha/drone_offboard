import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.nn.utils import spectral_norm
from glbatt.gtrxl_torch.gtrxl_torch import GTrXL

class CrossQ_GLBATT_NoFusion_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, latent_dim=128, belief_state_indices=None):
        super().__init__()
        
        self.d_model = 128
        self.nhead = 8
        self.belief_state_indices = belief_state_indices
        self.belief_dim = len(belief_state_indices) if belief_state_indices is not None else 4
        
        # State embedding
        self.state_embed = nn.Linear(state_dim, self.d_model)
        self.embed_norm = nn.LayerNorm(self.d_model)
        
        # The Backbone: GTrXL processes the sequence
        self.backbone = GTrXL(self.d_model, self.nhead, 2, hidden_dims=latent_dim, batch_first=True, dropout=0.0)
        self.backbone_norm = nn.BatchNorm1d(self.d_model)
        
        # NO FUSION: Just the 128-d output
        self.fusion_norm = nn.BatchNorm1d(self.d_model)
        self.l1 = nn.Linear(self.d_model, 800)
        self.bn1 = nn.BatchNorm1d(800)
        self.l2 = nn.Linear(800, 600)
        self.bn2 = nn.BatchNorm1d(600)
        self.dropout = nn.Dropout(0.0)
        self.l_mean = nn.Linear(600, action_dim)
        self.l_log_std = nn.Linear(600, action_dim)
        
        # BELIEF BOTTLENECK: Inform the Critic about hidden context
        self.belief_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.belief_dim),
            nn.BatchNorm1d(self.belief_dim)
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
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
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
        
        # BatchNorm1d expects (B, C, L) or (B, C). 
        # Here features is (B, L, D). We permute to (B, D, L) and back.
        features = self.backbone_norm(features.transpose(1, 2)).transpose(1, 2)
        
        # 1. Local View: Current timestep (last token)
        f_t = features[:, -1, :] # (B, D)
        
        # 2. Global View A: Average history (Deterministic Summary)
        # Apply Tanh to bound the summary features, preventing variance explosion
        f_mean = torch.tanh(torch.mean(features, dim=1)) # (B, D)
        
        # No Fusion: Only use the current timestep bounded feature
        f_t_bounded = torch.tanh(f_t)
        fused = self.fusion_norm(f_t_bounded)
        a = F.relu(self.bn1(self.l1(fused)))
        a = F.relu(self.bn2(self.l2(a)))
        a = self.dropout(a)
        
        # Belief bottleneck: Inform the Critic about hidden context
        belief = self.belief_head(f_mean)
        
        mean = self.l_mean(a)
        log_std = self.l_log_std(a)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std, belief, attn_weights

    def sample(self, state_seq, td3_mode=False):
        mean, log_std, belief, attn_weights = self.forward(state_seq)
        
        if td3_mode:
            # Deterministic for TD3 mode
            action = torch.tanh(mean) * self.max_action
            return action, 0.0, belief, attn_weights

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * epsilon)
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (Squashed Gaussian correction)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, belief, attn_weights

class CrossQ_GLBATT_NoFusion_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, belief_dim=4, num_critics=4):
        super().__init__()
        
        self.num_critics = num_critics
        self.critics = nn.ModuleList()
        
        for _ in range(num_critics):
            critic = nn.Sequential(
                nn.BatchNorm1d(state_dim + action_dim + belief_dim),
                nn.Linear(state_dim + action_dim + belief_dim, 800),
                nn.ReLU(),
                nn.BatchNorm1d(800),
                nn.Dropout(0.0),
                nn.Linear(800, 600),
                nn.ReLU(),
                nn.BatchNorm1d(600),
                nn.Dropout(0.0),
                nn.Linear(600, 1)
            )
            self.critics.append(critic)
        
        # Apply Orthogonal Initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2)) # ReLU/LeakyReLU gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state, action, belief):
        sa = torch.cat([state, action, belief], 1)
        q_values = [critic(sa) for critic in self.critics]
        return torch.stack(q_values, dim=0) # (num_critics, B, 1)

    def Q1(self, state, action, belief):
        sa = torch.cat([state, action, belief], 1)
        return self.critics[0](sa)

class CrossQ_GLBATT_NoFusion(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=1.0, policy_noise=0.3, noise_clip=0.6, policy_freq=3, belief_state_indices=None, td3_mode=False):
        self.td3_mode = td3_mode
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.belief_state_indices = belief_state_indices
        self.belief_dim = len(belief_state_indices) if belief_state_indices is not None else 4
        self.actor = CrossQ_GLBATT_NoFusion_Actor(state_dim, action_dim, max_action, belief_state_indices=belief_state_indices).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        
        # Automatic Entropy Tuning (Alpha Tuning)
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        # REDQ: Ensemble of critics
        self.critic = CrossQ_GLBATT_NoFusion_Critic(state_dim, action_dim, belief_dim=self.belief_dim, num_critics=4).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Differentiated Learning Rate & Weight Decay
        belief_params = list(self.actor.belief_head.parameters())
        belief_ids = list(map(id, belief_params))
        base_params = [p for p in self.actor.parameters() if id(p) not in belief_ids]
        
        self.actor_optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 3e-4, 'weight_decay': 0.0, 'betas': (0.5, 0.999)}, # CrossQ betas
            {'params': belief_params, 'lr': 5e-4, 'weight_decay': 1e-2, 'betas': (0.5, 0.999)}
        ])

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
        self.actor.eval()
        with torch.no_grad():
            mean, _, _, _ = self.actor(state_seq)
            action = torch.tanh(mean) * self.max_action
        self.actor.train()
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

        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            # Target actions
            if self.td3_mode:
                # TD3 Target Policy Smoothing
                next_action_mean, _, next_belief, _ = self.actor(next_state_seq)
                next_action = torch.tanh(next_action_mean) * self.max_action
                noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                next_log_prob = 0.0
            else:
                next_action, next_log_prob, next_belief, _ = self.actor.sample(next_state_seq)
            
            # INTRINSIC PENALTY
            if self.belief_state_indices is not None:
                intrinsic_penalty = 0.0
            else:
                intrinsic_penalty = -0.1 * torch.sigmoid(next_belief[:, 0:1])

            # Pre-calculate auxiliary targets
            if true_belief is not None:
                # Dynamic Hazards: node 0 predicts imminent termination (e.g. falling over)
                hazard_signal = 1.0 - not_done
                target_B = hazard_signal + 0.95 * torch.sigmoid(next_belief[:, 0:1]) * not_done
                target_B = target_B.clamp(0.0, 1.0)
            elif self.belief_state_indices is None:
                collision_signal = (reward < -50.0).float()
                target_B = collision_signal + 0.95 * torch.sigmoid(next_belief[:, 0:1]) * not_done
                target_B = target_B.clamp(0.0, 1.0)
            
            _, _, belief, _ = self.actor(state_seq)

        # ----- 2N BATCH TRICK -----
        # Concatenate current and next transitions for a single forward pass
        # This matches CrossQ vanilla precisely and stabilizes Norm layers
        obs_catted = torch.cat([state, next_state], dim=0)
        act_catted = torch.cat([action, next_action], dim=0)
        bel_catted = torch.cat([belief, next_belief], dim=0)
        
        # Single forward pass through Critic ensemble
        all_qs_catted = self.critic(obs_catted, act_catted, bel_catted) # (num_critics, 2*B, 1)
        
        current_Qs, next_Qs = torch.split(all_qs_catted, batch_size, dim=1)
        
        with torch.no_grad():
            # REDQ style: use subset of critics for target
            indices = torch.randperm(self.critic.num_critics)[:2]
            target_Q = torch.min(next_Qs[indices], dim=0)[0]
            
            # SAC Entropy term (skipped if td3_mode)
            if not self.td3_mode:
                target_Q = target_Q - alpha * next_log_prob
            
            Q_value = torch.mean(target_Q)
            
            # Bellman
            target_Q = reward + not_done * self.discount * target_Q
            target_Q = target_Q.clamp(-500.0, 500.0)

        # Critic Loss
        critic_loss = sum(F.mse_loss(current_Q, target_Q) for current_Q in current_Qs)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            # Actor Update
            if self.td3_mode:
                 curr_action_mean, _, curr_belief, curr_attn = self.actor(state_seq)
                 curr_action = torch.tanh(curr_action_mean) * self.max_action
                 log_prob = torch.tensor(0.0).to(device) # dummy
            else:
                 curr_action, log_prob, curr_belief, curr_attn = self.actor.sample(state_seq)
            
            self.critic.eval()
            q_vals = self.critic(state, curr_action, curr_belief)
            min_q = q_vals.min(dim=0)[0]
            
            if self.td3_mode:
                actor_loss = -min_q.mean()
            else:
                actor_loss = (alpha * log_prob - min_q).mean()
            self.critic.train()
            
            # Grounding Loss
            if true_belief is not None:
                # Joint formulation: Node 0 predicts Hazard Horizon, Node 1..N predict missing kinematics
                safety_loss_hazard = F.binary_cross_entropy(torch.sigmoid(curr_belief[:, 0:1]), target_B)
                safety_loss_kinematics = F.mse_loss(curr_belief[:, 1:], true_belief)
                safety_loss = safety_loss_hazard + safety_loss_kinematics
            elif self.belief_state_indices is not None:
                safety_loss = F.mse_loss(curr_belief, state_seq[:, -1, self.belief_state_indices])
            else:
                safety_loss = F.binary_cross_entropy(torch.sigmoid(curr_belief[:, 0:1]), target_B)
            
            self.actor_optimizer.zero_grad()
            (actor_loss + 1.0 * safety_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # Alpha Update (SAC only)
            if not self.td3_mode:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # Target updates (CrossQ has tau=1.0 by default now)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log metrics
        with torch.no_grad():
             _, _, belief_test, attn_test = self.actor(state_seq)
             z_var = torch.var(belief_test, dim=0).mean()
             if true_belief is not None:
                 intrinsic_avg = -0.1 * torch.sigmoid(belief_test[:, 0]).mean().item()
                 safety_acc = F.mse_loss(belief_test[:, 1:], true_belief).item()
             elif self.belief_state_indices is not None:
                 intrinsic_avg = 0.0
                 safety_acc = F.mse_loss(belief_test, state_seq[:, -1, self.belief_state_indices]).item()
             else:
                 intrinsic_avg = -0.1 * torch.sigmoid(belief_test[:, 0]).mean().item()
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
