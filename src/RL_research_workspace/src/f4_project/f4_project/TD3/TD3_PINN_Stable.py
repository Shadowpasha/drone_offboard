import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return self.max_action * torch.tanh(self.l3(a))

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
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)

class TD3_PINN_Stable(object):
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
        actor_lr=3e-4,
        critic_lr=1e-4,
        enable_pinn=True,
        lambda_pinn=2.0,
        lambda_actor_pinn=0.01, # Increased for better guidance
        pinn_every=1,
        dt=0.1
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

        self.enable_pinn = enable_pinn
        self.lambda_pinn = lambda_pinn
        self.lambda_actor_pinn = lambda_actor_pinn
        self.pinn_every = pinn_every
        self.dt = dt

        # Constants
        self.g = 9.81
        self.drag_coeff = 0.2 # Matches average (drag/mass) = (0.35/1.6)
        self.safe_margin = 1.0 
        self.ray_range = 12.0
        self.max_tilt = 0.4 # Matches environment limit

        # Placeholders for normalization (Should be set by main.py)
        self.state_mean = torch.zeros(state_dim, device=device)
        self.state_std = torch.ones(state_dim, device=device)
        self.action_mean = torch.zeros(action_dim, device=device)
        self.action_std = torch.ones(action_dim, device=device) * 0.3 # Scale to 0.3m/s
        self.max_speed = 0.3 # Matches environment

    def select_action(self, state_np):
        state = torch.FloatTensor(state_np.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def _denormalize_tensors(self, state, next_state, action):
        s_phys = state * self.state_std + self.state_mean
        s2_phys = next_state * self.state_std + self.state_mean
        a_phys = action * self.action_std + self.action_mean
        return s_phys, s2_phys, a_phys

    def physics_loss_batch(self, state_phys, next_state_phys, action_phys):
        dt, g = self.dt, self.g

        # State Indices (Verify these match your env!)
        vx_t, vy_t = state_phys[:, 64], state_phys[:, 65]
        vx_next, vy_next = next_state_phys[:, 64], next_state_phys[:, 65]
        x_t, y_t = state_phys[:, 66], state_phys[:, 67]
        x_next, y_next = next_state_phys[:, 66], next_state_phys[:, 67]
        roll_t, pitch_t = state_phys[:, 68], state_phys[:, 69]
        roll_next, pitch_next = next_state_phys[:, 68], next_state_phys[:, 69]
        
        cmd_vx, cmd_vy = action_phys[:, 0], action_phys[:, 1]

        # 1. Acceleration Model with Drag
        ax_measured = (vx_next - vx_t) / dt
        ay_measured = (vy_next - vy_t) / dt
        
        # Physics: Accel = Component of gravity - drag
        ax_model = g * torch.tan(pitch_t) - self.drag_coeff * vx_t
        ay_model = -g * torch.tan(roll_t) - self.drag_coeff * vy_t

        # 2. Kinematic Consistency (Pos-Vel relation)
        r_x = ((x_next - x_t) / dt) - vx_t
        r_y = ((y_next - y_t) / dt) - vy_t

        # 3. Holonomic Consistency (Ideal Control)
        # In this mode, velocity should perfectly track the action
        r_vx_tracking = vx_next - action_phys[:, 0]
        r_vy_tracking = vy_next - action_phys[:, 1]
        
        # Also kinematic consistency
        r_x_consist = ((x_next - x_t) / dt) - action_phys[:, 0]
        r_y_consist = ((y_next - y_t) / dt) - action_phys[:, 1]

        # 4. Safety Barrier (Stopping distance)
        speed = torch.sqrt(vx_t**2 + vy_t**2 + 1e-6)
        stopping_dist = (speed**2) / (2 * 3.0) 
        laser_phys = state_phys[:, :60] * self.ray_range
        min_ray, _ = torch.min(laser_phys, dim=1)
        # Penalize if stopping distance violates safe margin
        loss_collision = torch.mean(torch.relu(stopping_dist + self.safe_margin - min_ray)**2)

        # 5. Tilt and Action smoothness
        loss_tilt = torch.mean(torch.relu(torch.sqrt(roll_t**2 + pitch_t**2) - self.max_tilt)**2)
        action_smoothness = torch.mean(cmd_vx**2 + cmd_vy**2)

        # Total PINN Loss
        pinn_loss = (
            2.0 * torch.mean((ax_measured - ax_model)**2) +
            0.0 * torch.mean((ay_measured - ay_model)**2) + # Disabled for holonomic
            1.0 * torch.mean(r_x_consist**2 + r_y_consist**2) +
            1.0 * torch.mean(r_vx_tracking**2 + r_vy_tracking**2) +
            5.0 * loss_collision + # High priority safety
            2.0 * loss_tilt +
            0.1 * action_smoothness
        )

        return pinn_loss, {'pinn_coll': loss_collision.item(), 'pinn_tilt': loss_tilt.item()}

    def train(self, replay_buffer, batch_size=1024):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # CRITIC UPDATE
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        pinn_loss_val = torch.tensor(0.0, device=device)
        if self.enable_pinn:
            s_phys, s2_phys, a_phys = self._denormalize_tensors(state, next_state, action)
            pinn_loss_val, _ = self.physics_loss_batch(s_phys, s2_phys, a_phys)
            
            # Improved Adaptive Lambda: Never goes to zero
            adaptive_lambda = max(0.5, min(self.lambda_pinn, critic_loss.detach().item()))
            critic_loss += adaptive_lambda * pinn_loss_val

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # DELAYED ACTOR UPDATE
        out = {'Q_value': torch.mean(current_Q1).item(), 'critic_loss': critic_loss.item(), 'pinn_loss': pinn_loss_val.item()}
        
        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(state)
            actor_loss = -self.critic.Q1(state, actor_action).mean()

            if self.enable_pinn:
                s_phys, s2_phys, a_phys_actor = self._denormalize_tensors(state, next_state, actor_action)
                actor_pinn_loss, _ = self.physics_loss_batch(s_phys, s2_phys, a_phys_actor)
                actor_loss += self.lambda_actor_pinn * actor_pinn_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak Update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return out

    def save(self, prefix):
        torch.save(self.critic.state_dict(), prefix + "_critic.pth")
        torch.save(self.actor.state_dict(), prefix + "_actor.pth")

    def load(self, prefix):
        self.critic.load_state_dict(torch.load(prefix + "_critic.pth", map_location=device))
        self.actor.load_state_dict(torch.load(prefix + "_actor.pth", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)