import argparse
import gymnasium as gym
import numpy as np
import itertools
import torch
from datetime import datetime
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from train_env_disp_mem import DroneGazeboEnv
gym.register(
    id='GazeboIrisEnv-v0',
    entry_point='train_env_disp_mem:DroneGazeboEnv', 
)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="GazeboIrisEnv-v0",
                    help='Mujoco Gym environment (default: GazeboIrisEnv-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=500000, metavar='N',
                    help='maximum number of steps (default: 8000)')
parser.add_argument('--hidden_size', type=int, default=600, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 1000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument("--run_name", default="laser_2d_new_200_400") 
parser.add_argument("--load", default="")
parser.add_argument("--load_steps", default=0)
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)
datetime_now = datetime.now().strftime('%d_%m_%Y_%H_%M')
file_name = f"{args.run_name}_{args.policy}_{datetime_now}"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
state_dim = 34
action_dim = 2
# Agent
agent = SAC(state_dim, env.action_space, args)
       
if args.load != "":
    #Tesnorboard
    # writer = SummaryWriter('runs/2024-09-30_15-22-46_SAC_laser_top_bottom_transformer_Gaussian_')
    writer = SummaryWriter("runs/" + args.load)
else:
    #Tesnorboard
    writer = SummaryWriter("runs/" + file_name)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

total_numsteps = 0 + args.load_steps
updates = 0 + args.load_steps
rewards = np.ones(10)
rewards = np.negative(rewards)
steps_to_finish = args.load_steps + args.num_steps

if args.load != "":
    policy_file = args.load
    agent.load_checkpoint(f"checkpoints/sac_checkpoint_{args.env_name}_{policy_file}")
    memory.load_buffer(f"checkpoints/sac_buffer_{args.env_name}_{policy_file}")
    rewards = np.load(f"checkpoints/sac_rewards_{args.env_name}_{policy_file}.npy")
          
# Training Loop
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state, info = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state["laser"],state["goal"])  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, truncated, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = float(truncated)

        memory.push(state["laser"],state["goal"], action, reward, next_state["laser"], next_state["goal"], mask) # Append transition to memory

        state = next_state

    if total_numsteps > steps_to_finish:
        break
    rewards = np.roll(rewards,1,axis=0)
    if(episode_reward > 0):
        rewards[0] = 100
    else:
        rewards[0] = -100
    avg_reward = np.sum(rewards,axis=0)/10.0

    # if i_episode % 10 == 0:
    writer.add_scalar('avg_reward/test', avg_reward, total_numsteps)
    
    if i_episode % 10 == 0:
        memory.save_buffer(env_name=args.env_name,suffix=file_name)
        agent.save_checkpoint(env_name=args.env_name,suffix=file_name)
        np.save(f"checkpoints/sac_rewards_{args.env_name}_{file_name}",rewards)

    writer.add_scalar('reward/train', episode_reward, total_numsteps)
    print("Episode: {}, Total Numsteps: {}, Episode Steps: {}, Reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # if i_episode % 35 == 0 and args.eval is True:

    #     episodes = 10
    #     avg_reward_eval = 0
    #     memory.save_buffer(env_name=args.env_name,suffix=file_name)
    #     agent.save_checkpoint(env_name=args.env_name,suffix=file_name)
    #     np.save(f"checkpoints/sac_rewards_{args.env_name}_{file_name}",rewards)
    #     for _  in range(episodes):
    #         state, info = env.reset()
    #         episode_reward = 0
    #         done = False
    #         while not done:
    #             action = agent.select_action(state["laser"], state["goal"], evaluate=True)

    #             next_state, reward, done, truncated, _ = env.step(action)
    #             episode_reward += reward


    #             state = next_state
            
    #         if(episode_reward > 0):
    #             avg_reward_eval += 100
    #         else:
    #             avg_reward_eval += -100
    #         # avg_reward += episode_reward
    #     avg_reward_eval /= episodes

    #     writer.add_scalar('avg_reward/eval', avg_reward_eval, total_numsteps)

        # print("----------------------------------------")
        # print("Saved Models")
        # print("----------------------------------------")

# env.close()

