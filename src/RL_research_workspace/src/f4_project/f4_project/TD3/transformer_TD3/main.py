import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from datetime import datetime
import utils
import TD3
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from train_env_disp_mem import DroneGazeboEnv
import random
gym.register(
    id='GazeboIrisEnv-v0',
    entry_point='train_env_disp_mem:DroneGazeboEnv', 
)

# # Runs policy for X episodes and returns average reward
# # A fixed seed is used for the eval environment
# def eval_policy(policy, seed, eval_episodes=10):
# 	global env
# 	avg_reward = 0.
# 	for _ in range(eval_episodes):
# 		done = False
# 		state, info = env.reset()
# 		# state = np.concatenate((state["laser"].flatten(),state["goal"]),axis=0)
# 		while not done:
# 			action = policy.select_action(state["laser"],state["goal"])
# 			state, reward, done, truncated,  _ = env.step(action)
# 			avg_reward += reward

# 	avg_reward /= eval_episodes

# 	print("---------------------------------------")
# 	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
# 	print("---------------------------------------")
# 	return avg_reward
reward_array = []


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="GazeboIrisEnv-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=random.randint(0,9999), type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=0, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=526, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--expl_noise", default=1.0, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=40, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--run_name", default="transformer_single_critic")       # Frequency of delayed policy updates
	parser.add_argument("--load_model", default="")    # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--load_steps", default=0)
	args = parser.parse_args()

	datetime_now = datetime.now().strftime('%d_%m_%Y_%H_%M')
	file_name = f"{args.run_name}_{datetime_now}" if args.load_model == "" else args.load_model
	print("---------------------------------------")
	print(f" Run: {args.run_name}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Time: {datetime_now}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./rewards"):
		os.makedirs("./rewards")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./replays"):
		os.makedirs("./replays")

	env = gym.make(args.env)

	# Set seeds
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	# torch.manual_seed(args.seed)
	# np.random.seed(args.seed)
	
	state_dim = 34
	action_dim = 2
	max_action = 1.0

	rewards = np.ones(10)
	rewards = np.negative(rewards)
	avg_reward = 0


	if args.load_model != "":
    #Tesnorboard)
		writer = SummaryWriter("runs/" + args.load_model)
	else:
    #Tesnorboard
		writer = SummaryWriter("runs/" + file_name)

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "" else args.load_model
		policy.load(f"models/{policy_file}")
		replay_buffer.load(f"replays/{policy_file}")
		rewards = np.load(f"rewards/{policy_file}.npy")
	
	# Evaluate untrained policy
	# evaluations = [eval_policy(policy, args.seed)]
	evaluations = []
	done = False
	state, info = env.reset()
	state_array = np.zeros((5,state_dim))
	state_array = np.roll(state_array,1,axis=0)
	state_array[0] = state
	action_array = np.zeros((5, action_dim))
	action_array = np.roll(action_array,1,axis=0)
	action_array[0] = [0.0,0.0]
	episode_reward = 0
	episode_timesteps = 0 
	episode_num = 0
	total_timesteps = 0 + args.load_steps
	expl_min = 0.1
	expl_noise = args.expl_noise
	expl_decay_steps = 30000


	for t in range(150000):
		
		episode_timesteps += 1
		total_timesteps += 1
		# Select action randomly or according to policy
	
		if total_timesteps < args.start_timesteps:
			action = env.action_space.sample()
		else:
			if expl_noise > expl_min:
				expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
			a = policy.select_action(np.array(state), state_array)
			action = (
				a + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, truncated, info = env.step(action)
		reward_array.append(reward)

		next_state_array = np.roll(state_array,1,axis=0)
		next_state_array[0] = next_state

		next_action_array = np.roll(action_array,1,axis=0)
		next_action_array[0] = action

		done_bool = float(done)

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool, state_array, next_state_array, action_array, next_action_array)

		state = next_state
		state_array = next_state_array
		episode_reward += reward
		action_array = next_action_array

		# Train agent after collecting sufficient data
		if total_timesteps >= args.start_timesteps:
			Q_value, loss = policy.train(replay_buffer, args.batch_size)
			writer.add_scalar("training/Q",Q_value, total_timesteps)
			writer.add_scalar("training/loss",loss, total_timesteps)
		if done:
			# plt.plot(reward_array)
			# plt.show()
			reward_array = []
			rewards = np.roll(rewards,1,axis=0)
			if(info["reached"] == True):
				rewards[0] = 100
			else:
				rewards[0] = -100
			avg_reward = np.sum(rewards,axis=0)/10.0
			writer.add_scalar("training/average_raward",avg_reward, total_timesteps)
			
			print(f"Total Timesteps: {total_timesteps} Episode Num: {episode_num+1} Episode Timesteps: {episode_timesteps} Reward: {episode_reward:.3f} Average Reward: {avg_reward:.3f}")
			# Reset environment
			done = False
			state, info = env.reset()
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (total_timesteps + 1) % args.eval_freq == 0 and total_timesteps >= args.start_timesteps:
			np.save(f"rewards/{file_name}",rewards)
			policy.save(f"models/{file_name}")
			replay_buffer.save(f"replays/{file_name}")
			print("model_saved")
