import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

length = 10
height = 15
manhattan_reward = False
tabular = False
location_based  = True
encode_goal = True
sparsity = 0
location_normalized = True
num_colors = 8
num_placeholder_colors =10# 1
color_action_map  = [0, 1, 2, 3]*2
placeholder_color_prob = .5
goal_region_radius = 2

num_env_steps = 30
success_num_trials = 100
num_pg_steps = 10000
hidden_layer =10
stepsize = 1
trajectory_batch_size = 30
num_experiments = 20
averaging_window = 10


path = os.getcwd()

base_dir = "{}/figs/trajectory_feedback/".format(path)
results = pickle.load(open("{}/results_data.p".format(base_dir),  "rb"))


training_reward_evolution_summary = np.zeros((num_experiments, num_pg_steps))

for i in range(num_experiments):
	training_reward_evolution_summary[i, :] = results[i]



training_reward_evolution_mean = np.mean(training_reward_evolution_summary, axis = 0)
training_reward_evolution_mean = np.mean(training_reward_evolution_mean.reshape(-1, averaging_window), axis = 1)

training_reward_evolution_std = np.std(training_reward_evolution_summary, axis = 0)
training_reward_evolution_std = np.mean(training_reward_evolution_std.reshape(-1, averaging_window), axis = 1)

plt.rcParams.update({'font.size':15})
plt.ticklabel_format(axis = "x", style = "sci", scilimits = (0,0))

plt.title("Evolution of Rewards")
plt.xlabel("Number of trajectories (N)")
plt.ylabel("Reward")


plt.plot((np.arange(num_pg_steps/averaging_window) + 1)*averaging_window*trajectory_batch_size,  training_reward_evolution_mean, label = "Average Reward", linewidth = 3.5, color = "red")
plt.fill_between((np.arange(num_pg_steps/averaging_window) + 1)*averaging_window*trajectory_batch_size, training_reward_evolution_mean - training_reward_evolution_std, 
				training_reward_evolution_mean + training_reward_evolution_std, color = "red", alpha = .1)

# plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution_mean, label = "avg rewards", linewidth = 3.5, color = "red")
# plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_reward_evolution_mean - .5*training_reward_evolution_std, 
# 				training_reward_evolution_mean + .5*training_reward_evolution_std, color = "red", alpha = .1)

plt.legend(loc = "lower right")


plt.savefig("{}/avg_rewardevolutionPG_hidden{}.png".format(base_dir,hidden_layer))
plt.close('all')
