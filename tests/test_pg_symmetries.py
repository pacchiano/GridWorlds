from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np


import scipy.stats

import requests
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
from networkx import grid_graph, single_target_shortest_path_length, draw, spring_layout, draw_planar, grid_2d_graph, draw_shell, draw_spring, draw_spectral
from networkx.algorithms.components import is_connected
import random
import networkx as nx
import torch
import os
import time
import ray



import sys

sys.path.append('./')


import IPython



from gridworlds.environments import *
from gridworlds.policies import *
from gridworlds.pg_learning import *


use_ray = True

if use_ray:
	ray.init()

## Test Tabular Policy
@ray.remote
def run_reversal_experiments(length, height, num_env_steps, success_num_trials, num_pg_steps, stepsize, trajectory_batch_size, manhattan_reward, state_representation, location_normalized, 
	encode_goal, sparsity, 
	hidden_layer, model_path, base_dir):


	env = GridEnvironment(length, height, 
		manhattan_reward= manhattan_reward,
		state_representation = state_representation,
	 	location_normalized = location_normalized,
	 	encode_goal = encode_goal, 
	 	sparsity = sparsity)

	env.reset_initial_and_destination(hard_instances = True)

	state_dim = env.get_state_dim()
	num_actions = env.get_num_actions()
	policy = NNPolicy(state_dim, num_actions, hidden_layer = hidden_layer)


	### Save the top layer
	good_block_index = 0
	good_bias_block_index = 0
	good_parameter = None
	index = 0
	for parameter in policy.network.parameters():
		if tuple(parameter.shape) == (4,):
			print("Found special bias parameter")
			good_bias_block_index = index
		if tuple(parameter.shape) == (4, hidden_layer):
			print("Found special non bias parameter")
			good_block_index = index
			#good_parameter = parameter
		index += 1


	torch.save(policy.network,model_path)
	#torch.save(good_parameter,model_path)
	print("Saved Initial Policy!")



	base_rewards, base_success_num, trajectories = test_policy(env, policy, success_num_trials, num_env_steps)
	if not use_ray:
		save_graph_diagnostic_image( env, policy, num_env_steps, 10,"Initial sample paths" , "{}/initial_sample_paths_symmetries_hidden{}.png".format(base_dir, hidden_layer))
	print("Tested Initial Random Policy")
	#optimizer = torch.optim.Adam([policy.policy_params], lr=0.01)

	initial_time  = time.time()

	policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, 
		trajectory_batch_size, num_env_steps, verbose = verbose, supress_training_curve = True, logging_frequency = 10)
	final_time = time.time()

	vanilla_training_time = final_time - initial_time

	print("Trained Initial Policy")
	pg_rewards, pg_success_num, trajectories= test_policy(env, policy, success_num_trials, num_env_steps)
	
	


	if not use_ray:
		save_graph_diagnostic_image(env, policy, num_env_steps, 10, "After PG", "{}/after_pg_symmetries_hidden{}.png".format(base_dir,hidden_layer))



	print("DIAGNOSTICS FOR INITIAL VANILLA PG")
	print("Base success num ",  base_success_num)
	print("Base rewards ", base_rewards)
	print("PG success num ", pg_success_num)
	print("PG rewards ",pg_rewards )



	print("Starting test for random core")
	randomcore_policy_network = torch.load(model_path)


	with torch.no_grad():
		for parameter, i in zip(randomcore_policy_network.parameters(), range(index)):
			if i not in [good_block_index, good_bias_block_index]:
				parameter.requires_grad= False



	randomcore_policy = NNPolicy(state_dim, num_actions, hidden_layer = hidden_layer)
	randomcore_policy.network = randomcore_policy_network



	base_rewards_randomcore, base_success_num_randomcore, trajectories = test_policy(env, randomcore_policy, success_num_trials, num_env_steps)
	if not use_ray:
		save_graph_diagnostic_image( env, randomcore_policy, num_env_steps, 10,"Initial sample paths randomcore" , "{}/initial_sample_paths_randomcore_hidden{}.png".format(base_dir,hidden_layer))
	print("Tested Initial RandomCore Policy")
	initial_time  = time.time()

	randomcore_policy, training_reward_evolution_randomcore, training_success_evolution_randomcore = learn_pg(env, randomcore_policy, num_pg_steps, 
		trajectory_batch_size, num_env_steps, verbose = verbose, supress_training_curve = True, logging_frequency = 10)
	final_time  = time.time()
	randomcore_training_time = final_time - initial_time


	print("Trained RandomCore Policy")
	pg_rewards_randomcore, pg_success_num_randomcore, trajectories = test_policy(env, randomcore_policy, success_num_trials, num_env_steps)
	if not use_ray:
		save_graph_diagnostic_image(env, randomcore_policy, num_env_steps, 10, "After PG randomcore", "{}/after_pg_randomcore_hidden{}.png".format(base_dir,hidden_layer))


	print("DIAGNOSTICS FOR RANDOMCORE VANILLA PG")
	print("Base success num ",  base_success_num_randomcore)
	print("Base rewards ", base_rewards_randomcore)
	print("PG success num ", pg_success_num_randomcore)
	print("PG rewards ",pg_rewards_randomcore )


	print("parameter diagnostics")
	original_policy_network = torch.load(model_path)

	with torch.no_grad():
		for parameter, i, old_parameter in zip(randomcore_policy.network.parameters(), range(index), original_policy_network.parameters()):
			if i not in [good_block_index, good_bias_block_index]:
				print("New and old parameters should be the same.")
			else:
				print("New and old parameters should be different.")
			print("new param value ", parameter)
			print("old parameter ", old_parameter)




	env.reverse_environment()


	print("Starting the symmetries test")
	
	original_policy_network = torch.load(model_path)


	with torch.no_grad():
		for parameter, i, old_parameter in zip(policy.network.parameters(), range(index), original_policy_network.parameters()):
			if i == good_block_index or i == good_bias_block_index:
				print("Substituted parameter {}".format(i))
				#print("old param value ", parameter)#print("parameter now ", parameter)
				#print("old parameter ", old_parameter)
				parameter.copy_(old_parameter)
			
			else:
				parameter.requires_grad= False

		




	base_rewards_symmetries, base_success_num_symmetries, trajectories = test_policy(env, policy, success_num_trials, num_env_steps)

	if not use_ray:
		save_graph_diagnostic_image( env, policy, num_env_steps, 10,"Initial sample paths reversed" , "{}/initial_sample_paths_reversed_symmetries_hidden{}.png".format(base_dir, hidden_layer))

	initial_time  = time.time()


	policy, training_reward_evolution_symmetries, training_success_evolution_symmetries = learn_pg(env, policy, num_pg_steps, 
		trajectory_batch_size, num_env_steps, verbose = verbose, supress_training_curve = True)

	final_time = time.time()
	reversed_frozen_training_time = final_time - initial_time

	pg_rewards_symmetries, pg_success_num_symmetries, trajectories = test_policy(env, policy, success_num_trials, num_env_steps)

	if not use_ray:
		save_graph_diagnostic_image(env, policy, num_env_steps, 10, "After PG", "{}/after_pg_reversed_symmetries_hidden{}.png".format(base_dir, hidden_layer))


	#print("Sum policy params after PG ", torch.sum(policy.policy_params))


	print("DIAGNOSTICS FOR REVERSED SYMMETRIES VANILLA PG - FREEZE LEARNED CORE")
	print("Base success num ",  base_success_num_symmetries)
	print("Base rewards ", base_rewards_symmetries)
	print("PG success num ", pg_success_num_symmetries)
	print("PG rewards ",pg_rewards_symmetries)

	with torch.no_grad():
		for parameter, i, old_parameter in zip(policy.network.parameters(), range(index), original_policy_network.parameters()):
			if i == good_block_index or i==good_bias_block_index:
				print("New and old parameters should be different")
			else:
				print("New and old parameters should be the same")

			print("new param value ", parameter)
				#print("parameter now ", parameter)
			print("old parameter ", old_parameter)
				#parameter.copy_(old_parameter)
			# else:
			# 	#parameter.requires_grad= False
				



	### TEST SYMMETRIES GRAdIENTS THROUGH CORE ### 
	original_policy_network = torch.load(model_path)


	with torch.no_grad():
		for parameter, i, old_parameter in zip(policy.network.parameters(), range(index), original_policy_network.parameters()):
			if i == good_block_index or i == good_bias_block_index:
				print("Substituted parameter {}".format(i))
				parameter.copy_(old_parameter)
			
			else:
				parameter.requires_grad= True

		




	base_rewards_symmetries_grads, base_success_num_symmetries_grads, trajectories = test_policy(env, policy, success_num_trials, num_env_steps)

	if not use_ray:
		save_graph_diagnostic_image( env, policy, num_env_steps, 10,"Initial sample paths reversed grads" , "{}/initial_sample_paths_reversed_symmetries_grads_hidden{}.png".format(base_dir, hidden_layer))

	initial_time  = time.time()

	policy, training_reward_evolution_symmetries_grads, training_success_evolution_symmetries_grads = learn_pg(env, policy, num_pg_steps, 
		trajectory_batch_size, num_env_steps, verbose = verbose, supress_training_curve = True)

	final_time = time.time()
	reversed_unfrozen_training_time = final_time - initial_time

	pg_rewards_symmetries_grads, pg_success_num_symmetries_grads, trajectories = test_policy(env, policy, success_num_trials, num_env_steps)

	if not use_ray:
		save_graph_diagnostic_image(env, policy, num_env_steps, 10, "After PG", "{}/after_pg_reversed_symmetries_hidden{}.png".format(base_dir, hidden_layer))


	#print("Sum policy params after PG ", torch.sum(policy.policy_params))


	print("DIAGNOSTICS FOR REVERSED SYMMETRIES VANILLA PG - LEARN THROUGH CORE")
	print("Base success num ",  base_success_num_symmetries_grads)
	print("Base rewards ", base_rewards_symmetries_grads)
	print("PG success num ", pg_success_num_symmetries_grads)
	print("PG rewards ",pg_rewards_symmetries_grads)

	# IPython.embed()
	# raise ValueError("asdflkm")

	return training_success_evolution, training_success_evolution_symmetries, training_success_evolution_randomcore, training_success_evolution_symmetries_grads, training_reward_evolution, training_reward_evolution_symmetries, training_reward_evolution_randomcore, training_reward_evolution_symmetries_grads, vanilla_training_time, randomcore_training_time, reversed_frozen_training_time, reversed_unfrozen_training_time

length = 20
height = 20
num_env_steps = 30
success_num_trials = 100
num_pg_steps = 100
hidden_layer =10

stepsize = 1
trajectory_batch_size = 30
manhattan_reward = True




state_representation = "two-dim"


location_normalized = True#True
encode_goal = True
sparsity = 0

verbose = True
path = os.getcwd()

num_experiments = 2





base_dir = "./tests/figs/symmetries/T{}/grid{}_{}/Manhattan{}/".format(path,num_pg_steps, length, height,manhattan_reward)

if not os.path.isdir(base_dir):
			try:
				# os.makedirs("{}/figs/T{}".format(path,T))
				# os.makedirs("{}/figs/T{}/strlength{}/".format(path,T,string_length))
				os.makedirs(base_dir)
							
			except OSError:
				print ("Creation of the directories failed")
			else:
				print ("Successfully created the directory ")



for hidden_layer in [2, 3, 5, 10, 20, 30]:



	#model_path = "./models/policy_netowrk_symmetries_hidden_{}_expnum_{}.pt".format(hidden_layer)
	#model_path = "./models/policy_netowrk_symmetries_hidden_{}.pt".format(hidden_layer)


	if use_ray:

		results = [run_reversal_experiments.remote(length, height, num_env_steps, success_num_trials, num_pg_steps, stepsize, trajectory_batch_size, manhattan_reward, state_representation, location_normalized, encode_goal, sparsity, 
	hidden_layer, "./tests/models/policy_netowrk_symmetries_hidden_{}_expnum_{}.pt".format(hidden_layer, i), base_dir) for i in range(num_experiments)]

		results = ray.get(results)
	else:
		results = [run_reversal_experiments(length, height, num_env_steps, success_num_trials, num_pg_steps, stepsize, trajectory_batch_size, manhattan_reward, state_representation, location_normalized, encode_goal, sparsity, 
	hidden_layer, "./tests/models/policy_netowrk_symmetries_hidden_{}_expnum_{}.pt".format(hidden_layer, i), base_dir) for i in range(num_experiments)]

	
	training_success_evolution_summary = np.zeros((num_experiments, num_pg_steps))
	training_success_evolution_symmetries_summary = np.zeros((num_experiments, num_pg_steps))
	training_success_evolution_randomcore_summary = np.zeros((num_experiments, num_pg_steps))
	training_success_evolution_symmetries_grads_summary = np.zeros((num_experiments, num_pg_steps))

	training_reward_evolution_summary = np.zeros((num_experiments, num_pg_steps))
	training_reward_evolution_symmetries_summary = np.zeros((num_experiments, num_pg_steps))
	training_reward_evolution_randomcore_summary = np.zeros((num_experiments, num_pg_steps))
	training_reward_evolution_symmetries_grads_summary = np.zeros((num_experiments, num_pg_steps))

	vanilla_training_time_summary = []
	randomcore_training_time_summary = []
	reversed_frozen_training_time_summary = []
	reversed_unfrozen_training_time_summary = []




	for i in range(num_experiments):
		training_success_evolution, training_success_evolution_symmetries, training_success_evolution_randomcore, training_success_evolution_symmetries_grads, training_reward_evolution, training_reward_evolution_symmetries, training_reward_evolution_randomcore, training_reward_evolution_symmetries_grads, vanilla_training_time, randomcore_training_time, reversed_frozen_training_time, reversed_unfrozen_training_time = results[i]
		#training_success_evolution, training_success_evolution_symmetries, training_success_evolution_randomcore, training_reward_evolution, training_reward_evolution_symmetries, training_reward_evolution_randomcore = results[i]
		training_success_evolution_summary[i, :] = training_success_evolution
		training_success_evolution_symmetries_summary[i, :] = training_success_evolution_symmetries
		training_success_evolution_randomcore_summary[i, :]= training_success_evolution_randomcore
		training_success_evolution_symmetries_grads_summary[i,:] = training_success_evolution_symmetries_grads

		training_reward_evolution_summary[i, :] = training_reward_evolution
		training_reward_evolution_symmetries_summary[i,:] = training_reward_evolution_symmetries
		training_reward_evolution_randomcore_summary[i,:] = training_reward_evolution_randomcore
		training_reward_evolution_symmetries_grads_summary[i,:] = training_reward_evolution_symmetries_grads

		vanilla_training_time_summary.append(vanilla_training_time)
		randomcore_training_time_summary.append(randomcore_training_time)
		reversed_frozen_training_time_summary.append(reversed_frozen_training_time)
		reversed_unfrozen_training_time_summary.append(reversed_unfrozen_training_time)


	training_success_evolution_mean = np.mean(training_success_evolution_summary, axis = 0)
	training_success_evolution_std = np.std(training_success_evolution_summary, axis = 0)

	training_success_evolution_symmetries_mean = np.mean(training_success_evolution_symmetries_summary, axis = 0)
	training_success_evolution_symmetries_std = np.std(training_success_evolution_symmetries_summary, axis = 0)


	training_success_evolution_randomcore_mean = np.mean(training_success_evolution_randomcore_summary, axis = 0)
	training_success_evolution_randomcore_std = np.std(training_success_evolution_randomcore_summary, axis = 0)

	training_success_evolution_symmetries_grads_mean = np.mean(training_success_evolution_symmetries_grads_summary, axis = 0)	
	training_success_evolution_symmetries_grads_std = np.std(training_success_evolution_symmetries_grads_summary, axis = 0)

	training_reward_evolution_mean = np.mean(training_reward_evolution_summary, axis = 0) 
	training_reward_evolution_std = np.std(training_reward_evolution_summary, axis = 0) 


	training_reward_evolution_symmetries_mean = np.mean(training_reward_evolution_symmetries_summary, axis = 0) 
	training_reward_evolution_symmetries_std = np.std(training_reward_evolution_symmetries_summary, axis = 0) 

	training_reward_evolution_randomcore_mean = np.mean(training_reward_evolution_randomcore_summary, axis = 0) 
	training_reward_evolution_randomcore_std = np.std(training_reward_evolution_randomcore_summary, axis = 0) 

	training_reward_evolution_symmetries_grads_mean = np.mean(training_reward_evolution_symmetries_grads_summary, axis = 0)
	training_reward_evolution_symmetries_grads_std = np.std(training_reward_evolution_symmetries_grads_summary, axis = 0)


	plt.close("all")
	plt.title("Successes evolution")
	plt.xlabel("Num trajectories")
	plt.ylabel("Avg Successs")
	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_success_evolution_mean, label = "avg successes", linewidth = 3.5, color = "red")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_success_evolution_mean - .5*training_success_evolution_std, 
					training_success_evolution_mean + .5*training_success_evolution_std, color = "red", alpha = .1)

	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_success_evolution_symmetries_mean, label = "avg successes reversed", linewidth = 3.5, color = "blue")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_success_evolution_symmetries_mean - .5*training_success_evolution_symmetries_std, 
					training_success_evolution_symmetries_mean + .5*training_success_evolution_symmetries_std, color = "blue", alpha = .1)

	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_success_evolution_randomcore_mean, label = "avg successes randomcore", linewidth = 3.5, color = "green")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_success_evolution_randomcore_mean - .5*training_success_evolution_randomcore_std, 
					training_success_evolution_randomcore_mean + .5*training_success_evolution_randomcore_std, color = "green", alpha = .1)

	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_success_evolution_symmetries_grads_mean, label = "avg successes reversed grad", linewidth = 3.5, color = "violet")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_success_evolution_symmetries_grads_mean - .5*training_reward_evolution_symmetries_std, 
					training_success_evolution_symmetries_grads_mean + .5*training_reward_evolution_symmetries_std, color = "violet", alpha = .1)



	plt.legend(loc = "lower right")


	plt.savefig("{}/avg_successes_symmetries_hidden{}.png".format(base_dir,hidden_layer))
	plt.close('all')


	### Plot rewards
	plt.close("all")
	plt.title("Rewards evolution")
	plt.xlabel("Num trajectories")
	plt.ylabel("Avg Reward")
	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution_mean, label = "avg rewards", linewidth = 3.5, color = "red")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_reward_evolution_mean - .5*training_reward_evolution_std, 
					training_reward_evolution_mean + .5*training_reward_evolution_std, color = "red", alpha = .1)

	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution_symmetries_mean, label = "avg rewards reversed", linewidth = 3.5, color = "blue")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_reward_evolution_symmetries_mean - .5*training_reward_evolution_symmetries_std, 
					training_reward_evolution_symmetries_mean + .5*training_reward_evolution_symmetries_std, color = "blue", alpha = .1)

	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution_randomcore_mean, label = "avg rewards randomcore", linewidth = 3.5, color = "green")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_reward_evolution_randomcore_mean - .5*training_reward_evolution_randomcore_std, 
					training_reward_evolution_randomcore_mean + .5*training_reward_evolution_randomcore_std, color = "green", alpha = .1)

	plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution_symmetries_grads_mean, label = "avg rewards reversed grad", linewidth = 3.5, color = "violet")
	plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_reward_evolution_symmetries_grads_mean - .5*training_reward_evolution_symmetries_grads_std, 
					training_reward_evolution_symmetries_grads_mean + .5*training_reward_evolution_symmetries_grads_std, color = "violet", alpha = .1)


	plt.legend(loc = "lower right")
	plt.savefig("{}/avg_rewards_symmetries_hidden{}.png".format(base_dir, hidden_layer))
	plt.close('all')


	runtime_info = ""
	runtime_info += "Mean vanilla training time " + str(  np.mean(vanilla_training_time_summary)) +  "\n"
	runtime_info += "Mean randomcore training time " +  str(np.mean(randomcore_training_time_summary)) + "\n"
	runtime_info += "Mean reversed frozen training time " + str(np.mean(reversed_frozen_training_time_summary)) + "\n"
	runtime_info += "Mean reversed unfrozen training time " +  str(np.mean(reversed_unfrozen_training_time_summary)) + "\n"


	print(
		"Mean vanilla training time ",   np.mean(vanilla_training_time_summary), "\n",
		"Mean randomcore training time ", np.mean(randomcore_training_time_summary), "\n",
		"Mean reversed frozen training time ", np.mean(reversed_frozen_training_time_summary), "\n",
		"Mean reversed unfrozen training time ", np.mean(reversed_unfrozen_training_time_summary),"\n",
		)

	runtime_file = open("{}/runtime_info_hidden{}.txt".format(base_dir, hidden_layer), "w")
	runtime_file.write(runtime_info)
	runtime_file.close()
				
	#IPython.embed()
					









