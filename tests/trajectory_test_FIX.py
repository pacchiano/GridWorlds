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

from pg_learning import *


import IPython
from environments import *
from environments_color import *
from policies import *

USE_RAY = True

ray.init()



def train_trajectories( length, 
	height, tabular, location_based, location_normalized, 
	encode_goal, sparsity, goal_region_radius,
	num_env_steps, success_num_trials, num_pg_steps, hidden_layer, stepsize, trajectory_batch_size, base_dir, save_graph_image = True, graph_image_index = 0 ):


	env = GridEnvironmentNonMarkovian(length, 
			height, 
			manhattan_reward= False, 
			tabular = tabular,
			location_based = location_based,
			location_normalized = location_normalized,
			encode_goal = encode_goal, 
			sparsity = sparsity, 
			use_learned_reward_function = False,
			goal_region_radius = goal_region_radius)



	env.reset_initial_and_destination(hard_instances = True)


	verbose = True


	state_dim = env.get_state_dim()
	num_actions = env.get_num_actions()


	policy = NNPolicy(state_dim, num_actions, hidden_layer = hidden_layer)
	base_rewards, base_success_num = test_policy(env, policy, success_num_trials, num_env_steps, trajectory_feedback = True)

	if save_graph_image:
		save_graph_diagnostic_image( env, policy, num_env_steps, 2,"Initial sample paths - Trajectory" , "{}/initial_sample_paths_trajectory{}_{}.png".format(base_dir, hidden_layer, graph_image_index))

	policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, trajectory_batch_size, num_env_steps, 
		verbose = verbose, supress_training_curve = True, logging_frequency = 10, trajectory_feedback = True, reset_env = False)

	pg_rewards, pg_success_num = test_policy(env, policy, success_num_trials, num_env_steps, trajectory_feedback= True)
	
	if save_graph_image:
		save_graph_diagnostic_image(env, policy, num_env_steps, 2, "After PG", "{}/after_pg_sample_paths_trajectory{}_{}.png".format(base_dir,hidden_layer, graph_image_index))

	return training_reward_evolution

@ray.remote
def train_trajectories_remote( length, 
	height, tabular, location_based, location_normalized, 
	encode_goal, sparsity, goal_region_radius,
	num_env_steps, success_num_trials, num_pg_steps, hidden_layer, stepsize, trajectory_batch_size, base_dir, save_graph_image = True, graph_image_index = 0 ):

	return train_trajectories( length, 
	height, tabular, location_based, location_normalized, 
	encode_goal, sparsity, goal_region_radius,
	num_env_steps, success_num_trials, num_pg_steps, hidden_layer, stepsize, trajectory_batch_size, base_dir, save_graph_image = True , graph_image_index = graph_image_index)


def main():
	path = os.getcwd()

	base_dir = "{}/figs/trajectory_feedback/".format(path)

	if not os.path.isdir(base_dir):
				try:
					# os.makedirs("{}/figs/T{}".format(path,T))
					# os.makedirs("{}/figs/T{}/strlength{}/".format(path,T,string_length))
					os.makedirs(base_dir)
								
				except OSError:
					print ("Creation of the directories failed")
				else:
					print ("Successfully created the directory ")


	length = 10
	height = 15
	manhattan_reward = False
	tabular = False
	location_based  = True
	encode_goal = False
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
	if num_pg_steps < averaging_window:
		raise ValueError("Averaging window is smaller than num_pg_steps")
	if num_pg_steps%averaging_window != 0:
		raise ValueError("num_pg_steps is not divisible by averaging window size")

	# train_trajectories(length, 
	# 	height, tabular, location_based, location_normalized, 
	# 	encode_goal, sparsity, goal_region_radius,
	# 	num_env_steps, success_num_trials, num_pg_steps, hidden_layer, stepsize, trajectory_batch_size, base_dir)



	if USE_RAY:

		results = [	train_trajectories_remote.remote(length, 
		height, tabular, location_based, location_normalized, 
		encode_goal, sparsity, goal_region_radius,
		num_env_steps, success_num_trials, num_pg_steps, hidden_layer, stepsize, trajectory_batch_size, base_dir, graph_image_index = i) for i in range(num_experiments)]

		results = ray.get(results)
	else:
		results = [	train_trajectories(length, 
		height, tabular, location_based, location_normalized, 
		encode_goal, sparsity, goal_region_radius,
		num_env_steps, success_num_trials, num_pg_steps, hidden_layer, stepsize, trajectory_batch_size, base_dir, graph_image_index = i) for i in range(num_experiments)]

	
	training_reward_evolution_summary = np.zeros((num_experiments, num_pg_steps))

	for i in range(num_experiments):
		training_reward_evolution_summary[i, :] = results[i]



	training_reward_evolution_mean = np.mean(training_reward_evolution_summary, axis = 0)
	training_reward_evolution_mean = np.mean(training_reward_evolution_mean.reshape(-1, averaging_window), axis = 1)

	training_reward_evolution_std = np.std(training_reward_evolution_summary, axis = 0)
	training_reward_evolution_std = np.mean(training_reward_evolution_std.reshape(-1, averaging_window), axis = 1)


	# IPython.embed()



	plt.close("all")
	plt.title("Rewards evolution")
	plt.xlabel("Num trajectories")
	plt.ylabel("Rewards")
	plt.plot((np.arange(num_pg_steps/averaging_window) + 1)*averaging_window*trajectory_batch_size,  training_reward_evolution_mean, label = "avg rewards", linewidth = 3.5, color = "red")
	plt.fill_between((np.arange(num_pg_steps/averaging_window) + 1)*averaging_window*trajectory_batch_size, training_reward_evolution_mean - training_reward_evolution_std, 
					training_reward_evolution_mean + training_reward_evolution_std, color = "red", alpha = .1)

	# plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution_mean, label = "avg rewards", linewidth = 3.5, color = "red")
	# plt.fill_between((np.arange(num_pg_steps) + 1)*trajectory_batch_size, training_reward_evolution_mean - .5*training_reward_evolution_std, 
	# 				training_reward_evolution_mean + .5*training_reward_evolution_std, color = "red", alpha = .1)

	plt.legend(loc = "lower right")


	plt.savefig("{}/avg_rewardevolutionPG_hidden{}_nogoal.png".format(base_dir,hidden_layer))
	plt.close('all')


	import pickle
	pickle.dump( results, open("{}/results_data_nogoal.p".format(base_dir),  "wb")  )



	IPython.embed()
	raise ValueError("aslkdfm")


if __name__== "__main__":
	main()








