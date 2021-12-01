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
import IPython

import sys

sys.path.append('./')

from gridworlds.environments import *
from gridworlds.policies import *
from gridworlds.pg_learning import *
from gridworlds.environments_color import *
from gridworlds.policies_color import *
from gridworlds.pg_learning_color import *



length = 10
height = 15
manhattan_reward = False
sparsity = 0

num_colors = 8
num_placeholder_colors =10# 1
color_action_map  = [0, 1, 2, 3]*2
placeholder_color_prob = .5



env = ColorGridEnvironment(length, 
		height, 
		num_colors = num_colors,
		num_placeholder_colors = num_placeholder_colors,
		color_action_map = color_action_map,
		placeholder_color_prob = placeholder_color_prob,
		manhattan_reward= manhattan_reward, 
	 	sparsity = sparsity)


env.create_color_map()


env.reset_initial_and_destination(hard_instances = True, reinitialize_placeholders = False)




path = os.getcwd()
base_dir = "{}/tests/figs/colors/".format(path)
if not os.path.isdir(base_dir):
			try:
				# os.makedirs("{}/figs/T{}".format(path,T))
				# os.makedirs("{}/figs/T{}/strlength{}/".format(path,T,string_length))
				os.makedirs(base_dir)
							
			except OSError:
				print ("Creation of the directories failed")
			else:
				print ("Successfully created the directory ")



num_env_steps = 30
success_num_trials = 100
num_pg_steps = 50
hidden_layer =10
stepsize = 1
trajectory_batch_size = 30



verbose = True


state_dim = env.get_state_dim()



num_actions = env.get_num_actions()


policy = NNPolicy(state_dim, num_actions, hidden_layer = hidden_layer)
base_rewards, base_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)
save_graph_diagnostic_image( env, policy, num_env_steps, 10,"Initial sample paths" , "{}/initial_sample_paths_color_hidden{}.png".format(base_dir, hidden_layer))



policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, trajectory_batch_size, num_env_steps, verbose = verbose, supress_training_curve = True, logging_frequency = 10)

pg_rewards, pg_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)
save_graph_diagnostic_image(env, policy, num_env_steps, 10, "After PG", "{}/after_pg_symmetries_hidden{}.png".format(base_dir,hidden_layer))

optimal_policy = OptimalColorPolicy(env)
rew, suc, _ = test_policy(env, optimal_policy, success_num_trials, num_env_steps)

save_graph_diagnostic_image(env, optimal_policy, num_env_steps, 10, "Optimal policy", "{}/optimal_policy_paths.png".format(base_dir))
save_color_graph_diagnostic_image(env, env.color_map, "Color Map", "{}/color_map.png".format(base_dir))





string_with_placeholders = np.zeros((num_colors + num_placeholder_colors, env.get_num_actions()+1))
for i in range(num_colors + num_placeholder_colors):
	#index = -1
	if i < num_colors:
		index = i%4
	else:
		index = -1#random.choice(list(range(env.get_num_actions() + 1)))
	string_with_placeholders[i, index] = 1



color_policy = ColorPolicy(string_with_placeholders, state_dim, num_actions)

act = color_policy.get_action(env.get_state())


print("String with placeholders \n", string_with_placeholders)



policy, training_reward_evolution, training_success_evolution, all_rewards = learn_color_pg(env, color_policy, num_pg_steps, trajectory_batch_size, num_env_steps, verbose = verbose, supress_training_curve = True, logging_frequency = 10)



# IPython.embed()

env_multifood = ColorGridEnvironmentMultifood(
		length, 
		height, 
		num_food_sources = 1,
		num_colors = num_colors,
		num_placeholder_colors = num_placeholder_colors,
		color_action_map = color_action_map,
		placeholder_color_prob = placeholder_color_prob,
		pit = True,
		pit_colors = 4,
		manhattan_reward= manhattan_reward, 
		pit_type = "central",
	 	sparsity = sparsity)




save_color_graph_diagnostic_image(env_multifood, env_multifood.color_map, "Color Map Multifood", "{}/color_map_multifood.png".format(base_dir))


string_with_placeholders = np.zeros((num_colors + num_placeholder_colors, env.get_num_actions()+1))
for i in range(num_colors + num_placeholder_colors):
	#index = -1
	if i < num_colors:
		index = i%4
	else:
		index = -1#random.choice(list(range(env.get_num_actions() + 1)))
	string_with_placeholders[i, index] = 1



color_policy = ColorPolicy(string_with_placeholders, state_dim, num_actions)

num_multipolicy_pg_steps = 10
color_policy, training_reward_evolution_multifood, training_success_evolution_multifood, all_rewards = learn_color_pg(env_multifood, color_policy, num_multipolicy_pg_steps, trajectory_batch_size, num_env_steps, 
	multifood = True, verbose = verbose, supress_training_curve = True, logging_frequency = 10)





for i in range(8):
	save_graph_diagnostic_image(env_multifood, color_policy, num_env_steps, 1, "Optimal policy multifood", "{}/learned_policy_paths_multifood_day_{}.png".format(base_dir, i+1))
	env_multifood.start_day()
	# optimal_policy = OptimalColorPolicy(env_multifood)

env_multifood.reset_initial_and_food_sources()
env_multifood.create_color_map()


optimal_policy = OptimalColorPolicy(env_multifood)
for i in range(8):
	save_graph_diagnostic_image(env_multifood, optimal_policy, num_env_steps, 1, "Optimal policy multifood", "{}/optimal_policy_paths_multifood_day_{}.png".format(base_dir, i+1))
	env_multifood.start_day()
	optimal_policy = OptimalColorPolicy(env_multifood)



IPython.embed()






