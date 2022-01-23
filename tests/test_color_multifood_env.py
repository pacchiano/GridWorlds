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
from gridworlds.rendering_tools import save_color_grid_diagnostic_image, save_grid_diagnostic_image

length = 15
height = 10
num_fixed_colors = 8
num_placeholder_colors =4# 1
fixed_color_action_map  = [0, 1, 2, 3]*2
placeholder_color_prob = .5

num_food_sources = 2
pit_colors = 4

up_rim = 4
side_rim = 3





#path = os.getcwd()

base_dir = "./figs/colors/"#.format(path)

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
initialization_type = "avoiding_pit"



env_multifood = ColorGridEnvironmentMultifood(
		length, 
		height, 
		num_food_sources = num_food_sources,
		num_fixed_colors = num_fixed_colors,
		num_placeholder_colors = num_placeholder_colors,
		fixed_color_action_map = fixed_color_action_map,
		placeholder_color_prob = placeholder_color_prob,
		pit_colors = pit_colors,
		pit_type = "central",
	 	initialization_type = initialization_type,
	 	length_rim = up_rim,
	 	height_rim = side_rim)


state_dim = env_multifood.get_state_dim()
num_actions = env_multifood.get_num_actions()

# save_color_grid_diagnostic_image(env_multifood, env_multifood.color_map, "Color Map Multifood", "{}/color_map_multifood.png".format(base_dir))

#IPython.embed()


save_color_grid_diagnostic_image(env_multifood, env_multifood.color_map, "Color Map Multifood", "{}/color_map_multifood1.png".format(base_dir), 
	display_optimal_action = True, add_grid_colors = True)

save_color_grid_diagnostic_image(env_multifood, env_multifood.color_map, "Color Map Multifood", "{}/color_map_multifood2.png".format(base_dir), 
	display_optimal_action = False, add_grid_colors = True)
save_color_grid_diagnostic_image(env_multifood, env_multifood.color_map, "Color Map Multifood", "{}/color_map_multifood3.png".format(base_dir), 
	display_optimal_action = True, add_grid_colors = False)

save_color_grid_diagnostic_image(env_multifood, env_multifood.color_map, "Color Map Multifood", "{}/color_map_multifood4.png".format(base_dir), 
	display_optimal_action = False, add_grid_colors = False)



env_multifood.reset_initial_and_food_sources()
env_multifood.create_color_map()


optimal_policy = OptimalColorPolicy(env_multifood)
for i in range(15):
	save_grid_diagnostic_image(env_multifood, optimal_policy, num_env_steps, 
		1, "Optimal policy multifood", "{}/optimal_policy_paths_multifood_day_{}.png".format(base_dir, i+1))
	env_multifood.start_day()
	optimal_policy = OptimalColorPolicy(env_multifood)

env_multifood.reset_initial_and_food_sources()
env_multifood.create_color_map()






# IPython.embed()
# raise ValueError("asdlfkm")

string_with_placeholders = np.zeros((num_fixed_colors + num_placeholder_colors, env_multifood.get_num_actions()+1))
for i in range(num_fixed_colors + num_placeholder_colors):
	if i < num_fixed_colors:
		index = i%4
	else:
		index = -1
	string_with_placeholders[i, index] = 1



color_policy = ColorPolicy(string_with_placeholders, state_dim, num_actions)

num_multipolicy_pg_steps = 10
color_policy, training_reward_evolution_multifood, training_success_evolution_multifood, all_rewards = learn_color_pg(env_multifood, color_policy, num_multipolicy_pg_steps, trajectory_batch_size, num_env_steps, 
	multifood = True, verbose = verbose, supress_training_curve = True, logging_frequency = 10)





for i in range(15):
	save_grid_diagnostic_image(env_multifood, color_policy, num_env_steps, 1, "Optimal policy multifood", "{}/learned_policy_paths_multifood_day_{}.png".format(base_dir, i+1))
	env_multifood.start_day()
	# optimal_policy = OptimalColorPolicy(env_multifood)





IPython.embed()






