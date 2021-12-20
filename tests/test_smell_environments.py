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
import pkg_resources


import IPython

import sys

sys.path.append('./')



from gridworlds.environments import *
from gridworlds.policies import *
from gridworlds.pg_learning import *
from gridworlds.environments_color import *
from gridworlds.policies_color import *
from gridworlds.pg_learning_color import *
from gridworlds.environments_smell import *
from gridworlds.pg_learning_multifood import *


length = 15
height = 15
manhattan_reward = False
sparsity = 0
num_colors = 8
num_placeholder_colors =4# 1
color_action_map  = [0, 1, 2, 3]*2
placeholder_color_prob = .5

num_food_sources = 2
pit_colors = 4

up_rim = 5
side_rim = 4



IPython.embed()



#path = os.getcwd()

base_dir = "./tests/figs/multifood/"#.format(path)

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
initialization_type = 'random'#"near_pit"





def analysis(food_weight, pit_weight):

	env_multifood = GridEnvironmentMultifoodSmell(
			length, 
			height, 
			state_representation = "pit-foodsources",
			num_food_sources = num_food_sources,
			pit = False,
			manhattan_reward= manhattan_reward, 
			pit_type = "central",
		 	initialization_type = initialization_type,
		 	sparsity = sparsity,
		 	length_rim = up_rim,
		 	height_rim = side_rim)


	state_dim = env_multifood.get_state_dim()
	num_actions = env_multifood.get_num_actions()

	env_multifood.reset_initial_and_food_sources()

	reward_weight = torch.stack([torch.ones(2).float()*food_weight , torch.ones(2).float()*pit_weight]).flatten()
	reward_weight = torch.tensor([food_weight, pit_weight]).float()

	reward_weight = -torch.ones(num_food_sources)



	env_multifood.set_reward_weights(reward_weight)


	policy = NNSoftmaxPolicy(state_dim, num_actions)

	num_multipolicy_pg_steps = 50


	policy, training_reward_evolution_multifood, training_success_evolution_multifood, all_rewards = learn_multifood_pg(env_multifood, policy, num_multipolicy_pg_steps, 
		trajectory_batch_size, num_env_steps, 
		multifood = True, verbose = verbose, supress_training_curve = True, logging_frequency = 10)




	env_multifood.reset_initial_and_food_sources()

	for i in range(15):
		save_grid_diagnostic_image(env_multifood, policy, num_env_steps, 1, "Optimal policy multifood", "{}/{}_{}_learned_policy_paths_smell_multifood_day_{}.png".format(base_dir, food_weight, pit_weight, i+1))
		env_multifood.start_day()
		# optimal_policy = OptimalColorPolicy(env_multifood)



	return training_success_evolution_multifood, training_reward_evolution_multifood




food_weights = list(np.linspace(-1,1,3))
pit_weights = list(np.linspace(-1,1,3))



for food_weight, pit_weight in itertools.product(food_weights, pit_weights):
	success_evolution, reward_evolution = analysis(food_weight, pit_weight)

	print(" Success Evolution " , success_evolution, 'Reward evolution ', reward_evolution,  "food weight {} pit weight {}.".format(food_weight, pit_weight))



IPython.embed()






