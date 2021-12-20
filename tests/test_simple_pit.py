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
#from gridworlds.environments_color import *
#from gridworlds.policies_color import *
#from gridworlds.pg_learning_color import *
#from gridworlds.environments_smell import *
from gridworlds.pg_learning_multifood import *
from gridworlds.rendering_tools import *


from gridworlds.policies_multifood_grid import OptimalMultifoodPitPolicy


base_dir = "./tests/figs/pit/"#.format(path)
 
if not os.path.isdir(base_dir):
			try:
				os.makedirs(base_dir)
							
			except OSError:
				print ("Creation of the directories failed")
			else:
				print ("Successfully created the directory ")


length = 15
height = 15
verbose = True
num_env_steps = 30
success_num_trials = 100
num_pit_pg_steps = 30

hidden_layer =10
stepsize = 1
trajectory_batch_size = 30

state_representation = "two-dim-encode-goal-location-normalized" #


env_pit = GridEnvironmentPit(
		length, 
		height, 
		state_representation = state_representation,
		pit_type = "central",
		initialization_type = "avoiding_pit",
		length_rim = 3,
		height_rim = 2
		)



state_dim = env_pit.get_state_dim()
num_actions = env_pit.get_num_actions()
env_pit.reset_initial_and_destination(hard_instances = True)


policy = NNSoftmaxPolicy(state_dim, num_actions, hidden_layers = [50, 20])


policy, training_reward_evolution_pit, training_success_evolution_pit = learn_pg(env_pit, policy, 
	num_pit_pg_steps, 
	trajectory_batch_size, num_env_steps, verbose = verbose, 
	supress_training_curve = False, logging_frequency = 10)


env_pit.reset_initial_and_destination(hard_instances = True)


# for i in range(15):
# 	save_grid_diagnostic_image(env_pit, policy, num_env_steps, 1, 
# 		"Trained Policy Multifood", 
# 		"{}/learned_policy_paths_pit_trial_{}.png".format(base_dir, i+1))
# 	env_pit.reset_environment()



save_gif_diagnostic_image(env_pit, policy, num_env_steps, 1, 
	"Trained Policy Pit", "{}/learned_policy_paths_pit_trial".format(base_dir), 15)



policy = OptimalMultifoodPitPolicy(env_pit)





# for i in range(15):
# 	save_grid_diagnostic_image(env_pit, policy, num_env_steps, 1, 
# 		"Trained Policy Multifood", 
# 		"{}/optimal_policy_paths_pit_trial_{}.png".format(base_dir, i+1))
# 	env_pit.reset_environment()


save_gif_diagnostic_image(env_pit, policy, num_env_steps, 1, 
	"Trained Policy Pit", "{}/optimal_policy_paths_pit_trial".format(base_dir), 15)


IPython.embed()








