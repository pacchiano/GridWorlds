from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
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
from pg_learning import *

import IPython
from environments import *
from policies import *

## Test Tabular Policy

length = 10
height = 10
num_env_steps = 30
success_num_trials = 100
num_pg_steps = 100
stepsize = 1
trajectory_batch_size = 30
manhattan_reward = False


tabular = True
location_based = False#True
location_normalized = True#True
encode_goal = True
sparsity = 0

verbose = True


env = GridEnvironment(length, height, 
	manhattan_reward= manhattan_reward, 
	tabular = tabular,
 	location_based = location_based,
 	location_normalized = location_normalized,
 	encode_goal = encode_goal, 
 	sparsity = sparsity, 
 	use_learned_reward_function = False)

state_dim = env.get_state_dim()
num_actions = env.get_num_actions()
policy = NNPolicy(state_dim, num_actions)


base_rewards, base_success_num = test_policy(env, policy, success_num_trials, num_env_steps)
save_graph_diagnostic_image( env, policy, num_env_steps, 10,"Initial sample paths" , "./figs/initial_sample_paths.png")


#optimizer = torch.optim.Adam([policy.policy_params], lr=0.01)

policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, 
	trajectory_batch_size, num_env_steps, verbose = verbose)

pg_rewards, pg_success_num = test_policy(env, policy, success_num_trials, num_env_steps)


### Plot Successes
plt.close("all")
plt.title("Successes evolution")
plt.xlabel("Num trajectories")
plt.ylabel("Avg Successs")
plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_success_evolution, label = "avg successes", linewidth = 3.5, color = "red")
plt.savefig("./figs/avg_successes.png")
plt.close('all')


### Plot rewards
plt.close("all")
plt.title("Rewards evolution")
plt.xlabel("Num trajectories")
plt.ylabel("Avg Reward")
plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution, label = "avg rewards", linewidth = 3.5, color = "red")
plt.savefig("./figs/avg_rewards.png")
plt.close('all')



save_graph_diagnostic_image(env, policy, num_env_steps, 10, "After PG", "./figs/after_pg.png")

#print("Sum policy params after PG ", torch.sum(policy.policy_params))
print("Base success num ",  base_success_num)
print("Base rewards ", base_rewards)
print("PG success num ", pg_success_num)
print("PG rewards ",pg_rewards )