import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import numpy.random as npr
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

length = 30
height = 20
num_env_steps = 60
success_num_trials = 100
num_pg_steps = 300
stepsize = 1
trajectory_batch_size = 30
manhattan_reward = True
state_representation = "two-dim"
location_normalized = True
encode_goal = True
sparsity = 0

verbose = True


env = GridEnvironment(length, height, 
	manhattan_reward= manhattan_reward, 
	state_representation = state_representation,
 	location_normalized = location_normalized,
 	encode_goal = encode_goal, 
 	sparsity = sparsity)


state_dim = env.get_state_dim()
num_actions = env.get_num_actions()
policy = NNPolicy(state_dim, num_actions)


base_rewards, base_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)
save_graph_diagnostic_image( env, policy, num_env_steps, 2,"Initial sample paths" , "./figs/initial_sample_paths.png")
save_grid_diagnostic_image(env, policy, num_env_steps, 2, "Initial sample paths", "./figs/initial_sample_paths_grid.png")


policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, 
	trajectory_batch_size, num_env_steps, verbose = verbose)

pg_rewards, pg_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)


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






save_graph_diagnostic_image(env, policy, num_env_steps, 2, "After PG", "./figs/after_pg.png")
save_grid_diagnostic_image(env, policy, num_env_steps, 2, "After PG", "./figs/after_pg_grid.png")



#print("Sum policy params after PG ", torch.sum(policy.policy_params))
print("Base success num ",  base_success_num)
print("Base rewards ", base_rewards)
print("PG success num ", pg_success_num)
print("PG rewards ",pg_rewards )





### Change the linear multiplier for the environment. 
P = np.arange(state_dim) + 1
P = P/(1.0*state_dim)
P = np.diag(P)
P = torch.tensor(P).float()
env.add_linear_transformation(P)


P_rewards, P_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)


print("Rewards after P change ", P_rewards)
print("Successes after P change ", P_success_num)


### Plot Successes
plt.close("all")
plt.title("Successes evolution")
plt.xlabel("Num trajectories")
plt.ylabel("Avg Successs")
plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_success_evolution, label = "avg successes", linewidth = 3.5, color = "red")
plt.savefig("./figs/avg_successes_P.png")
plt.close('all')


### Plot rewards
plt.close("all")
plt.title("Rewards evolution")
plt.xlabel("Num trajectories")
plt.ylabel("Avg Reward")
plt.plot((np.arange(num_pg_steps) + 1)*trajectory_batch_size,  training_reward_evolution, label = "avg rewards", linewidth = 3.5, color = "red")
plt.savefig("./figs/avg_rewards_P.png")
plt.close('all')





save_graph_diagnostic_image(env, policy, num_env_steps, 2, "P change", "./figs/after_P_change.png")
save_grid_diagnostic_image(env, policy, num_env_steps, 2, "P change", "./figs/after_P_change_grid.png")



#print("Sum policy params after PG ", torch.sum(policy.policy_params))
print("P change Base success num ",  base_success_num)
print("P change Base rewards ", base_rewards)
print("P change PG success num ", P_success_num)
print("P change PG rewards ",P_rewards )





