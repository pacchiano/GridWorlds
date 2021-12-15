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
import IPython

import sys

sys.path.append('./')

from gridworlds.environments import *
from gridworlds.policies import *
from gridworlds.pg_learning import *


from gridworlds.do_undo_maps import *

## Test Tabular Policy

length = 10
height = 10
num_env_steps = 30
success_num_trials = 100
num_pg_steps = 1000
stepsize = 1
trajectory_batch_size = 30
manhattan_reward = False
state_representation = "two-dim-encode-goal-location-normalized"

verbose = True


env = GridEnvironment(length, height, 
	manhattan_reward= manhattan_reward, 
	state_representation = state_representation)





state_dim = env.get_state_dim()
num_actions = env.get_num_actions()
policy = NNPolicy(state_dim, num_actions)



#IPython.embed()

base_rewards, base_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)
save_graph_diagnostic_image( env, policy, num_env_steps, 10,"Initial sample paths" , "./tests/figs/initial_sample_paths.png")


#optimizer = torch.optim.Adam([policy.policy_params], lr=0.01)

policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, 
	trajectory_batch_size, num_env_steps, verbose = verbose)

pg_rewards, pg_success_num, optimal_policy_trajectories = test_policy(env, policy, success_num_trials, num_env_steps)


#print("Sum policy params after PG ", torch.sum(policy.policy_params))
print("Base success num ",  base_success_num)
print("Base rewards ", base_rewards)
print("PG success num ", pg_success_num)
print("PG rewards ",pg_rewards )


do_undo_map = ReverseActionsDoUndoDiscrete()


env.add_do_undo(do_undo_map)


reversed_rewards, reversed_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)




print("Rewards after reversed actions change ", reversed_rewards)
print("Successes after reversed actions change ", reversed_success_num)



policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, 
	trajectory_batch_size, num_env_steps, verbose = verbose)



pg_rewards, pg_success_num, optimal_policy_trajectories = test_policy(env, policy, success_num_trials, num_env_steps)




print("PG success num after retraining", pg_success_num)
print("PG rewards after retraining",pg_rewards )




IPython.embed()














