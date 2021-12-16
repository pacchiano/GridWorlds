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
from gridworlds.rewards import SimpleIndicatorReward, ManhattanReward
from gridworlds.rendering_tools import save_grid_diagnostic_image

## Test Tabular Policy

length = 10
height = 10
num_env_steps = 30
success_num_trials = 100
num_pg_steps = 700
stepsize = 1
trajectory_batch_size = 30
state_representation = "two-dim-encode-goal-location-normalized"

verbose = True


reward_function = ManhattanReward()
reward_function = SimpleIndicatorReward()


env = GridEnvironment(length, height, 
	state_representation = state_representation)

env.add_reward_function(reward_function)






state_dim = env.get_state_dim()
num_actions = env.get_num_actions()
policy = NNSoftmaxPolicy(state_dim, num_actions,  hidden_layers = [12,20])



base_rewards, base_success_num, _ = test_policy(env, policy, success_num_trials, num_env_steps)
save_grid_diagnostic_image( env, policy, num_env_steps, 
	10,"Initial sample paths" , "./tests/figs/reverse_initial_sample_paths.png")


#optimizer = torch.optim.Adam([policy.policy_params], lr=0.01)

policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, 
	trajectory_batch_size, num_env_steps, verbose = verbose)

pg_rewards, pg_success_num, optimal_policy_trajectories = test_policy(env, policy, 
	success_num_trials, num_env_steps)


#print("Sum policy params after PG ", torch.sum(policy.policy_params))
print("Base success num ",  base_success_num)
print("Base rewards ", base_rewards)
print("PG success num ", pg_success_num)
print("PG rewards ",pg_rewards )



save_grid_diagnostic_image( env, policy, num_env_steps, 
	10,"After PG sample paths" , "./tests/figs/reverse_afterPG_sample_paths.png")


do_undo_map = ReverseActionsDoUndoDiscrete()
env.add_do_undo(do_undo_map)

IPython.embed()
reversed_rewards, reversed_success_num, _ = test_policy(env, policy, 
	success_num_trials, num_env_steps)

save_grid_diagnostic_image( env, policy, num_env_steps, 
	10,"After reverse sample paths" , "./tests/figs/reverse_reversed_sample_paths.png")



print("Rewards after reversed actions change ", reversed_rewards)
print("Successes after reversed actions change ", reversed_success_num)



policy, training_reward_evolution, training_success_evolution = learn_pg(env, policy, num_pg_steps, 
	trajectory_batch_size, num_env_steps, verbose = verbose)



pg_rewards, pg_success_num, optimal_policy_trajectories = test_policy(env, policy, 
	success_num_trials, num_env_steps)




print("PG success num after retraining", pg_success_num)
print("PG rewards after retraining",pg_rewards )

save_grid_diagnostic_image( env, policy, num_env_steps, 
	10,"After reverse sample paths" , "./tests/figs/reverse_afterPGreversed_sample_paths.png")




IPython.embed()














