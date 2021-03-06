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
import random
import networkx as nx
import torch
import IPython
import string

import itertools


from matplotlib import colors as colors_matplotlib
from copy import deepcopy


from .policies import *
from .environments import *
from .environments_color import *

def learn_color_pg(env, color_policy, num_pg_steps, trajectory_batch_size, 
	num_env_steps, multifood = False, verbose = False, 
	supress_training_curve = True, logging_frequency = None, reset_env = True):
	#optimizer = torch.optim.Adam([policy.policy_params], lr=0.01)
	if logging_frequency == None:
		logging_frequency = num_pg_steps


	is_cuda = next(color_policy.network.parameters()).is_cuda
	# IPython.embed()
	# raise ValueError("asdflkm")
	print("Using CUDA ", is_cuda)

	optimizer = torch.optim.Adam(color_policy.network.parameters(), lr = 0.01)

	training_success_evolution = []
	training_reward_evolution = []
	all_rewards = []

	if multifood and reset_env:
			env.reset_initial_and_food_sources()
			env.create_color_map()


	for i in range(num_pg_steps):
		# if verbose:
		# 	print(i)
		pg_data_list = []
		baseline = 0
		states = []
		action_indices = []
		baseline = 0
		weights= []
		placeholder_color_masks = []
		successes_per_batch = 0
		rewards_per_batch = 0


		for _ in range(trajectory_batch_size):
			if multifood:
				env.start_day()
			else:
				env.restart_env()
				if reset_env:
					env.reset_initial_and_destination(hard_instances = True)
			
			node_path1, edge_path1,states_info1, action_indices1, rewards1  = run_color_walk(env, color_policy, num_env_steps)
			#print("Info ", states_info1)

			rewards_per_batch += np.sum(rewards1)
			rewards_to_go = np.cumsum(rewards1[::-1])[::-1] 
			baseline += np.sum(rewards1)	
			weights += list(rewards_to_go)
			placeholder_color_masks += [info[1] for info in states_info1]

			### SUCCESSES EQUALS REWARDS IN THE MULTIFOOD ENV
			if multifood:
				successes_per_batch = rewards_per_batch
			else:
				successes_per_batch += (tuple(node_path1[-1].cpu().numpy())==env.destination_node)*1.0
			states += [state.flatten() for state, is_placeholder_color in states_info1] 
			action_indices += action_indices1

			all_rewards.append(np.sum(rewards1)) 
			#pg_data_list.append((states1, action_indices1, rewards1))

		training_success_evolution.append(successes_per_batch*1.0/trajectory_batch_size)
		training_reward_evolution.append(rewards_per_batch*1.0/trajectory_batch_size)



		baseline = baseline*1.0/trajectory_batch_size
		num_states = len(states)

		states = torch.cat(states).to(DEVICE)
		states = states.view(num_states, -1)
		action_indices = torch.tensor(action_indices).to(DEVICE)
		weights = torch.tensor(weights).float().to(DEVICE) - baseline
		placeholder_color_masks = torch.tensor(placeholder_color_masks).float().to(DEVICE)
		# IPython.embed()
		# raise ValueError("Asdlkm")
		weights = weights*placeholder_color_masks

		optimizer.zero_grad()
		loss = color_policy.log_prob_loss(states, action_indices, weights )
		loss.backward()
		optimizer.step()

		for parameter in color_policy.network.parameters():
			parameter.data.clamp_(-2, 2)
			#parameter.detach()
			#print("parameter norm ", torch.norm(parameter) )
		if verbose and (i+1)%logging_frequency ==0:
			print("PG step {}".format(i+1))
			if not supress_training_curve:
				print("training reward evolution ", training_reward_evolution)
				print("training success evolution ", training_success_evolution)
	return color_policy, training_reward_evolution, training_success_evolution, all_rewards


