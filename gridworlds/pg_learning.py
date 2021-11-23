import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import numpy.random as npr
import random
import networkx as nx
import torch
import IPython
from environments import run_walk

def learn_pg(env, policy, num_pg_steps, trajectory_batch_size, num_env_steps, verbose = False, supress_training_curve = True, 
	logging_frequency = None, reset_env = True, trajectory_feedback = False):
	#optimizer = torch.optim.Adam([policy.policy_params], lr=0.01)
	if logging_frequency == None:
		logging_frequency = num_pg_steps
	optimizer = torch.optim.Adam(policy.network.parameters(), lr = 0.01)

	training_success_evolution = []
	training_reward_evolution = []

	for i in range(num_pg_steps):
		# if verbose:
		# 	print(i)
		pg_data_list = []
		baseline = 0
		states = []
		action_indices = []
		baseline = 0
		weights= []
		successes_per_batch = 0
		rewards_per_batch = 0

		for _ in range(trajectory_batch_size):
			env.restart_env()
			if reset_env:
				env.reset_initial_and_destination(hard_instances = True)
			
			node_path1, edge_path1,states1, action_indices1, rewards1  = run_walk(env, policy, num_env_steps)
			#IPython.embed()
			# raise ValueError("asdflkm")
			if trajectory_feedback:
				#trajectory_feedback = env.trajectory_reward

				trajectory_reward = env.trajectory_reward
				rewards_per_batch += trajectory_reward*1.0
				#print("trajectory reward ", trajectory_reward)
				baseline += trajectory_reward
				weights += [trajectory_reward]*len(rewards1)

			else:
				rewards_per_batch += np.sum(rewards1)
				rewards_to_go = np.cumsum(rewards1[::-1])[::-1] 
				baseline += np.sum(rewards1)	
				weights += list(rewards_to_go)
			#IPython.embed()


			successes_per_batch += (tuple(node_path1[-1].numpy())==env.destination_node)*1.0
			states += [state.flatten() for state in states1] 
			action_indices += action_indices1

			#pg_data_list.append((states1, action_indices1, rewards1))

		training_success_evolution.append(successes_per_batch*1.0/trajectory_batch_size)
		training_reward_evolution.append(rewards_per_batch*1.0/trajectory_batch_size)



		baseline = baseline*1.0/trajectory_batch_size
		num_states = len(states)

		states = torch.cat(states)
		states = states.view(num_states, -1)
		action_indices = torch.tensor(action_indices)
		weights = torch.tensor(weights).float() - baseline


		optimizer.zero_grad()
		loss = policy.log_prob_loss(states, action_indices, weights )
		loss.backward()
		optimizer.step()

		for parameter in policy.network.parameters():
			parameter.data.clamp_(-2, 2)
			#parameter.detach()
			#print("parameter norm ", torch.norm(parameter) )
		if verbose and (i+1)%logging_frequency ==0:
			print("PG step {}".format(i+1))
			if not supress_training_curve:
				print("training reward evolution ", training_reward_evolution)
				print("training success evolution ", training_success_evolution)
	return policy, training_reward_evolution, training_success_evolution

