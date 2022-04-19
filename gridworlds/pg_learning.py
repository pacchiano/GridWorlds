import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pdb

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import numpy.random as npr
import random
import networkx as nx
import torch
import IPython
from .environments import run_walk

DEBUG = False

def learn_pg_step(env, policy, optimizer, trajectory_batch_size, num_env_steps,
	verbose = False, trajectory_feedback = False, entropy_regularization=False,
	device='cpu'):

	pg_data_list = []
	baseline = 0
	all_states = []
	all_action_indices = []
	baseline = 0
	weights= []
	successes_per_batch = 0
	rewards_per_batch = 0

	for _ in range(trajectory_batch_size):
		env.restart_env()

		# FIXME: We probably don't need nodes and edges to be in GPU, only states
		nodes, edges, states, action_indices, rewards  = run_walk(
			env, policy, num_env_steps)
		if DEBUG: pdb.set_trace()
		if trajectory_feedback:
			trajectory_reward = env.trajectory_reward
			rewards_per_batch += trajectory_reward*1.0
			#print("trajectory reward ", trajectory_reward)
			baseline += trajectory_reward
			weights += [trajectory_reward]*len(rewards)
		else:
			rewards_per_batch += np.sum(rewards)
			rewards_to_go = np.cumsum(rewards[::-1])[::-1]
			baseline += np.sum(rewards)
			weights += list(rewards_to_go)

		successes_per_batch += (tuple(nodes[-1].cpu().numpy())==env.destination_node)*1.0
		all_states += [state.flatten() for state in states]
		all_action_indices += action_indices

	success_rate = successes_per_batch*1.0/trajectory_batch_size
	mean_rewards = rewards_per_batch*1.0/trajectory_batch_size

	baseline = baseline*1.0/trajectory_batch_size
	num_states = len(all_states)
	if DEBUG: pdb.set_trace()
	all_states = torch.cat(all_states)
	all_states = all_states.view(num_states, -1)
	if DEBUG: print(trajectory_batch_size, num_env_steps, all_states.shape)
	all_action_indices = torch.tensor(all_action_indices, device=device)
	weights = torch.tensor(weights, device=device).float() - baseline

	optimizer.zero_grad()
	#pdb.set_trace()

	loss = policy.log_prob_loss(all_states, all_action_indices, weights,
								entropy_regularization=entropy_regularization)

	#pdb.set_trace()
	loss.backward()
	optimizer.step()

	return success_rate, mean_rewards, loss.item()

### TODO: Make this a class (PG Learning) to be able to share params between global and step methods

def learn_pg(env, policy, num_pg_steps, trajectory_batch_size, num_env_steps,
	verbose = False, supress_training_curve = True, logging_frequency = None,
	trajectory_feedback = False, entropy_regularization = False, lr=1e-3, device='cpu'):

	policy = policy.to(device)

	if logging_frequency == None:
		logging_frequency = num_pg_steps
	optimizer = torch.optim.Adam(policy.network.parameters(), lr = lr)

	training_success_evolution = []
	training_reward_evolution = []

	for i in range(num_pg_steps):
		success_rate, mean_rewards, loss = learn_pg_step(env, policy, optimizer,
			trajectory_batch_size, num_env_steps, verbose, trajectory_feedback,
			entropy_regularization=entropy_regularization, device=device)

		training_success_evolution.append(success_rate)
		training_reward_evolution.append(mean_rewards)

		for parameter in policy.network.parameters():
			parameter.data.clamp_(-2, 2)

		if verbose and (i+1)%logging_frequency ==0:
			print("PG step {}. Rewards={:4.2f}, Success={:4.2f}".format(i+1,mean_rewards,success_rate))
			#print(loss, states.shape, action_indices.shape, weights.shape, weights.min(), weights.max())

	return policy, training_reward_evolution, training_success_evolution


### Aldo's Version
# def learn_pg(env, policy, num_pg_steps, trajectory_batch_size, num_env_steps,
# 	verbose = False, supress_training_curve = True,
# 	logging_frequency = None, reset_env = True,
# 	trajectory_feedback = False, lr=1e-3):
#
# 	if logging_frequency == None:
# 		logging_frequency = num_pg_steps
# 	optimizer = torch.optim.Adam(policy.network.parameters(), lr=lr)
#
# 	training_success_evolution = []
# 	training_reward_evolution = []
#
# 	for i in range(num_pg_steps):
# 		pg_data_list = []
# 		baseline = 0
# 		states = []
# 		action_indices = []
# 		baseline = 0
# 		weights= []
# 		successes_per_batch = 0
# 		rewards_per_batch = 0
#
# 		for _ in range(trajectory_batch_size):
# 			env.restart_env()
# 			nodes, edges, states, action_indices, rewards  = run_walk(env, policy, num_env_steps)
# 			if trajectory_feedback:
# 				trajectory_reward = env.trajectory_reward
# 				rewards_per_batch += trajectory_reward*1.0
# 				#print("trajectory reward ", trajectory_reward)
# 				baseline += trajectory_reward
# 				weights += [trajectory_reward]*len(rewards)
#
# 			else:
# 				rewards_per_batch += np.sum(rewards)
# 				rewards_to_go = np.cumsum(rewards[::-1])[::-1]
# 				baseline += np.sum(rewards)
# 				weights += list(rewards_to_go)
#
# 			pdb.set_trace()
# 			successes_per_batch += (tuple(nodes[-1].numpy())==env.destination_node)*1.0
# 			states += [state.flatten() for state in states]
# 			action_indices += action_indices
#
# 			#pg_data_list.append((states, action_indices, rewards))
#
# 		training_success_evolution.append(successes_per_batch*1.0/trajectory_batch_size)
# 		training_reward_evolution.append(rewards_per_batch*1.0/trajectory_batch_size)
#
# 		baseline = baseline*1.0/trajectory_batch_size
# 		num_states = len(states)
# 		states = torch.cat(states)
# 		states = states.view(num_states, -1)
# 		action_indices = torch.tensor(action_indices)
# 		weights = torch.tensor(weights).float() - baseline
#
# 		optimizer.zero_grad()
#
# 		loss = policy.log_prob_loss(states, action_indices, weights )
# 		loss.backward()
# 		optimizer.step()
#
# 		for parameter in policy.network.parameters():
# 			parameter.data.clamp_(-2, 2)
# 			#parameter.detach()
# 			#print("parameter norm ", torch.norm(parameter) )
# 		if verbose and (i+1)%logging_frequency ==0:
# 			print("PG step {}. Rewards={:4.2f}, Success={:4.2f}".format(i+1,training_reward_evolution[-1],training_success_evolution[-1]))
# 			print(loss.item(), states.shape, action_indices.shape, weights.shape, weights.min(), weights.max())
# 			if not supress_training_curve:
# 				print("training reward evolution ", training_reward_evolution)
# 				print("training success evolution ", training_success_evolution)
# 	return policy, training_reward_evolution, training_success_evolution
