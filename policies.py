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
import numpy.random as npr
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical
import IPython


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
                        nn.Linear(input_dim, hidden_size, bias = False),
                        nn.ReLU(),
                        nn.Linear(hidden_size, hidden_size, bias = False),
                        nn.ReLU(),

                        # nn.Linear(hidden_size, hidden_size),
                        # nn.ReLU(),
                        # nn.Linear(hidden_size, hidden_size),
                        # nn.ReLU(),
                        nn.Linear(hidden_size, output_dim, bias = False),
                        )
    
    def forward(self, x):
        return self.network(x)

class LinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearNetwork, self).__init__()
        self.network = nn.Linear(input_dim, output_dim)
                           
    def forward(self, x):
        return self.network(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=50):
        super(PolicyNetwork, self).__init__()
        self.action_dim = action_dim

        self.network = MLPNetwork(state_dim, action_dim , hidden_size)

    def forward(self, x, action_indices = None, get_logprob=False):
        logp = self.network(x)
        dist = Categorical(logits =logp)
        action = dist.sample()
        if get_logprob:
          logprob = dist.log_prob(action_indices)
          return logprob  
        return action


class NNPolicy:
  def __init__(self, state_dim, num_actions, softmax= True, max_policy_param_absval =100, hidden_layer = 50 ):
    self.num_actions = num_actions
    self.max_policy_param_absval = max_policy_param_absval
    self.state_dim = state_dim
    if not softmax:
      raise ValueError("Not implemented")    
    self.network = PolicyNetwork(state_dim, num_actions, hidden_size = hidden_layer)

  def log_prob_loss(self, states, action_indices, weights):
    logprob =  self.network(states, action_indices = action_indices, get_logprob = True)
    loss = -(logprob*weights).mean()
    return loss

  def get_action(self, state):
    action = self.network(state)
    return action

  # def reset_parameters(self):
  #   for _, module in self.network.named_children():
  #     if hasattr(module, 'reset_parameters'):
  #       print("resetting parameters")
  #       module.reset_parameters()

  def get_parameters_norm(self):
    norm_sum = 0
    for parameter in self.network.parameters():
      norm_sum += torch.norm(parameter)
    return norm_sum

class RandomPolicy:
  def __init__(self, num_actions = 4):
    self.num_actions = num_actions

  def get_action(self, state):
    return random.choice(range(self.num_actions))





def test_policy(env, policy, num_trials, num_env_steps, trajectory_feedback = False):
  base_success_nums = []
  collected_base_rewards = []

  for i in range(num_trials):
    env.restart_env()
    node_path1, _,states, _, rewards1  = run_walk(env, policy, num_env_steps, trajectory_feedback)
    base_success_nums.append( tuple(node_path1[-1].numpy())==env.destination_node)
    if trajectory_feedback:
      collected_base_rewards.append(env.trajectory_reward)
    else:
      collected_base_rewards.append(sum(rewards1))
  base_success_num= np.mean(base_success_nums)
  base_rewards = np.mean(collected_base_rewards)

  return base_rewards, base_success_num

