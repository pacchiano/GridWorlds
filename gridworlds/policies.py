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
from .environments import run_walk



class FeedforwardMultiLayerRepresentation(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_type = "relu",
      batch_norm = False, device = torch.device("cpu")):
        super(FeedforwardMultiLayerRepresentation, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.batch_norm = batch_norm
        self.sigmoid = torch.nn.Sigmoid()
        self.hidden_layers = hidden_layers
        if activation_type == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise ValueError("Unrecognized activation type.")

        self.layers = torch.nn.ModuleList()

        sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(sizes)-1):
          self.layers = self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))

        # self.layers = self.layers.append(torch.nn.Linear(self.input_size, self.hidden_sizes[0]))

        # for i in range(len(self.hidden_sizes)-1):
        #     self.layers.append(torch.nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))

        self.layers.to(device)

        if self.batch_norm:
            raise ValueError("Not implemented properly yet.")
            self.batch_norms = torch.nn.ModuleList()
            output_sizes = hidden_layers + [output_size]
            for i in range(len(output_sizes)):
              self.batch_norms.append(torch.nn.BatchNorm1d(output_sizes[i]))
            #self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_sizes[0]))
            #for i in range(len(self.hidden_sizes)-1):
            #    self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_sizes[i+1]))
            self.batch_norms.to(device)

    def forward(self, x):
        representation = x

        #IPython.embed()
        #raise ValueError("asldkfm")
        for i in range(len(self.layers)):
            representation = self.layers[i](representation)
            if self.batch_norm:
                representation = self.batch_norms[i](representation)
            if i != len(self.layers)-1:
                representation = self.activation(representation)
        return representation










# class MLPNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_size=256):
#         super(MLPNetwork, self).__init__()
#         self.network = nn.Sequential(
#                         nn.Linear(input_dim, hidden_size, bias = False),
#                         nn.ReLU(),
#                         nn.Linear(hidden_size, hidden_size, bias = False),
#                         nn.ReLU(),

#                         # nn.Linear(hidden_size, hidden_size),
#                         # nn.ReLU(),
#                         # nn.Linear(hidden_size, hidden_size),
#                         # nn.ReLU(),
#                         nn.Linear(hidden_size, output_dim, bias = False),
#                         )

#     def forward(self, x):
#         return self.network(x)

# class LinearNetwork(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearNetwork, self).__init__()
#         self.network = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.network(x)



# self, input_size, hidden_sizes, activation_type = "sigmoid",
#       batch_norm = False, device = torch.device("cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[50], activation_type = "relu", device = torch.device("cpu")):
        super(PolicyNetwork, self).__init__()
        self.action_dim = action_dim
        self.network = FeedforwardMultiLayerRepresentation(input_size = state_dim,
          hidden_layers = hidden_layers, output_size = action_dim , activation_type = activation_type,
          device = device)

    def forward(self, x, action_indices = None, get_logprob=False):
        logp = self.network(x)
        dist = Categorical(logits =logp)
        action = dist.sample()
        if get_logprob:
          logprob = dist.log_prob(action_indices)
          return logprob
        return action








class NNSoftmaxPolicy:
  def __init__(self, state_dim, num_actions, hidden_layers = [50],
    activation_type = "relu", device = torch.device("cpu")):
    self.num_actions = num_actions
    self.state_dim = state_dim
    self.network = PolicyNetwork(state_dim, num_actions, hidden_layers = hidden_layers,
      activation_type = "relu", device = torch.device("cpu"))

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





def test_policy(env, policy, num_trials, num_env_steps, reset_goal=False):
    base_success_nums = []
    collected_base_rewards = []
    trajectories = []

    device = policy.device

    for i in range(num_trials):
        env.restart_env()
        if reset_goal:
        	print('WARNING: Reseting env goal')
        	env.reset_initial_and_destination(hard_instances = True)
        node_path1, _,states, _, rewards1  = run_walk(env, policy, num_env_steps)
        base_success_nums.append( tuple(node_path1[-1].numpy())==env.destination_node)
        collected_base_rewards.append(sum(rewards1))
        trajectories.append(states)

    base_success_num= np.mean(base_success_nums)
    base_rewards = np.mean(collected_base_rewards)

    return base_rewards, base_success_num, trajectories
