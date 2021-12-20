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







class OptimalColorPolicy():
	def __init__(self, env):
		self.env = env
	def get_action(self, state, with_info = False):
		color_indicator = state.cpu().numpy()
		color_index = int(np.sum((color_indicator > 0)*np.arange(self.env.get_state_dim())))
		if color_index < self.env.num_fixed_colors:
			return self.env.fixed_color_action_map[color_index]
		else:
			return self.env.placeholder_action_map[color_index - self.env.num_fixed_colors]




class ColorPolicy(NNSoftmaxPolicy):
	def __init__(self, string_with_placeholders, state_dim, num_actions, reactive = False, 
		activation_type = "softmax", hidden_layers = [50], device = torch.device("cpu")):
		
		self.string_with_placeholders = string_with_placeholders
		self.reactive = reactive

		super().__init__(state_dim, num_actions, hidden_layers = [50], activation_type = activation_type, device =  device)



	def get_action(self, state, with_info = False):
		# color_indicator = state[self.color_indicator_dimensions_in_state].numpy()
		# color_index = int(np.sum((color_indicator > 0)*np.arange(len(self.color_indicator_dimensions_in_state))))
		color_indicator = state.cpu().numpy()
		color_index = int(np.sum((color_indicator > 0)*np.arange(self.state_dim)))

		# IPython.embed()
		# raise ValueError("asdflkm")


		advice_action_index = int(np.sum(self.string_with_placeholders[color_index, :]*np.arange(self.num_actions + 1)))
		is_placeholder_color = False
		
		if advice_action_index == self.num_actions:
			#print("asldfkmasdlfkmasldkfmalsdkfmalksdmf")
			is_placeholder_color = True
			if self.reactive:
				advice_action_index = random.choice(list(range(self.num_actions)))
			else:
				advice_action_index = self.network(state)
		
		if with_info:
			return advice_action_index, is_placeholder_color 
		return advice_action_index

