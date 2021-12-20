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
from networkx import grid_graph, single_target_shortest_path_length, draw, spring_layout, draw_planar, grid_2d_graph, draw_shell, draw_spring, draw_spectral, shortest_path_length



import random


class OptimalGridPitPolicy():
	def __init__(self, env):
		self.env = env
		

	def get_action(self, state, with_info = False):

		return self.env.get_optimal_action_and_index(self.env.curr_node)



class OptimalMultifoodPitPolicy():
	def __init__(self, env):
		self.env = env
		

	def get_action(self, state, with_info = False):

		return self.env.get_optimal_action_and_index(self.env.curr_node)
