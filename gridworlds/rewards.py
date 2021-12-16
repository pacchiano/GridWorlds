import matplotlib 
import matplotlib.pyplot as plt
import torch 

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy.random as npr

from abc import ABCMeta, abstractmethod



class RewardFunction:

	@abstractmethod
	def evaluate(self, env):
		pass

	@abstractmethod
	def evaluate_trajectory(self, env):
		pass





class SimpleIndicatorReward(RewardFunction):
	def evaluate(self, env, reward_info):
		if env.get_curr_node() == env.get_destination_node():
			return 1
		else:
			return 0


	def evaluate_trajectory(self, env):
		raise ValueError("Not implemented evaluate_trajectory for ManhattanReward")




class ManhattanReward(RewardFunction):

	def evaluate(self, env, reward_info):
		curr_node = env.get_curr_node()
		destination_node = env.get_destination_node()
		action = reward_info["action"]	
		normalization_factor = env.get_length() + env.get_height()
		dist = -1.0*(np.abs(curr_node[0]  - destination_node[0]) +  np.abs(curr_node[1]  - destination_node[1]))
		return dist/normalization_factor

	def evaluate_trajectory(self, env):
		raise ValueError("Not implemented evaluate_trajectory for ManhattanReward")
