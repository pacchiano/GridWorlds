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
from networkx import grid_graph, single_target_shortest_path_length, draw, spring_layout, draw_planar, grid_2d_graph, draw_shell, draw_spring, draw_spectral, shortest_path_length
from copy import deepcopy


from .policies import *
from .environments import *
from .rewards import MultifoodIndicatorReward, SimpleIndicatorReward

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
DEVICE = torch.device("cpu") 





### TODO: Write a description of this Class.
class ColorGridEnvironment(GridEnvironment):
	def __init__(			self,  
							length, 
							height, 
							num_fixed_colors = 4,
							num_placeholder_colors = 1,
							fixed_color_action_map  = [0, 1, 2, 3], ### the action map format is just a list of indices in {0,1,2,3} of size num_colors 
							placeholder_color_prob = .2,
							randomization_threshold = 0,
							state_representation = "colors",
							do_undo_map = IdentityDoUndo(),
							reward_function = SimpleIndicatorReward() 
							):
 

		self.num_fixed_colors = num_fixed_colors
		self.num_placeholder_colors = num_placeholder_colors
		self.num_colors_with_placeholders = self.num_fixed_colors + self.num_placeholder_colors

		if state_representation not in ["colors", "overwritten"]:
			raise ValueError("State representation for ColorSimple is set to {} and therefore unavailable.".format(state_representation))


		GridEnvironment.__init__(   
							self,
							length, 
							height, 
							randomization_threshold = randomization_threshold, 
							state_representation = "overwritten",
							do_undo_map = do_undo_map,
							reward_function = reward_function
							)


		self.name = "ColorSimple"
		self.state_representation = state_representation

		self.placeholder_color_prob = placeholder_color_prob

		if len(fixed_color_action_map) != num_fixed_colors:
			raise ValueError("Number of colors and size of action map are different.")
		for action_index in fixed_color_action_map:
			if action_index not in list(range(len(self.actions))):
				raise ValueError("Action index not in the valid action index set")
		self.fixed_color_action_map = fixed_color_action_map

		self.action_color_map = dict([(i, []) for i in range(len(self.actions)) ])


		## The variable all_color_action_map_vectorized encodes the map between colors to actions and placeholders 
		## to the placeholder action. This map is in vectorized form so that we can use it to compute its distance 
		## to the strings produced by the meta learning procedure.
		self.all_color_action_map_vectorized = np.zeros((self.num_colors_with_placeholders, len(self.actions) + 1))
		for i in range(self.num_fixed_colors):
			self.all_color_action_map_vectorized[i, self.fixed_color_action_map[i]] = 1
		for i in range(self.num_fixed_colors, self.num_fixed_colors + self.num_placeholder_colors):
			self.all_color_action_map_vectorized[i, -1] = 1


		for color_index in range(self.num_fixed_colors):
			self.action_color_map[self.fixed_color_action_map[color_index]].append(color_index)
		self.state_dim = self.get_state_dim()

		self.initialize_placeholders()

		if self.state_representation != "overwritten":
			#GridEnvironment.reset_environment(self)
			#self.create_color_map()
			self.reset_environment(info = dict([("hard_instances", False), ("reinitialize_placeholders", True)]))

	def initialize_placeholders(self):

		self.placeholder_action_map = [0]*self.num_placeholder_colors
		self.action_placeholder_map = dict([(i, []) for i in range(len(self.actions)) ])

		for i in range(self.num_placeholder_colors):
			action_corresponding_to_placeholder_i = random.choice(list(range(len(self.actions))))
			self.placeholder_action_map[i] = action_corresponding_to_placeholder_i
			self.action_placeholder_map[action_corresponding_to_placeholder_i].append(i)


	def get_state_dim(self):
		return self.num_colors_with_placeholders



	def create_color_map(self):
		self.color_map = np.zeros((self.length, self.height))
		for i in range(self.length):
			for j in range(self.height):
				optimal_action_index, _ = self.get_optimal_action_and_index_single_destination((i,j), self.destination_node, self.graph)
				#optimal_action_index, _ = self.get_optimal_action_and_index((i,j), self.destination_node, self.graph)
				if np.random.random() < self.placeholder_color_prob and len(self.action_placeholder_map[optimal_action_index]) > 0:
							#print("aslkdmfalskdfmalskdmfalskdmfalksdmflaskdmflaksdmflaksdmflaksdmf")
							self.color_map[i,j] = np.random.choice(self.action_placeholder_map[optimal_action_index]) + self.num_fixed_colors
				else	:
					self.color_map[i, j] = np.random.choice(self.action_color_map[optimal_action_index])
							



	def reset_environment(self, info = dict([("hard_instances", True), ("reinitialize_placeholders", False)])):
		GridEnvironment.reset_initial_and_destination(self, hard_instances = info["hard_instances"])
		if info["reinitialize_placeholders"]:
			self.initialize_placeholders()
		self.create_color_map()


	def get_state_helper(self, curr_node):
		indicator_color = [0]*self.num_colors_with_placeholders
		indicator_color[int(self.color_map[curr_node[0], curr_node[1]])] = 1
		state = torch.tensor(indicator_color).float().to(DEVICE)
		#IPython.embed()
		return self.do_undo_map.do_state(state)

	def compute_agreement_ratio(self, string_with_placeholders):
		agreement = np.sum(string_with_placeholders*self.all_color_action_map_vectorized)
		agreement*=1.0/self.num_colors_with_placeholders
		return agreement


	def get_optimal_action_index(self, i,j):
		#raise ValueError("this function may not be needed. Get optimal action index.")
		color_index = int(self.color_map[i,j] )
		#color_indicator = state.cpu().numpy()
		#color_index = int(np.sum((color_indicator > 0)*np.arange(self.env.get_state_dim())))
		#IPython.embed()
		if color_index < self.num_fixed_colors:
			return self.fixed_color_action_map[color_index]
		else:
			return self.placeholder_action_map[color_index - self.num_fixed_colors]



class ColorGridEnvironmentMultifood(ColorGridEnvironment,GridEnvironmentPitMultifood):
	def __init__(			
					self,  
					length, 
					height,
					num_food_sources = 1, 
					num_fixed_colors = 4,
					num_placeholder_colors = 1,
					fixed_color_action_map  = [0, 1, 2, 3], ### the action map format is just a set of indices in {0,1,2,3} of size num_colors 
					placeholder_color_prob = .2,
					#pit = False, 
					state_representation = "colors",
					pit_type = "border",
					pit_colors = 0,
					initialization_type = "avoiding_pit",
					randomization_threshold = 0, 
					length_rim = 3,
					height_rim = 3,
					do_undo_map = IdentityDoUndo(),
					reward_function = MultifoodIndicatorReward()
							):
 

		GridEnvironmentPitMultifood.__init__(
				  self,
				  length=length, 
				  height = height, 
				  state_representation = "overwritten",
				  pit_type = pit_type,
				  initialization_type = initialization_type,              
				  length_rim = length_rim,
				  height_rim = height_rim,
				  randomization_threshold = randomization_threshold,
				  do_undo_map = do_undo_map,
				  reward_function = reward_function
				  )


		ColorGridEnvironment.__init__(
							self,
							length=length, 
							height = height, 
							num_fixed_colors = num_fixed_colors,
							num_placeholder_colors = num_placeholder_colors,
							fixed_color_action_map  = fixed_color_action_map,
							placeholder_color_prob = placeholder_color_prob,
							randomization_threshold = randomization_threshold, 
							state_representation = "overwritten",
							do_undo_map = do_undo_map,
							reward_function = reward_function
							)
		





		self.name = "ColorSimpleMultifood"
		self.state_representation = state_representation
		self.num_food_sources = num_food_sources
		self.destination_node = None ## This is to ensure that no function using self.destination_node as 


		#self.near_pit_initialization = near_pit_initialization
		self.initialization_type = initialization_type

		self.length_rim = length_rim
		self.height_rim = height_rim

		### FIGURE OUT IF THERE IS A PIT
		if 2*self.height_rim >= self.height or 2*self.length_rim >= self.length:	
			self.pit = False
		else:
			self.pit = True



		self.pit_type = pit_type
		if pit_type not in ["border", "central"]:
			raise ValueError("pit type set to unknown pit type")
		# if not pit and near_pit_initialization:
		# 	raise ValueError("Pit set to False and near pit initialization set to True")
		if not self.pit and initialization_type == "near_pit":
			raise ValueError("Pit set to False and near pit initialization set to True")
		if not self.pit and initialization_type == "avoiding_pit":
			raise ValueError("Pit set to False and initialization type set to avoiding_pit")

		if initialization_type not in ["near_pit", "avoiding_pit"]:
			raise ValueError("Initialization_type not recognized {}".format(initialization_type))

		self.set_pit_nodes()
		self.pit_colors = pit_colors ### the pit color index is that of the first pit_colors part of num_colors
		if not self.pit and self.pit_colors >0:
			raise ValueError("pit colors >0 but pit set to false")
		if self.pit_colors >= self.num_fixed_colors:
			raise ValueError("Number of pit colors equals or surpases number of colors")
		if self.pit_colors%len(self.actions) != 0:
				raise ValueError("Number of pit colors is not a multiple of the number of actions")
		for action_index in fixed_color_action_map[:self.pit_colors]:
			if action_index not in list(range(len(self.actions))):
				raise ValueError("Pit color action index not in the valid action index set")
		for action_index in fixed_color_action_map[self.pit_colors:]:
			if action_index not in list(range(len(self.actions))):
				raise ValueError("Pit color action index not in the valid action index set")
		if self.pit_colors > 0 and len(set(fixed_color_action_map[:self.pit_colors])) != len(self.actions):
			raise ValueError("Number of actions in pit colors different from all actions")
		if len(set(fixed_color_action_map[self.pit_colors:])) != len(self.actions):
			raise ValueError("Number of actions in non pit colors different from all actions")


		if self.initialization_type == "near_pit":
		  self.valid_initial_nodes = list(self.outer_rim)
		elif self.initialization_type == "avoiding_pit":
		  self.valid_initial_nodes = list(self.pit_avoiding_graph.nodes)
		else:
		  raise ValueError("Initialization type not recognized {}".format(initialization_type))

		self.valid_destination_nodes = list(self.pit_avoiding_graph.nodes)



		if self.state_representation != "overwritten":
			self.reset_environment( info = dict([("reinitialize_placeholders", True)]))


	def get_state_helper(self, curr_node):
		return ColorGridEnvironment.get_state_helper(self, curr_node)


	def get_state(self):
		if self.state_representation != "colors":
			raise ValueError("State representation different from colors")
		else:
			return self.get_state_helper(self.curr_node)
	### Overwriting the restart env function to do nothing.
	def restart_env(self):
	  self.end = False
	  self.curr_node = self.initial_node


	def reset_environment(self, info = dict([("reinitialize_placeholders", False)])):
		GridEnvironmentPitMultifood.reset_environment(self)
		if info["reinitialize_placeholders"]:
			self.initialize_placeholders()
		self.create_color_map()



	def create_color_map(self):
		self.color_map = np.zeros((self.length, self.height))
		for i in range(self.length):
			for j in range(self.height):
				if self.is_pit(i,j):
					optimal_action_index = 0
				else:
					optimal_action_index = GridEnvironmentPitMultifood.get_optimal_action_and_index(self, (i,j))

				# get_optimal_action_and_index(self, vertex)
				# if self.pit and (i,j) in self.pit_adjacent_nodes and (i,j) in self.pit_nodes: 
				# 	optimal_action_index = 0
				# elif self.pit and (i,j) in self.pit_adjacent_nodes and (i,j) not in self.pit_nodes:
				# 	for l in range(self.num_actions):
				# 		if (i,j) in self.pit_four_rims[l]:
				# 			into_pit_action = self.actions[l]

				# 	optimal_action = (-into_pit_action[0], -into_pit_action[1])
				# 	optimal_action_index = self.actions.index(optimal_action)

				# else:
				# 	min_dist_food_source = float("inf")
				# 	optimal_action_index = 0

				# 	for food_source in self.food_sources:

				# 		if self.pit and (i,j) not in self.pit_adjacent_nodes:
				# 			curr_food_source_action_index, _ = self.get_optimal_action_and_index((i,j), food_source, self.pit_avoiding_graph)
				# 			curr_food_source_distance = shortest_path_length(self.pit_avoiding_graph, (i,j), food_source)


				# 		if not self.pit:
				# 			curr_food_source_action_index, _ = self.get_optimal_action_and_index((i,j), food_source, self.graph)
				# 			curr_food_source_distance = np.abs(i-food_source[0]) + np.abs(j - food_source[1])



				# 		if curr_food_source_distance < min_dist_food_source:
				# 			min_dist_food_source = curr_food_source_distance
				# 			optimal_action_index = curr_food_source_action_index



				
				non_placeholder_color_choices = self.action_color_map[optimal_action_index]
				if self.pit_colors > 0 and (i,j) in self.outer_rim:
				#if self.pit and self.pit_colors >0 and self.is_pit_adjacent(i,j):
						non_placeholder_color_choices = list(set(non_placeholder_color_choices)&set(range(self.pit_colors)))
						self.color_map[i, j] = np.random.choice(non_placeholder_color_choices)

				else:					
					if np.random.random() < self.placeholder_color_prob and len(self.action_placeholder_map[optimal_action_index]) > 0:
								#print("aslkdmfalskdfmalskdmfalskdmfalksdmflaskdmflaksdmflaksdmflaksdmf")
								self.color_map[i,j] = np.random.choice(self.action_placeholder_map[optimal_action_index]) + self.num_fixed_colors
					else:
						non_placeholder_color_choices = list(set(non_placeholder_color_choices)&set(range(self.pit_colors, self.num_fixed_colors)))
						self.color_map[i, j] = np.random.choice(non_placeholder_color_choices)
						if not self.pit and self.pit_colors > 0:
							raise ValueError("Pit is not on but pit colors are more than zero ")

	def step(self, action_index):
		action = self.actions[action_index]


		if random.random() > self.randomization_threshold:
		  next_vertex = self.get_next_vertex(self.curr_node[0], self.curr_node[1], action_index)
		else:
		  next_vertex = self.curr_node

		# next_vertex = ( (self.curr_node[0] + action[0])%self.length ,  (self.curr_node[1] + action[1])%self.height)
		# neighbors =  list(self.graph.neighbors(self.curr_node))



		reward_info = dict([("action", action)])
		reward = self.reward_function.evaluate(self, reward_info)
		
		## Only return a reward the first time we reach the destination node.


		if self.curr_node in self.pit_nodes:
				self.end = True

		if self.curr_node in self.food_sources:
				self.end = True

		## Dynamics:
		if self.curr_node not in self.food_sources + self.pit_nodes:
			 self.curr_node = next_vertex
		

		step_info = dict([])
		step_info['curr_node'] = self.curr_node
		step_info['reward'] = reward

		step_info["state"] = self.get_state()
		step_info["action"] = action
		step_info["action_index"] = action_index
		step_info["end"] = self.end
		step_info["is_pit"] = self.curr_node in self.pit_nodes
		step_info["is_food_source"] = self.curr_node in self.food_sources

		return step_info
		#return self.curr_node, reward


	def start_day(self):
			self.end = False
			self.initial_node = self.curr_node

			if self.curr_node in self.food_sources:
				#self.food_sources.remove(self.curr_node)
				self.remove_and_reset_one_food_source(self.curr_node)

				self.create_color_map()
			
			if len(self.food_sources) == 0:
				self.reset_food_sources()
				self.create_color_map()



def run_color_walk(env, policy, max_time = 1000):
	time_counter = 0
	node_path =  []
	states_info = []
	edge_path = []
	action_indices = []

	rewards = []


	while not env.end:
		# print(env.get_state().flatten())

		action_index, is_placeholder_color = policy.get_action(env.get_state().flatten(), with_info = True)
		# print("Action ", action_index, " is placeholder color ", is_placeholder_color)
		node_path.append(torch.from_numpy(np.array(env.curr_node)).to(DEVICE))
		states_info.append((env.get_state(), is_placeholder_color))

		old_vertex = env.curr_node
		step_info = env.step(action_index)
		
		r = step_info["reward"]
		action_indices.append(action_index)
		edge_path.append(torch.from_numpy(np.array((old_vertex, env.curr_node))).to(DEVICE))
		#node_path.append(env.curr_node)
		rewards.append(r)

		time_counter += 1
		if time_counter > max_time:
			break
	#action_indices.append(policy.get_action(env.get_state()))
	# IPython.embed()
	# raise ValueError("asldfkm")
	return node_path, edge_path, states_info, action_indices, rewards

















