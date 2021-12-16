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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
DEVICE = torch.device("cpu") 



class GridEnvironmentMultifoodSmell(GridEnvironment):
	def __init__(self,  
							length, 
							height,
							state_representation = "potential-foodsources",
				            location_normalized = False,
				            encode_goal = False, 
							num_food_sources = 1, 
							pit = False, 
							pit_type = "border",
							initialization_type = "avoiding_pit",
							randomization_threshold = 0, 
							manhattan_reward = False, 
							sparsity = 0,
							combine_with_sparse = False,
							reversed_actions = False,
							length_rim = 3,
							height_rim = 3,
							reward_weights = None
							):
 

		if num_food_sources > 1 and encode_goal:
			raise ValueError("Number of food sources is more than one and state representation has encode goal option on")

		self.pit = pit
		self.num_food_sources = num_food_sources


		super().__init__(length=length, 
							height = height, 
							state_representation = "overwritten",
							location_normalized = location_normalized,
							encode_goal = encode_goal,
							randomization_threshold = randomization_threshold, 
							manhattan_reward = manhattan_reward, 
							sparsity = sparsity,
							combine_with_sparse = combine_with_sparse,
							reversed_actions = reversed_actions,
							return_state_info = True)
		



		self.name = "SimpleMultifoodSmell"

		self.state_representation = state_representation


		### This needs to change. 
		if num_food_sources > 1 and encode_goal:
			raise ValueError("Number of food sources is more than one and state representation has encode goal option on")


		self.name = "SimpleMultifood"
		self.num_food_sources = num_food_sources
		self.destination_node = None ## This is to ensure that no function using self.destination_node as 

		if self.manhattan_reward:
			raise ValueError("Manhattan reward is not supported for the multi-food environment")


		#self.near_pit_initialization = near_pit_initialization
		self.initialization_type = initialization_type

		self.length_rim = length_rim
		self.height_rim = height_rim

		self.pit = pit
		self.pit_type = pit_type
		if pit_type not in ["border", "central"]:
			raise ValueError("pit type set to unknown pit type")
		# if not pit and near_pit_initialization:
		# 	raise ValueError("Pit set to False and near pit initialization set to True")
		if not pit and initialization_type == "near_pit":
			raise ValueError("Pit set to False and near pit initialization set to True")
		if not pit and initialization_type == "avoiding_pit":
			raise ValueError("Pit set to False and initialization type set to avoiding_pit")

		if initialization_type not in ["near_pit", "avoiding_pit", "random"]:
			raise ValueError("Initialization_type not recognized {}".format(initialization_type))

		self.set_pit_nodes()


		self.reset_initial_and_food_sources()



		if reward_weights == None:
			self.reward_weights = torch.zeros(self.get_state_dim()).float()

		else:
			self.set_reward_weights(reward_weights)




	def set_reward_weights(self, reward_weights):
		if len(reward_weights) != self.get_state_dim():
			raise ValueError("Dimension of reward_weights does not match state dim {}".format(len(reward_weights)))
		self.reward_weights = reward_weights


	def set_pit_nodes(self):
		self.pit_nodes = []
		self.pit_adjacent_nodes = []
		self.pit_boundary = [] ### nodes not in the pit but one action away

		## The i-th list in self.pit_four_rims
		## corresponds to the pit_rim nodes for which action i
		## takes us into the pit.
		self.pit_four_rims = [[] for _ in range(4)]

		
		if self.pit:
			for i in range(self.length):
				for j in range(self.height):
					#print("asldkfm ", self.is_pit(i,j), " ", self.is_pit_adjacent(i,j))
					if self.is_pit(i,j):
						self.pit_nodes.append((i,j))
					if self.is_pit_adjacent(i,j):
						self.pit_adjacent_nodes.append((i,j))

					if not self.is_pit(i,j) and self.is_pit_adjacent(i,j):
						self.pit_boundary.append((i,j))
						### Find what is the action that makes it be part of the pit
						for action in self.actions:
							neighbor = ( (i + action[0])%self.length ,  (j + action[1])%self.height) 							


			for k, action in enumerate(self.actions):
				for (i,j) in self.pit_boundary:
					neighbor = ( (i + action[0])%self.length ,  (j + action[1])%self.height)					
					if neighbor in self.pit_nodes:
						self.pit_four_rims[k].append((i,j))


		self.pit_avoiding_graph = deepcopy(self.graph)
		for node in self.pit_adjacent_nodes:
			self.pit_avoiding_graph.remove_node(node)




		#IPython.embed()


	### Overwriting the restart env function to do nothing.
	def restart_env(self):
		pass

	def is_border_pit(self, i, j):
		if i==0:
			return True
		else:
			return False

	def is_border_pit_adjacent(self, i,j):
		if self.is_pit(i,j):
			return True
		else:
			if i ==1:
				return True 
			else:
				return False


	def get_central_pit_indices(self):
		# length_rim = int(self.length*(3/8.0))
		# height_rim = int(self.height*(3/8.0))

		# pit_length_indices = list(range(max(int(self.length*(3/8.0)), 1) , int((6.0/8)*self.length)))
		# pit_height_indices = list(range(max(int(self.height*(3/8.0)), 1) , int((6.0/8)*self.height)))
		
		pit_length_indices = list(range(self.length_rim, self.length-self.length_rim))
		pit_height_indices = list(range(self.height_rim, self.height - self.height_rim))

		return pit_length_indices, pit_height_indices

	def is_central_pit(self, i,j):
		pit_length_indices, pit_height_indices = self.get_central_pit_indices()
		if i in  pit_length_indices and j in pit_height_indices:
			return True
		else:
			return False





	def is_central_pit_adjacent(self, i,j):
		
		pit_length_indices, pit_height_indices = self.get_central_pit_indices()


		pit_nodes = list(itertools.product( pit_length_indices, pit_height_indices) )
		adjacent_nodes = [ (i,j), (max(0, i-1), j), (i, max(0, j-1)), (min(i+1, self.length-1), j), (i, min(j+1, self.height-1)) ]

		# IPython.embed()
		# raise ValueError("asdflkm")

		is_pit_adjacent= False
		for node in adjacent_nodes:
			if node in pit_nodes:
				is_pit_adjacent = True

		return is_pit_adjacent




	def is_pit(self, i, j):
		if self.pit_type == "border":
			return self.is_border_pit(i,j)
		elif self.pit_type == "central":
			return self.is_central_pit(i,j)
		else:
			raise ValueError("unrecognized pit_type {}".format(self.pit_type))

	def is_pit_adjacent(self, i,j):
		if self.pit_type == "border":
			return self.is_border_pit_adjacent(i,j)

		elif self.pit_type == "central":
			return self.is_central_pit_adjacent(i,j)
		else:
			raise ValueError("unrecognized type {} ".format(self.pit_type))


	def reset_food_sources(self):
		self.food_sources = []
		valid_states =[]
		for i in range(self.length):
			for j in range(self.height):
				if (i,j) not in self.pit_adjacent_nodes + [self.initial_node]:
					valid_states.append((i,j))
		self.food_sources = random.sample(valid_states, self.num_food_sources)

		# for i in range(self.num_food_sources):
		# 		food_source = random.choice(list(self.graph.nodes))
		# 		while food_source in [self.initial_node] + self.food_sources or food_source in self.pit_nodes: ### This we may want to change so that it doesn't cause an infinite loop 
		# 			food_source = random.choice(list(self.graph.nodes))
		# 		self.food_sources.append(food_source)
		for food_source in self.food_sources:
			if food_source in self.pit_nodes:
				raise ValueError("Food source is in a pit node")

	def remove_and_reset_one_food_source(self, food_source_node):
		
		self.food_sources.remove(food_source_node)
		if len(self.food_sources) != self.num_food_sources-1:
			raise ValueError("Num food sources after removal inconsistent.")

		valid_states =[]
		for i in range(self.length):
			for j in range(self.height):
				if (i,j) not in self.pit_adjacent_nodes + [self.curr_node] + self.food_sources:
					valid_states.append((i,j))
		self.food_sources += random.sample(valid_states, 1)

		for food_source in self.food_sources:
			if food_source in self.pit_nodes:
				raise ValueError("Food source is in a pit node")



	def reset_initial_and_food_sources(self):
		if self.initialization_type == "near_pit":
		#if self.near_pit_initialization:
			self.initial_node = random.choice(list(self.pit_adjacent_nodes))
		elif self.initialization_type == "avoiding_pit":
			self.initial_node = random.choice(list(self.pit_avoiding_graph.nodes))

		else:
			self.initial_node = random.choice(list(self.graph.nodes))
		while self.initial_node in self.pit_nodes:
					self.initial_node = random.choice(list(self.graph.nodes))

		self.curr_node = self.initial_node
		self.end = False
		self.reset_food_sources()
			


	### CREATE A FUNCTION CALLED get_state_info


	def get_state(self):
		if self.state_representation == "pit-foodsources":
			### Compute the closest distance to the pit
			
			if self.pit:
				min_dist = float("inf")
				for pit_node in self.pit_nodes:
					distance = np.abs(pit_node[0] - self.curr_node[0]) + np.abs(pit_node[1] - self.curr_node[1])
					if distance < min_dist:
						min_dist = distance

			### Compute distances to all the food sources.
			distances_food = [np.abs(a-self.curr_node[0]) + np.abs(b-self.curr_node[1]) for (a,b) in self.food_sources]
			if self.pit:
				state = torch.tensor([min_dist] + distances_food)
			else:
				state = torch.tensor(distances_food)
			return state.float()


		elif self.state_representation == "potential-pit-foodsources":
			distances_food = [np.abs(a-self.curr_node[0]) + np.abs(b-self.curr_node[1]) for (a,b) in self.food_sources]
			food_sources_tensor = [torch.tensor(food_source).float() for food_source in self.food_sources]
			curr_node_tensor = torch.tensor(self.curr_node).float()
			food_potential_gradients = [torch.exp(-torch.norm(food_source - curr_node_tensor))*(food_source - curr_node_tensor)/(.1+torch.norm(food_source - curr_node_tensor) ) for food_source in food_sources_tensor  ]
			sum_food_potential_gradients = torch.stack(food_potential_gradients).sum(0)
			pit_nodes_tensor = [torch.tensor(pit_node).float() for pit_node in self.pit_nodes]
			pit_potential_gradients = [torch.exp(-torch.norm(pit_node - curr_node_tensor))*(pit_node - curr_node_tensor)/(.1+torch.norm(pit_node - curr_node_tensor) ) for pit_node in pit_nodes_tensor  ]
			sum_pit_potential_gradients = torch.stack(pit_potential_gradients).sum(0)
			state = torch.stack([sum_food_potential_gradients, sum_pit_potential_gradients])
			return state.flatten()

		elif self.state_representation == 'potential-foodsources':
			distances_food = [np.abs(a-self.curr_node[0]) + np.abs(b-self.curr_node[1]) for (a,b) in self.food_sources]
			food_sources_tensor = [torch.tensor(food_source).float() for food_source in self.food_sources]
			curr_node_tensor = torch.tensor(self.curr_node).float()
			food_potential_gradients = [torch.exp(-torch.norm(food_source - curr_node_tensor))*(food_source - curr_node_tensor)/(.1+torch.norm(food_source - curr_node_tensor) ) for food_source in food_sources_tensor  ]
			sum_food_potential_gradients = torch.stack(food_potential_gradients).sum(0)		
			return sum_food_potential_gradients


			raise ValueError("potential-foodsources is not yet implemented" )
		else:
			raise ValueError("State representation not implemented {}".format(self.state_representation))

	def get_state_dim(self):
		if self.state_representation == "pit-foodsources":
			return self.num_food_sources + self.pit
		elif self.state_representation == 'potential-pit-foodsources':
			return 4
		elif self.state_representation == "potential-foodsources":
			return 2
		else:
			raise ValueError("State representation not implemented {}".format(self.state_representation))


	def reward(self, curr_node, action, next_node):
		return torch.dot(self.get_state(), self.reward_weights)



	def step(self, action_index):
		action = self.actions[action_index]
		next_vertex = ( (self.curr_node[0] + action[0])%self.length ,  (self.curr_node[1] + action[1])%self.height)
		neighbors =  list(self.graph.neighbors(self.curr_node))

		reward = self.reward(self.curr_node, action, next_vertex)
		
		## Only return a reward the first time we reach the destination node.
		if self.curr_node in self.food_sources and not self.end:
			#print("reached here! ", self.curr_node, self.destination_node)
			self.end = True

		if self.curr_node in self.pit_nodes:
				self.end = True

		## Dynamics:
		if next_vertex in neighbors and self.curr_node not in self.food_sources and self.curr_node not in self.pit_nodes:
			 self.curr_node = next_vertex
		
		step_info = dict([])
		step_info['curr_node'] = self.curr_node
		step_info['reward'] = reward
		# if self.curr_node in self.food_sources:
		# 	IPython.embed()
		# 	raise ValueError("Asdflkm")
		step_info['is_food_source'] =  (self.curr_node in self.food_sources)
		return step_info 


	def start_day(self):
			self.end = False
			self.initial_node = self.curr_node

			if self.curr_node in self.food_sources:
				#self.food_sources.remove(self.curr_node)
				self.remove_and_reset_one_food_source(self.curr_node)

			
			if len(self.food_sources) == 0:
				self.reset_food_sources()





### Except for state_info this is the same function as run_walk
def run_multifood_walk(env, policy, max_time = 1000):
	time_counter = 0
	node_path =  []
	states_info = []
	edge_path = []
	action_indices = []

	rewards = []
	is_food_source_list = []

	while env.manhattan_reward or not env.end:
		# print(env.get_state().flatten())


		action_index = policy.get_action(env.get_state().flatten())

		# print("Action ", action_index, " is placeholder color ", is_placeholder_color)
		node_path.append(torch.from_numpy(np.array(env.curr_node)).to(DEVICE))
		states_info.append(env.get_state())

		old_vertex = env.curr_node
		step_info = env.step(action_index)
		r = step_info["reward"]
		is_food_source = step_info["is_food_source"]
		is_food_source_list.append(is_food_source)
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
	return node_path, edge_path, states_info, action_indices, rewards, is_food_source_list





