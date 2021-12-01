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





### TODO: Write a description of this Class.

class ColorGridEnvironment(GridEnvironment):
	def __init__(self,  
							length, height, 
							num_colors = 4,
							num_placeholder_colors = 1,
							color_action_map  = [0, 1, 2, 3], ### the action map format is just a list of indices in {0,1,2,3} of size num_colors 
							placeholder_color_prob = .2,
							state_representation = "two-dim",
							manhattan_reward = False, 
							location_normalized = False,
							encode_goal = False, 
							sparsity = 0,
							combine_with_sparse = False,
							reversed_actions = False,
							initialize_color_map = True,
							randomization_threshold = 0, 
							):
 

		self.num_colors = num_colors
		self.num_placeholder_colors = num_placeholder_colors
		self.num_colors_with_placeholders = self.num_colors + self.num_placeholder_colors



		super().__init__( length, height, 
							randomization_threshold = randomization_threshold, 
							manhattan_reward = manhattan_reward, 
							state_representation = state_representation,
							location_normalized = location_normalized,
							encode_goal = encode_goal, 
							sparsity = sparsity,
							combine_with_sparse =combine_with_sparse,
							reversed_actions = reversed_actions)

		self.name = "ColorSimple"
		self.placeholder_color_prob = placeholder_color_prob

		if len(color_action_map) != num_colors:
			raise ValueError("Number of colors and size of action map are different.")
		for action_index in color_action_map:
			if action_index not in list(range(len(self.actions))):
				raise ValueError("Action index not in the valid action index set")
		self.color_action_map = color_action_map

		self.action_color_map = dict([(i, []) for i in range(len(self.actions)) ])


		## The variable all_color_action_map_vectorized encodes the map between colors to actions and placeholders 
		## to the placeholder action. This map is in vectorized form so that we can use it to compute its distance 
		## to the strings produced by the meta learning procedure.
		self.all_color_action_map_vectorized = np.zeros((self.num_colors_with_placeholders, len(self.actions) + 1))
		for i in range(self.num_colors):
			self.all_color_action_map_vectorized[i, self.color_action_map[i]] = 1
		for i in range(self.num_colors, self.num_colors + self.num_placeholder_colors):
			self.all_color_action_map_vectorized[i, -1] = 1


		for color_index in range(self.num_colors):
			self.action_color_map[self.color_action_map[color_index]].append(color_index)
		self.state_dim = self.get_state_dim()

		self.initialize_placeholders()
		if initialize_color_map:

			self.create_color_map()

	def initialize_placeholders(self):

		self.placeholder_action_map = [0]*self.num_placeholder_colors
		self.action_placeholder_map = dict([(i, []) for i in range(len(self.actions)) ])

		for i in range(self.num_placeholder_colors):
			action_corresponding_to_placeholder_i = random.choice(list(range(len(self.actions))))
			self.placeholder_action_map[i] = action_corresponding_to_placeholder_i
			self.action_placeholder_map[action_corresponding_to_placeholder_i].append(i)


	def get_state_dim(self):
		return self.num_colors_with_placeholders

	def get_optimal_action_and_index(self, vertex, food_source, graph ):
			(i,j) = vertex
			optimal_action_index = 0
			min_distance  = float("inf")
			dist = float("inf")
			shortest_paths_map = dict(single_target_shortest_path_length(graph, food_source))

			for l in range(len(self.actions)):
					action = self.actions[l]							

					next_vertex = ( (i + action[0])%self.length ,  (j + action[1])%self.height)
					if next_vertex in list(graph.neighbors((i,j))):
						dist = shortest_paths_map[next_vertex]

						#dist = np.abs(next_vertex[0]-food_source[0]) + np.abs(next_vertex[1] - food_source[1])
					if dist < min_distance:
							optimal_action_index  =  l
							min_distance = dist
			return optimal_action_index, self.actions[optimal_action_index]


	def create_color_map(self):
		self.color_map = np.zeros((self.length, self.height))
		for i in range(self.length):
			for j in range(self.height):
				optimal_action_index, _ = self.get_optimal_action_and_index((i,j), self.destination_node, self.graph)
				if np.random.random() < self.placeholder_color_prob and len(self.action_placeholder_map[optimal_action_index]) > 0:
							#print("aslkdmfalskdfmalskdmfalskdmfalksdmflaskdmflaksdmflaksdmflaksdmf")
							self.color_map[i,j] = np.random.choice(self.action_placeholder_map[optimal_action_index]) + self.num_colors
				else	:
					self.color_map[i, j] = np.random.choice(self.action_color_map[optimal_action_index])
							


	def reset_initial_and_destination(self, hard_instances, reinitialize_placeholders = False):
		super().reset_initial_and_destination(hard_instances)
		if reinitialize_placeholders:
			self.initialize_placeholders()
		self.create_color_map()



	def get_state_helper(self, curr_node):
		indicator_color = [0]*self.num_colors_with_placeholders
		indicator_color[int(self.color_map[curr_node[0], curr_node[1]])] = 1
		state = torch.tensor(indicator_color).float().to(DEVICE)
		return state

	def compute_agreement_ratio(self, string_with_placeholders):
		agreement = np.sum(string_with_placeholders*self.all_color_action_map_vectorized)
		agreement*=1.0/self.num_colors_with_placeholders
		return agreement


	def get_optimal_action_index(self, i,j):
		color_index = int(self.color_map[i,j] )
		#color_indicator = state.cpu().numpy()
		#color_index = int(np.sum((color_indicator > 0)*np.arange(self.env.get_state_dim())))
		#IPython.embed()
		if color_index < self.num_colors:
			return self.color_action_map[color_index]
		else:
			return self.placeholder_action_map[color_index - self.num_colors]



class ColorGridEnvironmentMultifood(ColorGridEnvironment):
	def __init__(self,  
							length, 
							height,
							num_food_sources = 1, 
							num_colors = 4,
							num_placeholder_colors = 1,
							color_action_map  = [0, 1, 2, 3], ### the action map format is just a set of indices in {0,1,2,3} of size num_colors 
							placeholder_color_prob = .2,
							pit = False, 
							pit_type = "border",
							pit_colors = 0,
							#near_pit_initialization = True,
							initialization_type = "avoiding_pit",
							randomization_threshold = 0, 
							state_representation = "two-dim",
							manhattan_reward = False, 
							location_normalized = False,
							encode_goal = False, 
							sparsity = 0,
							combine_with_sparse = False,
							reversed_actions = False,
							):
 

		super().__init__(length=length, 
							height = height, 
							num_colors = num_colors,
							num_placeholder_colors = num_placeholder_colors,
							color_action_map  = color_action_map,
							placeholder_color_prob = placeholder_color_prob,
							randomization_threshold = randomization_threshold, 
							manhattan_reward = manhattan_reward, 
							state_representation = state_representation,
							location_normalized = location_normalized,
							encode_goal = encode_goal, 
							sparsity = sparsity,
							combine_with_sparse = combine_with_sparse,
							reversed_actions = reversed_actions,
							initialize_color_map = False)
		
		self.name = "ColorSimpleMultifood"
		self.num_food_sources = num_food_sources
		self.destination_node = None ## This is to ensure that no function using self.destination_node as 

		if self.manhattan_reward:
			raise ValueError("Manhattan reward is not supported for the multi-food environment")


		#self.near_pit_initialization = near_pit_initialization
		self.initialization_type = initialization_type

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

		if initialization_type not in ["near_pit", "avoiding_pit"]:
			raise ValueError("Initialization_type not recognized {}".format(initialization_type))

		self.set_pit_nodes()
		self.pit_colors = pit_colors ### the pit color index is that of the first pit_colors part of num_colors
		if not self.pit and self.pit_colors >0:
			raise ValueError("pit colors >0 but pit set to false")
		if self.pit_colors >= self.num_colors:
			raise ValueError("Number of pit colors equals or surpases number of colors")
		if self.pit_colors%len(self.actions) != 0:
				raise ValueError("Number of pit colors is not a multiple of the number of actions")
		for action_index in color_action_map[:self.pit_colors]:
			if action_index not in list(range(len(self.actions))):
				raise ValueError("Pit color action index not in the valid action index set")
		for action_index in color_action_map[self.pit_colors:]:
			if action_index not in list(range(len(self.actions))):
				raise ValueError("Pit color action index not in the valid action index set")
		if self.pit_colors > 0 and len(set(color_action_map[:self.pit_colors])) != len(self.actions):
			raise ValueError("Number of actions in pit colors different from all actions")
		if len(set(color_action_map[self.pit_colors:])) != len(self.actions):
			raise ValueError("Number of actions in non pit colors different from all actions")


		self.reset_initial_and_food_sources()
		self.create_color_map()







	def set_pit_nodes(self):
		self.pit_nodes = []
		self.pit_adjacent_nodes = []
		self.pit_boundary = []

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
		length_rim = int(self.length*(3/8.0))
		height_rim = int(self.height*(3/8.0))

		# pit_length_indices = list(range(max(int(self.length*(3/8.0)), 1) , int((6.0/8)*self.length)))
		# pit_height_indices = list(range(max(int(self.height*(3/8.0)), 1) , int((6.0/8)*self.height)))
		
		pit_length_indices = list(range(length_rim, self.length-length_rim))
		pit_height_indices = list(range(height_rim, self.height - height_rim))

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

		# for i in range(self.num_food_sources):
		# 		food_source = random.choice(list(self.graph.nodes))
		# 		while food_source in [self.initial_node] + self.food_sources or food_source in self.pit_nodes: ### This we may want to change so that it doesn't cause an infinite loop 
		# 			food_source = random.choice(list(self.graph.nodes))
		# 		self.food_sources.append(food_source)
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
			




	def create_color_map(self):
		self.color_map = np.zeros((self.length, self.height))
		for i in range(self.length):
			for j in range(self.height):

				if self.pit and (i,j) in self.pit_adjacent_nodes and (i,j) in self.pit_nodes: 
					optimal_action_index = 0
				elif self.pit and (i,j) in self.pit_adjacent_nodes and (i,j) not in self.pit_nodes:
					for l in range(self.num_actions):
						if (i,j) in self.pit_four_rims[l]:
							into_pit_action = self.actions[l]

					optimal_action = (-into_pit_action[0], -into_pit_action[1])
					optimal_action_index = self.actions.index(optimal_action)

					# for l in range(len(self.actions)):
					# 		action = self.actions[l]							
					# 		next_vertex = ( (i + action[0])%self.length ,  (j + action[1])%self.height)
					# 		if next_vertex in list(self.graph.neighbors((i,j))) and next_vertex not in self.pit_adjacent_nodes:
					# 			optimal_action_index = l
				else:
					min_dist_food_source = float("inf")
					optimal_action_index = 0

					for food_source in self.food_sources:

						if self.pit and (i,j) not in self.pit_adjacent_nodes:
							curr_food_source_action_index, _ = self.get_optimal_action_and_index((i,j), food_source, self.pit_avoiding_graph)
							curr_food_source_distance = shortest_path_length(self.pit_avoiding_graph, (i,j), food_source)


						if not self.pit:
							curr_food_source_action_index, _ = self.get_optimal_action_and_index((i,j), food_source, self.graph)
							curr_food_source_distance = np.abs(i-food_source[0]) + np.abs(j - food_source[1])



						if curr_food_source_distance < min_dist_food_source:
							min_dist_food_source = curr_food_source_distance
							optimal_action_index = curr_food_source_action_index



				
				non_placeholder_color_choices = self.action_color_map[optimal_action_index]
				if self.pit and self.pit_colors >0 and self.is_pit_adjacent(i,j):
						non_placeholder_color_choices = list(set(non_placeholder_color_choices)&set(range(self.pit_colors)))
						self.color_map[i, j] = np.random.choice(non_placeholder_color_choices)

				else:					
					if np.random.random() < self.placeholder_color_prob and len(self.action_placeholder_map[optimal_action_index]) > 0:
								#print("aslkdmfalskdfmalskdmfalskdmfalksdmflaskdmflaksdmflaksdmflaksdmf")
								self.color_map[i,j] = np.random.choice(self.action_placeholder_map[optimal_action_index]) + self.num_colors
					else:
						non_placeholder_color_choices = list(set(non_placeholder_color_choices)&set(range(self.pit_colors, self.num_colors)))
						self.color_map[i, j] = np.random.choice(non_placeholder_color_choices)
						if not self.pit and self.pit_colors > 0:
							raise ValueError("Pit is not on but pit colors are more than zero ")

	def step(self, action_index):
		action = self.actions[action_index]
		next_vertex = ( (self.curr_node[0] + action[0])%self.length ,  (self.curr_node[1] + action[1])%self.height)
		neighbors =  list(self.graph.neighbors(self.curr_node))
		reward = self.reward(self.curr_node, action)
		
		## Only return a reward the first time we reach the destination node.



		if self.curr_node in self.food_sources and not self.end:
			#print("reached here! ", self.curr_node, self.destination_node)
			self.end = True

		if self.curr_node in self.pit_nodes:
				self.end = True

		## Dynamics:
		if next_vertex in neighbors and self.curr_node not in self.food_sources and self.curr_node not in self.pit_nodes:
			 self.curr_node = next_vertex
		

		return self.curr_node, reward

	def reward(self, node, action):
		if node in self.food_sources:
			return 1
		else:
			return 0

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


	while env.manhattan_reward or not env.end:
		# print(env.get_state().flatten())

		action_index, is_placeholder_color = policy.get_action(env.get_state().flatten(), with_info = True)
		# print("Action ", action_index, " is placeholder color ", is_placeholder_color)
		node_path.append(torch.from_numpy(np.array(env.curr_node)).to(DEVICE))
		states_info.append((env.get_state(), is_placeholder_color))

		old_vertex = env.curr_node
		_, r = env.step(action_index)
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














def save_color_graph_diagnostic_image(env, color_map, title, filename):
	directed_graph= env.graph.to_directed()
	pos = nx.spectral_layout(env.graph)
	draw(directed_graph, pos = pos,  node_size =10, arrows= False)


	node_lists_map = dict([])
	
	for i in range(color_map.shape[0]):
		for j in range(color_map.shape[1]):
			if color_map[i,j] not in node_lists_map.keys():
				node_lists_map[color_map[i,j]] = [(i,j)]
			else:
				node_lists_map[color_map[i,j]].append((i,j))
	num_colors_with_placeholders = len(node_lists_map.keys())

	#colors = ["orange", "purple"]
	if num_colors_with_placeholders <= 7:
		color_map = mcolors.BASE_COLORS
	elif num_colors_with_placeholders <= 10:
		color_map = mcolors.TABLEAU_COLORS
	else:
		color_map = mcolors.CSS4_COLORS
	colors = list(color_map.keys())
	if num_colors_with_placeholders > len(colors):
		raise ValueError("Too many num_colors_with_placeholders, I don't have enough colors")


	for i in range(len(node_lists_map.keys())):
		 
		nx.draw_networkx_nodes(directed_graph, pos = pos, nodelist = node_lists_map[list(node_lists_map.keys())[i]], node_size = 20, node_color = colors[i])


	nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=70, node_color = "blue")
	if env.destination_node == None:
		nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=env.food_sources, node_size=70, node_color = "black")
		if len(env.pit_nodes) > 0:
				nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=env.pit_nodes, node_size=70, node_color = "red")

	else:
		nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=70, node_color = "black")

	plt.title(title)
	plt.savefig(filename)
	plt.close("all")





def save_color_grid_diagnostic_image(env, color_map, title, filename, display_optimal_action = True, add_grid_colors = True):

	node_lists_map = dict([])
	
	for i in range(color_map.shape[0]):
		for j in range(color_map.shape[1]):
			if color_map[i,j] not in node_lists_map.keys():
				node_lists_map[color_map[i,j]] = [(i,j)]
			else:
				node_lists_map[color_map[i,j]].append((i,j))
	num_colors_with_placeholders = int(max(node_lists_map.keys()))



	if add_grid_colors:
		#colors = ["orange", "purple"]
		if num_colors_with_placeholders <= 7:
			cmap = mcolors.BASE_COLORS
		elif num_colors_with_placeholders <= 10:
			cmap = mcolors.TABLEAU_COLORS
		else:
			cmap = mcolors.CSS4_COLORS
		all_colors = list(cmap.keys())
		if num_colors_with_placeholders > len(all_colors):
			raise ValueError("Too many num_colors_with_placeholders, I don't have enough colors")
		cmap = colors_matplotlib.ListedColormap(all_colors[:int(max(node_lists_map.keys()))])



		bounds = np.arange(cmap.N + 1)
		norm = colors_matplotlib.BoundaryNorm(bounds, cmap.N)
		fig, ax = plt.subplots()
		ax.imshow(color_map, cmap = cmap, norm = norm)

	else:
		white_map = np.zeros(env.color_map.shape)
		cmap = colors_matplotlib.ListedColormap(["white"])
		bounds = np.arange(cmap.N + 1)
		norm = colors_matplotlib.BoundaryNorm(bounds, cmap.N)
		fig, ax = plt.subplots()
		ax.imshow(white_map, cmap = cmap, norm = norm)





	ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

	ax.tick_params(axis = 'x', labelbottom = False)
	ax.tick_params(axis = 'y', labelleft = False)
	#IPython.embed()

	ax.set_xticks(np.arange(-.5, color_map.shape[1], 1));
	ax.set_yticks(np.arange(-.5, color_map.shape[0], 1));

	#ax.arrow(1,1, )
	#node_lists_map.keys()
	alphabet_symbols = list(string.ascii_lowercase)
	display_symbol_dictionary = dict([])

	if len(node_lists_map.keys()) > len(alphabet_symbols):
		raise ValueError("Number of colors is larger than number of alphabet symbols. ")
		## TODO: Implement a solution to circumvent this issue.

	
	for i in range(len(node_lists_map.keys())):
		display_symbol_dictionary[list(node_lists_map.keys())[i]] = alphabet_symbols[i]



	for i in range(color_map.shape[0]):
		for j in range(color_map.shape[1]):
			optimal_action_index = env.get_optimal_action_index(i,j)
			if display_optimal_action:
				ax.text(j,i, env.action_names[optimal_action_index], fontsize = 8)				
			else:
				ax.text(j,i, display_symbol_dictionary[color_map[i,j]], fontsize = 8)


	ax.plot([env.initial_node[1]], [env.initial_node[0]], 'o', color = "blue", markersize = 12)
	if env.destination_node == None:
		ax.plot([food_source[1] for food_source in env.food_sources ], [food_source[0] for food_source in env.food_sources], 'o', color = 'black', markersize = 12)
		if len(env.pit_nodes) > 0:
			ax.plot([pit[1] for pit in env.pit_nodes], [pit[0] for pit in env.pit_nodes], 'o', color = 'red', markersize = 12)
	else:
		ax.plot([env.destination_node[1]], [env.destination_node[0]], 'o', color = 'black', markersize = 12)

	# plt.savefig("./slkdfmdslfkm.png")
	# IPython.embed()

	# for i in range(len(node_lists_map.keys())):


	# 	print("Nodes ", node_lists_map[list(node_lists_map.keys())[i]], " color ", colors[i])
	# 	nx.draw_networkx_nodes(directed_graph, pos = pos, nodelist = node_lists_map[list(node_lists_map.keys())[i]], node_size = 20, node_color = colors[i])



	# nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=70, node_color = "blue")
	# if env.destination_node == None:
	# 	nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=env.food_sources, node_size=70, node_color = "black")
	# 	if len(env.pit_nodes) > 0:
	# 			nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=env.pit_nodes, node_size=70, node_color = "red")

	# else:
	# 	nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=70, node_color = "black")

	plt.title(title)
	plt.savefig(filename)
	plt.close("all")


