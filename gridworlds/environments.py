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
from networkx import grid_graph, single_target_shortest_path_length, draw, spring_layout, draw_planar, grid_2d_graph, draw_shell, draw_spring, draw_spectral
from networkx.algorithms.components import is_connected
import random
import networkx as nx
import copy
import IPython
#%matplotlib inline
from matplotlib import colors as colors_matplotlib



def sparsify_graph(graph, sparsity):
  shuffled_edges = random.sample(list(graph.edges), len(graph.edges))
  for edge in graph.edges:
    if random.random() < sparsity:
      graph.remove_edge(*edge)

      if not is_connected(graph):
        graph.add_edge(*edge)
  return graph


def get_grid_graph(length, height, sparse = True, sparsity = .1):
  graph = grid_graph((length, height))
  if sparse:
	  graph = sparsify_graph(graph, sparsity)
  return graph


def node_list_to_tuples(node_list):
 return [tuple(node.numpy()) for x in node_list]

def edge_list_to_tuples(edge_list):
  return [ [tuple(node.numpy()) for node in list(edge)] for edge in edge_list  ]

## This class implements a simple grid environment that consists of a 
## grid derived graph and with features equal to the grid locations.
### The grid has a single goal location. 
### The environment's actions are UP, DOWN, LEFT and RIGHT encoded as the two dimensional vectors 
### [(1,0), (-1, 0), (0,1), (0,-1)]
### Parameter desciptions.
### length (int) = GridEnvironment length
### height (int) = GridEnvironment height
### manhattan_reward (boolean) = If True the reward equals the negative manhattan distance between the current location and the goal.
### state_representation (string) = This parameter can take the form of either  "two-dim" or "tabular". In the first case the native state 
###                                 representation simply takes the form of a two dimensional location tuple. 
### location_normalized = If True the state features are normalized.
### encode_goal (boolean) = If True the state also contains the goal. The dimensionality of the state doubles in this case.  
### sparsity (double) = Probability of sparsity per edge. 
### reversed_actions (boolean) = If True, the indexing of the actions is different from the standard one. 

class GridEnvironment:
  def __init__(self,  
              length, 
              height, 
              manhattan_reward = False, 
              state_representation = "two-dim",
              location_normalized = False,
              encode_goal = False, 
              sparsity = 0,
              combine_with_sparse = False,
              reversed_actions = False):
    
    self.graph = get_grid_graph(height, length, sparsity = sparsity)
    self.initial_graph_edges = list(self.graph.edges) 
    


    self.location_normalized = location_normalized  

    self.state_representation = state_representation

    self.reversed_actions = reversed_actions

    if self.reversed_actions:
      self.actions = [(1, 0), (-1,0), (0,1), (0, -1)]
      self.action_names = ["D", "U", "R", "L"]

    else:
      self.actions = [(-1, 0), (1,0), (0,-1), (0, 1)]
      self.action_names = ["U", "D", "L", "R"]
      
    self.length = length
    self.height = height

    self.encode_goal = encode_goal
    self.manhattan_reward = manhattan_reward

    ### Perhpas change the initial and destination nodes.
    self.initial_node = random.choice(list(self.graph.nodes))
    self.destination_node = random.choice(list(self.graph.nodes))
    
    self.curr_node = self.initial_node
    self.end = False
    self.shortest_paths = dict(single_target_shortest_path_length(self.graph,self.destination_node))


    self.P = torch.eye(self.get_state_dim())
    self.inverse_P = torch.eye(self.get_state_dim())


  def add_linear_transformation(self, P):
    self.P = P
    self.inverse_P = torch.inverse(self.P)


  def reset_linear_transformation(self):
    self.P = torch.eye(self.get_state_dim())  
    self.inverse_P = torch.eye(self.get_state_dim())


  ### Applies P{-1} to the trajectory
  def apply_undo_map(self, trajectory):
    undone_trajectory = copy.deepcopy(trajectory)
    undone_trajectory['states'] = torch.matmul(undone_trajectory['states'], self.inverse_P)
    # for key in trajectory.keys():
    #   pass
    return undone_trajectory

  def reverse_environment(self):
    self.reversed_actions = not self.reversed_actions
    if self.reversed_actions:
      self.actions = [(1, 0), (-1,0), (0,1), (0, -1)]
    else:
      self.actions = [(-1, 0), (1,0), (0,-1), (0, 1)]

  def reset_initial(self, hard_instances):
    self.initial_node = random.choice(list(self.graph.nodes))
    #self.destination_node = random.choice(list(self.graph.nodes))
    if hard_instances:
      while np.abs(self.initial_node[0] - self.destination_node[0]) + np.abs(self.initial_node[1] - self.destination_node[1]) < (self.length + self.height)/3:
        self.initial_node = random.choice(list(self.graph.nodes))
        #self.destination_node = random.choice(list(self.graph.nodes))
    self.restart_env()


  def reset_initial_and_destination(self, hard_instances):
    self.initial_node = random.choice(list(self.graph.nodes))
    destination_node = random.choice(list(self.graph.nodes))
    if hard_instances:
      while np.abs(self.initial_node[0] - self.destination_node[0]) + np.abs(self.initial_node[1] - self.destination_node[1]) < (self.length + self.height)/3:
        self.initial_node = random.choice(list(self.graph.nodes))
        destination_node = random.choice(list(self.graph.nodes))
    
    #self.destination_node = destination_node
    self.set_destination_node(destination_node)
    self.restart_env()

  def get_name(self):
    return "{}_{}".format(self.length, self.height)
  
  def restore_initial_graph(self):
    self.graph = nx.Graph()
    self.graph.add_edges_from(self.initial_graph_edges)

  def sparsify_graph(self, sparsity):
    self.graph = sparsify_graph(self.graph, sparsity)

  def remove_random_edge(self):
    edge_to_remove = random.choice(list(self.graph.edges))
    self.graph.remove_edge(*edge_to_remove)

  def restart_env(self):
    self.curr_node = self.initial_node 
    self.end = False
  
  def get_state_dim(self):
    if self.state_representation == "tabular":
      return self.height*self.length
    elif self.state_representation == "two-dim":
      if self.encode_goal:
        return 4
      else:
        return 2
    else:
      raise ValueError("State representation not available - {}".format(self.state_representation))

  def get_num_actions(self):
    return 4


  ### OUTPUTS the current state as a pytorch tensor.
  def get_state(self):    
    return self.get_state_helper(self.curr_node)

  def get_state_helper(self, curr_node):
    if self.state_representation == "tabular":
      state = torch.zeros(self.length, self.height)
      state[curr_node[0], curr_node[1]] = 1
      if self.encode_goal:
        state[self.destination_node[0], self.destination_node[1] ] =1
      
    elif self.state_representation == "two-dim":
      state = torch.tensor(curr_node).float()
      if self.encode_goal:
        state = torch.tensor(list(curr_node) + list(self.destination_node)).float()
      if self.location_normalized:
        state = state/(max(self.length, self.height)*1.0)
    else:
      raise ValueError("State representation type not availale - {}".format(self.state_representation))

    return torch.matmul(self.P, state.flatten())


  ### outputs the reward of executing action on node.
  ### node 
  def reward(self, node, action):
    if self.manhattan_reward:
      return -1.0*(np.abs(node[0]  - self.destination_node[0]) +  np.abs(node[1]  - self.destination_node[1]))/(self.length + self.height)
    else:
      if node == self.destination_node:
        return 1
      else:
        return 0
      
  ### Takes an action and returns the next state and reward value.
  ### Inputs
  ### action_index (int) = index of the action to take.  
  def step(self, action_index):
    action = self.actions[action_index]
    next_vertex = ( (self.curr_node[0] + action[0])%self.length ,  (self.curr_node[1] + action[1])%self.height)
    neighbors =  list(self.graph.neighbors(self.curr_node))
    reward = self.reward(self.curr_node, action)
    
    ## Only return a reward the first time we reach the destination node.
    ## end the episode immediately after reaching a destination node.
    if self.curr_node == self.destination_node and not self.end:
      self.end = True

    ## Dynamics.
    ## If the destination_node has been reached the agent cannot move out of this node.
    if next_vertex in neighbors and self.curr_node != self.destination_node:
       self.curr_node = next_vertex
    

    return self.curr_node, reward

  ### Sets the self.initial_node to initial_node
  def set_initial_node(self, initial_node):
    self.initial_node = initial_node
    self.restart_env()

  ### Sets the self.destination_node to destination_node  
  def set_destination_node(self, destination_node):
    self.destination_node = destination_node
    self.shortest_paths = dict(single_target_shortest_path_length(self.graph,self.destination_node))
    self.restart_env()












    



def run_walk(env, policy, max_time = 1000):
  time_counter = 0
  node_path =  []
  states = []
  edge_path = []
  action_indices = []

  rewards = []
  while env.manhattan_reward or not env.end:
    node_path.append(torch.from_numpy(np.array(env.curr_node)))
    states.append(env.get_state())
      
    action_index = policy.get_action(env.get_state().flatten())
    old_vertex = env.curr_node
    _, r = env.step(action_index)
    action_indices.append(action_index)
    edge_path.append(torch.from_numpy(np.array((old_vertex, env.curr_node))))
    rewards.append(r)

    time_counter += 1
    if time_counter > max_time:
      break
  return node_path, edge_path, states, action_indices, rewards







def save_graph_diagnostic_image(env, policy, num_steps, num_paths, title, filename):
  directed_graph= env.graph.to_directed()
  pos = nx.spectral_layout(env.graph)
  draw(directed_graph, pos = pos,  node_size =10, arrows= False)

  nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=90, node_color = "blue")
  if env.destination_node != None:
    nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=90, node_color = "black")


  else:
    nx.draw_networkx_nodes(directed_graph, pos = pos, nodelist= env.food_sources, node_size = 90, node_color = "black")
    if len(env.pit_nodes) > 0:
      nx.draw_networkx_nodes(directed_graph, pos = pos, nodelist= env.pit_nodes, node_size = 90, node_color = "red")
      if env.initial_node in env.pit_nodes:
          nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=90, node_color = "purple")


  
  if num_paths == 2:
    colors_small = ["orange", "purple"]
  if num_paths <= 7:
    color_map = mcolors.BASE_COLORS
  elif num_paths <= 10:
    color_map = mcolors.TABLEAU_COLORS
  else:
    color_map = mcolors.CSS4_COLORS
  colors = list(color_map.keys())
  if num_paths > len(colors):
    raise ValueError("Too many paths, I don't have enough colors")

  for i in range(num_paths):

    env.restart_env()
    node_path1, edge_path1,states1, action_indices1, rewards1  = run_walk(env, policy, num_steps)
    if num_paths > 2:
      _=nx.draw_networkx_edges(directed_graph, pos = pos, edgelist = edge_list_to_tuples(edge_path1), width = 4, 
                             edge_color = color_map[colors[i]],  arrows = True, arrowsize=5)
    else:
      _=nx.draw_networkx_edges(directed_graph, pos = pos, edgelist = edge_list_to_tuples(edge_path1), width = 4, 
                             edge_color = colors_small[i],  arrows = True, arrowsize=5)


  plt.title(title)
  plt.savefig(filename)
  plt.close("all")





def save_grid_diagnostic_image(env, policy, num_steps, num_paths, title, filename):
  white_map = np.zeros((env.length, env.height))

  cmap = colors_matplotlib.ListedColormap(["white"])

  bounds = np.arange(cmap.N + 1)
  norm = colors_matplotlib.BoundaryNorm(bounds, cmap.N)

  fig, ax = plt.subplots()
  ax.imshow(white_map, cmap = cmap, norm = norm)
  ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

  ax.tick_params(axis = 'x', labelbottom = False)
  ax.tick_params(axis = 'y', labelleft = False)
  #IPython.embed()

  ax.set_xticks(np.arange(-.5, white_map.shape[1], 1));
  ax.set_yticks(np.arange(-.5, white_map.shape[0], 1));


  ax.plot([env.initial_node[1]], [env.initial_node[0]], 'o', color = "blue", markersize = 12)
  if env.destination_node == None:
    ax.plot([food_source[1] for food_source in env.food_sources ], [food_source[0] for food_source in env.food_sources], 'o', color = 'black', markersize = 12)
    if len(env.pit_nodes) > 0:
      ax.plot([pit[1] for pit in env.pit_nodes], [pit[0] for pit in env.pit_nodes], 'o', color = 'red', markersize = 12)
  else:
    ax.plot([env.destination_node[1]], [env.destination_node[0]], 'o', color = 'black', markersize = 12)


  if num_paths == 1:
    colors_small = ["orange"]

  elif num_paths == 2:
    colors_small = ["orange", "purple"]
  if num_paths <= 7:
    color_map = mcolors.BASE_COLORS
  elif num_paths <= 10:
    color_map = mcolors.TABLEAU_COLORS
  else:
    color_map = mcolors.CSS4_COLORS
  colors = list(color_map.keys())
  if num_paths > len(colors):
    raise ValueError("Too many paths, I don't have enough colors")

  for i in range(num_paths):

    env.restart_env()
    node_path1, edge_path1,states1, action_indices1, rewards1  = run_walk(env, policy, num_steps)
    

    if num_paths > 2:
      color = color_map[colors[i]]
    else:
      color = colors_small[i]
    for (node1, node2) in edge_list_to_tuples(edge_path1):
      ax.arrow(node1[1], node1[0], node2[1]-node1[1], node2[0]-node1[0], length_includes_head=True,
          head_width=0.2, head_length=0.3, color = color)

  plt.title(title)
  plt.savefig(filename)
  plt.close("all")










