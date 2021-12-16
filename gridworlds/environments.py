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


from .do_undo_maps import IdentityDoUndo
from .rewards import SimpleIndicatorReward


def get_grid_graph(length, height):
  graph = grid_graph((length, height))
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
###                                 "two-dim", "tabular", "overwritten",
###                                 "two-dim-location-normalized", "two-dim-encode-goal-location-normalized", "two-dim-encode-goal",
###                                 "tabular-encode-goal"]

class GridEnvironment:
  def __init__(self,  
              length, 
              height, 
              state_representation = "two-dim",
              randomization_threshold = 0,
              do_undo_map = IdentityDoUndo(),
              reward_function = SimpleIndicatorReward(),
              ):
    

    self.name = "GridSimple"
    self.reward_function = reward_function

    self.graph = get_grid_graph(height, length)
    self.initial_graph_edges = list(self.graph.edges) 
    self.randomization_threshold = randomization_threshold


    self.state_representation = state_representation


    state_representations_types = ["two-dim", "tabular", "overwritten",
      "two-dim-location-normalized", "two-dim-encode-goal-location-normalized", "two-dim-encode-goal",
      "tabular-encode-goal"]

    if self.state_representation not in state_representations_types:
      raise ValueError("The state representation provided is not available {}".format(self.state_representation))


    self.actions = [(1, 0), (-1,0), (0,1), (0, -1)]
    self.action_names = ["D", "U", "R", "L"]
  
    self.num_actions = len(self.actions)
    self.length = length
    self.height = height

    ### Perhpas change the initial and destination nodes. Consider making these fields private.
    self.initial_node = random.choice(list(self.graph.nodes))
    self.destination_node = random.choice(list(self.graph.nodes))
    
    self.curr_node = self.initial_node
    self.end = False
    self.shortest_paths = dict(single_target_shortest_path_length(self.graph,self.destination_node))

    self.do_undo_map = do_undo_map



  def get_curr_node(self):
    return self.curr_node

  def get_destination_node(self):
    return self.destination_node

  def get_height(self):
    return self.height

  def get_length(self):
    return self.length

  def reset_transformation(self):
    self.do_undo_map = IdentityDoUndo()


  def add_do_undo(self, do_undo_map):
    self.do_undo_map = do_undo_map

  def add_reward_function(self, reward_function):
    self.reward_function = reward_function

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
  

  def restart_env(self):
    self.curr_node = self.initial_node 
    self.end = False
  
  def get_state_dim(self):
    if "tabular" in self.state_representation:
      return self.height*self.length
    elif "two-dim" in self.state_representation:
      if "encode-goal" in self.state_representation:
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


  def get_initial_destination_states(self):
    initial_state = self.get_state_helper(self.initial_node)
    destination_state = self.get_state_helper(self.destination_node)
    return dict([('initial_state', initial_state), ("destination_state", destination_state)])

  def get_state_helper(self, curr_node):
    if "tabular" in self.state_representation:
      state = torch.zeros(self.length, self.height)
      state[curr_node[0], curr_node[1]] = 1
      if "encode-goal" in self.state_representation:
        state[self.destination_node[0], self.destination_node[1] ] =1
      
    elif "two-dim" in self.state_representation:
      state = torch.tensor(curr_node).float()
      if "encode-goal" in self.state_representation:
        state = torch.tensor(list(curr_node) + list(self.destination_node)).float()
      if "location-normalized" in self.state_representation:
        state = state/(max(self.length, self.height)*1.0)
    else:
      raise ValueError("State representation type not availale - {}".format(self.state_representation))


    return self.do_undo_map.do_state(state.flatten())

      
  ### Takes an action and returns the next state and reward value.
  ### Inputs
  ### action_index (int) = index of the action to take.  
  def step(self, action_index):
    


    action = self.actions[action_index]
    action = self.do_undo_map.do_action(action)
    new_action_index = self.actions.index(action)
    

    # IPython.embed()
    # raise ValueError("Asdflkm")

    if random.random() > self.randomization_threshold:
      next_vertex = self.get_next_vertex(self.curr_node[0], self.curr_node[1], action_index)
    else:
      next_vertex = self.curr_node

    reward_info = dict([("action", action)])


    reward = self.reward_function.evaluate(self, reward_info)

    #reward = self.reward(self.curr_node, action)
    
    ## Only return a reward the first time we reach the destination node.
    ## end the episode immediately after reaching a destination node.
    if self.curr_node == self.destination_node and not self.end:
      self.end = True

    ## Dynamics.
    ## If the destination_node has been reached the agent cannot move out of this node.
    if self.curr_node != self.destination_node:
       self.curr_node = next_vertex
    
    step_info = dict([])
    step_info["curr_node"] = self.curr_node
    step_info["reward"] = reward
    step_info["state"] = self.get_state()
    step_info["action"] = action
    step_info["action_index"] = action_index
    step_info["end"] = self.end

    return step_info

  def get_next_vertex(self, i,j, action_index):
    action = self.actions[action_index]

    next_vertex_0 = i+action[0]
    next_vertex_0 = max(0, next_vertex_0)
    next_vertex_0 = min(next_vertex_0, self.length-1)

    next_vertex_1 = j+action[1]
    next_vertex_1 = max(0, next_vertex_1)
    next_vertex_1 = min(next_vertex_1, self.height-1)

    next_vertex = (next_vertex_0, next_vertex_1)

    return next_vertex



  ### Sets the self.initial_node to initial_node
  def set_initial_node(self, initial_node):
    self.initial_node = initial_node
    self.restart_env()

  ### Sets the self.destination_node to destination_node  
  def set_destination_node(self, destination_node):
    self.destination_node = destination_node
    self.shortest_paths = dict(single_target_shortest_path_length(self.graph,self.destination_node))
    self.restart_env()





class GridEnvironmentNonMarkovian(GridEnvironment):
  def __init__(self,  
              length, height, 
              randomize = False, 
              randomization_threshold = 0, 
              manhattan_reward = False, 
              tabular = True, 
              location_based = False,
              location_normalized = False,
              encode_goal = False, 
              sparsity = 0,
              use_learned_reward_function = True,
              reward_network_type = "MLP",
              combine_with_sparse = False,
              reversed_actions = False,
              goal_region_radius = 1):
    
    super().__init__(length, height, randomize, randomization_threshold, manhattan_reward, tabular, location_based, 
      location_normalized, encode_goal, sparsity, use_learned_reward_function, reward_network_type, combine_with_sparse, reversed_actions)

    raise ValueError("Grid Environment Non Markovian not properly implemented. Needs to be updated to match the implementation of GridEnvironment.")

    self.name = "GridNonMarkovian"
    self.state_dim = self.get_state_dim()
    self.goal_region_radius = goal_region_radius
    self.trajectory_reward = 0
    self.set_goal_region()
    self.last_three_steps = []

  def restart_env(self):
    self.curr_node = self.initial_node 
    self.end = False
    self.trajectory_reward = 0
    self.last_three_steps = []


  def set_goal_region(self):
      goal_region = []
      for i in range(self.length):
        for j in range(self.height):
          if np.abs(i-self.destination_node[0]) + np.abs(j - self.destination_node[1]) <= self.goal_region_radius:
            goal_region.append((i,j))

      self.goal_region = goal_region

  def reset_initial_and_destination(self, hard_instances):
    self.initial_node = random.choice(list(self.graph.nodes))
    destination_node = random.choice(list(self.graph.nodes))
    if hard_instances:
      while np.abs(self.initial_node[0] - self.destination_node[0]) + np.abs(self.initial_node[1] - self.destination_node[1]) < (self.length + self.height)/2:
        self.initial_node = random.choice(list(self.graph.nodes))
        destination_node = random.choice(list(self.graph.nodes))
    
    #self.destination_node = destination_node
    self.set_destination_node(destination_node)
    self.restart_env()
    self.set_goal_region()

  def step(self, action_index):
      action = self.actions[action_index]
      next_vertex = ( (self.curr_node[0] + action[0])%self.length ,  (self.curr_node[1] + action[1])%self.height)
      neighbors =  list(self.graph.neighbors(self.curr_node))
      #reward = self.reward(self.curr_node, action)
      # if  self.curr_node in self.goal_region:
      #   self.trajectory_reward = 1
      # else:
      #   self.trajectory_reward = 0

      if len(self.last_three_steps) == 3:
        self.last_three_steps = self.last_three_steps[1:] + [self.curr_node]
      elif len(self.last_three_steps) < 3:
        self.last_three_steps += [self.curr_node]
      else:
        raise ValueError("The last three steps list is larger than 3")
      
      rew = 1
      for node in self.last_three_steps:
        if node not in self.goal_region:
          rew = 0
      self.trajectory_reward = rew
      ## Dynamics:
      
      if next_vertex in neighbors:
         self.curr_node = next_vertex
      
      return self.curr_node, None



class GridEnvironmentPit(GridEnvironment):
  def __init__(self,  
              length, 
              height,
              state_representation = "pit", ### modify this representation value.
              location_normalized = False,
              encode_goal = False, 
              pit = False, 
              pit_type = "border",
              initialization_type = "avoiding_pit",
              randomization_threshold = 0, 
              manhattan_reward = False, 
              sparsity = 0,
              combine_with_sparse = False,
              reversed_actions = False,
              length_rim = 3,
              height_rim = 3
              ):
 


    self.pit = pit

    if not self.pit:
      raise ValueError("There is no pit -- State dimension stuff will fail") 


    super().__init__(length=length, 
              height = height, 
              state_representation = "overwritten",
              location_normalized = location_normalized,
              encode_goal = encode_goal,
              randomization_threshold = randomization_threshold, 
              manhattan_reward = manhattan_reward, 
              sparsity = sparsity,
              combine_with_sparse = combine_with_sparse,
              reversed_actions = reversed_actions)
    
    self.name = "PitEnvironment"


    if not self.pit:
      raise ValueError("There is no pit -- State dimension stuff will fail") 


    self.destination_node = None ## This is to ensure that no function using self.destination_node works and interferes with the environment works. 

    if self.manhattan_reward:
      raise ValueError("Manhattan reward is not supported for the pit environment")


    #self.near_pit_initialization = near_pit_initialization
    self.initialization_type = initialization_type

    self.length_rim = length_rim
    self.height_rim = height_rim

    self.pit = pit
    self.pit_type = pit_type
    if pit_type not in ["border", "central"]:
      raise ValueError("pit type set to unknown pit type")
    # if not pit and near_pit_initialization:
    #   raise ValueError("Pit set to False and near pit initialization set to True")
    if not pit and initialization_type == "near_pit":
      raise ValueError("Pit set to False and near pit initialization set to True")
    if not pit and initialization_type == "avoiding_pit":
      raise ValueError("Pit set to False and initialization type set to avoiding_pit")

    if initialization_type not in ["near_pit", "avoiding_pit"]:
      raise ValueError("Initialization_type not recognized {}".format(initialization_type))

    self.set_pit_nodes()


    self.reset_initial_and_destination(hard_instances = False)


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



  def reset_initial(self):
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

      


  def get_state(self):
    ### Compute the closest distance to the pit
    min_dist = float("inf")
    for pit_node in self.pit_nodes:
      distance = np.abs(pit_node[0] - self.curr_node[0]) + np.abs(pit_node[1] - self.curr_node[1])
      if distance < min_dist:
        min_dist = distance

    ### Compute distances to all the food sources.

    distances_food = np.abs(self.destination_node[0]-self.curr_node[0]) + np.abs(self.destination_node[0]-self.curr_node[1]) 

    ### add an if else statement that loops over the different values. 
    state = torch.tensor([min_dist, distances_food])

    return state


  def get_state_dim(self):
    return 2


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








class GridEnvironmentMultifood(GridEnvironment):
  def __init__(self,  
              length, 
              height,
              state_representation = "pit-foodsources",
              location_normalized = False,
              encode_goal = False, 
              num_food_sources = 1, 
              randomization_threshold = 0, 
              manhattan_reward = False, 
              sparsity = 0,
              combine_with_sparse = False,
              reversed_actions = False,
              length_rim = 3,
              height_rim = 3
              ):
 

    if num_food_sources > 1 and encode_goal:
      raise ValueError("Number of food sources is more than one and state representation has encode goal option on")

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
              reversed_actions = reversed_actions)
    
    self.name = "SimpleMultifood"

    ### This needs to change. 
    if num_food_sources > 1 and encode_goal:
      raise ValueError("Number of food sources is more than one and state representation has encode goal option on")



    self.name = "SimpleMultifood"
    self.num_food_sources = num_food_sources
    self.destination_node = None ## This is to ensure that no function using self.destination_node as 

    if self.manhattan_reward:
      raise ValueError("Manhattan reward is not supported for the multi-food environment")


    #self.near_pit_initialization = near_pit_initialization

    self.length_rim = length_rim
    self.height_rim = height_rim


    self.reset_initial_and_food_sources()


  ### Overwriting the restart env function to do nothing.
  def restart_env(self):
    pass


  def reset_food_sources(self):
    self.food_sources = []
    valid_states = list(self.graph.nodes)
    valid_states.remove(self.initial_node)
    self.food_sources = random.sample(valid_states, self.num_food_sources)

  def remove_and_reset_one_food_source(self, food_source_node):    
    self.food_sources.remove(food_source_node)
    if len(self.food_sources) != self.num_food_sources-1:
      raise ValueError("Num food sources after removal inconsistent.")

    valid_states =[]
    for i in range(self.length):
      for j in range(self.height):
        if (i,j) not in [self.curr_node] + self.food_sources:
          valid_states.append((i,j))
    self.food_sources += random.sample(valid_states, 1)




  def reset_initial_and_food_sources(self):
    self.initial_node = random.choice(list(self.graph.nodes))
    self.curr_node = self.initial_node
    self.end = False
    self.reset_food_sources()
      


  def get_state(self):
    ### Compute distances to all the food sources.
    distances_food = [np.abs(a-self.curr_node[0]) + np.abs(b-self.curr_node[1]) for (a,b) in self.food_sources]

    state = torch.tensor(distances_food)

    return state


  def get_state_dim(self):
    return self.num_food_sources 


  def reward(self):
    return 0


  def step(self, action_index):
    next_vertex = self.get_next_vertex(self.curr_node[0], self.curr_node[1], action_index)    
    reward = self.reward(self.curr_node, action)
    
    ## Only return a reward the first time we reach the destination node.



    if self.curr_node in self.food_sources and not self.end:
      #print("reached here! ", self.curr_node, self.destination_node)
      self.end = True

    if self.curr_node in self.pit_nodes:
        self.end = True

    ## Dynamics:
    if self.curr_node not in self.food_sources:
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

      
      if len(self.food_sources) == 0:
        self.reset_food_sources()









    



def run_walk(env, policy, max_time = 1000):
  time_counter = 0
  node_path =  []
  states = []
  edge_path = []
  action_indices = []

  rewards = []
  while not env.end:
    node_path.append(torch.from_numpy(np.array(env.curr_node)))
    states.append(env.get_state())
    
    #print("state ", env.get_state())

    action_index = policy.get_action(env.get_state().flatten())
    old_vertex = env.curr_node
    step_info = env.step(action_index)
    r = step_info["reward"]
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


  
  if num_paths <= 2:
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










