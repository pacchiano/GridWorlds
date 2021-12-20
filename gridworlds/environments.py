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
from networkx import grid_graph, single_target_shortest_path_length, draw, spring_layout, draw_planar, grid_2d_graph, draw_shell, draw_spring, draw_spectral, shortest_path_length
from networkx.algorithms.components import is_connected
import random
import networkx as nx
import copy
import IPython
#%matplotlib inline
import itertools
from copy import deepcopy
from matplotlib import colors as colors_matplotlib


from .do_undo_maps import IdentityDoUndo
from .rewards import SimpleIndicatorReward, MultifoodIndicatorReward


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
DEVICE = torch.device("cpu") 




def get_grid_graph(length, height):
  graph = grid_graph((length, height))
  return graph



## This class implements a simple grid environment that consists of a 
## grid derived graph and with features equal to the grid locations.
### The grid has a single goal location. 
### The environment's actions are UP, DOWN, LEFT and RIGHT encoded as the two dimensional vectors 
### [(1,0), (-1, 0), (0,1), (0,-1)]
### Parameter desciptions.
### length (int) = GridEnvironment length
### height (int) = GridEnvironment height
### state_representation (string) = This parameter can take the form of either  "two-dim" or "tabular". In the first case the native state 
###                                 representation simply takes the form of a two dimensional location tuple. 
###                                 "two-dim", "tabular", "overwritten",
###                                 "two-dim-location-normalized", "two-dim-encode-goal-location-normalized", "two-dim-encode-goal",
###                                 "tabular-encode-goal"]
### 
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

    self.do_undo_map = do_undo_map


    ### Perhpas change the initial and destination nodes. Consider making these fields private.
    self.valid_initial_nodes = list(self.graph.nodes)
    self.valid_destination_nodes = list(self.graph.nodes)


    if state_representation != "overwritten":
      self.reset_environment()




    # self.initial_node = random.choice(list(self.graph.nodes))
    # self.destination_node = random.choice(list(self.graph.nodes))
    
    # self.curr_node = self.initial_node
    # self.end = False
    # self.shortest_paths = dict(single_target_shortest_path_length(self.graph,self.destination_node))







  def add_do_undo(self, do_undo_map):
    self.do_undo_map = do_undo_map

  def add_reward_function(self, reward_function):
    self.reward_function = reward_function

  def get_curr_node(self):
    return self.curr_node

  def get_destination_node(self):
    return self.destination_node

  def get_height(self):
    return self.height


  def get_initial_destination_states(self):
    initial_state = self.get_state_helper(self.initial_node)
    destination_state = self.get_state_helper(self.destination_node)
    return dict([('initial_state', initial_state), ("destination_state", destination_state)])





  def get_length(self):
    return self.length

  def get_name(self):
    return "{}_{}".format(self.length, self.height)


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


  def get_num_actions(self):
    return len(self.actions)



  def get_optimal_action_and_index(self, vertex):
    return get_optimal_action_and_index_single_destination(vertex, self.destination_node, self.graph)[0]


  def get_optimal_action_and_index_single_destination(self, vertex, destination_node, graph ):
      (i,j) = vertex
      optimal_action_index = 0
      min_distance  = float("inf")
      dist = float("inf")
      shortest_paths_map = dict(single_target_shortest_path_length(graph, destination_node))

      for l in range(len(self.actions)):
          action = self.actions[l]              

          next_vertex = self.get_next_vertex(i,j, l)

          #next_vertex = ( (i + action[0])%self.length ,  (j + action[1])%self.height)
          if next_vertex in list(graph.neighbors((i,j))):
            dist = shortest_paths_map[next_vertex]

            #dist = np.abs(next_vertex[0]-food_source[0]) + np.abs(next_vertex[1] - food_source[1])
          if dist < min_distance:
              optimal_action_index  =  l
              min_distance = dist
      return optimal_action_index, self.actions[optimal_action_index]



  ### OUTPUTS the current state as a pytorch tensor.
  def get_state(self):    
    return self.get_state_helper(self.curr_node)

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

  def reset_environment(self, info = dict([("hard_instances", True)])):
    self.reset_initial_and_destination(info["hard_instances"])



  def reset_initial_and_destination(self, hard_instances):
    self.initial_node = random.choice(self.valid_initial_nodes)

    self.destination_node = random.choice([node for node in self.valid_destination_nodes if node  != self.initial_node])
    if hard_instances:
      while np.abs(self.initial_node[0] - self.destination_node[0]) + np.abs(self.initial_node[1] - self.destination_node[1]) < (self.length + self.height)/3:
        self.initial_node = random.choice(self.valid_initial_nodes)
        self.destination_node = random.choice([node for node in self.valid_destination_nodes if node  != self.initial_node])
    
    #self.destination_node = destination_node
    self.shortest_paths = dict(single_target_shortest_path_length(self.graph,self.destination_node))
    self.restart_env()



  def reset_transformation(self):
    self.do_undo_map = IdentityDoUndo()

  
  ### restarts the environment
  def restart_env(self):
    self.curr_node = self.initial_node 
    self.end = False
  

  ### Sets the self.initial_node to initial_node
  def set_initial_node(self, initial_node):
    self.initial_node = initial_node
    self.restart_env()

      
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









## This class implements a simple grid environment that consists of a 
## grid derived graph and with features equal to the grid locations.
### The grid has a multiple food sources. 
### The environment's actions are UP, DOWN, LEFT and RIGHT encoded as the two dimensional vectors 
### [(1,0), (-1, 0), (0,1), (0,-1)]
### Parameter desciptions.
### length (int) = GridEnvironment length
### height (int) = GridEnvironment height
### state_representation (string) = This parameter can take the form of either  "two-dim" or "tabular". In the first case the native state 
###                                 representation simply takes the form of a two dimensional location tuple. 
###                                 "two-dim", "tabular", "overwritten",
###                                 "two-dim-location-normalized", "two-dim-encode-goal-location-normalized", "two-dim-encode-goal",
###                                 "tabular-encode-goal"]

class GridEnvironmentMultifood(GridEnvironment):
  def __init__(self,  
              length, 
              height,
              state_representation = "two-dim",
              num_food_sources = 1, 
              randomization_threshold = 0, 
              do_undo_map = IdentityDoUndo(),
              reward_function = MultifoodIndicatorReward()              
              ):
 


    valid_state_representations = ["two-dim", "tabular", "overwritten",
      "two-dim-location-normalized", "two-dim-encode-food-location-normalized", "two-dim-encode-food",
      "tabular-encode-food", "food-distances", "food-distances-encode-food", "food-distances-encode-food-normalized",
      "food-distances-normalized"]


    if "encode-goal" in state_representation:
      raise ValueError("Encode goal in state representation is not allowed for the Multifood Environment")


    if state_representation not in valid_state_representations:
      raise ValueError("State representation type not availale - {}".format(state_representation))




    GridEnvironment.__init__(
              self,
              length=length, 
              height = height, 
              state_representation = "overwritten",
              randomization_threshold = randomization_threshold,
              do_undo_map = do_undo_map,
              reward_function = reward_function
              )
    


    self.num_food_sources = num_food_sources
    self.state_representation = state_representation
    self.name = "SimpleMultifood"
    self.destination_node = None ## This is to ensure that no function uses self.destination_node as in the parent class.

    self.valid_initial_nodes = list(self.graph.nodes)
    self.valid_destination_nodes = list(self.graph.nodes)

    if self.state_representation != "overwritten":
      self.reset_environment()





  def get_state(self):

    if self.state_representation in ["two-dim", "tabular",
      "two-dim-location-normalized"]:
        state = GridEnvironment.get_state(self)

    elif self.state_representation in ["two-dim-encode-food-location-normalized", "two-dim-encode-food",
      "tabular-encode-food"]:
        if self.state_representation == "tabular-encode-food":
          state = torch.zeros(self.length, self.height).float()
          state[self.curr_node[0], self.curr_node[1]] = 1

          for (a,b) in self.food_sources:
            state[a,b] = 1          
          
        elif "two-dim" in self.state_representation:
          state = list(self.curr_node)
          if "encode-food" in self.state_representation:
            for food_source in self.food_sources:
              state += list(food_source)
          state = torch.tensor(state).float()

          if "location-normalized" in self.state_representation:
            state = state/(max(self.length, self.height)*1.0)


    elif "food-distances" in self.state_representation:
      state = [np.abs(a-self.curr_node[0]) + np.abs(b-self.curr_node[1]) for (a,b) in self.food_sources]
      
      if "encode-food" in self.state_representation:
          for food_source in self.food_sources:
            state += list(food_source)
      state = torch.tensor(state).float()
      if "normalized" in self.state_representation:
          state = state/((self.length + self.height)*1.0)


    else:
      raise ValueError("State representation type not availale - {}".format(self.state_representation))

    return self.do_undo_map.do_state(state.flatten())



  def get_initial_destination_states(self):
    raise ValueError("This function is not defined for the multifood environment")


  def get_optimal_action_and_index(self, vertex):
      return self.get_optimal_multifood_action_index_custom_graph(vertex, self.graph)


  def get_optimal_multifood_action_index_custom_graph(self, vertex, graph):
      min_dist_food_source = float('inf')
      optimal_action_index = 0
      for food_source in self.food_sources:
        if food_source not in graph.nodes:
          raise ValueError("food source is not in graph nodes {} vertex {}".format(food_source, vertex))
        curr_food_source_action_index, _ = self.get_optimal_action_and_index_single_destination(vertex, food_source, graph)
        curr_food_source_distance = shortest_path_length(graph, vertex, food_source)

        if curr_food_source_distance < min_dist_food_source:
          min_dist_food_source = curr_food_source_distance
          optimal_action_index = curr_food_source_action_index
      return optimal_action_index



  def get_state_dim(self):

    if self.state_representation in ["two-dim", "tabular",
      "two-dim-location-normalized"]:
        return GridEnvironment.get_state_dim(self)

    elif self.state_representation in ["two-dim-encode-food-location-normalized", "two-dim-encode-food",
      "tabular-encode-food"]:
        if self.state_representation == "tabular-encode-food":
          return self.length*self.height        
          
        elif "two-dim" in self.state_representation:
          if "encode-food" in self.state_representation:
            return 2*(self.num_food_sources+1)            
            

    elif "food-distances" in self.state_representation:
      ### Compute distances to all the food sources.
      state_dim = self.num_food_sources

      if "encode-food" in self.state_representation:
        state_dim += 2*self.num_food_sources
      return state_dim
    else:
      raise ValueError("State representation type not availale - {}".format(self.state_representation))





  ### Remove provided food source
  ### The new food source can only be places in a valid node, including
  ### all squares,  that do not contain a food source. 
  def remove_and_reset_one_food_source(self, food_source_node):    
    if food_source_node not in self.food_sources:
      raise ValueError("Attempted to remove a food source node {} that is not in the list of food sources {}.".format(food_source_node, self.food_sources))
    food_source_to_remove_index = self.food_sources.index(food_source_node)
    left_food_sources = self.food_sources[:food_source_to_remove_index]
    right_food_sources = self.food_sources[food_source_to_remove_index+1 :]
    valid_states = [node for node in self.valid_state_respawn_single_food if node != self.curr_node]
    new_food_sources = random.sample(valid_states, 1)
    self.food_sources = left_food_sources + new_food_sources + right_food_sources
    self.reset_valid_state_respawn_single_food()

  ### Brings the environment back to pristine conditions 
  def reset_environment(self, info = dict([])):
    self.reset_initial_and_food_sources()



  def reset_initial_and_food_sources(self):
    self.initial_node = random.choice(self.valid_initial_nodes)
    ##### 
    self.reset_valid_states_respawn_all_food_sources()
    self.food_sources = random.sample(self.valid_states_respawn_all_food_sources, self.num_food_sources)
    self.reset_valid_state_respawn_single_food()
    self.restart_env()


  def reset_valid_states_respawn_all_food_sources(self):
    self.valid_states_respawn_all_food_sources = deepcopy(self.valid_destination_nodes)
    self.valid_states_respawn_all_food_sources.remove(self.initial_node)


  def reset_valid_state_respawn_single_food(self):
    self.valid_state_respawn_single_food = [node for node in self.valid_destination_nodes if node not in self.food_sources]


  ### RESTART ENV BRINGS THE CURRENT NODE BACK TO THE INITIAL ONE.
  def restart_env(self):
      self.end = False
      self.curr_node = self.initial_node


  def start_day(self):
      self.end = False
      self.initial_node = self.curr_node

      if self.curr_node in self.food_sources:
        self.remove_and_reset_one_food_source(self.curr_node)

      
      if len(self.food_sources) == 0:
        raise ValueError("Somehow food sources went down to zero")


  def step(self, action_index):
    next_vertex = self.get_next_vertex(self.curr_node[0], self.curr_node[1], action_index)    

    reward_info = dict([])
    reward = self.reward_function.evaluate(self, reward_info)
    
    ## Only return a reward the first time we reach the destination node.
    if self.curr_node in self.food_sources and not self.end:
      self.end = True

    ## Dynamics:
    if self.curr_node not in self.food_sources:
       self.curr_node = next_vertex
    

    step_info = dict([])
    step_info["curr_node"] = self.curr_node
    step_info["reward"] = reward
    step_info["state"] = self.get_state()
    step_info["action"] = self.actions[action_index]
    step_info["action_index"] = action_index
    step_info["end"] = self.end

    step_info["is_food_source"] = self.curr_node in self.food_sources

    return step_info











## This class implements a simple grid environment that consists of a 
## grid derived graph and with features equal to the grid locations.
### The grid has a multiple food sources. 
### The environment's actions are UP, DOWN, LEFT and RIGHT encoded as the two dimensional vectors 
### [(1,0), (-1, 0), (0,1), (0,-1)]
### Parameter desciptions.
### length (int) = GridEnvironment length
### height (int) = GridEnvironment height
### state_representation (string) = 


class GridEnvironmentPit(GridEnvironment):
  def __init__(self,  
              length, 
              height,
              state_representation = "pit", ### modify this representation value.
              pit_type = "border",
              initialization_type = "avoiding_pit",
              randomization_threshold = 0, 
              length_rim = 3,
              height_rim = 3,
              do_undo_map = IdentityDoUndo(),
              reward_function = SimpleIndicatorReward(),
              ):
 
    valid_state_representations = ["two-dim", "tabular", "overwritten",
      "two-dim-location-normalized", "two-dim-encode-goal-location-normalized", "two-dim-encode-goal",
      "tabular-encode-goal"]

    if state_representation not in valid_state_representations:
      raise ValueError("State representation type not availale - {}".format(state_representation))

    #IPython.embed()

    GridEnvironment.__init__(
              self,
              length=length, 
              height = height, 
              state_representation = "overwritten",
              randomization_threshold = randomization_threshold,
              do_undo_map = do_undo_map,
              reward_function = reward_function
              )

    
    self.name = "PitEnvironment"
    self.state_representation = state_representation


    #self.near_pit_initialization = near_pit_initialization
    self.initialization_type = initialization_type

    self.length_rim = length_rim
    self.height_rim = height_rim

    self.pit_type = pit_type
    if pit_type not in ["border", "central"]:
      raise ValueError("pit type set to unknown pit type")

    if initialization_type not in ["near_pit", "avoiding_pit"]:
      raise ValueError("Initialization_type not recognized {}".format(initialization_type))


    #pself.pit = True
    self.set_pit_nodes()

    if self.initialization_type == "near_pit":
      self.valid_initial_nodes = list(self.outer_rim)
    elif self.initialization_type == "avoiding_pit":
      self.valid_initial_nodes = list(self.pit_avoiding_graph.nodes)
    else:
      raise ValueError("Initialization type not recognized {}".format(initialization_type))

    self.valid_destination_nodes = list(self.pit_avoiding_graph.nodes)



    if self.state_representation != "overwritten":
      self.reset_environment()



  def get_optimal_action_and_index(self, vertex):
    if self.is_pit(vertex[0], vertex[1]):
      raise ValueError("Asked for optimal action for a pit vertex.")
    if vertex in list(self.pit_avoiding_graph):
      return self.get_optimal_action_and_index_single_destination(vertex, self.destination_node, self.pit_avoiding_graph)[0]
    else:
      in_pit_four_rims = [vertex in rim for rim in self.pit_four_rims]
      return in_pit_four_rims.index(True)


  def set_pit_nodes(self):
    self.pit_nodes = []
    self.outer_rim = []

    ## The i-th list in self.pit_four_rims
    ## corresponds to the pit_rim nodes for which action i
    ## takes us into the pit.
    self.pit_four_rims = [[] for _ in range(4)]

    
    for i in range(self.length):
      for j in range(self.height):
        if self.is_pit(i,j):
          self.pit_nodes.append((i,j))

        if self.is_outer_rim(i,j):
          self.outer_rim.append((i,j))

    for (i,j) in self.outer_rim:
      ### find the node in the pit that 
      adjacent_nodes = [self.get_next_vertex(i,j, m) for m in range(self.num_actions)]

      adjacent_pit_node_index = [self.is_pit(i,j) for (i,j) in adjacent_nodes].index(True)
      (i_pit, j_pit) = adjacent_nodes[adjacent_pit_node_index]

      neighbors_of_pit_node = [self.get_next_vertex(i_pit, j_pit, m) for m in range(self.num_actions)]
      i_j_index = neighbors_of_pit_node.index((i,j))

      self.pit_four_rims[i_j_index].append((i,j))


    # if len(self.pit_nodes) == 0:
    #   raise ValueError("Num of pit nodes is zero in a pit environment")

    self.pit_avoiding_graph = deepcopy(self.graph)
    for node in self.pit_nodes + self.outer_rim:
      self.pit_avoiding_graph.remove_node(node)




  def is_border_pit(self, i, j):
    if i==0:
      return True
    else:
      return False


  def is_central_pit(self, i,j):
    pit_length_indices, pit_height_indices = self.get_central_pit_indices()
    if i in  pit_length_indices and j in pit_height_indices:
      return True
    else:
      return False



  def is_outer_rim(self, i,j):
    if self.is_pit(i,j):
      return False
    else:
     adjacent_nodes = [ (max(0, i-1), j), (i, max(0, j-1)), (min(i+1, self.length-1), j), (i, min(j+1, self.height-1)) ]
     if np.sum([self.is_pit(node[0], node[1]) for node in adjacent_nodes]) > 0:
        return True
     else:
        return False



  def get_central_pit_indices(self):    
    pit_length_indices = list(range(self.length_rim, self.length-self.length_rim))
    pit_height_indices = list(range(self.height_rim, self.height - self.height_rim))
    return pit_length_indices, pit_height_indices



  def is_pit(self, i, j):
    if self.pit_type == "border":
      return self.is_border_pit(i,j)
    elif self.pit_type == "central":
      return self.is_central_pit(i,j)
    else:
      raise ValueError("unrecognized pit_type {}".format(self.pit_type))

      


  def get_state(self):

    if self.state_representation in ["two-dim", "tabular", "overwritten",
      "two-dim-location-normalized", "two-dim-encode-goal-location-normalized", "two-dim-encode-goal",
      "tabular-encode-goal"]:
        return GridEnvironment.get_state(self)
    else:
      raise ValueError("Not implemented state representation {}".format(self.state_representation))




  def step(self, action_index):
    action = self.actions[action_index]
    
    if random.random() > self.randomization_threshold:
      next_vertex = self.get_next_vertex(self.curr_node[0], self.curr_node[1], action_index)
    else:
      next_vertex = self.curr_node

    #next_vertex = ( (self.curr_node[0] + action[0])%self.length ,  (self.curr_node[1] + action[1])%self.height)
    #neighbors =  list(self.graph.neighbors(self.curr_node))
    reward_info = dict([("action", action)])

    reward = self.reward_function.evaluate(self, reward_info)
    
    ## Only return a reward the first time we reach the destination node.

    if self.curr_node in self.pit_nodes:
        self.end = True

    ## Dynamics:
    if self.curr_node not in self.pit_nodes and self.curr_node != self.destination_node:
       self.curr_node = next_vertex
    



    step_info = dict([])
    step_info["curr_node"] = self.curr_node
    step_info["reward"] = reward
    step_info["state"] = self.get_state()
    step_info["action"] = action
    step_info["action_index"] = action_index
    step_info["end"] = self.end
    step_info["is_pit"] = self.curr_node in self.pit_nodes

    return step_info










class GridEnvironmentPitMultifood(GridEnvironmentPit,GridEnvironmentMultifood):
  def __init__(self,  
              length, 
              height,
              state_representation = "pit", ### modify this representation value.
              pit_type = "central",
              initialization_type = "avoiding_pit",
              randomization_threshold = 0, 
              num_food_sources = 1,
              length_rim = 3,
              height_rim = 3,
              do_undo_map = IdentityDoUndo(),
              reward_function = MultifoodIndicatorReward(),
              ):
 
    valid_multifood_state_representations = ["two-dim", "tabular", "overwritten",
      "two-dim-location-normalized", "two-dim-encode-food-location-normalized", "two-dim-encode-food",
      "tabular-encode-food", "food-distances", "food-distances-encode-food", "food-distances-encode-food-normalized",
      "food-distances-normalized"]


    if state_representation not in valid_multifood_state_representations:
      raise ValueError("State representation type not availale - {}".format(state_representation))



    GridEnvironmentPit.__init__(
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


    GridEnvironmentMultifood.__init__(
              self,
              length=length, 
              height = height, 
              state_representation = "overwritten",
              randomization_threshold = randomization_threshold,
              num_food_sources = num_food_sources,
              do_undo_map = do_undo_map,
              reward_function = reward_function
              )

    
    self.name = "MultifoodPitEnvironment"
    self.state_representation = state_representation
    self.destination_node = None

    if self.initialization_type == "near_pit":
      self.valid_initial_nodes = list(self.outer_rim)
    elif self.initialization_type == "avoiding_pit":
      self.valid_initial_nodes = list(self.pit_avoiding_graph.nodes)
    else:
      raise ValueError("Initialization type not recognized {}".format(initialization_type))

    self.valid_destination_nodes = list(self.pit_avoiding_graph.nodes)



    if self.state_representation != "overwritten":
      self.reset_environment()


    #IPython.embed()








  ### Brings the environment back to pristine conditions 
  def reset_environment(self, info = dict([])):
    GridEnvironmentMultifood.reset_initial_and_food_sources(self)

      
  def restart_env(self):
      self.end = False
      self.curr_node = self.initial_node

  def get_optimal_action_and_index(self, vertex):

      if self.is_pit(vertex[0], vertex[1]):
        raise ValueError("Asked for optimal action for a pit vertex.")

      if vertex in list(self.pit_avoiding_graph.nodes):
        return self.get_optimal_multifood_action_index_custom_graph(vertex, self.pit_avoiding_graph)
      else:
        in_pit_four_rims = [vertex in rim for rim in self.pit_four_rims]
        return in_pit_four_rims.index(True)




  def get_state(self):

    if self.state_representation in ["two-dim", "tabular", "overwritten",
      "two-dim-location-normalized", "two-dim-encode-food-location-normalized", "two-dim-encode-food",
      "tabular-encode-food", "food-distances", "food-distances-encode-food", "food-distances-encode-food-normalized",
      "food-distances-normalized"]:
        return GridEnvironmentMultifood.get_state(self)
    else:
      raise ValueError("Not implemented state representation {}".format(self.state_representation))




  def step(self, action_index):
    action = self.actions[action_index]
    
    if random.random() > self.randomization_threshold:
      next_vertex = self.get_next_vertex(self.curr_node[0], self.curr_node[1], action_index)
    else:
      next_vertex = self.curr_node

    neighbors =  list(self.graph.neighbors(self.curr_node))
    reward_info = dict([("action", action)])

    reward = self.reward_function.evaluate(self, reward_info)
    

    if self.curr_node in self.pit_nodes:
        self.end = True

    if self.curr_node in self.food_sources:
      self.end = True

    ## Dynamics:
    if self.curr_node not in self.food_sources + self.pit_nodes:
       self.curr_node = next_vertex
    

    step_info = dict([])
    step_info["curr_node"] = self.curr_node
    step_info["reward"] = reward
    step_info["state"] = self.get_state()
    step_info["action"] = action
    step_info["action_index"] = action_index
    step_info["end"] = self.end
    step_info["is_pit"] = self.curr_node in self.pit_nodes
    step_info["is_food_source"] = self.curr_node in self.food_sources

    return step_info




  def start_day(self):
    GridEnvironmentMultifood.start_day(self)



  # def reset_initial_and_destination(self, hard_instances = True):
  #   raise ValueError("Reset initial and destination is not available for MultifoodPitEnvironment.")



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







### Except for state_info this is the same function as run_walk
def run_multifood_walk(env, policy, max_time = 1000):
  time_counter = 0
  node_path =  []
  states_info = []
  edge_path = []
  action_indices = []

  rewards = []
  is_food_source_list = []

  while not env.end:
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














