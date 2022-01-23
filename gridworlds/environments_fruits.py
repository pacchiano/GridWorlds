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

from .environments import GridEnvironmentMultifood

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
DEVICE = torch.device("cpu") 




class GridEnvironmentPoisonFruits(GridEnvironmentMultifood):
  def __init__(self,  
              length, 
              height,
              state_representation = "potentials",
              num_food_sources = 1, 
              randomization_threshold = 0, 
              do_undo_map = IdentityDoUndo(),
              reward_function = MultifoodIndicatorReward()              
              ):
 


    valid_state_representations = ["potential-differences"]


    if state_representation not in valid_state_representations:
      raise ValueError("State representation type not availale - {}".format(state_representation))



    GridEnvironmentMultifood.__init__(
              self,
              length=length, 
              height = height, 
              state_representation = "overwritten",
              num_food_sources = num_food_sources,
              randomization_threshold = randomization_threshold,
              do_undo_map = do_undo_map,
              reward_function = reward_function
              )
    

    self.state_representation = state_representation
    self.sick = None
    self.name = "PoisonMultifood"

    if self.state_representation != "overwritten":
      self.reset_environment()

    if self.num_food_sources%2 == 1:
      raise ValueError("Odd number of food sources in ")


    self.food_source_types = ["Good"]*int(self.num_food_sources/2) + ["Bad"]*int(self.num_food_sources/2)

    self.recovery_probabilities = [1]*int(self.num_food_sources/2) + [.1]*int(self.num_food_sources/2)




    def get_reward_info(self):
      neighbors = [self.get_next_vertex(self.curr_node[0], self.curr_node[1], action_index) for action_index in range(self.num_actions)]

      info = dict([])
      info["neighbors"] = neighbors
      info["curr_node"] = self.curr_node

      info["is_sick"] = self.sick
      if self.curr_node in self.food_sources and not self.sick:
        info["is_food"] = True
      else:
        info["is_food"] = False

      return info



    def get_potentials(self, i, j, food_label):
      potential_value = 0
      for i in range self.food_sources:
         if self.food_source_types[i] == food_label:
            food_source = self.food_sources[i]
            potential_value += np.exp(-np.sqrt( (food_source[0]-i)**2 + (food_source[1]-j)**2  ))
      return potential_value


    def get_potential_differences(self, i,j):
      neighbors = self.get_next_vertex(self.curr_node)
      good_potential_differences = [0]*self.num_actions
      bad_potential_differences = [0]*self.num_actions

      good_potential_value_curr_node = self.get_potentials(self.curr_node[0], self.curr_node[1], "Good")
      bad_potential_value_curr_node = self.get_potentials(self.curr_node[0], self.curr_node[1], "Bad")



      for action_index in range(self.num_actions):

        good_potential_differences[action_index] = self.get_potentials(neighbors[action_index], neighbors[action_index], "Good") - good_potential_value_curr_node
        bad_potential_differences[action_index] = self.get_potentials(neighbors[action_index], neighbors[action_index], "Bad") - bad_potential_value_curr_node


      return good_potential_differences, bad_potential_differences





    def get_state_dim(self):
      if self.state_representation == "potentials":
        return 2*self.num_actions 

      else:
        raise ValueError("state representation not recognized")


    def get_state(self):
      if self.state_representation ==  "potentials":
          good_potential_differences, bad_potential_differences = self.get_potential_differences(self.curr_node[0],self.curr_node[1])
          return 
      else:
        raise ValueError("state representation not recognized")
      


    def set_food_source_type(self, food_source_types):
      self.food_source_types = food_source_types

    ### Brings the environment back to pristine conditions 
    def reset_environment(self, info = dict([])):
      self.reset_initial_and_food_sources()
      self.sick = False



    ### RESTART ENV BRINGS THE CURRENT NODE BACK TO THE INITIAL ONE.
    def restart_env(self):
        self.end = False
        self.curr_node = self.initial_node
        self.sick = False

    def start_day(self):
        self.end = False
        self.initial_node = self.curr_node

        if self.curr_node in self.food_sources:
          food_source_index = self.food_sources.index(self.curr_node)
          if random.random()  <= self.recovery_probabilities[food_source_index]:
            self.sick = False
            self.remove_and_reset_one_food_source(self.curr_node)
          else:
            self.sick = True
        
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
      if self.curr_node not in self.food_sources and not self.sick:
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























