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
import string
import imageio

import IPython
#%matplotlib inline
from matplotlib import colors as colors_matplotlib


from .environments import run_walk, run_multifood_walk


def node_list_to_tuples(node_list):
 return [tuple(node.numpy()) for x in node_list]

def edge_list_to_tuples(edge_list):
  return [ [tuple(node.numpy()) for node in list(edge)] for edge in edge_list  ]



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


  if env.name in ["SimpleMultifood", "MultifoodPitEnvironment", "ColorSimpleMultifood"]:
    ax.plot([food_source[1] for food_source in env.food_sources ], [food_source[0] for food_source in env.food_sources], 'x',markeredgewidth = 5, color = 'black', markersize = 12)
  
  if env.name in ["PitEnvironment", "MultifoodPitEnvironment", "ColorSimpleMultifood"]: 
      if len(env.pit_nodes) > 0:
        ax.plot([pit[1] for pit in env.pit_nodes], [pit[0] for pit in env.pit_nodes], 'o', color = 'red', markersize = 12)
  if env.name in ["GridSimple", "PitEnvironment", "ColorSimple"]:
    ax.plot([env.destination_node[1]], [env.destination_node[0]], 'x', markeredgewidth = 5, color = 'black', markersize = 12)

  ax.plot([env.initial_node[1]], [env.initial_node[0]], 'o', color = "blue", markersize = 12)

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
    if env.name in ["GridSimple", "PitEnvironment", "ColorSimple"]:
      node_path1, edge_path1,states1, action_indices1, rewards1  = run_walk(env, policy, num_steps)
  
    elif env.name in ["SimpleMultifood", "MultifoodPitEnvironment", "ColorSimpleMultifood"]:
      env.start_day()
      #print("info start ", env.end)
      node_path1, edge_path1,states1, action_indices1, rewards1, is_food_source_list1  = run_multifood_walk(env, policy, num_steps)

      #print("info ", env.end)
    #print("action indices ", action_indices1)
    #print("node path 1", node_path1)    
    #print("states1 ", states1)

    # print("Node path ", node_path1)
    # print("Destination node ", env.destination_node)

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



## filename_stub needs to be of the form path/filename without .png
def save_gif_diagnostic_image(env, policy, num_steps, num_paths, title, filename_stub, num_frames):
  diagnostic_images_filenames = []
  for i in range(num_frames):

      diagnostic_image_file = "{}_{}.png".format(filename_stub, i+1)
      diagnostic_images_filenames.append(diagnostic_image_file)
      save_grid_diagnostic_image(env, policy, num_steps, num_paths, title, diagnostic_image_file)
      if env.name in ["GridSimple", "PitEnvironment", "ColorSimple"]:
        env.reset_environment()
      elif env.name in ["SimpleMultifood", "MultifoodPitEnvironment", "ColorSimpleMultifood"]:
        env.start_day()

  images = []
  for filename in diagnostic_images_filenames:
    images.append(imageio.imread(filename))
  imageio.mimsave('{}.gif'.format(filename_stub), images)
  #IPython.embed()






def save_color_grid_diagnostic_image(env, color_map, title, filename, display_optimal_action = True, add_grid_colors = True):



  if env.name not in ["ColorSimple", "ColorSimpleMultifood"]:
    raise ValueError("The environment does not support ColorEnv rendering.")

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
    ax.plot([food_source[1] for food_source in env.food_sources ], [food_source[0] for food_source in env.food_sources], 'x',markeredgewidth = 5, color = 'black', markersize = 12)
    if len(env.pit_nodes) > 0:
      ax.plot([pit[1] for pit in env.pit_nodes], [pit[0] for pit in env.pit_nodes], 'o', color = 'red', markersize = 12)
  else:
    ax.plot([env.destination_node[1]], [env.destination_node[0]], 'x',markeredgewidth = 5, color = 'black', markersize = 12)

  # plt.savefig("./slkdfmdslfkm.png")
  # IPython.embed()

  # for i in range(len(node_lists_map.keys())):


  #   print("Nodes ", node_lists_map[list(node_lists_map.keys())[i]], " color ", colors[i])
  #   nx.draw_networkx_nodes(directed_graph, pos = pos, nodelist = node_lists_map[list(node_lists_map.keys())[i]], node_size = 20, node_color = colors[i])



  # nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=70, node_color = "blue")
  # if env.destination_node == None:
  #   nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=env.food_sources, node_size=70, node_color = "black")
  #   if len(env.pit_nodes) > 0:
  #       nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=env.pit_nodes, node_size=70, node_color = "red")

  # else:
  #   nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=70, node_color = "black")

  plt.title(title)
  plt.savefig(filename)
  plt.close("all")









