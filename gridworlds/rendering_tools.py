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


from .environments import run_walk


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
