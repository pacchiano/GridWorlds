import matplotlib
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

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
from matplotlib import cm
import pdb

from .environments import run_walk


def node_list_to_tuples(node_list):
 return [tuple(node.cpu().numpy()) for x in node_list]

def edge_list_to_tuples(edge_list):
  return [ [tuple(node.cpu().numpy()) for node in list(edge)] for edge in edge_list  ]


def joint_dual_potential_plot(env_tgt, env_src, π_tgt, π_src, φ, ψ, num_actions=4,
                              horizon_eval=20, ntraj=10, save_path=None):
    fig = plt.figure(figsize=((num_actions+1)*5, 5))
    axes = ImageGrid(fig, 111, nrows_ncols=(2,num_actions+1),
                     axes_pad= (0.15,0.5), share_all=True,
                     cbar_location="right",
                     cbar_mode="edge",
                     cbar_size="7%",
                     cbar_pad=0.15,
                     )
    plot_grid_sampled_paths(env_tgt, π_tgt, horizon_eval, ntraj, "Tπ trajs.", ax=axes[0],show=False)
    plot_grid_sampled_paths(env_src, π_src, horizon_eval, ntraj, "π_src opt trajs.", ax=axes[5],show=False)
    plot_reward_heatmap(env_tgt, φ._f, list(range(num_actions)), 'φ(s,', axes = axes[1:5], show=False, device = φ.get_device())
    plot_reward_heatmap(env_src, ψ._f, list(range(num_actions)), 'ψ(s,', axes = axes[6:],show=False,  device = ψ.get_device())
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_reward_heatmap(env, reward_function, action_index, title='Reward',
                        axes = None, save_path=None, show=True, device='cpu'):
    states = []

    #actions_onehot = torch.zeros((env.get_num_actions()+1, env.length*env.height))
    #actions_onehot[action_index,:] = 1
    #actions[action_index, :] = 1
    for i in range(env.length):
        for j in range(env.height):
            states.append( env.get_state_helper((i,j)))

    states = torch.vstack(states).to(device)

    #action_onehot = torch.zeros(env.get_num_actions() + 1)
    #action_onehot = F.one_hot(actions, self.num_actions + 1)
    if type(action_index) is int:
        action_index = [action_index]

    if action_index != 'all':
        rewards = reward_function(states)[:,action_index].cpu().detach()
    else:
        rewards = reward_function(states).cpu().detach()

    #rewards = np.random.randn(env.length, env.height) rewards = rewards.numpy()
    #_, bins = np.histogram(rewards, num_bins)
    #bin_indices = np.digitize(rewards, bins, right = False)
    #reward_bin_map = bin_indices.reshape((env.length, env.height))
    cmap = cm.get_cmap('viridis', 256)
    if axes is None:
        # Set up figure and image grid
        fig = plt.figure(figsize=(len(action_index)*4, 4))
        axes = ImageGrid(fig, 111,          # as in plt.subplot(111)
                         nrows_ncols=(1,len(action_index)),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
        #fig, axes = plt.subplots(1,len(action_index), constrained_layout=True)
    else:
        fig = plt.gcf()
    for idx,ax in enumerate(axes):
        #ax = axes[idx]
        im = ax.imshow(rewards[:,idx].reshape(env.length,env.height), cmap = cmap)#, norm = norm)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.tick_params(axis = 'x', labelbottom = False)
        ax.tick_params(axis = 'y', labelleft = False)
        ax.set_xticks(np.arange(-.5, env.height, 1));
        ax.set_yticks(np.arange(-.5, env.length, 1))
        ax.plot([env.destination_node[1]], [env.destination_node[0]], 'x', markeredgewidth = 5, color = 'black', markersize = 12)
        ax.plot([env.initial_node[1]], [env.initial_node[0]], 'o', color = "blue", markersize = 12)
        ax.set_title(f"{title} action {idx}={env.action_names[idx]})")

    ax.cax.colorbar(im)
    #fig.colorbar(im, ax=axes.ravel().tolist())
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    #plt.show()
    if show:
        plt.show(block=False)
        plt.pause(3)
        plt.close()



def plot_grid_sampled_paths(env, policy, num_steps, num_paths, title,
                            save_path=None, ax=None, show=True):
    white_map = np.zeros((env.length, env.height))

    cmap = colors_matplotlib.ListedColormap(["white"])

    bounds = np.arange(cmap.N + 1)
    norm = colors_matplotlib.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(white_map, cmap = cmap, norm = norm)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    ax.tick_params(axis = 'x', labelbottom = False)
    ax.tick_params(axis = 'y', labelleft = False)
    ax.set_xticks(np.arange(-.5, white_map.shape[1], 1));
    ax.set_yticks(np.arange(-.5, white_map.shape[0], 1));

    ax.plot([env.initial_node[1]], [env.initial_node[0]], 'o', color = "blue", markersize = 12)
    if env.destination_node == None:
        ax.plot([food_source[1] for food_source in env.food_sources ], [food_source[0] for food_source in env.food_sources], 'o', color = 'black', markersize = 12)
        if len(env.pit_nodes) > 0:
            ax.plot([pit[1] for pit in env.pit_nodes], [pit[0] for pit in env.pit_nodes], 'o', color = 'red', markersize = 12)
    else:
        ax.plot([env.destination_node[1]], [env.destination_node[0]], 'o', color = 'black', markersize = 12)

    if num_paths == 1: colors_small = ["orange"]
    elif num_paths == 2: colors_small = ["orange", "purple"]

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
        nodes, edges, states, action_indices, rewards  = run_walk(env, policy, num_steps)

        if num_paths > 2:
            color = color_map[colors[i]]
        else:
            color = colors_small[i]
        for (node1, node2) in edge_list_to_tuples(edges):
            ax.arrow(node1[1], node1[0], node2[1]-node1[1], node2[0]-node1[0],
                     length_includes_head=True, head_width=0.2, head_length=0.3,
                     color = color)

    ax.set_title(title)
    if save_path: plt.savefig(save_path)
    if show: plt.close("all")
