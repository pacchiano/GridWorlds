import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import numpy.random as npr
from networkx import grid_graph, single_target_shortest_path_length, draw, spring_layout, draw_planar, grid_2d_graph, draw_shell, draw_spring, draw_spectral
import random
import networkx as nx

import IPython
import torch

from .environments import *
from .policies import *


length = 10
height = 10

env = GridEnvironment(length, height)
#policy = RandomGridPolicy()
policy = RandomPolicy()



directed_graph= env.graph.to_directed()
pos = nx.spectral_layout(env.graph)
draw(directed_graph, pos = pos,  node_size =10, arrows= False)
# initial_node = random.choice(list(graph.nodes))

# destination_node = random.choice(list(graph.nodes))

nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=40, node_color = "red")
nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=40, node_color = "black")

shortest_paths = dict(single_target_shortest_path_length(env.graph,env.destination_node))
#node_path1, edge_path1 = run_walk_vintage(env.initial_node, env.destination_node, env.graph, shortest_paths)


env.restart_env()
node_path1, edge_path1, states1, action_indices1, rewards1 = run_walk(env, policy)
_=nx.draw_networkx_edges(directed_graph, pos = pos, edgelist = edge_list_to_tuples(edge_path1), width = 4, 
                       edge_color = "orange",  arrows = True, arrowsize=5)


env.restart_env()
node_path2, edge_path2, states2, action_indices2, rewards2 = run_walk(env, policy)
_=nx.draw_networkx_edges(directed_graph, pos = pos, edgelist = edge_list_to_tuples(edge_path2), width = 4, 
                       edge_color = "purple",  arrows = True, arrowsize=5)
plt.savefig("./figs/test_env.png")

## 

plt.clf()


env.sparsify_graph(.05)
directed_graph= env.graph.to_directed()
pos = nx.spectral_layout(env.graph)
draw(directed_graph, pos = pos,  node_size =10, arrows= False)
nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=40, node_color = "red")
nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=40, node_color = "black")
plt.savefig("./figs/test_env_sparsify.png")



plt.clf()



env.restore_initial_graph()
directed_graph= env.graph.to_directed()
pos = nx.spectral_layout(env.graph)
draw(directed_graph, pos = pos,  node_size =10, arrows= False)
nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=40, node_color = "red")
nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=40, node_color = "black")
plt.savefig("./figs/test_restore_graph.png")




plt.clf()

env.remove_random_edge()

directed_graph= env.graph.to_directed()
pos = nx.spectral_layout(env.graph)
draw(directed_graph, pos = pos,  node_size =10, arrows= False)
nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.initial_node], node_size=40, node_color = "red")
nx.draw_networkx_nodes(directed_graph, pos = pos,nodelist=[env.destination_node], node_size=40, node_color = "black")
plt.savefig("./figs/remove_random_edge_graph.png")


