

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


