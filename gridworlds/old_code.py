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


