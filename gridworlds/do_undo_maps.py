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

from abc import ABCMeta, abstractmethod

class DoUndoMap:

	@abstractmethod
	def do_state(self, state):
		pass

	@abstractmethod
	def do_action(self, action):
		pass


	@abstractmethod
	def undo_state(self, state):
		pass

	@abstractmethod
	def undo_action(self, action):
		pass


	## This will implement a per state, action application of the undo map of a trajectory
	@abstractmethod
	def undo_trajectory(self, trajectory):
		pass

	## This will implement a per state, action application of the do map of a trajectory
	@abstractmethod
	def do_trajectory(self, trajectory):
		pass

	def do(self, state, action):
		return (self.do_state(state), self.do_action(action))

	def undo(self, state, action):
		return (self.undo_state(state), self.undo_action(action))


class IdentityDoUndo(DoUndoMap):

	def do_state(self, state):
		return state

	def do_action(self, action):
		return action


	def undo_state(self, state):
		return state

	def undo_action(self, action):
		return action

	def undo_trajectory(self, trajectory):
		return trajectory

	def do_trajectory(self, trajectory):
		return trajectory



class LinearDoUndoDiscrete(IdentityDoUndo):
	def __init__(self, linear_do):

		if len(linear_do.shape) != 2:
			raise ValueError("The shape of the linear_do map is not a matrix. It equals {}.".format(linear_do.shape))

		if linear_do.shape[0] != linear_do.shape[1]:
			raise ValueError("""The shape of the linear map is a matrix but the
				first dimension does not equal the second.
				First dimension {}. Second dimension {}.""".format(linear_do.shape[0], linear_do.shape[1]))

		self.linear_do = linear_do
		self.linear_undo = torch.inverse(linear_do)

	def do_state(self, state):
		return torch.matmul(self.linear_do, state.flatten())


	def undo_state(self, state):
		return torch.matmul(self.linear_undo, state.flatten())

	def undo_trajectory(self, trajectory):
		raise ValueError("undo trajectory not implemented.")
	    # undone_trajectory = copy.deepcopy(trajectory)
	    # undone_trajectory['states'] = torch.matmul(undone_trajectory['states'], self.linear_undo)
	    # return undone_trajectory

	def do_trajectory(self, trajectory):
		raise ValueError("do trajectory not implemented.")
	    # done_trajectory = copy.deepcopy(trajectory)
	    # done_trajectory['states'] = torch.matmul(done_trajectory['states'], self.linear_do)
	    # # for key in trajectory.keys():
	    # #   pass
	    # return done_trajectory


class TorchModuleDoUndoDiscrete(IdentityDoUndo):
	def __init__(self, net):
		self.do   = net.eval()#.clone()
		for param in net.parameters():
			param.requires_grad = False
		self.undo = None

	def do_state(self, state):
		return self.do(state)

	def undo_state(self, state):
		raise ValueError("undo not implemented.")

	def undo_trajectory(self, trajectory):
		raise ValueError("undo trajectory not implemented.")

	def do_trajectory(self, trajectory):
		raise ValueError("do trajectory not implemented.")



class ReverseActionsDoUndoDiscrete(IdentityDoUndo):


	def __init__(self):
	    self.reversed_actions = [(1, 0), (-1,0), (0,1), (0, -1)]
	    self.reversed_action_names = ["D", "U", "R", "L"]
	    self.actions = [(-1, 0), (1,0), (0,-1), (0, 1)]
	    self.action_names = ["U", "D", "L", "R"]


	def do_action(self, action):
		action_index = self.actions.index(action)
		return self.reversed_actions[action_index]

	def undo_action(self, action):
		reversed_action_index = self.reversed_actions.index(action)
		return self.actions[reversed_action_index]

	def undo_trajectory(self, trajectory):
		raise ValueError("undo trajectory not implemented.")


	def do_trajectory(self, trajectory):
		raise ValueError("Do trajectory not implemneneted.")
