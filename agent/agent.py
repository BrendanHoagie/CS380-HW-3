import json
import os
import random

from .state import State
import random


class Q_State(State):
	'''Augments the game state with Q-learning information'''

	def __init__(self, string):
		super().__init__(string)

		# key stores the state's key string (see notes in _compute_key())
		self.key = self._compute_key()

	def _compute_key(self):
		'''
		Returns a key used to index this state.

		The key should reduce the entire game state to something much smaller
		that can be used for learning. When implementing a Q table as a
		dictionary, this key is used for accessing the Q values for this
		state within the dictionary.
		'''

		# this simple key uses the 8 object characters surrounding the frog
		# and combines them into a key string
		#  X  X  X
        #  X  F  X
		#  X  X  X
		return ''.join([
			self.get(self.frog_x - 1, self.frog_y - 1) or '_',
			self.get(self.frog_x, self.frog_y - 1) or '_',
			self.get(self.frog_x + 1, self.frog_y - 1) or '_',
			self.get(self.frog_x - 1, self.frog_y) or '_',
			self.get(self.frog_x, self.frog_y) or '_',
			self.get(self.frog_x - 1, self.frog_y + 1) or '_',
			self.get(self.frog_x, self.frog_y + 1) or '_',
			self.get(self.frog_x + 1, self.frog_y + 1) or '_',
		])

	def reward(self):
		'''Returns a reward value for the state.'''

		if self.at_goal:
			return self.score
		elif self.is_done:
			return -10
		else:
			return 0


class Agent:

	def __init__(self, train=None):

		# train is either a string denoting the name of the saved
		# Q-table file, or None if running without training
		self.train = train

		# q is the dictionary representing the Q-table
		self.q = {}

		# name is the Q-table filename
		# (you likely don't need to use or change this)
		self.name = train or 'q'

		# path is the path to the Q-table file
		# (you likely don't need to use or change this)
		self.path = os.path.join(os.path.dirname(
			os.path.realpath(__file__)), 'train', self.name + '.json')

		self.load()

		self._alpha = 0.1
		self._gamma = 0.9
		self._epsilon = 0.1

		self._prev_state = None
		self._prev_action = None

	def load(self):
		'''Loads the Q-table from the JSON file'''
		try:
			with open(self.path, 'r') as f:
				self.q = json.load(f)
			if self.train:
				print('Training {}'.format(self.path))
			else:
				print('Loaded {}'.format(self.path))
		except IOError:
			if self.train:
				print('Training {}'.format(self.path))
			else:
				raise Exception('File does not exist: {}'.format(self.path))
		return self

	def save(self):
		'''Saves the Q-table to the JSON file'''
		with open(self.path, 'w') as f:
			json.dump(self.q, f)
		return self

	def choose_action(self, state_string: str) -> str:
		'''
		Returns the action to perform.

		Args:
			state_string: a string representing the cur_staterent game state

		Returns:
			choice: a string from State.ACTIONS representing the chosen best state
		'''
		cur_state = Q_State(state_string)
		cur_state_index = cur_state._compute_key()
		
		# Look to update the table
		if self._prev_state and self._prev_action:
			reward = cur_state.reward()
			prev_state_index = self._prev_state._compute_key()

			# create blank entry if we've not seen it before -> really only used at the start
			if not prev_state_index in self.q:
				self.q[prev_state_index] = {}
				for action in State.ACTIONS:
					self.q[prev_state_index][action] = 0

			# get value from table 
			prev_val = self.q[prev_state_index].get(self._prev_action)

			# get maximum expected value for cur_state's actions, create state if it doesn't exist already
			if not cur_state_index in self.q:
				self.q[cur_state_index] = {}
				for action in State.ACTIONS:
					self.q[cur_state_index][action] = 0
				max_actions_val = 0
			else:
				max_actions_val = max([self.q[cur_state_index].get(action) for action in State.ACTIONS])
			
			# formula
			q_value = (1 - self._alpha) * prev_val + self._alpha * (reward + self._gamma * max_actions_val)

			# update table
			self.q[self._prev_state._compute_key()][self._prev_action] = q_value
			self.save()

		# exploration vs exploitation
		if random.random() < self._epsilon:
			choice = random.choice(State.ACTIONS)
		else:
			# if there's nothing in the table for cur_state, create an entry and choose something
			# random. This is slightly better (in theory) than always choosing action 0
			if not cur_state_index in self.q:
				self.q[cur_state_index] = {}
				for action in State.ACTIONS:
					self.q[cur_state_index][action] = 0
				choice = random.choice(State.ACTIONS)
			
			else:
				# get the best action
				best_action_value = max([self.q[cur_state_index].get(action) for action in State.ACTIONS])
				choice = next((action for action, q_value in self.q[cur_state_index].items() if q_value == best_action_value), None)

		self._prev_state = cur_state
		self._prev_action = choice
		return choice
