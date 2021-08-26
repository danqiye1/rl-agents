"""
Implementation of a Replay Buffer for Deep Q Learning and other things.
The goal of the replay buffer is to ensure that samples for training are less correlated.
It is like the shuffle method in Supervised Deep Learning
"""
import random
from collections import namedtuple

class ReplayBuffer:

    def __init__(self):
        """ Initialize a replay buffer """
        self.buffer = []

    def push(self, s, a, s_prime, r):
        """ 
        Save a transition. All parameters are in the context of a RL agent.
        
        :param s: Current state of agent
        :param a: Current action of agent
        :param s_prime: Next state of agent
        :param r: Reward obtained by agent
        """
        # Using a namedtuple allows us to retrieve the parameters easily later
        transition = namedtuple("Transition", ("s", "a", "s_prime", "r"))
        self.buffer.append(transition(s, a, s_prime, r))

    def sample(self, batch_size):
        """
        Sample a batch for training.

        :param batch_size: Batch size of the training data to prepare. Need to be smaller than self.buffer or will raise ValueError.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """ Overwrite length method """
        return len(self.buffer)
