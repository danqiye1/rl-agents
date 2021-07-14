"""
A set of agents for solving the OpenAI Cartpole Environment
"""
import numpy as np
from matplotlib import pyplot as plt
from Cartpole.models import RBFRegressionModel

class QLearningAgent:
    """ A Q Learning Agent """
    def __init__(self, env, use_sklearn=False):
        """ Constructor 
        
        :param env: OpenAI gym cartpole environment
        :param use_sklearn: Use Scikit-Learn SGD regressor if True, use CustomSGDRegressor if False
        """

        # Initialize agent's environment and Q(s,a) model
        self.env = env
        self.model = RBFRegressionModel(env, use_sklearn=use_sklearn)

    def play(self, epsilon=0.3, gamma=0.9, max_steps=2000, render=False):
        """ 
        Play one episode of CartPole
        
        :param epsilon: Parameter for epsilon greedy algorithm which determines the explore-exploit selection
        :param gamma: Discount rate for future rewards
        :param max_steps: Maximum iterations for an episode. The original 200 max steps of OpenAI Gym is too small
        :param render: Flag for rendering if needed
        """

        # Initialize metrics and state for episode
        s = self.env.reset()
        done = False
        total_rewards = 0
        step = 0

        while not done and step < max_steps:
            action = self.select_action(s, epsilon)
            s_prime, reward, done, _ = self.env.step(action)

            # Give a huge negative reward for finishing early
            if done:
                reward = -200

            # Get Q(s_prime,a_prime) for all actions in action space
            Q_values = self.model.predict(s_prime)
            G = reward + gamma * np.max(Q_values)

            # Train one epoch of the function approximator model
            self.model.update(s, action, G)

            if reward == 1:
                # Add rewards only of the non-terminal steps
                total_rewards += reward

            if render:
                # Render to show play if needed
                plt.imshow(self.env.render('rgb_array'))

            # Don't forget to update state
            # Or agent won't learn
            s = s_prime
            step += 1

        return total_rewards

    def select_action(self, s, epsilon):
        """
        Bandit algorithm to solve the exploration-exploitation dilemma. 
        Epsilon is the probability of choosing a random action.

        :param s: State parameter to choose action from pi(a|s)
        :param epsilon: Probability to choose action randomly for epsilon-greedy algorithm
        """
        if np.random.random() < epsilon:
            # Select random action for exploration
            return self.env.action_space.sample()
        else:
            # Select action from policy pi(a|s)
            return np.argmax(self.model.predict(s))

