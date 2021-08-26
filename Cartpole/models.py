"""
A set of custom models for predicting a Q(s,a) value from a state s
"""
from sklearn import linear_model
import torch
from torch import nn
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

from pdb import set_trace as bp

class CustomSGDRegressor:
    """
    A custom implementation of a linear regressor.
    """
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def partial_fit(self, X, Y):
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

class RBFRegressionModel:
    """
    Classic single layer RBF Linear Regression Model
    """
    def __init__(self, env, use_sklearn=False, dim=20000):
        """
        Constructor for RBF Linear Regression Model

        :param env: OpenAI Gym environment
        :param use_sklearn: Use Scikit-Learn SGD regressor if True, use CustomSGDRegressor if False
        :param dim: Dimensionality of RBF Kernel
        :type dim: int
        """

        # Initialize environment
        self.env = env

        # Initialize observations/states for Cartpole environment
        # This is done by randomly sampling over a uniform distribution over [-1, 1]
        # The state is represented by [x, vx, y, vy]
        # The reason why env.observation_space.sample() is not used is because if wrongly gives very large numbers for vx, vy.
        sample_states = np.random.random((dim, 4)) * 2 - 1

        # Initialize the scaler
        self.scaler = StandardScaler()
        self.scaler.fit(sample_states)

        # Initialize featurizer and scaler
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000))
        ])
        

        # Get accurate dimensions after featurizer transform
        sample_features = self.featurizer.fit_transform(self.scaler.transform(sample_states))
        self.dimensions = sample_features.shape[1]

        # Initialize the regression models that map state to Q(s,a)
        # Scikit Learn regressor's parameter needs to be initialized to right dimensions with a partial_fit
        self.models = []
        for _ in range(env.action_space.n):
            if use_sklearn:
                model = SGDRegressor()
                model.partial_fit(self.featurizer.transform(self.scaler.transform([env.reset()])), [0])
            else:
                model = CustomSGDRegressor(self.dimensions)
                
            self.models.append(model)

    def _transform(self, obs):
        """
        Helper function for transforming state observations into RBF features

        :param obs: State observations. For CartPole it will be dim 1x4
        """
        return self.featurizer.transform(self.scaler.transform([obs]))

    def predict(self, s):
        """
        Predict Q(s,a) given state observation s.

        :param s: A single state observation. For CartPole it will be dim 1x4
        """
        X = self._transform(s)
        Y_hat = np.stack([m.predict(X) for m in self.models]).T
        return Y_hat

    def update(self, s, a, G):
        """
        Update the model parameters
        
        :param s: A single state observation. For CartPole it will be dim 1x4
        :param a: Action selected
        :param G: The actual value of Q(s,a)
        """
        X = self._transform(s)
        self.models[a].partial_fit(X, [G])

class PolicyModel(nn.Module):
    """ A Pytorch model approximation of pi(a|s) for Policy Gradient Method """

    def __init__(self, n_states, n_actions):
        """ Constructor 
        
        :param n_states: State/observation space of the environment
        :param n_actions: Action space of the agent
        """
        super(PolicyModel, self).__init__()

        # Neural Network
        self.network = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, s):
        """ 
        Forward propagation for policy model pi(a|s)
        
        :param s: Input state of dim [batch_size, n_states]
        :type s: numpy.ndarray

        :returns:
            - no_grad_logits - probabilities of each action without gradient (for action selection)
            - logits - probabilities of each action with gradient (for backpropagation)
            - log_probs - log probabilities of each action with gradient (for backpropagation)
        """
        input_tensor = torch.as_tensor(s, dtype=torch.float32)
        x = self.network(input_tensor)

        with torch.no_grad():
            no_grad_logits = nn.functional.softmax(x, -1)

        logits = nn.functional.softmax(x, -1)
        log_prob = nn.functional.log_softmax(x, -1)
        return no_grad_logits.numpy(), logits, log_prob

class DeepQNetwork(nn.Module):
    """
    Convolution Neural Network used for Deep Q Learning
    """

    def __init__(self, w, h, c, n_actions):
        """ Constructor

        :param w: Width of the image of agent state.
        :param h: Height of the image of agent state.
        :param c: Number of channels of the image of agent state.
        :param n_actions: Action space of agent, which is also the number of outputs of network.
        """
        super(DeepQNetwork, self).__init__()

        self.ConvNet = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Hacky way to get the dimensions
        conv_output = self.ConvNet(torch.rand(1, c, h, w))
        linear_dim = len(conv_output.view(-1))

        self.LinearNet = nn.Sequential(
            nn.Linear(linear_dim, linear_dim // 2),
            nn.ReLU(),
            nn.Linear(linear_dim // 2, linear_dim // 4),
            nn.ReLU(),
            nn.Linear(linear_dim // 4, n_actions)
        )

    def forward(self, x):
        """ Forward pass """
        x = self.ConvNet(x)
        output = self.LinearNet(x.view(-1))

        return output