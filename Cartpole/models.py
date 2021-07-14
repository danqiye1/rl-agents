"""
A set of custom models for predicting a Q(s,a) value from a state s
"""
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

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



       

        