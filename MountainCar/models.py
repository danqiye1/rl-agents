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
    def __init__(self, env, use_sklearn=False):
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
        sample_states = np.array([env.observation_space.sample() for _ in range(10000)])

        # Initialize the scaler
        self.scaler = StandardScaler()
        self.scaler.fit(sample_states)

        # Initialize featurizer and scaler
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=500))
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

class EligibilityRegressor:
    """
    Implementation of an RBF Regressor for TD lambda algorithm
    This is different from the normal RBF Regressor as it needs to incorporate eligibility trace.
    """
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1

    def partial_fit(self, X, Y, eligibility):
        self.w += self.lr*(Y - X.dot(self.w))*eligibility

    def predict(self, X):
        return X.dot(self.w)

class TDLambdaRegressionModel:
    """
    Implementation of an RBF Regressor for TD lambda algorithm
    This is different from the normal RBF Regressor as it needs to incorporate eligibility trace.
    """

    def __init__(self, env):
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
        sample_states = np.array([env.observation_space.sample() for _ in range(10000)])

        # Initialize the scaler
        self.scaler = StandardScaler()
        self.scaler.fit(sample_states)

        # Initialize featurizer and scaler
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=500))
        ])
        

        # Get accurate dimensions after featurizer transform
        sample_features = self.featurizer.fit_transform(self.scaler.transform(sample_states))
        self.dimensions = sample_features.shape[1]

        # Initialize eligibility traces
        self.eligibilities = np.zeros((env.action_space.n, self.dimensions))

        # Initialize the regression models that map state to Q(s,a)
        # Scikit Learn regressor's parameter needs to be initialized to right dimensions with a partial_fit
        self.models = []
        for _ in range(env.action_space.n):
            model = EligibilityRegressor(self.dimensions)  
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

    def update(self, s, a, G, lambda_, gamma):
        """
        Update the model parameters
        
        :param s: A single state observation. For CartPole it will be dim 1x4
        :param a: Action selected
        :param G: The actual value of Q(s,a)
        :param lambda_: Lambda value of TD(lambda)
        :param gamma: Discount factor
        """
        X = self._transform(s)
        # Update eligibilities. Note this might be more difficult if it is not a linear regressor
        # The gradient of w x + c is pre-calculated to be x.
        self.eligibilities = X[0] + self.eligibilities*lambda_*gamma
        self.models[a].partial_fit(X, [G], self.eligibilities[a])
        


       

        