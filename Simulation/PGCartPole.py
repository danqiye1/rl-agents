import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from AgentUtils.plots import plot_running_avg

env = gym.make("CartPole-v0")
# gym compatibility: unwrap TimeLimit
# Without this, the environment caps the 
if hasattr(env, '_max_episode_steps'):
    env = env.env

class Model(nn.Module):

    def __init__(self, n_states, n_actions):
        super(Model, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        self.model = nn.Sequential(
            nn.Linear(n_states, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )

    def forward(self, states):
        state_tensor = torch.as_tensor(states, dtype=torch.float32)
        x = self.model(state_tensor)

        with torch.no_grad():
            no_grad_logits = nn.functional.softmax(x, -1)

        grad_logits = nn.functional.softmax(x, -1)
        log_probs = nn.functional.log_softmax(x, -1)

        # Provide 2 kind of outputs. One for select action (no grad) and one for backpropagation (with grad)
        return no_grad_logits.numpy(), grad_logits, log_probs

# Initialise the Policy Gradient Network
network = Model(env.observation_space.shape[0],env.action_space.n)

class Agent:

    def __init__(self, model, gamma=0.99, learning_rate=1e-3, entropy_coeff=1e-2):
        self.model = model
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.entropy_coeff = entropy_coeff

    def play_episode(self, env, render=False, max_steps=1000):
        states, actions, rewards = [], [], []
        s = env.reset()

        for t in range(max_steps):
            # Select action
            action_probs, _, _ = self.model(np.array(s))
            action = np.random.choice(len(action_probs), p=action_probs)

            # Step on action
            s_prime, r, done, info = env.step(action)

            if render:
                # Render to show play if needed
                plt.imshow(env.render('rgb_array'))

            # Record session
            states.append(s)
            actions.append(action)
            rewards.append(r)

            s = s_prime

            if done:
                break

        # Get cummulative rewards
        G = 0
        returns = []
        for r in rewards[::-1]:
            G = r + self.gamma*G
            returns.append(G)

        returns.reverse()

        return states, actions, rewards, returns

    def train(self, states, actions, rewards, returns):

        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        returns = torch.as_tensor(returns, dtype=torch.float32)

        _, action_probs, log_probs = self.model(states)

        # Get the log_probs for the selected actions of the session
        log_probs_for_actions = (log_probs * nn.functional.one_hot(actions)).sum(1)

        # Calculate "loss"
        entropy = (action_probs*log_probs).sum()
        loss = -(log_probs_for_actions*returns).mean() + self.entropy_coeff * entropy

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return np.sum(rewards)

# Instantiate the agent
agent = Agent(network)

# Training the agent
rewards_history = []
for i in tqdm(range(10000)):
    states, actions, rewards, returns = agent.play_episode(env)
    total_reward = agent.train(states, actions, rewards, returns)
    rewards_history.append(total_reward)
    if i%100 == 0:
        tqdm.write("Episode {} Total Reward: {}".format(i, total_reward))
        tqdm.write("Average Reward for past 100 episodes: {}".format(np.mean(rewards_history[-100:])))
        if np.mean(rewards_history[-100:]) > 300:
            tqdm.write("We won!")
        
plot_running_avg(np.array(rewards_history))

agent.play_episode(env, render=True)



