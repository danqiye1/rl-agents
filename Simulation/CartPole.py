"""
Experiments for CartPole
"""
from Cartpole.models import PolicyModel
import gym
import sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from Cartpole.agents import QLearningAgent, PolicyGradientAgent
from Cartpole.models import PolicyModel
from AgentUtils.plots import plot_running_avg

sys.path.insert(0, "../Cartpole")

def main():
    env = gym.make("CartPole-v0")
    policy_gradient_learning(env)

def policy_gradient_learning(env, N=10000):

    # gym compatibility: unwrap TimeLimit
    if hasattr(env, '_max_episode_steps'):
        env = env.env
    
    env.reset()

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    model = PolicyModel(n_states, n_actions)
    agent = PolicyGradientAgent(model)
    total_rewards = []
    for n in tqdm(range(N)):
        states, actions, rewards, returns = agent.play(env)
        total_reward = agent.train(states, actions, rewards, returns)
        total_rewards.append(total_reward)
        if n%100 == 0:
            tqdm.write(" Episode {} Reward: {}. Average Reward: {}".format(n, total_reward, np.mean(total_rewards[-100:])))

    # Sanity check on training
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(np.array(total_rewards))

     # Test if the agent is properly trained
    agent.play(env, render=True)

if __name__=="__main__":
    main()
    