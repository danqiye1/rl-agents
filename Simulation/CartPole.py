"""
Experiments for CartPole
"""
import gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from Cartpole.agents import QLearningAgent, PolicyGradientAgent
from AgentUtils.plots import plot_running_avg

def main():
    env = gym.make("CartPole-v0")
    policy_gradient_learning(env)

def policy_gradient_learning(env, N=1000):
    env.reset()
    agent = PolicyGradientAgent(env)
    total_rewards = np.empty(N)
    for n in tqdm(range(N)):
        total_reward = agent.play(env)
        total_rewards[n] = total_reward

    # Sanity check on training
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(total_rewards)

     # Test if the agent is properly trained
    agent.play(env, render=True)

if __name__=="__main__":
    main()
    