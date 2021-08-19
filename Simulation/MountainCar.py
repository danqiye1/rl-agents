"""
Experiments for MountainCar
"""
import gym
import telegram_send
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from MountainCar.agents import QLearningAgent, TDLambdaAgent
from AgentUtils.plots import plot_running_avg

def main():
    env = gym.make("MountainCar-v0")
    #q_learning_simulation(env)
    td_lambda_simulation(env)

def q_learning_simulation(env, N=1000):
    """ Experiment with Q Learning """
    env.reset()
    agent = QLearningAgent(env, use_sklearn=True)
    total_rewards = np.empty(N)

    for n in tqdm(range(N)):
        # There are 3 different kinds of epsilon to try
        #eps = 1.0/np.sqrt(n+1)
        #eps = 0.3
        eps = 0.1*(0.97**n)
        total_reward = agent.play(epsilon=eps, env=env, gamma=0.99)
        total_rewards[n] = total_reward
        if (n + 1) % 100 == 0:
            telegram_send.send(messages=["episode: {}, total reward: {}".format(n, total_reward)])

    avgReward = total_rewards[-100:].mean()
    totalsteps = -total_rewards.sum()
    telegram_send.send(messages=[
        "Agent training complete! Please check your plots.",
        "Average reward for last 100 episodes: {}.".format(avgReward),
        "Total steps: {}".format(totalsteps)
    ])

    # Sanity check on training
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()
    plot_running_avg(total_rewards)

    # Test if the agent is properly trained
    agent.play(epsilon=0, env=env, gamma=0.99, render=True)

def td_lambda_simulation(env, N=1000):
    """ Experiment with Q Learning """
    env.reset()
    agent = TDLambdaAgent(env)
    total_rewards = np.empty(N)

    for n in tqdm(range(N)):
        # There are 3 different kinds of epsilon to try
        #eps = 1.0/np.sqrt(n+1)
        #eps = 0.3
        eps = 0.1*(0.97**n)
        total_reward = agent.play(epsilon=eps, env=env, gamma=0.99)
        total_rewards[n] = total_reward
        if (n + 1) % 100 == 0:
            telegram_send.send(messages=["episode: {}, total reward: {}".format(n, total_reward)])

    avgReward = total_rewards[-100:].mean()
    totalsteps = -total_rewards.sum()
    telegram_send.send(messages=[
        "Agent training complete! Please check your plots.",
        "Average reward for last 100 episodes: {}.".format(avgReward),
        "Total steps: {}".format(totalsteps)
    ])

    # Sanity check on training
    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()
    plot_running_avg(total_rewards)

    # Test if the agent is properly trained
    agent.play(epsilon=0, env=env, gamma=0.99, render=True)

if __name__ == "__main__":
    main()