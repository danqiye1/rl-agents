# Reinforcement Learning Agents

## Introduction
This package is a set of RL agents that are specifically built to solve classic RL problems. Current problems include:
1. CartPole
2. MountainCar

## Installation
To use the agent packages in your python virtual environment, simply activate your virtual environment install it using pip:

```bash
pip install .
```

or pipenv:

```bash
pipenv install .
```

## Usage
After installation, RL agents can be imported and experimented on:

```python
import gym
from CartPole.agents import QLearningAgent

env = gym.make("CartPole-v0")
agent = QLearningAgent(env)
```

For more examples of how to train and test the agent, see the `Simulation` folder
