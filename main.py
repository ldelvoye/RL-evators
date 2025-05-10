# %% 
# Imports
import gymnasium as gym
import random

# %%
env = gym.make("CartPole-v1", render_mode="human")
states = env.observation_space.shape[0]
actions = env.action_space.n

# %%
episodes = 10
for episode in range(1, episodes + 1):
    state, _ = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0, 1])
        n_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward

    print(f"Episode:{episode} Score:{score}")

# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model

model = build_model(states, actions)
model.summary()

# %%
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
# %%
