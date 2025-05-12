#%%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# %%
# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode=None)  # use "human" if you want to see it

# %%
episode_rewards = []

# Run 100 episodes
for episode in range(100):
    state, _ = env.reset()
    total_reward = 0

    for t in range(500):  # Max steps per episode
        action = env.action_space.sample()  # Random action
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

        state = next_state

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: reward = {total_reward}")

env.close()

# %%
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Random Agent on CartPole")
plt.grid(True)
plt.show()