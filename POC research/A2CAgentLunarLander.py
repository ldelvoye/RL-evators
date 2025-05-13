#%% Imports & Setup
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#%% Visualization Settings
render_simulation = True  # Toggle for rendering

#%% Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

#%% A2C Agent
class A2CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, _ = self.model(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1. - done)
            returns.insert(0, R)
        return returns

    def update(self, trajectories):
        states, actions, log_probs, rewards, dones, values = zip(*trajectories)

        states = torch.FloatTensor(states).to(self.device)
        values = torch.stack(values).squeeze().to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)

        with torch.no_grad():
            _, last_value = self.model(torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device))
            returns = self.compute_returns(rewards, dones, last_value.item())
            returns = torch.FloatTensor(returns).to(self.device)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#%% Initialize Environment & Agent
env = gym.make("LunarLander-v3", render_mode="human" if render_simulation else None)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = A2CAgent(state_dim, action_dim)

#%% Training Loop
num_episodes = 1000
reward_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    trajectories = []

    for t in range(1000):
        if render_simulation:
            env.render()

        action, log_prob = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        _, value = agent.model(torch.FloatTensor(state).unsqueeze(0).to(agent.device))

        trajectories.append((state, action, log_prob, reward, done, value))
        state = next_state
        total_reward += reward

        if done:
            break

    agent.update(trajectories)
    reward_history.append(total_reward)
    print(f"Episode {episode + 1}, Reward: {total_reward:.2f}")

env.close()

#%% Plotting Results
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("A2C Agent on LunarLander-v2")
plt.grid(True)
plt.show()
