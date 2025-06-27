import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym
import cec2017.functions as functions
import math
import matplotlib.pyplot as plt
def func1(x):
    result = functions.f1
    return result([x])[0]
class ActorCritic(nn.Module):  # Define the Actor-Critic model
    def __init__(self, state_dim, action_dim):  # Initialize with state and action dimensions
        super(ActorCritic, self).__init__()  # Call parent class constructor
        self.shared_layer = nn.Sequential(  # Shared network layers for feature extraction
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # 增加一层网络
            nn.ReLU()
        )
        self.actor = nn.Sequential(  # Define the actor (policy) network
            nn.Linear(128, action_dim),  # Fully connected layer to output action probabilities
            nn.Softmax(dim=-1)  # Softmax to ensure output is a probability distribution
        )
        self.critic = nn.Linear(128, 1)  # Define the critic (value) network to output state value

    def forward(self, state):  # Forward pass for the model
        shared = self.shared_layer(state)  # Pass state through shared layers
        action_probs = self.actor(shared)  # Get action probabilities from actor network
        state_value = self.critic(shared)  # Get state value from critic network
        return action_probs, state_value  # Return action probabilities and state value
# Memory to store experiences
class Memory:  # Class to store agent's experience
    def __init__(self):  # Initialize memory
        self.states = []  # List to store states
        self.actions = []  # List to store actions
        self.logprobs = []  # List to store log probabilities of actions
        self.rewards = []  # List to store rewards
        self.is_terminals = []  # List to store terminal state flags
    def clear(self):  # Clear memory after an update
        self.states = []  # Clear stored states
        self.actions = []  # Clear stored actions
        self.logprobs = []  # Clear stored log probabilities
        self.rewards = []  # Clear stored rewards
        self.is_terminals = []  # Clear terminal state flags
# PPO Agent
class PPO:  # Define the PPO agent
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=4,ent_coef_init=0.1, ent_decay=0.995):
        self.policy = ActorCritic(state_dim, action_dim).to(device)  # Initialize the Actor-Critic model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)  # Adam optimizer for parameter updates
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)  # Copy of the policy for stability
        self.policy_old.load_state_dict(self.policy.state_dict())  # Synchronize parameters
        self.MseLoss = nn.MSELoss()  # Mean Squared Error loss for critic updates
        self.gamma = gamma  # Discount factor for rewards
        self.eps_clip = eps_clip  # Clipping parameter for PPO
        self.K_epochs = K_epochs  # Number of epochs for optimization
        self.ent_coef = ent_coef_init  # 初始熵系数
        self.ent_decay = ent_decay  # 衰减率
    def select_action(self, state, memory):
        state = torch.FloatTensor(state).to(device)  # Convert state to PyTorch tensor
        action_probs, _ = self.policy_old(state)  # Get action probabilities from old policy
        dist = Categorical(action_probs)  # Create a categorical distribution
        action = dist.sample()  # Sample an action from the distribution

        memory.states.append(state)  # Store state in memory
        memory.actions.append(action)  # Store action in memory
        memory.logprobs.append(dist.log_prob(action))  # Store log probability of the action
        return action.item()  # Return action as a scalar value
    def select(self, state):
        state = torch.FloatTensor(state).to(device)  # Convert state to PyTorch tensor
        action_probs, _ = self.policy_old(state)  # Get action probabilities from old policy
        dist = Categorical(action_probs)  # Create a categorical distribution
        action = dist.sample()  # Sample an action from the distribution
        return action.item()  # Return action as a scalar value

    def update(self, memory):
        self.ent_coef=self.ent_coef*self.ent_decay
        # Convert memory to tensors
        old_states = torch.stack(memory.states).to(device).detach()  # Convert states to tensor
        old_actions = torch.stack(memory.actions).to(device).detach()  # Convert actions to tensor
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()  # Convert log probabilities to tensor
        # Monte Carlo rewards
        rewards = []  # Initialize rewards list
        discounted_reward = 0  # Initialize discounted reward
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)  # Compute discounted reward
            rewards.insert(0, discounted_reward)  # Insert at the beginning of the list
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # Convert rewards to tensor
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # Normalize rewards
        # Update for K epochs
        for _ in range(self.K_epochs):
            self.ent_coef *= self.ent_decay
            # Get action probabilities and state values
            action_probs, state_values = self.policy(old_states)  # Get action probabilities and state values
            dist = Categorical(action_probs)  # Create a categorical distribution
            new_logprobs = dist.log_prob(old_actions)  # Compute new log probabilities of actions
            entropy = dist.entropy()  # Compute entropy for exploration
            # Calculate ratios
            ratios = torch.exp(new_logprobs - old_logprobs.detach())  # Compute probability ratios
            # Advantages
            advantages = rewards - state_values.detach().squeeze()  # Compute advantages
            # Surrogate loss
            surr1 = ratios * advantages  # Surrogate loss 1
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # Clipped loss
            loss_actor = -torch.min(surr1, surr2).mean()  # Actor loss
            # Critic loss
            loss_critic = self.MseLoss(state_values.squeeze(), rewards)  # Critic loss
            # Total loss
            loss = loss_actor + 0.5 * loss_critic - 0.1 * entropy.mean()  # Combined loss
            # Update policy
            self.optimizer.zero_grad()  # Zero the gradient buffers
            loss.backward()  # Backpropagate loss
            self.optimizer.step()  # Perform a parameter update
        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())  # Copy new policy parameters to old policy
state_dim = 3
action_dim = 3
lr = 0.01  # Learning rate
gamma = 0.99  # Discount factor
eps_clip = 0.2  # Clipping parameter
K_epochs = 10  # Number of epochs for policy update
max_episodes = 10  # Maximum number of episodes
max_timesteps =800  # Maximum timesteps per episode
reuslt = []
n = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
loaded_ppo = PPO(state_dim, action_dim, lr, gamma, eps_clip, K_epochs)
loaded_ppo.policy.load_state_dict(torch.load('D:/ppo_model.pth'))
loaded_ppo.policy.to(device)
loaded_ppo.policy.eval()  # 设置为评估模式
matrix = np.random.uniform(-n, n, size=(30, 100))
count = 0
best = float('inf')
row = len(matrix)
col = len(matrix[0])
bestIndex = 0
f = np.zeros((row, 1))
func = func1
for i in range(0, row):
    f[i][0] = func(matrix[i])
    if f[i][0] < best:
        best = f[i][0]
        bestIndex = i
Div = 0
for j in range(0, col):
    column_sum = np.sum(matrix[:, j])
    mean = np.median(matrix[:, j])
    Div = Div + abs(column_sum - row * mean) / (row * col)
NDiv = Div / (2 * n)
r = np.random.uniform(0, 1)
a = 2 - 0 * ((2) / (2 * n))
A = 2 * a * r - a
C = 2 * r
state = [np.var(matrix[0]), A, f[0][0]]
result = []
for t in range(max_timesteps):  # Loop over timesteps
    index = 0
    count = count + 1
    while index < row:
        action = loaded_ppo.select(state)  # Select action using PPO
        if action == 0:
            D = abs(C * matrix[bestIndex] - matrix[index])
            matrix[index] = matrix[bestIndex] - A * D
        elif action == 1:
            D = abs(matrix[bestIndex] - matrix[index])
            b = 1
            l = np.random.uniform(-1, 1)
            matrix[index] = D * math.exp(b * l) * math.cos(2 * math.pi * l) + matrix[bestIndex]
        else:
            rand = np.random.randint(0, row)
            D = abs(C * matrix[rand] - matrix[index])
            matrix[index] = matrix[rand] - A * D
        for j in range(0, col):
            if matrix[index][j] > n:
                matrix[index][j] = n
            if matrix[index][j] < -n:
                matrix[index][j] = -n
        Div = 0
        for j in range(0, col):
            column_sum = np.sum(matrix[:, j])
            mean = np.median(matrix[:, j])
            Div = Div + abs(column_sum - row * mean) / (row * col)
        NDiv = Div / 200
        r = np.random.uniform(0, 1)
        a = 2 - count * ((2) / (max_timesteps))
        A = 2 * a * r - a
        C = 2 * r
        f[index][0] = func(matrix[index])
        state = [np.var(matrix[index]), A, f[index][0]]
        for i in range(0, row):
            if f[i][0] < best:
                best = f[i][0]
                bestIndex = i
        index = index + 1
        result.append(best)
        print(f"Episode {t + 1},最小值: {best}")
mean_result = np.mean(result)
std_result = np.std(result)
print(f"F1 函数的均值: {mean_result}")
print(f"F1 函数的标准差: {std_result}")


if __name__ == "__main__":
    main()