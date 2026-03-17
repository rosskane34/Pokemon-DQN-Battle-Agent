import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# ─── Simple feed-forward neural network ───
class QNetwork(nn.Module):
    def __init__(self, state_size=4, action_size=4, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# ─── Replay buffer (stores experiences) ───
class ReplayBuffer:
    def __init__(self, capacity=10000):             #constructor of size 10k
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):        #add a new state to the replay buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)          #return our batch from memory buffer
        states, actions, rewards, next_states, dones = zip(*batch)  

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)   # 0.0 or 1.0
        )

    def __len__(self):
        return len(self.buffer)

# ─── The DQN Agent ───
class DQNAgent:
    def __init__(self, state_size=4, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.90           # discount factor 
        self.epsilon = 1.0          # exploration
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.995

        # Two networks
        self.policy_net = QNetwork(state_size, action_size)     #define our target and policy networks
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # copy weights
        self.target_net.eval()  # target net doesn't train

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.003) #optimize
        self.memory = ReplayBuffer()
        self.batch_size = 64    
        self.target_update_freq = 300   # how often to update target net
        self.steps_done = 0

    def choose_action(self, state, force_greedy=False):
        sample = random.random()
        if not force_greedy and sample < self.epsilon:      #choose random if epsilon is high enough
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)  # shape: [1, state_size]
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.max(1)[1].item()  # argmax

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.steps_done += 1

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        #Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Target Q-values (using target network)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]  # choose action using policy net
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)


        # Calculate loss with Huber Loss
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="dqn_pokemon.pth"):
        torch.save(self.policy_net.state_dict(), path)          #save the DQN

    def load(self, path="dqn_pokemon.pth"):
        try:                                                    #load if a file is found
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("DQN loaded successfully")                    #make a new one if not
        except:
            print("No saved DQN model found — starting fresh")
