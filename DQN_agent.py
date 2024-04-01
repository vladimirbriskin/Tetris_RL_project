import numpy as np
from DQN import DQN
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Check for CUDA availability and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tetris Agent with memory replay and epsilon-greedy policy
class DQN_Agent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state, next_state = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
            action, reward, done = torch.LongTensor([action]).to(device), torch.FloatTensor([reward]).to(device), torch.FloatTensor([done]).to(device)

            target = reward + (1 - done) * self.gamma * torch.max(self.model(next_state).detach(), 1)[0].unsqueeze(1)
            expected = self.model(state).gather(1, action.unsqueeze(1))

            loss = nn.functional.mse_loss(expected, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss