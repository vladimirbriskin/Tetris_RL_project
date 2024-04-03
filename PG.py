import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PolicyGradient(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyGradient, self).__init__()
        self.affine1 = nn.Linear(input_dim, 128)
        self.affine2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=-1)

class PG_Agent:
    def __init__(self, input_dim, output_dim, device):
        self.model = PolicyGradient(input_dim, output_dim).to(device)
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.saved_log_probs = []
        self.rewards = []
        self.device = device

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        state = state.view(-1)
        probs = self.model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]