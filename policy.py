import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym

# 超参数设置
ALPHA = 0.2  # 探索和利用之间的权衡系数
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4
LOG_STD_MIN = -20
LOG_STD_MAX = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.q1 = MLP(state_dim + action_dim, 1)
        self.q2 = MLP(state_dim + action_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.randn_like(mean)
        z = mean + std * normal
        action = torch.tanh(z)

        # Compute log_prob (注意tanh的Jacobian修正项)
        log_prob = (
            -0.5 * ((normal ** 2) + 2 * log_std + np.log(2 * np.pi))
        ).sum(dim=-1, keepdim=True)

        # 修正log_prob（因为action经过了tanh）
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob
    
class SACAgent:
    def __init__(self, state_dim, action_dim, alpha=0.2, gamma=0.99, tau=0.005,
                 buffer_size=1000000, batch_size=256, automatic_entropy_tuning=False):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_q_net = QNetwork(state_dim, action_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # 自动调节 alpha（可选）
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim  # 推荐设为 -|A|
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

    def get_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy_net(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy_net.sample(state_tensor)
        return action.squeeze(0).cpu().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)  # [B, 1]
        dones = dones.to(device)      # [B, 1]
        next_states = next_states.to(device)

        with torch.no_grad():
            next_actions, next_log_prob = self.policy_net.sample(next_states)
            next_q1, next_q2 = self.target_q_net(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            q_target = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_prob)

        q1_pred, q2_pred = self.q_net(states, actions)
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_actions, log_prob = self.policy_net.sample(states)
        q1_new, q2_new = self.q_net(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # soft update
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return q_loss.item(), policy_loss.item()

    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'alpha': self.alpha,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.alpha = checkpoint.get('alpha', self.alpha)


from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array([done], dtype=np.float32)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buffer)
