import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class MultiModalNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, image_channels=1, image_height=64, image_width=64, hidden_dim=256):
        super(MultiModalNetwork, self).__init__()
        self.image_height = image_height
        self.image_width = image_width

        # MLP for relative position + lidar data (assuming 38 features for state)
        self.mlp1 = nn.Sequential(
            nn.Linear(state_dim - (image_height * image_width), hidden_dim),  # State + Lidar data
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # CNN for image data
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # Calculate output dimension of CNN layer based on input image size
        self.fc1 = nn.Linear(64 * (image_height // 4) * (image_width // 4), hidden_dim)

        # Merging both MLP and CNN features
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim)

        # Final layers for Q-network and Policy-network
        self.q1 = nn.Linear(hidden_dim + action_dim, 1)
        self.q2 = nn.Linear(hidden_dim + action_dim, 1)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state) -> torch.Tensor:
        # Separate state into lidar (position + radar) and image parts
        vec_state = state[:, :38]  # Assuming first 38 values are the lidar/state data
        image = state[:, 38:]  # The rest is image data

        # Process lidar/radar data using MLP
        state_features = self.mlp1(vec_state)

        # Process image using CNN (ensure image shape is correct)
        image = image.view(-1, 1, self.image_height, self.image_width)  # Reshape image for CNN
        image_features = F.relu(self.conv1(image))
        image_features = F.relu(self.conv2(image_features))
        image_features = image_features.view(image_features.size(0), -1)  # Flatten image features
        image_features = F.relu(self.fc1(image_features))

        # Merge features from MLP and CNN
        merged_features = torch.cat([state_features, image_features], dim=-1)
        merged_features = F.relu(self.fc2(merged_features))

        return merged_features

    def q_network(self, state, action):
        features = self.forward(state)
        sa = torch.cat([features, action], dim=-1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2

    def policy_network(self, state):
        features = self.forward(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.policy_network(state)
        std = log_std.exp()

        normal = torch.randn_like(mean)
        z = mean + std * normal
        action = torch.tanh(z)

        log_prob = (-0.5 * ((normal ** 2) + 2 * log_std + np.log(2 * np.pi))).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob
