import numpy as np
import random
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta annealing

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        data = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array([done], dtype=np.float32)
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        if np.sum(probs) == 0 or np.any(np.isnan(probs)):
            probs = np.ones_like(probs)
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self._beta_by_frame(self.frame)
        self.frame += 1

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.FloatTensor(np.stack(states)),
            torch.FloatTensor(np.stack(actions)),
            torch.FloatTensor(np.stack(rewards)),
            torch.FloatTensor(np.stack(next_states)),
            torch.FloatTensor(np.stack(dones)),
            torch.FloatTensor(weights),
            torch.FloatTensor(indices)
        )
        
    def update_priorities(self, indices, td_errors):
        # 转换为 numpy 数组（若为 Tensor）
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        for i, td_error in zip(indices, td_errors):
            self.priorities[int(i)] = abs(td_error) + 1e-5  # 强制索引为 Python int

    def __len__(self):
        return len(self.buffer)

    def _beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
