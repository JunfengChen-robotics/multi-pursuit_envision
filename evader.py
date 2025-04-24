import numpy as np

def simple_thief_policy(thief_pos, police_positions):
    # 远离最近的警察
    nearest = min(police_positions, key=lambda p: np.linalg.norm(p - thief_pos))
    direction = thief_pos - nearest
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm
    return np.clip(thief_pos + direction * 0.15, 0, 8.0)


class ThiefAgent:
    def __init__(self, init_pos):
        self.pos = np.array(init_pos, dtype=np.float32)

    def update(self, police_positions):
        # 替换成复杂规则时只需要改这里
        direction = np.random.uniform(-1, 1, size=2)
        self.pos += 0.1 * direction / np.linalg.norm(direction)
        self.pos = np.clip(self.pos, 0, 8)
        
        
        

        



