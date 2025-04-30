import numpy as np
from shapely.geometry import Point, Polygon
from matplotlib.path import Path

class PoliceAgent:
    def __init__(self, init_pos, id, obstacles, boundary):
        self.state = np.array(init_pos, dtype=np.float32)
        self.id = id
        self.robot_radius = 0.1
        self.obstacles = obstacles
        self.lidar_range = 3
        self.max_velocity = 1.2
        self.capture_range = 0.8
        self.robot_distance = 0.1
        self.liard_readings = 36
        self.boundary = [Path(ob) for ob in boundary]
        self.init_state = self.state

    def update(self, action, dt):
       self.velocity = np.clip(action[:2], -self.max_velocity, self.max_velocity)
       self.state += self.velocity * dt
       self.state = np.clip(self.state, 0, 8)

    def get_obs(self, thief_pos, obstacle_image):
        pos =self.state
        rel_thief = thief_pos - pos
        lidar_scan = self._lidar(pos)
        
        return np.concatenate([rel_thief, lidar_scan, obstacle_image.flatten()])
    
    def _lidar(self, pos):
        readings = []
        for angle in np.linspace(0, 2 * np.pi, self.liard_readings, endpoint=False):
            for r in np.linspace(self.robot_radius, self.lidar_range, 100):
                point = pos + r * np.array([np.cos(angle), np.sin(angle)])
                if any(obs.contains_point(point) for obs in self.obstacles) or any(boundary.contains_point(point) for boundary in self.boundary):
                    readings.append(r)
                    break
            else:
                readings.append(self.lidar_range)
        return np.array(readings, dtype=np.float32)
    
    
