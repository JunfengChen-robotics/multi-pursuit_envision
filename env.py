import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import Point, Polygon
from evader import simple_thief_policy
import yaml, os
from police import PoliceAgent
from evader import ThiefAgent
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, RegularPolygon
from matplotlib.path import Path

CASE = 'case5'
COLOR = [
    "darkgreen",
    "darkblue",
    "indigo",
    "orange",
    "hotpink",
    "gold",
    "cyan",
    "red"
]

path = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(path, "coord.yaml")
with open(yaml_path, "r") as stream:
    try:
        coords = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
OBSTACLES = coords[CASE]
SIZE = 8.0 if CASE == 'case5' else 10.0
Boundary = [[[0,0], [0, SIZE], [-0.2, SIZE], [-0.2, 0]], 
                         [[0, SIZE], [SIZE, SIZE], [SIZE, SIZE+0.2], [0, SIZE+0.2]],
                         [[SIZE, 0], [SIZE+0.2, 0], [SIZE+0.2, SIZE], [SIZE, SIZE]],
                         [[0, -0.2], [SIZE, -0.2], [SIZE, 0], [0, 0]]
                         ]


class ChaseEnv(gym.Env):
    def __init__(self, num_police=3):
        super().__init__()
        self.num_police = num_police
        self.size = SIZE
        self.dt = 0.05
        self.capture_distance = 0.8
        self.robot_radius = 0.1
        # action: [dx, dy, turn_rate]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # observation: lidar (360 directions) + self pos + thief pos
        self.observation_space = spaces.Box(low=0, high=10, shape=(40,), dtype=np.float32)

        self.obstacles = [Path(ob) for ob in OBSTACLES]
        self.vis_obstacles = OBSTACLES
        

    def reset(self, seed=None, options=None):
        self.police_agents = [PoliceAgent(init_pos=np.random.uniform(1, 7, size=2), 
                                          id=i, 
                                          obstacles=self.obstacles,
                                          boundary=Boundary)
                              for i in range(self.num_police)]
        self.thief = ThiefAgent(init_pos=np.random.uniform(1, 7, size=2))
        self.steps = 0
        obs = {f"agent_{p.id}": p.get_obs(self.thief.pos) for p in self.police_agents}
        return obs, {}

    def step(self, action_dict):
        self.steps += 1
        for i in range(self.num_police):
            v = np.clip(action_dict[f"agent_{i}"][:2], -1, 1.0)
            self.police_agents[i].update(v, self.dt)

        self.thief.update([agent.pos for agent in self.police_agents])

        obs = {f"agent_{i}": self.police_agents[i].get_obs(self.thief.pos) for i in range(self.num_police)}
        reward = {f"agent_{i}": self.compute_reward(i) for i in range(self.num_police)}
        done = any(np.linalg.norm(p.pos - self.thief.pos) < self.capture_distance for p in self.police_agents)
        done_dict = {f"agent_{i}": done for i in range(self.num_police)}
        done_dict["__all__"] = done

        return obs, reward, done_dict, {}, {}
    
    def compute_reward(self, idx):
        agent = self.police_agents[idx]
        reward = 0.0

        # 1. 与小偷距离
        dist_to_thief = np.linalg.norm(agent.pos - self.thief.pos)
        reward += -0.1 * dist_to_thief

        # 2. 是否抓住小偷
        if dist_to_thief < self.capture_distance:
            reward += 100.0

        # 3. 是否撞墙（激光雷达值小于机器人半径）
        lidar_scan = agent._lidar(self.thief.pos)
        if np.any(lidar_scan <= self.robot_radius):
            reward -= 2.0

        # 4. 是否撞到其他警察
        for j, other_agent in enumerate(self.police_agents):
            if j != idx:
                dist = np.linalg.norm(agent.pos - other_agent.pos)
                if dist < self.robot_radius:
                    reward -= 5.0
        
        # 5. 是否撞到障碍物
        from shapely.geometry import Polygon, Point
        agent_point = Point(agent.pos)
        collided_with_boundary = False
        for poly_pts in Boundary:  # self.boundaries 应设置为 Boundary
            boundary_poly = Polygon(poly_pts)
            if boundary_poly.contains(agent_point) or boundary_poly.exterior.distance(agent_point) <= self.robot_radius:
                collided_with_boundary = True
                break
        if collided_with_boundary:
            reward -= 2.0

        return reward
    
    
    def render(self,):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()
            self.ax.set_xlim(0, self.size)
            self.ax.set_ylim(0, self.size)
            self.ax.set_aspect('equal')

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_title("Police vs Thief")

        # 画障碍物
        for poly in self.vis_obstacles:
            p = Polygon(poly, closed=True, facecolor='black', edgecolor='black')
            self.ax.add_patch(p)

        # 画警察（绿色圆形）
        for id, agent in enumerate(self.police_agents):
            c = Circle(agent.pos, radius=0.1, color=COLOR[id])
            self.ax.add_patch(c)
            self.ax.text(agent.pos[0]+0.1, agent.pos[1]+0.1, str(agent.id), fontsize=12, ha='center', va='center', color='red')
            c_a = Circle(agent.pos, radius=self.capture_distance, color=COLOR[id], alpha=0.2)
            self.ax.add_patch(c_a)
            # 画雷达线束
            readings = agent._lidar(agent.pos)
            angles = np.linspace(0, 2 * np.pi, agent.liard_readings, endpoint=False)
            for r, angle in zip(readings, angles):
                end_point = agent.pos + r * np.array([np.cos(angle), np.sin(angle)])
                self.ax.plot([agent.pos[0], end_point[0]], [agent.pos[1], end_point[1]], color='black', linewidth=1.0, linestyle='--', zorder=1)
                self.ax.plot(end_point[0], end_point[1], 'yo', markersize=6, zorder=5)

        # 画小偷（红色三角形）
        t = self.thief.pos
        tri = RegularPolygon((t[0], t[1]), numVertices=3, radius=0.1, color=COLOR[-1])
        self.ax.add_patch(tri)

        # 可选：网格背景
        self.ax.set_xticks(np.arange(0, self.size + 1, 1))
        self.ax.set_yticks(np.arange(0, self.size + 1, 1))

        plt.pause(0.01)
        
        
