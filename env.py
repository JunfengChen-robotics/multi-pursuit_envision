import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import Point, Polygon
import yaml, os
from police import PoliceAgent
from thief import ThiefAgent
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, RegularPolygon
from matplotlib.path import Path
from evader.core.evader import Evader
from config.position import InitPos

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
coord_yaml_path = os.path.join(path, "config","coord.yaml")


with open(coord_yaml_path, "r") as stream:
    try:
        coords = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
OBSTACLES = coords[CASE]

world_yaml_path = os.path.join(path, "config", "world.yaml")
with open(world_yaml_path, "r") as stream:
    try:
        world = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
SIZE = world['map']['size'][CASE]*2
Boundary = [[[0,0], [0, SIZE], [-0.2, SIZE], [-0.2, 0]], 
                         [[0, SIZE], [SIZE, SIZE], [SIZE, SIZE+0.2], [0, SIZE+0.2]],
                         [[SIZE, 0], [SIZE+0.2, 0], [SIZE+0.2, SIZE], [SIZE, SIZE]],
                         [[0, -0.2], [SIZE, -0.2], [SIZE, 0], [0, 0]]
                         ]


def generate_random_positions(robot_num, obstacles):
    def is_valid(pos, existing_positions, min_dist=0.5):
        """检查位置是否在地图内、在障碍物外，并且离已有位置不太近"""
        x, y = pos
        if not (0 < x < SIZE and 0 < y < SIZE):
            return False
        point = np.array([x, y])
        if any(poly.contains_point(point) for poly in obstacles):
            return False
        if any(np.linalg.norm(np.array(pos) - np.array(p)) < min_dist for p in existing_positions):
            return False
        return True
    
    import random
    positions = []
    max_trials = 1000
    for _ in range(robot_num):
        for _ in range(max_trials):
            x = random.uniform(0.5, SIZE - 0.5)
            y = random.uniform(0.5, SIZE - 0.5)
            if is_valid((x, y), positions):
                positions.append((x, y))
                break
        else:
            raise RuntimeError("Failed to generate valid initial positions.")
    return np.array(positions).round(1)
    


class ChaseEnv(gym.Env):
    
    def __init__(self, num_police=3, train_or_test='train'):
        super().__init__()
        self.num_police = num_police
        self.size = SIZE
        self.dt = 0.05
        # action: [dx, dy, turn_rate]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # observation: lidar (360 directions) + self pos + thief pos
        self.observation_space = spaces.Box(low=0, high=10, shape=(40,), dtype=np.float32)

        self.obstacles = [Path(ob) for ob in OBSTACLES]
        self.vis_obstacles = OBSTACLES
        self.train_or_test = train_or_test
        

    def reset(self, seed=None, options=None):
        if self.train_or_test == 'test':
            init_pos = InitPos.get_predefined_positions(case_number=CASE)
        if self.train_or_test == 'train':
            init_pos = generate_random_positions(self.num_police + 1, self.obstacles)
            
        self.police_agents = [PoliceAgent(init_pos[i], 
                                          id=i, 
                                          obstacles=self.obstacles,
                                          boundary=Boundary)
                              for i in range(self.num_police)]
        # initialization of evader
        # ===================================
        # self.thief = ThiefAgent(init_pos=np.random.uniform(1, 7, size=2))
        self.thief = Evader(pos=init_pos[-1])
        self.thief.get_env_info(CASE)
        self.thief.update_env(self.police_agents)
        # ====================================
        self.steps = 0
        obs = {f"agent_{p.id}": p.get_obs(self.thief.state) for p in self.police_agents}
        return obs, {}

    def step(self, action_dict):
        self.steps += 1
        for i in range(self.num_police):
            v = np.clip(action_dict[f"agent_{i}"][:2], -1, 1.0)
            self.police_agents[i].update(v, self.dt)

        # evader execution
        # ==============================
        # self.thief.update([agent.state for agent in self.police_agents])
        self.thief.local_info_and_collision_check(self.police_agents) 
        self.thief.update_env(self.police_agents)
        self.thief.update_memory()
        self.thief.select_goal()
        self.thief.sel_stgy()
        self.thief.move()
        # ==============================

        obs = {f"agent_{i}": self.police_agents[i].get_obs(self.thief.state) for i in range(self.num_police)}
        reward = {f"agent_{i}": self.compute_reward(i) for i in range(self.num_police)}
        done = any(np.linalg.norm(p.state- self.thief.state) < p.capture_range for p in self.police_agents)
        done_dict = {f"agent_{i}": np.linalg.norm(p.state- self.thief.state) < p.capture_range for i, p in enumerate(self.police_agents)}
        done_dict["__all__"] = done

        return obs, reward, done_dict, {}, {}
    
    def compute_reward(self, idx):
        agent = self.police_agents[idx]
        reward = 0.0

        # 1. 与小偷距离
        dist_to_thief = np.linalg.norm(agent.state - self.thief.state)
        reward += -0.1 * dist_to_thief

        # 2. 是否抓住小偷
        if dist_to_thief < agent.capture_range:
            reward += 100.0

        # 3. 是否撞墙（激光雷达值小于机器人半径）
        lidar_scan = agent._lidar(self.thief.state)
        if np.any(lidar_scan <= agent.robot_radius):
            reward -= 5.0

        # 4. 是否撞到其他警察
        for j, other_agent in enumerate(self.police_agents):
            if j != idx:
                dist = np.linalg.norm(agent.state - other_agent.state)
                if dist < agent.robot_radius:
                    reward -= 5.0
        
        # 5. 是否撞到障碍物
        from shapely.geometry import Polygon, Point
        agent_point = Point(agent.state)
        collided_with_boundary = False
        for poly_pts in Boundary:  # self.boundaries 应设置为 Boundary
            boundary_poly = Polygon(poly_pts)
            if boundary_poly.contains(agent_point) or boundary_poly.exterior.distance(agent_point) <= agent.robot_radius:
                collided_with_boundary = True
                break
        if collided_with_boundary:
            reward -= 10.0

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
            c = Circle(agent.state, radius=0.1, color=COLOR[id])
            self.ax.add_patch(c)
            self.ax.text(agent.state[0]+0.1, agent.state[1]+0.1, str(agent.id), fontsize=12, ha='center', va='center', color='red')
            c_a = Circle(agent.state, radius=agent.capture_range, color=COLOR[id], alpha=0.2)
            self.ax.add_patch(c_a)
            # 画雷达线束
            readings = agent._lidar(agent.state)
            angles = np.linspace(0, 2 * np.pi, agent.liard_readings, endpoint=False)
            for r, angle in zip(readings, angles):
                end_point = agent.state + r * np.array([np.cos(angle), np.sin(angle)])
                self.ax.plot([agent.state[0], end_point[0]], [agent.state[1], end_point[1]], color='black', linewidth=1.0, linestyle='--', zorder=1)
                self.ax.plot(end_point[0], end_point[1], 'yo', markersize=6, zorder=5)

        # 画小偷（红色三角形）
        t = self.thief.state
        tri = RegularPolygon((t[0], t[1]), numVertices=3, radius=0.1, color=COLOR[-1])
        self.ax.add_patch(tri)
        self.ax.plot(self.thief.modified_goal_pos[0], self.thief.modified_goal_pos[1], marker='s', color='red', markersize=6, fillstyle='none', alpha=0.3)

        # 可选：网格背景
        self.ax.set_xticks(np.arange(0, self.size + 1, 1))
        self.ax.set_yticks(np.arange(0, self.size + 1, 1))

        plt.pause(0.01)
        
        
