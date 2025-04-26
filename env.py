import gymnasium as gym
import numpy as np
from gymnasium import spaces
import yaml, os
from police import PoliceAgent
from thief import ThiefAgent
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.path import Path
from evader.core.evader import Evader
from config.position import InitPos


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

class ChaseEnv(gym.Env):
    
    def __init__(self, args):
        super().__init__()
        self.num_police = args.num_police
        self.case = args.case
        self.construct_world()
        self.size = self.size
        self.dt = 0.05
        # action: [dx, dy, turn_rate]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # observation: lidar (360 directions) + self pos + thief pos
        self.observation_space = spaces.Box(low=0, high=10, shape=(40,), dtype=np.float32)

        self.obstacles = [Path(ob) for ob in self.vis_obstacles]
        self.train_or_test = args.train_or_test
        
    
    def construct_world(self):
        path = os.path.dirname(os.path.abspath(__file__))
        coord_yaml_path = os.path.join(path, "config","coord.yaml")
        with open(coord_yaml_path, "r") as stream:
            try:
                coords = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.vis_obstacles = coords[self.case]

        world_yaml_path = os.path.join(path, "config", "world.yaml")
        with open(world_yaml_path, "r") as stream:
            try:
                world = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.size = world['map']['size'][self.case]*2
        self.boundary = [[[0,0], [0, self.size], [-0.2, self.size], [-0.2, 0]], 
                                [[0, self.size], [self.size, self.size], [self.size, self.size+0.2], [0, self.size+0.2]],
                                [[self.size, 0], [self.size+0.2, 0], [self.size+0.2, self.size], [self.size, self.size]],
                                [[0, -0.2], [self.size, -0.2], [self.size, 0], [0, 0]]
                         ]
        
    def generate_random_positions(self, robot_num):
        def is_valid(pos, existing_positions, min_dist=0.5):
            """检查位置是否在地图内、在障碍物外，并且离已有位置不太近"""
            x, y = pos
            if not (0 < x < self.size and 0 < y < self.size):
                return False
            point = np.array([x, y])
            if any(poly.contains_point(point) for poly in self.obstacles):
                return False
            if any(np.linalg.norm(np.array(pos) - np.array(p)) < min_dist for p in existing_positions):
                return False
            return True
        
        import random
        positions = []
        max_trials = 1000
        for _ in range(robot_num):
            for _ in range(max_trials):
                x = random.uniform(0.5, self.size - 0.5)
                y = random.uniform(0.5, self.size - 0.5)
                if is_valid((x, y), positions):
                    positions.append((x, y))
                    break
            else:
                raise RuntimeError("Failed to generate valid initial positions.")
        return np.array(positions).round(1)
        

    def reset(self, seed=None, options=None):
        if self.train_or_test == 'test':
            init_pos = InitPos.get_predefined_positions(case_number=self.case)
        if self.train_or_test == 'train':
            init_pos = self.generate_random_positions(self.num_police + 1)
            
        self.police_agents = [PoliceAgent(init_pos[i], 
                                          id=i, 
                                          obstacles=self.obstacles,
                                          boundary=self.boundary)
                              for i in range(self.num_police)]
        self.trajectories= np.empty((self.num_police+1, 2, 0))
        # initialization of evader
        # ===================================
        # self.thief = ThiefAgent(init_pos=init_pos[-1])
        self.thief = Evader(pos=init_pos[-1])
        self.thief.get_env_info(self.case)
        self.thief.update_env(self.police_agents)
        # ====================================
        self.steps = 0
        obs = {f"agent_{p.id}": p.get_obs(self.thief.state) for p in self.police_agents}
        return obs, {}

    def step(self, action_dict):
        self.steps += 1
        current_positions = np.zeros((self.num_police+1, 2, 1))
        for i in range(self.num_police):
            v = np.clip(action_dict[f"agent_{i}"][:2], -1, 1.0)
            self.police_agents[i].update(v, self.dt)
            current_positions[i, :, 0] = self.police_agents[i].state

        # evader execution
        # ==============================
        # self.thief.update([agent.state for agent in self.police_agents])
        self.thief.local_info_and_collision_check(self.police_agents) 
        self.thief.update_env(self.police_agents)
        self.thief.update_memory()
        self.thief.select_goal()
        self.thief.sel_stgy()
        self.thief.move()
        current_positions[-1, :, 0] = self.thief.state
        self.trajectories = np.concatenate([self.trajectories, current_positions], axis=2)
        # ==============================

        obs = {f"agent_{i}": self.police_agents[i].get_obs(self.thief.state) for i in range(self.num_police)}
        reward = {f"agent_{i}": self.compute_reward(i) for i in range(self.num_police)}
        done = any(np.linalg.norm(p.state- self.thief.state) < p.capture_range and self.is_in_sight(p.state.tolist(), self.thief.state.tolist(), self.vis_obstacles,  p.capture_range) for p in self.police_agents)
        done_dict = {f"agent_{i}": np.linalg.norm(p.state- self.thief.state) < p.capture_range for i, p in enumerate(self.police_agents)}
        done_dict["__all__"] = done

        return obs, reward, done_dict, {}, {}
    
    
    def is_in_sight(self, pos1, pos2, obs_info_local_,scout_range):
        """
        判断端口是否在 Evader 的视线范围内。判断evader 是否在pursuer 视线范围内。或pursuer是否在evader视线范围内。
        即端口到 Evader 之间的距离是否小于感知evader的范围，以及连线是否没有障碍物遮挡。
        """
        
        def line_intersects_obstacle(pos1, pos2, obstacle):
            """
            检查由 pos2 和 pos1 定义的线段是否与障碍物相交。
            
            :param pos2: 线段的一个端点，格式为 (x, y)
            :param pos1: 线段的另一个端点，格式为 (x, y)
            :param obstacle: 障碍物的顶点列表，格式为 [(x1, y1), (x2, y2), ...]
            :return: 如果线段与障碍物相交，则返回 True，否则返回 False
            """
            import shapely
            
            line = shapely.geometry.LineString([pos2, pos1])
            poly = shapely.geometry.Polygon(obstacle)
            
            return line.intersects(poly)

        if np.linalg.norm(np.array(pos1) - np.array(pos2)) > scout_range:
            return False

        for obs_verts in obs_info_local_:
            # if np.array(obs_verts).shape[0] == 2:
            #     obs_verts = obs_verts.T
            if line_intersects_obstacle(pos1, pos2, obs_verts):
                return False

        return True
    
    
    
    
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
        if np.any(lidar_scan <= agent.robot_radius+0.1):
            reward -= 10.0

        # 4. 是否撞到其他警察
        for j, other_agent in enumerate(self.police_agents):
            if j != idx:
                dist = np.linalg.norm(agent.state - other_agent.state)
                if dist <= 2* agent.robot_radius:
                    reward -= 5.0
        
        # 5. 是否撞到障碍物
        from shapely.geometry import Polygon, Point
        agent_point = Point(agent.state)
        collided_with_boundary = False
        for poly_pts in self.boundary:  # self.boundaries 应设置为 Boundary
            boundary_poly = Polygon(poly_pts)
            if boundary_poly.contains(agent_point) or boundary_poly.exterior.distance(agent_point) <= agent.robot_radius:
                collided_with_boundary = True
                break
        if collided_with_boundary:
            reward -= 10.0

        return reward
    
    
    def render(self):
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
            x, y = agent.state[0], agent.state[1]
            self.ax.scatter(x, y, marker='o', color=COLOR[id], s=200, edgecolor=COLOR[id], linewidth=2)
            self.ax.text(agent.state[0]+0.1, agent.state[1]+0.1, str(agent.id), fontsize=12, ha='center', va='center', color='red')
            c_a = Circle(agent.state, radius=agent.capture_range, color=COLOR[id], alpha=0.2)
            self.ax.add_patch(c_a)
            if self.train_or_test == "test":
                self.ax.plot(self.trajectories[id, 0, :], self.trajectories[id, 1, :], color=COLOR[id], linewidth=5.0, linestyle='-')
            
            if self.train_or_test == "train":
                # 画雷达线束
                readings = agent._lidar(agent.state)
                angles = np.linspace(0, 2 * np.pi, agent.liard_readings, endpoint=False)
                for r, angle in zip(readings, angles):
                    end_point = agent.state + r * np.array([np.cos(angle), np.sin(angle)])
                    self.ax.plot([agent.state[0], end_point[0]], [agent.state[1], end_point[1]], color='black', linewidth=1.0, linestyle='--', zorder=1)
                    self.ax.plot(end_point[0], end_point[1], 'yo', markersize=6, zorder=5)

        # 画小偷（红色三角形）
        t_x, t_y = self.thief.state[0], self.thief.state[1]
        self.ax.scatter(t_x, t_y, marker='^', color=COLOR[-1], s=200, edgecolor=COLOR[-1], linewidth=2)
        self.ax.plot(self.thief.modified_goal_pos[0], self.thief.modified_goal_pos[1], marker='s', color='red', markersize=6, fillstyle='none', alpha=0.3)
        if self.train_or_test == "test":
            self.ax.plot(self.trajectories[-1, 0, :], self.trajectories[-1, 1, :], color=COLOR[-1], linewidth=5.0, linestyle='-')

        # 可选：网格背景
        self.ax.set_xticks(np.arange(0, self.size + 1, 1))
        self.ax.set_yticks(np.arange(0, self.size + 1, 1))

        plt.pause(0.01)
        
        
