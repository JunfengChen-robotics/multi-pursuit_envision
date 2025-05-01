import gymnasium as gym
import numpy as np
from gymnasium import spaces
import shapely
import yaml, os
from police import PoliceAgent
from thief import ThiefAgent
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.path import Path
from evader.core.evader import Evader
from config.position import InitPos
import random


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
        
        # —— 新增：分阶段＆难度控制参数 —— 
        self.training_phase = args.training_phase    # 0:避障, 1:静态目标, 2:动态对抗
        self.num_police = args.num_police
        self.case = args.case
        self.construct_world()
        self.size = self.size
        self.dt = 0.05
        self.max_velocity = 1.2
        # action: [dx, dy, turn_rate]
        self.action_space = spaces.Box(low=-self.max_velocity, high=self.max_velocity, shape=(2,), dtype=np.float32)

        # 修改观察空间定义
        # observation: lidar (360 directions) + self pos + thief pos
        self.observation_space = spaces.Box(low=0, high=10, shape=(38+int(self.size/self.discrete_size)**2,), dtype=np.float32)

        self.obstacles = [Path(ob) for ob in self.vis_obstacles]
        self.train_or_test = args.train_or_test
        self.reward_func = {0:self.compute_reward_from_phase0, 1:self.compute_reward_from_phase1, 2:self.compute_reward_from_phase1}

    
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
        self.discrete_size = world['map']['dx'][self.case]
        

    def generate_random_positions(self, robot_num):
        def is_valid(pos, existing_positions, min_dist=0.5, obs_margin=0.1):
            """检查位置是否合法"""
            x, y = pos
            if not (0 < x < self.size and 0 < y < self.size):
                return False
            point = shapely.geometry.Point(x, y)

            # 检查与障碍物的关系
            for poly in self.obstacles:
                if not isinstance(poly, shapely.geometry.Polygon):
                    poly = shapely.geometry.Polygon(poly.vertices)
                if poly.contains(point):  # 在障碍物内部
                    return False
                if poly.exterior.distance(point) < obs_margin:  # 离边界太近
                    return False

            # 检查与已有点的最小距离
            for p in existing_positions:
                if np.linalg.norm(np.array(pos) - np.array(p)) < min_dist:
                    return False

            return True

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


    def generate_obstacle_image(self):
        # 获取网格大小（基于环境的大小和离散化参数）
        grid_size = int(self.size / self.discrete_size)  # 假设self.size是环境的尺寸，self.discrete_size是每个网格的尺寸
        
        # 创建一个全白的图像，表示空白区域
        image = np.zeros((grid_size, grid_size))  # 使用1表示空白区域
        
        # 将每个障碍物转化为网格位置，填充图像
        for obstacle in self.vis_obstacles:
            # 创建一个多边形对象，表示障碍物
            poly = Polygon(obstacle, closed=True)
            
            # 获取障碍物的多边形包围盒（bounding box），用于确定哪些网格会被占据
            min_x, min_y = np.min(obstacle, axis=0)
            max_x, max_y = np.max(obstacle, axis=0)
            
            # 计算这些坐标在网格中的对应位置
            min_grid_x = int(min_x / self.discrete_size)
            max_grid_x = int(max_x / self.discrete_size)
            min_grid_y = int(min_y / self.discrete_size)
            max_grid_y = int(max_y / self.discrete_size)
            
            # 遍历多边形所在的网格并填充图像
            for i in range(min_grid_x, max_grid_x + 1):
                for j in range(min_grid_y, max_grid_y + 1):
                    # 计算当前网格的中心点
                    grid_center = (i * self.discrete_size + self.discrete_size / 2, j * self.discrete_size + self.discrete_size / 2)
                    # 判断网格中心点是否在障碍物内，如果在，则将该网格位置设为0（障碍物）
                    if poly.contains_point(grid_center):
                        image[j, i] = 1  # 使用1表示障碍物
        
        return image

    
    def reset(self, seed=None, options=None):
        # —— 定位模式 & 随机采样 —— 
        if self.train_or_test == 'test':
            init_pos = InitPos.get_predefined_positions(case_number=self.case)
        if self.train_or_test == 'train':
            init_pos = self.generate_random_positions(self.num_police+1)      
        
        self.police_agents = [PoliceAgent(init_pos[i], 
                        id=i, 
                        obstacles=self.obstacles,
                        boundary=self.boundary,
                        mapsize=self.size)
            for i in range(self.num_police)]
        
        #  —— Phase0: 生成一个静态“目标点”给警察，No thief —— 
        if self.training_phase == 1:
            self.thief = ThiefAgent(init_pos=init_pos[-1], mapsize=self.size)
        else:
            self.thief = Evader(pos=init_pos[-1])
            self.thief.get_env_info(self.case)
            self.thief.update_env(self.police_agents)      

        self.trajectories= np.empty((self.num_police+1, 2, 0))
        self.police_to_thief_dict  = {i: 0 for i in range(len(self.police_agents))}
        self.steps = 0
        obs = {}
        # 获取障碍物图像
        obstacle_image = self.generate_obstacle_image()
        for p in self.police_agents:
            # 原： lidar + 相对小偷向量
            obs[f"agent_{p.id}"] = p.get_obs(self.thief.state, obstacle_image)
        return obs, {}


    def step(self, action_dict):
        
        self.steps += 1
        current_positions = np.zeros((self.num_police+1, 2, 1))
        for i in range(self.num_police):
            v = np.clip(action_dict[f"agent_{i}"][:2], -self.police_agents[i].max_velocity, self.police_agents[i].max_velocity)
            # 计算目标位置
            start_position = self.police_agents[i].state
            target_position = self.police_agents[i].state + v * self.dt
            if self.is_valid_transition(start_position, target_position):
                # 如果目标位置有效，则更新警察位置
                self.police_agents[i].update(v, self.dt)
            else:
                # 如果目标位置无效，则警察位置不更新
                pass
                # print(f"警察{i}的目标位置无效，保持原位置")
            current_positions[i, :, 0] = self.police_agents[i].state
        
        obs, reward, done_dict = {}, {}, {}
        obstacle_image = self.generate_obstacle_image()
        # —— Phase0: 纯避障 + 静态目标追踪 —— 
        if self.training_phase == 0:
            pass
        # —— Phase1: 简单小偷策略 —— 
        if self.training_phase == 1:
            self.thief.update([agent.state for agent in self.police_agents])
        # —— Phase2: 动态对抗 ——
        elif self.training_phase == 2 or self.train_or_test == "test":
            self.thief.local_info_and_collision_check(self.police_agents)
            self.thief.update_env(self.police_agents)
            self.thief.update_memory()
            self.thief.select_goal()
            self.thief.sel_stgy()
            self.thief.move()
        current_positions[-1, :, 0] = self.thief.state
        self.trajectories = np.concatenate([self.trajectories, current_positions], axis=2)
        
        # assemble obs & reward & done（沿用原 compute_reward）
        obs = {f"agent_{i}": self.police_agents[i].get_obs(self.thief.state, obstacle_image)
               for i in range(self.num_police)}
        
        if self.train_or_test == "train":
            reward = {f"agent_{i}": self.reward_func[self.training_phase](i)
                    for i in range(self.num_police)}
        elif self.train_or_test == "test":
            reward = {f"agent_{i}": self.reward_func[2](i)
                    for i in range(self.num_police)}
        
        
        done_dict = {f"agent_{i}": 
                         self.police_to_thief_dict[i] < self.police_agents[i].capture_range \
                         and self.is_in_sight(self.police_agents[i].state, self.thief.state, self.vis_obstacles, self.police_agents[i].capture_range)
                     for i in range(self.num_police)}
        
        for i in range(self.num_police):
            dist = self.police_to_thief_dict[i]
            in_sight = self.is_in_sight(self.police_agents[i].state, self.thief.state, self.vis_obstacles, self.police_agents[i].capture_range)
            
        done_dict["__all__"] = any(done_dict.values())
        return obs, reward, done_dict, {}, {}
    
    
    def compute_reward_from_phase0(self, idx):
        agent = self.police_agents[idx]
        reward = 0.0
        # reward：到 goal 的距离 & 撞墙惩罚
        dist_to_thief = self.compute_a_star_path(agent.state, self.thief.state)
        # --- 2. 引导奖励（仅前期启用）---
        if self.steps < 5000:
            # 鼓励探索远处（与初始位置保持一定距离）
            init_pos = agent.init_state
            dist_from_start = np.linalg.norm(agent.state - init_pos)
            reward += 0.05 * dist_from_start  # 奖励走得远
            # 额外奖励靠近小偷的行为（高权重）
            reward += 0.2 * (1.0 / (dist_to_thief + 1e-6))
        
        self.police_to_thief_dict[idx] = dist_to_thief
        reward = -0.1 * dist_to_thief
        if dist_to_thief < agent.capture_range:     # 当作“到达”
            reward += 100.0
        # 撞墙
        if np.any(agent._lidar(agent.state) <= agent.robot_radius+0.02):
            reward -= 10.0
        # 撞到边界
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
    
       
    def compute_reward_from_phase1(self, idx):
        agent = self.police_agents[idx]
        reward = 0.0

        # 1. 与小偷距离
        dist_to_thief = self.compute_a_star_path(agent.state, self.thief.state)
        reward += -0.1 * dist_to_thief
        
        # --- 2. 引导奖励（仅前期启用）---
        if self.steps < 5000:
            # 鼓励探索远处（与初始位置保持一定距离）
            init_pos = agent.init_state
            dist_from_start = np.linalg.norm(agent.state - init_pos)
            reward += 0.05 * dist_from_start  # 奖励走得远

            # 额外奖励靠近小偷的行为（高权重）
            reward += 0.2 * (1.0 / (dist_to_thief + 1e-6))
        
        # 2. 逃脱惩罚：如果警察离小偷越来越远，则给予惩罚
        previous_dist = self.police_to_thief_dict[idx]
        if dist_to_thief > previous_dist:
            reward -= 5
        
        self.police_to_thief_dict[idx] = dist_to_thief

        # 2. 是否抓住小偷
        if dist_to_thief < agent.capture_range:
            reward += 100.0

        # 3. 是否撞墙（激光雷达值小于机器人半径）
        lidar_scan = agent._lidar(self.thief.state)
        if np.any(lidar_scan <= agent.robot_radius+0.02):
            reward -= 5.0

        # 5. 是否撞到边界
        from shapely.geometry import Polygon, Point
        agent_point = Point(agent.state)
        collided_with_boundary = False
        for poly_pts in self.boundary:  # self.boundaries 应设置为 Boundary
            boundary_poly = Polygon(poly_pts)
            if boundary_poly.contains(agent_point) or boundary_poly.exterior.distance(agent_point) <= agent.robot_radius:
                collided_with_boundary = True
                break
        if collided_with_boundary:
            reward -= 5.0

        return reward
    
        
    def compute_reward_from_phase2(self, idx):
        agent = self.police_agents[idx]
        reward = 0.0

        # 1. 与小偷的距离
        dist_to_thief = self.compute_a_star_path(agent.state, self.thief.state)
        reward += -0.1 * dist_to_thief  # 距离越远，惩罚越大

        # 2. 是否抓住小偷
        if dist_to_thief < agent.capture_range:
            reward += 200.0  # 抓到小偷，奖励100分

        # 3. 逃脱惩罚：如果小偷逃脱，警察受到惩罚
        previous_dist = self.police_to_thief_dict.get(idx, dist_to_thief)
        if dist_to_thief > previous_dist:  # 如果距离变远了
            reward -= 20.0  # 小偷逃脱，警察惩罚

        self.police_to_thief_dict[idx] = dist_to_thief  # 更新记录

        # 4. 撞墙惩罚
        lidar_scan = agent._lidar(self.thief.state)
        if np.any(lidar_scan <= agent.robot_radius):
            reward -= 10.0  # 撞墙的惩罚

        # 5. 撞到边界惩罚
        from shapely.geometry import Polygon, Point
        agent_point = Point(agent.state)
        collided_with_boundary = False
        for poly_pts in self.boundary:
            boundary_poly = Polygon(poly_pts)
            if boundary_poly.contains(agent_point) or boundary_poly.exterior.distance(agent_point) <= agent.robot_radius:
                collided_with_boundary = True
                break
        if collided_with_boundary:
            reward -= 10.0  # 撞到边界的惩罚

        # 6. 避障奖励：如果警察避开了障碍物并接近小偷，给予奖励
        if dist_to_thief < previous_dist and np.all(agent._lidar(agent.state) > agent.robot_radius):
            reward += 5.0  # 避开障碍物并接近小偷，增加奖励

        return reward
    

    def compute_a_star_path(self, start, goal):
        """ 使用 networkx 计算A*路径，返回真实欧式距离（非步数），自动修正非法点 """
        import networkx as nx 
        import shapely.geometry as shageo

        def euclidean_heuristic(a, b):
            a, b = np.array(a), np.array(b)
            return np.linalg.norm(a - b)

        # 创建网格图
        grid_size = int(self.size / self.discrete_size)
        G = nx.grid_2d_graph(grid_size, grid_size)

        # 将障碍物转换为shapely的多边形
        obstacle_polygons = [shageo.Polygon(obstacle) for obstacle in self.vis_obstacles]

        # 删除与障碍物重叠的节点
        for obstacle in obstacle_polygons:
            for (x, y) in list(G.nodes):
                point = (x * self.discrete_size + self.discrete_size / 2, y * self.discrete_size + self.discrete_size / 2)
                if obstacle.contains(shageo.Point(point)):
                    G.remove_node((x, y))

        # 起点终点网格坐标
        start_grid = (int(start[0] / self.discrete_size), int(start[1] / self.discrete_size))
        goal_grid = (int(goal[0] / self.discrete_size), int(goal[1] / self.discrete_size))

        def find_nearest_valid_node(pos_grid, graph_nodes):
            """寻找距离 pos_grid 最近的合法图中节点"""
            min_dist = float('inf')
            best_node = None
            for node in graph_nodes:
                dist = euclidean_heuristic(pos_grid, node)
                if dist < min_dist:
                    min_dist = dist
                    best_node = node
            return best_node

        # 如果起点或终点不在图中，尝试修正为最近的合法节点
        if start_grid not in G:
            new_start = find_nearest_valid_node(start_grid, G.nodes)
            if new_start is not None:
                start_grid = new_start
            else:
                print("无法修复起点")
                return float('inf')

        if goal_grid not in G:
            new_goal = find_nearest_valid_node(goal_grid, G.nodes)
            if new_goal is not None:
                goal_grid = new_goal
            else:
                print("无法修复终点")
                return float('inf')

        # A*搜索
        try:
            path = nx.astar_path(G, start_grid, goal_grid, heuristic=euclidean_heuristic)

            # 计算路径欧式长度（乘以 discrete_size 恢复物理单位）
            total_length = 0.0
            for i in range(len(path) - 1):
                p1 = np.array(path[i])
                p2 = np.array(path[i + 1])
                total_length += np.linalg.norm((p1 - p2) * self.discrete_size)

            return total_length

        except nx.NetworkXNoPath:
            print("没有找到路径")
            return float('inf')

        
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
    
    
    def render(self, episode, episode_step):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()
            self.ax.set_xlim(0, self.size)
            self.ax.set_ylim(0, self.size)
            self.ax.set_aspect('equal')
            self.ax.set_title(f"Police vs Thief in {self.training_phase} in episode {episode} step {episode_step}")

            
            # 画障碍物，只画一次
            self.static_elements = []
            for poly in self.vis_obstacles:
                p = Polygon(poly, closed=True, facecolor='black', edgecolor='black')
                self.ax.add_patch(p)
                self.static_elements.append(p)
                
            # 可选：网格背景
            self.ax.set_xticks(np.arange(0, self.size + 1, 1))
            self.ax.set_yticks(np.arange(0, self.size + 1, 1))

        self.ax.set_title(f"Police vs Thief in {self.training_phase} in episode {episode} step {episode_step}")

        # 清除上一次的动态元素（警察、小偷、轨迹等）
        if hasattr(self, 'dynamic_elements'):
            for element in self.dynamic_elements:
                element.remove()
                
        self.dynamic_elements = []

        # 画警察（绿色圆形）
        for id, agent in enumerate(self.police_agents):
            x, y = agent.state[0], agent.state[1]
            sc = self.ax.scatter(x, y, marker='o', color=COLOR[id], s=200, edgecolor=COLOR[id], linewidth=2)
            txt = self.ax.text(agent.state[0]+0.1, agent.state[1]+0.1, str(agent.id), fontsize=12, ha='center', va='center', color='red')
            c_a = Circle(agent.state, radius=agent.capture_range, color=COLOR[id], alpha=0.2)
            self.ax.add_patch(c_a)
            self.dynamic_elements.extend([sc, txt, c_a])
            if self.train_or_test == "test":
                traj_line, = self.ax.plot(self.trajectories[id, 0, :], self.trajectories[id, 1, :], color=COLOR[id], linewidth=5.0, linestyle='-')
                self.dynamic_elements.append(traj_line)
                
            if self.train_or_test == "train":
                # 画雷达线束
                readings = agent._lidar(agent.state)
                angles = np.linspace(0, 2 * np.pi, agent.liard_readings, endpoint=False)
                for r, angle in zip(readings, angles):
                    end_point = agent.state + r * np.array([np.cos(angle), np.sin(angle)])
                    radar_line, = self.ax.plot([agent.state[0], end_point[0]], [agent.state[1], end_point[1]], color='black', linewidth=1.0, linestyle='--', zorder=1)
                    radar_dot, = self.ax.plot(end_point[0], end_point[1], 'yo', markersize=6, zorder=5)
                    self.dynamic_elements.extend([radar_line, radar_dot])

        # 画小偷（红色三角形）
        t_x, t_y = self.thief.state[0], self.thief.state[1]
        thief_marker = self.ax.scatter(t_x, t_y, marker='^', color=COLOR[-1], s=200, edgecolor=COLOR[-1], linewidth=2)
        self.dynamic_elements.append(thief_marker)
        goal_marker, = self.ax.plot(self.thief.modified_goal_pos[0], self.thief.modified_goal_pos[1], marker='s', color='red', markersize=6, fillstyle='none', alpha=0.3)
        self.dynamic_elements.append(goal_marker)
        if self.train_or_test == "test":
            thief_traj, = self.ax.plot(self.trajectories[-1, 0, :], self.trajectories[-1, 1, :], color=COLOR[-1], linewidth=5.0, linestyle='-')
            self.dynamic_elements.append(thief_traj)

        plt.pause(0.01)
    
    
    def is_valid_transition(self, start_pos, end_pos):
        """
        判断从 start_pos 移动到 end_pos 是否会穿过障碍物。
        插值路径点，使用 contains + exterior.distance 检查。
        """

        import numpy as np
        from shapely.geometry import Point, Polygon

        def interpolate_path(start, end, step_size=0.05):
            direction = end - start
            distance = np.linalg.norm(direction)
            if distance == 0:
                return [start]
            steps = int(distance / step_size)
            return [start + direction * (i / steps) for i in range(1, steps + 1)]

        path_points = interpolate_path(np.array(start_pos), np.array(end_pos))
        path_points.append(np.array(end_pos))  # 添加终点也参与检测

        for point_array in path_points:
            point = Point(point_array)

            for obs_pts in self.vis_obstacles:
                poly = Polygon(obs_pts)

                # 1. 点是否在障碍物内
                if poly.contains(point):
                    return False

                # 2. 点是否过于靠近障碍边界（用于“贴墙”检测）
                if poly.exterior.distance(point) <= 0.05:
                    return False

        return True
