from env import ChaseEnv
import numpy as np
from policy import SACAgent
from torch.utils.tensorboard import SummaryWriter
import os
import torch



MAX_STEPS = 10000000


def train(env, render=False):
    
    # 设置TensorBoard日志目录
    log_dir = "logs/sac"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    
    
    sac_agent = SACAgent(state_dim, action_dim)
    total_steps = 0
    while total_steps < MAX_STEPS:
        obs, _ = env.reset()
        episode_rewards = {f"agent_{i}": 0.0 for i in range(env.num_police)}
        done = False
        episode = 0
        episode_length = 0
        while not done:
            if render:
                env.render()
            action_dict = {}
            # 为每个警察代理选择动作
            for i in range(env.num_police):
                agent_state = obs[f"agent_{i}"]
                action = sac_agent.get_action(agent_state)
                action_dict[f"agent_{i}"] = action

            # 执行动作并获得新的观察和奖励
            next_obs, reward_dict, done_dict, _, _ = env.step(action_dict)

            # 存储经验并更新网络
            for i in range(env.num_police):
                agent_state = obs[f"agent_{i}"]
                agent_action = action_dict[f"agent_{i}"]
                agent_reward = reward_dict[f"agent_{i}"]
                agent_next_state = next_obs[f"agent_{i}"]
                agent_done = done_dict[f"agent_{i}"]

                sac_agent.store_transition(agent_state, agent_action, agent_reward, agent_next_state, agent_done)
                episode_rewards[f"agent_{i}"] += agent_reward
                writer.add_scalar(f"Agent_{i}_Reward", agent_reward, total_steps)
            
            # 更新SAC代理
            q_loss, policy_loss = sac_agent.update()
            
            if q_loss is not None:
                writer.add_scalar("Q_Loss", q_loss, total_steps)
                writer.add_scalar("Policy_Loss", policy_loss, total_steps)
                

            obs = next_obs
            total_steps += 1
            episode_length += 1

            # 检查所有警察是否都抓住了小偷
            done = done_dict["__all__"]

            if total_steps % 100 == 0:
                print(f"Steps: {total_steps}, Episode Rewards: {episode_rewards}")
                
        episode += 1
        
        avg_reward = np.mean(list(episode_rewards.values()))
        writer.add_scalar("Episode_Reward", avg_reward, total_steps)
        writer.add_scalar("Episode_Length", episode_length, episode)
        
        if total_steps % 1000 == 0:
            print(f"Total steps: {total_steps}, Average Episode Reward: {avg_reward}")
            model_path = os.path.join(save_dir, f"sac_checkpoint_{total_steps}.pth")
            sac_agent.save(model_path)
            print(f"[Checkpoint] Saved at step {total_steps}")

    

if __name__ == "__main__":
    env = ChaseEnv(num_police=3)
    train(env, render=True)
