from env import ChaseEnv
import numpy as np
from policy import SACAgent
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import re

def get_latest_checkpoint(checkpoint_dir, phase):
    """
    获取指定阶段的最新模型检查点文件，选择最大步数的文件。
    """
    # 获取所有的模型文件
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f"sac_phase{phase}_")]
    
    if not checkpoint_files:
        return None
    
    # 使用正则表达式提取步数
    step_pattern = re.compile(r'sac_phase\d+_(\d+).pth')
    
    # 找到最大步数的文件
    latest_step = -1
    latest_ckpt = None
    for checkpoint in checkpoint_files:
        match = step_pattern.search(checkpoint)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_ckpt = checkpoint

    return os.path.join(checkpoint_dir, latest_ckpt)


def train(env, render=False, resume=False, init_ckpt=None):
    
    # 设置TensorBoard日志目录
    log_dir = "logs/"+args.case
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    save_dir = "checkpoints/"+args.case
    os.makedirs(save_dir, exist_ok=True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    sac_agent = SACAgent(state_dim, action_dim, args.case)
    if resume and init_ckpt is not None:
        print(f"→ Loading weights from {init_ckpt}")
        sac_agent.load(init_ckpt)
        
    total_steps = 0
    episode = 0
    while total_steps < args.max_steps:
        episode_rewards = {f"agent_{i}": 0.0 for i in range(env.num_police)}
        done = False
        episode_length = 0
        obs, _ = env.reset()
        print("="*20)
        print(f"Episode {episode} started.")
        while not done and episode_length < args.max_episode_steps:
            if render:
                env.render()
            action_dict = {}
            # 为每个警察代理选择动作
            for i in range(env.num_police):
                agent_state = obs[f"agent_{i}"]
                if total_steps < 10000:
                    action = env.action_space.sample()  # 随机动作
                else:
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
            q_loss, policy_loss, alpha_loss = sac_agent.update()
            
            if q_loss is not None :
                writer.add_scalar("Q_Loss", q_loss, total_steps)
                writer.add_scalar("Policy_Loss", policy_loss, total_steps)
            if alpha_loss is not None:
                writer.add_scalar("Alpha_Loss", alpha_loss, total_steps)
                

            obs = next_obs
            total_steps += 1
            episode_length += 1

            # 检查所有警察是否都抓住了小偷
            done = done_dict["__all__"]

            if total_steps % 100 == 0:
                print(f"Steps: {total_steps}, Total Rewards: {episode_rewards}")
                
        episode += 1
        print("="*20)
        avg_reward = np.mean(list(episode_rewards.values()))
        writer.add_scalar("Episode_Reward", avg_reward, episode)
        writer.add_scalar("Episode_Length", episode_length, episode)
        
        if episode % 10 == 0:
            print(f"Current Episode: {episode}, Average Episode Reward: {avg_reward}")
            model_path = os.path.join(save_dir, f"sac_phase{args.training_phase}_{total_steps}.pth")

            sac_agent.save(model_path)
            print(f"[Checkpoint] Saved at step {total_steps}")
    
    # 返回最后一个 checkpoint 路径，供下个阶段 warm-start
    return model_path
    

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Train the SAC agent in the Chase environment.")
    argparser.add_argument("--num_police", type=int, default=3, help="Number of police agents.")
    argparser.add_argument("--train_or_test", type=str, default="train", choices=["train", "test"], help="Train or test mode.")
    argparser.add_argument("--render", action="store_true", help="Render the environment.")
    argparser.add_argument("--case", type=str, default="case5", choices=["case5", "case6", "case10"], help="Case number for predefined positions.")
    argparser.add_argument("--max_episode_steps", type=int, default=2000, help="Maximum steps per episode.")
    # argparser.add_argument("--max_steps", type=int, default=100000000, help="Maximum number of steps.")
    # —— 新增分阶段步数控制 —— 
    argparser.add_argument("--phase0_steps", type=int, default=1000000, help="Phase0 (避障) 步数")
    argparser.add_argument("--phase1_steps", type=int, default=5000000, help="Phase1 (静态目标) 步数")
    argparser.add_argument("--phase2_steps", type=int, default=10000000, help="Phase2 (动态对抗) 步数")
    # 由 Curricumulum Loop 赋值，不需要用户直接设置
    argparser.add_argument("--training_phase", type=int, default=0, help="Curriculum training phase")
    args = argparser.parse_args()
    # —— 三阶段循环训练 —— 
    phases = [
        (0, args.phase0_steps),
        (1, args.phase1_steps),
        (2, args.phase2_steps),
    ]
    phases = phases[args.training_phase:]
    prev_ckpt = None
    for phase, steps in phases:
        print(f"\n===== Starting Phase {phase} ({['避障','静态目标','动态对抗'][phase]}) =====")
        args.training_phase = phase
        args.max_steps = steps
        env = ChaseEnv(args)
        
        if phase > 0:
            # 根据阶段号和 case 名称来构建路径
            prev_ckpt = get_latest_checkpoint(f"checkpoints/{args.case}", phase-1)
            if prev_ckpt:
                print(f"Loading checkpoint from {prev_ckpt}")
        
        # 第一阶段无需载入模型，后面阶段都从上一次 checkpoint warm-start
        ckpt = train(env,
                     render=True,
                     resume=(prev_ckpt is not None),
                     init_ckpt=prev_ckpt)
        prev_ckpt = ckpt
    print("=== Curriculum Training Completed ===")