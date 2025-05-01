from env import ChaseEnv
import numpy as np
from policy import SACAgent
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import glob

MAX_TEST_EPISODES = 1  # 测试多少个episode

def find_latest_checkpoint(case):
    base_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint_dir = os.path.join(base_path, "checkpoints", case)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # 找到步数最大的那个模型
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_checkpoint = checkpoint_files[-1]
    return latest_checkpoint

def test(env, model_path, render=False):
    log_dir = "logs_test/" + args.case
    os.makedirs(log_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=log_dir)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    sac_agent = SACAgent(state_dim, action_dim)
    sac_agent.load(model_path)
    print(f"Loaded model from {model_path}")

    total_rewards = []
    
    for episode in range(MAX_TEST_EPISODES):
        episode_rewards = {f"agent_{i}": 0.0 for i in range(env.num_police)}
        done = False
        obs, _ = env.reset()
        episode_length = 0
        print("="*20)
        print(f"Test Episode {episode} started.")
        
        while not done:
            if render:
                env.render()
          
            action_dict = {}
            for i in range(env.num_police):
                agent_state = obs[f"agent_{i}"]
                action = sac_agent.get_action(agent_state, evaluate=True)
                action_dict[f"agent_{i}"] = action
            
            next_obs, reward_dict, done_dict, _, _ = env.step(action_dict)
            
            for i in range(env.num_police):
                agent_reward = reward_dict[f"agent_{i}"]
                episode_rewards[f"agent_{i}"] += agent_reward
            
            obs = next_obs
            done = done_dict["__all__"]
            episode_length += 1
        
        avg_reward = np.mean(list(episode_rewards.values()))
        total_rewards.append(avg_reward)
        # writer.add_scalar("Test_Episode_Reward", avg_reward, episode)
        # writer.add_scalar("Test_Episode_Length", episode_length, episode)

        print(f"Test Episode {episode} finished. Average Reward: {avg_reward}. Average Length: {episode_length}")
        print("="*20)
    
    overall_avg_reward = np.mean(total_rewards)
    print(f"Testing finished over {MAX_TEST_EPISODES} episodes. Overall Average Reward: {overall_avg_reward}")

    # writer.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Test the SAC agent in the Chase environment.")
    argparser.add_argument("--num_police", type=int, default=3, help="Number of police agents.")
    argparser.add_argument("--render", action="store_true", help="Render the environment during testing.")
    argparser.add_argument("--case", type=str, default="case5", choices=["case5", "case6", "case10"], help="Case number for predefined positions.")
    argparser.add_argument("--model_path", type=str, default=None, help="Optional: manually specify model path.")
    argparser.add_argument("--train_or_test", type=str, default="test", choices=["train", "test"], help="Train or test mode.")
    argparser.add_argument("--phase0_steps", type=int, default=1000000, help="Phase0 (避障) 步数")
    argparser.add_argument("--phase1_steps", type=int, default=5000000, help="Phase1 (静态目标) 步数")
    argparser.add_argument("--phase2_steps", type=int, default=10000000, help="Phase2 (动态对抗) 步数")
    # 由 Curricumulum Loop 赋值，不需要用户直接设置
    argparser.add_argument("--training_phase", type=int, default=2, help="Curriculum training phase")
    args = argparser.parse_args()

    env = ChaseEnv(args)

    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = find_latest_checkpoint(args.case)

    test(env, model_path=model_path, render=args.render)
