import matplotlib.pyplot as plt
import torch
import os
import argparse
import time
from rl_td3 import PIDEnv, TD3Agent


def test_agent(agent, env, num_episodes=1):
    all_setpoints = []
    all_outputs = []
    all_kp = []
    all_ki = []
    all_kd = []

    for episode in range(num_episodes):
        state = env.reset()
        setpoints = []
        outputs = []
        kp_values = []
        ki_values = []
        kd_values = []
        for step in range(env.max_steps):
            action = agent.select_action(state, noise_scale=0.0)  # 不添加噪声
            next_state, reward, done, _ = env.step(action)
            state = next_state

            setpoints.append(env.setpoint)
            outputs.append(env.x)
            kp_values.append(env.Kp)
            ki_values.append(env.Ki)
            kd_values.append(env.Kd)

            if done:
                break

        all_setpoints.append(setpoints)
        all_outputs.append(outputs)
        all_kp.append(kp_values)
        all_ki.append(ki_values)
        all_kd.append(kd_values)

    return all_setpoints, all_outputs, all_kp, all_ki, all_kd


def plot_results(
    setpoints, outputs, kp_values, ki_values, kd_values, ckpt_name, episode
):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    for i in range(len(setpoints)):
        plt.plot(setpoints[i], label=f"Episode {i+1} Setpoint")
        plt.plot(outputs[i], label=f"Episode {i+1} Output")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Setpoint vs Output")
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(len(kp_values)):
        plt.plot(kp_values[i], label=f"Episode {i+1} Kp")
        plt.plot(ki_values[i], label=f"Episode {i+1} Ki")
        plt.plot(kd_values[i], label=f"Episode {i+1} Kd")
    plt.xlabel("Time Steps")
    plt.ylabel("PID Parameters")
    plt.title("PID Parameters over Time")
    plt.legend()
    plt.tight_layout()

    timestamp = time.strftime("%m%d-%H%M%S")
    plt.savefig(f"ckpt_td3/{ckpt_name}/test_{episode}_{timestamp}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="checkpoint name")
    args = parser.parse_args()

    env = PIDEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3Agent(state_dim, action_dim)

    ckpt_name = args.ckpt

    latest_episode = max(
        [
            int(f.split("_")[-1].split(".")[0])
            for f in os.listdir(f"ckpt_td3/{ckpt_name}")
            if f.startswith("actor_td3")
        ]
    )

    # 加载训练好的模型权重
    agent.actor.load_state_dict(
        torch.load(f"ckpt_td3/{ckpt_name}/actor_td3_{latest_episode}.pth")
    )
    agent.critic.load_state_dict(
        torch.load(f"ckpt_td3/{ckpt_name}/critic_td3_{latest_episode}.pth")
    )

    setpoints, outputs, kp_values, ki_values, kd_values = test_agent(agent, env)
    plot_results(
        setpoints, outputs, kp_values, ki_values, kd_values, ckpt_name, latest_episode
    )
