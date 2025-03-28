import matplotlib.pyplot as plt
import torch
from rl import PIDEnv, DDPGAgent


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


def plot_results(setpoints, outputs, kp_values, ki_values, kd_values):
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
    plt.show()


if __name__ == "__main__":
    env = PIDEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim)

    # 加载训练好的模型权重
    agent.actor.load_state_dict(torch.load("actor.pth"))
    agent.critic.load_state_dict(torch.load("critic.pth"))

    setpoints, outputs, kp_values, ki_values, kd_values = test_agent(agent, env)
    plot_results(setpoints, outputs, kp_values, ki_values, kd_values)
