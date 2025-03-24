import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
import matplotlib.pyplot as plt


# 定义一个用于PID控制的环境
class PIDEnv(gym.Env):
    def __init__(self):
        super(PIDEnv, self).__init__()
        # 定义状态：[误差, 积分, 微分, 当前测量值, 目标值]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        # 定义动作：PID 参数的调整量 [dKp, dKi, dKd]，范围为-1到1
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.dt = 0.1  # 时间步长
        self.tau = 1.0  # 系统时间常数
        self.max_steps = 100  # 每个episode最大步数
        self.reset()

    def reset(self):
        self.step_count = 0
        self.x = 0.0  # 系统初始值
        self.setpoint = 0.0  # 初始目标值（常值）
        self.target1 = 0.0  # 初始目标
        self.target2 = 1.0  # 突变后的目标值
        self.change_step = 50  # 在第50步时改变目标值
        self.integral = 0.0
        self.prev_error = self.setpoint - self.x
        # 初始PID参数
        self.Kp = 1.0
        self.Ki = 0.0
        self.Kd = 0.0
        return self._get_state()

    def _get_state(self):
        error = self.setpoint - self.x
        derivative = error - self.prev_error
        # 返回状态向量
        return np.array(
            [error, self.integral, derivative, self.x, self.setpoint], dtype=np.float32
        )

    def step(self, action):
        # 根据动作更新PID参数，动作为调整量，乘以一个缩放因子来控制更新幅度
        scale = 0.1
        self.Kp = max(0.0, self.Kp + scale * action[0])
        self.Ki = max(0.0, self.Ki + scale * action[1])
        self.Kd = max(0.0, self.Kd + scale * action[2])

        # 计算误差及其积分、微分
        error = self.setpoint - self.x
        self.integral += error * self.dt
        derivative = error - self.prev_error

        # 根据PID公式计算控制信号
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # 模拟一阶系统动态： x_{t+1} = x_t + (dt/tau)*(-x_t + u)
        self.x = self.x + (self.dt / self.tau) * (-self.x + u)
        self.prev_error = error

        self.step_count += 1
        # 当达到设定步数时改变目标值
        if self.step_count == self.change_step:
            self.setpoint = self.target2

        # 奖励设计：负的误差平方（期望误差最小）
        reward = -(error**2)
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done, {}


# Actor网络（策略网络）：将状态映射至动作（PID参数的调整值）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.out(x))


# Critic网络：估计状态-动作对的价值
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# 经验回放Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# DDPG算法代理
class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        # 添加噪声以便探索
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1)

        # Critic更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


def main():
    env = PIDEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim)

    num_episodes = 1500
    max_steps = env.max_steps
    batch_size = 128

    rewards = []
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train(batch_size)
            state = next_state
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Episode {episode}, Reward: {episode_reward:.2f}, Time: {elapsed_time:.2f}s"
            )
            start_time = time.time()

    torch.save(agent.actor.state_dict(), "actor.pth")
    torch.save(agent.critic.state_dict(), "critic.pth")

    # 绘制reward曲线 俞皓然
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.savefig("reward_curve.png")
    plt.show()


if __name__ == "__main__":
    main()

