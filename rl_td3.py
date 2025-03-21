import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time


# 定义一个用于PID控制的环境，目标切换和突变时刻均带有随机性
class PIDEnv(gym.Env):
    def __init__(self):
        super(PIDEnv, self).__init__()
        # 状态维度：[误差, 积分, 微分, 当前测量值, 目标值]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        # 动作：PID参数的调整量 [dKp, dKi, dKd]，范围为 -1 到 1
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.dt = 0.1  # 时间步长
        self.tau = 1.0  # 系统时间常数
        self.max_steps = 100  # 每个 episode 最大步数
        self.reset()

    def reset(self):
        self.step_count = 0
        self.x = 0.0  # 系统初始值

        # 为 target1、target2 以及突变时刻添加随机性
        self.target1 = random.uniform(-1.0, 1.0)
        self.target2 = random.choice(
            [val for val in np.linspace(-1.0, 1.0, 21) if abs(val - self.target1) > 0.5]
        )
        self.change_step = random.randint(30, 70)
        self.setpoint = self.target1

        self.integral = 0.0
        self.prev_error = self.setpoint - self.x
        # 初始化 PID 参数
        self.Kp = 1.0
        self.Ki = 0.0
        self.Kd = 0.0
        return self._get_state()

    def _get_state(self):
        error = self.setpoint - self.x
        derivative = error - self.prev_error
        return np.array(
            [error, self.integral, derivative, self.x, self.setpoint], dtype=np.float32
        )

    def step(self, action):
        # 动作为 PID 参数调整量，乘以缩放因子
        scale = 0.1
        self.Kp = max(0.0, self.Kp + scale * action[0])
        self.Ki = max(0.0, self.Ki + scale * action[1])
        self.Kd = max(0.0, self.Kd + scale * action[2])

        error = self.setpoint - self.x
        self.integral += error * self.dt
        derivative = error - self.prev_error

        # 根据 PID 公式计算控制信号
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # 模拟一阶系统动态： x_{t+1} = x_t + (dt/tau)*(-x_t + u)
        self.x = self.x + (self.dt / self.tau) * (-self.x + u)
        self.prev_error = error

        self.step_count += 1
        if self.step_count == self.change_step:
            self.setpoint = self.target2

        # 奖励设计：负的误差平方，期望误差最小
        reward = -(error**2)
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done, {}


# Actor 网络：将状态映射到动作空间
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.out(x))


# Critic 网络（TD3 使用双 critic）合并在一个类中
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 第一个 Critic
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out1 = nn.Linear(64, 1)
        # 第二个 Critic
        self.fc3 = nn.Linear(state_dim + action_dim, 64)
        self.fc4 = nn.Linear(64, 64)
        self.out2 = nn.Linear(64, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        # 第一条分支
        x1 = torch.relu(self.fc1(xu))
        x1 = torch.relu(self.fc2(x1))
        q1 = self.out1(x1)
        # 第二条分支
        x2 = torch.relu(self.fc3(xu))
        x2 = torch.relu(self.fc4(x2))
        q2 = self.out2(x2)
        return q1, q2


# 经验回放 Buffer
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


# TD3 代理，结合了双 Critic、目标策略平滑和延迟更新策略
class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
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
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0  # 用于延迟更新计数

        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        # 添加外部探索噪声
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1.0, 1.0)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1)

        with torch.no_grad():
            # 对目标动作添加噪声，并进行 clip 限制
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(
            current_q2, target_q
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新 actor 网络
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic(states, self.actor(states))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )


def main():
    env = PIDEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3Agent(state_dim, action_dim)

    num_episodes = 5000
    max_steps = env.max_steps
    batch_size = 64
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
        if episode % 20 == 0:
            elapsed_time = time.time() - start_time
            print(
                f"Episode {episode}, Reward: {episode_reward:.2f}, Change Step: {env.change_step}, Target1: {env.target1:.2f}, Target2: {env.target2:.2f}, Elapsed Time: {elapsed_time:.2f} seconds"
            )
            start_time = time.time()

    torch.save(agent.actor.state_dict(), "ckpt_td3/0321_1/actor_td3.pth")
    torch.save(agent.critic.state_dict(), "ckpt_td3/0321_1/critic_td3.pth")


if __name__ == "__main__":
    main()
