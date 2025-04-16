import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import copy
import argparse # 导入 argparse

# --- 环境定义 ---
class BuckBoostEnv:
    """
    简化的 Buck-Boost 变换器环境模拟 (离散时间)
    用于测试 PI 控制器参数调整
    """
    def __init__(self, Vref=12.0, Vin=5.0, L=1e-4, C=1e-4, R=10.0, dt=1e-5, max_steps=500):
        self.Vref = Vref
        self.Vin = Vin
        self.L = L
        self.C = C
        self.R = R
        self.dt = dt

        self.Vout = 0.0
        self.IL = 0.0
        self.error_integral = 0.0
        self.error = 0.0

        self.max_steps = max_steps
        self.current_step = 0

        self.state_dim = 3
        self.action_dim = 2
        # --- 修改: 扩大参数范围 ---
        self.action_low = np.array([0.001, 1.0])  # Kp_min, Ki_min
        self.action_high = np.array([0.1, 1000.0]) # Kp_max, Ki_max (扩大范围)
        # --------------------------
        print(f"Action range set to: Kp [{self.action_low[0]}, {self.action_high[0]}], Ki [{self.action_low[1]}, {self.action_high[1]}]")


    def reset(self):
        """重置环境状态"""
        self.Vout = 0.0
        self.IL = 0.0
        self.error_integral = 0.0
        self.error = self.Vref - self.Vout
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        """获取当前状态"""
        return np.array([self.error, self.error_integral, self.Vout])

    def step(self, action):
        """执行一个时间步"""
        Kp, Ki = action
        # 使用实例变量 self.action_low/high 进行 clip
        Kp = np.clip(Kp, self.action_low[0], self.action_high[0])
        Ki = np.clip(Ki, self.action_low[1], self.action_high[1])

        self.current_step += 1

        self.error = self.Vref - self.Vout
        self.error_integral += self.error * self.dt

        duty_cycle = Kp * self.error + Ki * self.error_integral
        duty_cycle = np.clip(duty_cycle, 0.01, 0.99)

        dIL_on = (self.Vin / self.L) * self.dt * duty_cycle
        dIL_off = (-self.Vout / self.L) * self.dt * (1 - duty_cycle) if self.Vout > 0 else 0
        self.IL += dIL_on + dIL_off
        self.IL = max(0, self.IL)

        dVout = ((self.IL / self.C) - (self.Vout / (self.R * self.C))) * self.dt
        self.Vout += dVout
        self.Vout = max(0, self.Vout)

        reward = -abs(self.error)
        overshoot = max(0, self.Vout - self.Vref * 1.05)
        reward -= overshoot * 5.0

        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        return next_state, reward, done, {'duty_cycle': duty_cycle}

    def get_trajectory(self, Kp, Ki, steps):
        """使用给定的 PI 参数模拟轨迹"""
        self.reset()
        Vout_history = [self.Vout]
        time_history = [0.0]
        action = np.array([Kp, Ki])
        for i in range(steps):
            _, _, done, _ = self.step(action)
            Vout_history.append(self.Vout)
            time_history.append(self.dt * (i + 1))
            if done:
                break
        return time_history, Vout_history


# --- Replay Buffer ---
# (Replay Buffer 代码保持不变)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# --- 网络定义 ---
# (Actor 和 Critic 网络代码保持不变, Actor 会自动使用 env 传入的新范围)
class Actor(nn.Module):
    """策略网络，输出动作 (Kp, Ki)"""
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_dim=256):
        super(Actor, self).__init__()
        # 将 numpy array 转换为 tensor
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        # 确保在同一设备上计算 scale 和 bias
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # 获取当前设备
        device = self.mean_layer.weight.device
        # 确保 scale 和 bias 在同一设备
        action_scale_dev = self.action_scale.to(device)
        action_bias_dev = self.action_bias.to(device)

        action = y_t * action_scale_dev + action_bias_dev

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(action_scale_dev * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * action_scale_dev + action_bias_dev
        return action, log_prob, mean_action

    def to(self, device):
        """将模型和内部张量移动到指定设备"""
        # action_low/high 本身不需要移动，因为它们在 sample 中会根据网络参数的 device 被移动
        # 但 scale 和 bias 需要在初始化时就确定设备，或者在 sample 中动态移动
        # 这里选择在 sample 中动态移动
        return super().to(device)


class Critic(nn.Module):
    """Q 值网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        return q1, q2


# --- SAC Agent ---
# (SACAgent 代码保持不变)
class SACAgent:
    def __init__(self, env, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4,
                 hidden_dim=256, buffer_capacity=100000, batch_size=256,
                 auto_entropy_tuning=True, target_update_interval=1):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        # 从 env 获取 action bounds
        self.action_low = env.action_low
        self.action_high = env.action_high

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha # Initial alpha value
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.auto_entropy_tuning = auto_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks - Pass action bounds to Actor
        self.actor = Actor(self.state_dim, self.action_dim, self.action_low, self.action_high, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Entropy Tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([self.env.action_dim]).to(self.device)).item() # Use float tensor
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item() # Update alpha based on initial log_alpha
        # else: self.alpha remains the initial value passed

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.train_step_count = 0 # Initialize training step counter

    def select_action(self, state, evaluate=False):
        """Selects an action based on the current state"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not evaluate:
            action, _, _ = self.actor.sample(state) # Sample action for exploration/training
        else:
            _, _, action = self.actor.sample(state) # Use mean action for evaluation
        return action.detach().cpu().numpy()[0]

    def update(self):
        """Updates the agent's networks"""
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough samples to update

        self.train_step_count += 1 # Increment update counter
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # --- Update Critic ---
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state_batch)
            q1_target_next, q2_target_next = self.critic_target(next_state_batch, next_action)
            min_q_target_next = torch.min(q1_target_next, q2_target_next) - self.alpha * next_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_q_target_next

        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        pi, log_pi, _ = self.actor.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Alpha ---
        if self.auto_entropy_tuning:
            # Detach log_pi here as per SAC algorithm
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item() # Update alpha value

        # --- Soft update target networks ---
        if self.train_step_count % self.target_update_interval == 0:
            self._soft_update(self.critic_target, self.critic, self.tau)

        # Return losses and alpha for logging
        return critic_loss.item(), actor_loss.item(), self.alpha


    def _soft_update(self, target, source, tau):
        """Performs soft update of target network parameters"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_checkpoint(self, filename="sac_checkpoint.pth"):
        """Saves the agent's state"""
        print(f"--- Saving checkpoint to {filename} ---")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning and hasattr(self, 'alpha_optimizer') else None,
            'train_step_count': self.train_step_count,
            'alpha': self.alpha # Save current alpha value
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename="sac_checkpoint.pth"):
        """Loads the agent's state from a checkpoint"""
        if os.path.isfile(filename):
            print(f"--- Loading checkpoint from {filename} ---")
            try:
                checkpoint = torch.load(filename, map_location=self.device)

                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

                if self.auto_entropy_tuning:
                    if 'log_alpha' in checkpoint and checkpoint['log_alpha'] is not None:
                        self.log_alpha.data = checkpoint['log_alpha'].to(self.device).data # Load log_alpha data
                        # Re-initialize optimizer only if state dict exists
                        if 'alpha_optimizer_state_dict' in checkpoint and checkpoint['alpha_optimizer_state_dict'] is not None:
                             self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                        self.alpha = self.log_alpha.exp().item() # Update alpha from loaded log_alpha
                    else:
                        print("Warning: Checkpoint missing log_alpha for auto entropy tuning. Using default.")
                        # Keep the initialized log_alpha and optimizer
                        self.alpha = self.log_alpha.exp().item()
                elif 'alpha' in checkpoint: # Use fixed alpha from checkpoint if not auto-tuning
                     self.alpha = checkpoint['alpha']

                self.train_step_count = checkpoint.get('train_step_count', 0) # Load step count

                # Ensure target networks are correctly loaded (hard update)
                self._soft_update(self.critic_target, self.critic, tau=1.0)

                print(f"--- Checkpoint loaded successfully. Resuming from step {self.train_step_count}, Alpha: {self.alpha:.4f} ---")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with uninitialized agent.")
                return False
        else:
            print(f"--- No checkpoint found at {filename} ---")
            return False

# --- 训练和测试 ---
# (train 函数保持不变)
def train(env, agent, max_episodes=100, max_steps=500, update_every=1, start_steps=1000, checkpoint_interval=50, checkpoint_dir="checkpoints", initial_episode=0):
    """训练 SAC 代理"""
    episode_rewards = []
    total_steps = agent.train_step_count # Start from loaded step count

    # Adjust episode loop to start from initial_episode
    for episode in range(initial_episode, max_episodes):
        state = env.reset()
        episode_reward = 0
        critic_losses = []
        actor_losses = []
        alphas = []

        for step in range(max_steps):
            # Use total_steps for start_steps comparison
            if total_steps < start_steps:
                action = env.action_low + np.random.rand(env.action_dim) * (env.action_high - env.action_low)
            else:
                action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)
            action = np.clip(action, env.action_low, env.action_high)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1 # Increment total steps

            # Update based on total_steps and update_every
            if total_steps >= start_steps and len(agent.replay_buffer) >= agent.batch_size and total_steps % update_every == 0:
               update_results = agent.update()
               if update_results:
                   c_loss, a_loss, alpha_val = update_results
                   critic_losses.append(c_loss)
                   actor_losses.append(a_loss)
                   alphas.append(alpha_val)

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_c_loss = np.mean(critic_losses) if critic_losses else 0
        avg_a_loss = np.mean(actor_losses) if actor_losses else 0
        avg_alpha = np.mean(alphas) if alphas else agent.alpha # Use current agent alpha if no updates

        print(f"Episode: {episode+1}/{max_episodes}, Steps: {step+1}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}, Avg Critic Loss: {avg_c_loss:.4f}, Avg Actor Loss: {avg_a_loss:.4f}, Alpha: {avg_alpha:.4f}")

        # Save checkpoint periodically
        if (episode + 1) % checkpoint_interval == 0:
            agent.save_checkpoint(os.path.join(checkpoint_dir, f"sac_checkpoint_ep{episode+1}.pth"))

    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "sac_final.pth")
    agent.save_checkpoint(final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    return episode_rewards, final_checkpoint_path # Return path to final model

# (test_agent 函数保持不变)
def test_agent(env, agent, num_steps=500, checkpoint_path=None, fixed_Kp=0.5, fixed_Ki=30):
    """测试训练好的代理，并绘制 Vout, PI 参数和占空比, 同时对比固定 PI 控制器"""
    agent_loaded = False
    if checkpoint_path:
        # test_agent should only load if the agent wasn't already loaded externally
        # However, loading again ensures the correct state is tested.
        if agent.load_checkpoint(checkpoint_path):
            agent_loaded = True
            image_save_dir = os.path.join(os.path.dirname(checkpoint_path), "result.png")
        else:
            print("Could not load checkpoint specified for testing.")
            return # Exit if testing requires a specific checkpoint that failed to load
    else:
        print("No specific checkpoint path provided for testing, using agent's current state.")
        # Assume agent is ready if no path is given (e.g., after training)
        agent_loaded = True # Or check if agent has been trained/loaded before calling test_agent

    # --- SAC Agent Simulation ---
    state = env.reset()
    Vout_history_agent = [env.Vout]
    time_history_agent = [0.0]
    Kp_history_agent = []
    Ki_history_agent = []
    duty_cycle_history_agent = []
    total_reward_agent = 0

    if agent_loaded:
        agent.actor.eval()
        agent.critic.eval()

        initial_action = agent.select_action(state, evaluate=True)
        Kp_history_agent.append(initial_action[0])
        Ki_history_agent.append(initial_action[1])
        initial_duty = initial_action[0] * env.error + initial_action[1] * env.error_integral
        initial_duty = np.clip(initial_duty, 0.01, 0.99)
        duty_cycle_history_agent.append(initial_duty)

        with torch.no_grad():
            for i in range(num_steps):
                action = agent.select_action(state, evaluate=True)
                Kp, Ki = action
                next_state, reward, done, info = env.step(action)
                current_duty_cycle = info.get('duty_cycle', np.nan)

                state = next_state
                total_reward_agent += reward
                Vout_history_agent.append(env.Vout)
                time_history_agent.append(env.dt * (i + 1))
                Kp_history_agent.append(Kp)
                Ki_history_agent.append(Ki)
                duty_cycle_history_agent.append(current_duty_cycle)

                if done:
                    break
        agent.actor.train()
        agent.critic.train()

        print(f"\n--- SAC Agent Testing Finished ---")
        print(f"Total Steps Simulated: {len(time_history_agent) - 1}")
        print(f"Total Reward: {total_reward_agent:.2f}")
        final_Kp_agent = Kp_history_agent[-1] if Kp_history_agent else np.nan
        final_Ki_agent = Ki_history_agent[-1] if Ki_history_agent else np.nan
        print(f"Final PI Parameters (Agent approx): Kp={final_Kp_agent:.4f}, Ki={final_Ki_agent:.4f}")

    # --- Fixed PI Controller Simulation ---
    print(f"\n--- Simulating Fixed PI Controller (Kp={fixed_Kp}, Ki={fixed_Ki}) ---")
    time_history_fixed, Vout_history_fixed = env.get_trajectory(fixed_Kp, fixed_Ki, steps=num_steps)
    print(f"Fixed PI Simulation Finished. Steps: {len(time_history_fixed) - 1}")


    # --- 绘图 ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 子图 1: 输出电压 Vout
    if agent_loaded:
        axs[0].plot(time_history_agent, Vout_history_agent, label='SAC Agent Vout', color='blue', linewidth=2)
    axs[0].plot(time_history_fixed, Vout_history_fixed, label=f'Fixed PI (Kp={fixed_Kp:.2f}, Ki={fixed_Ki:.1f})', color='grey', linestyle='-.') # Format numbers in label
    axs[0].axhline(y=env.Vref, color='r', linestyle='--', label=f'Reference Voltage (Vref={env.Vref}V)')
    axs[0].set_ylabel('Output Voltage (V)')
    axs[0].set_title('Buck-Boost Step Response Comparison: SAC Agent vs Fixed PI')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_ylim(bottom=-1)

    # 子图 2: PI 参数
    if agent_loaded:
        line_kp_agent, = axs[1].plot(time_history_agent, Kp_history_agent, label='SAC Agent Kp', color='green')
        ax1_twin = axs[1].twinx() # Create twin axis only if agent data exists
        line_ki_agent, = ax1_twin.plot(time_history_agent, Ki_history_agent, label='SAC Agent Ki', color='purple', linestyle=':')
        axs[1].set_ylabel('Kp Gain (Agent)', color='green')
        ax1_twin.set_ylabel('Ki Gain (Agent)', color='purple')
        line_kp_fixed = axs[1].axhline(y=fixed_Kp, color='lime', linestyle='--', label=f'Fixed Kp={fixed_Kp:.2f}')
        line_ki_fixed = ax1_twin.axhline(y=fixed_Ki, color='magenta', linestyle='--', label=f'Fixed Ki={fixed_Ki:.1f}')
        lines1 = [line_kp_agent, line_kp_fixed]
        labels1 = [l.get_label() for l in lines1]
        axs[1].legend(lines1, labels1, loc='center left')
        lines2 = [line_ki_agent, line_ki_fixed]
        labels2 = [l.get_label() for l in lines2]
        ax1_twin.legend(lines2, labels2, loc='center right')
    else: # Only plot fixed lines if no agent data
        ax1_twin = axs[1].twinx() # Still need twin axis for Ki line
        line_kp_fixed = axs[1].axhline(y=fixed_Kp, color='lime', linestyle='--', label=f'Fixed Kp={fixed_Kp:.2f}')
        line_ki_fixed = ax1_twin.axhline(y=fixed_Ki, color='magenta', linestyle='--', label=f'Fixed Ki={fixed_Ki:.1f}')
        axs[1].set_ylabel('Kp Gain')
        ax1_twin.set_ylabel('Ki Gain')
        axs[1].legend([line_kp_fixed], [line_kp_fixed.get_label()], loc='center left')
        ax1_twin.legend([line_ki_fixed], [line_ki_fixed.get_label()], loc='center right')

    axs[1].set_title('PI Parameters (SAC Agent vs Fixed)')
    axs[1].grid(True)


    # 子图 3: 占空比 Duty Cycle
    if agent_loaded:
        axs[2].plot(time_history_agent, duty_cycle_history_agent, label='SAC Agent Duty Cycle', color='orange')
        axs[2].set_title('Duty Cycle (SAC Agent)')
        axs[2].set_ylabel('Duty Cycle')
        axs[2].set_ylim(0, 1)
        axs[2].legend()
    else:
        axs[2].set_title('Duty Cycle (No Agent Data)')
        axs[2].text(0.5, 0.5, 'No SAC Agent data to display', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)

    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(image_save_dir)

# (plot_rewards 函数保持不变)
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Episode Rewards during Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()


# --- 主程序 (重构以使用 argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test SAC agent for Buck-Boost PI tuning.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'continue', 'test'],
                        help='Execution mode: train from scratch, continue training, or test.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint file for continue or test mode, or to start training from.')
    parser.add_argument('--episodes', type=int, default=150,
                        help='Number of episodes to train.')
    parser.add_argument('--max_steps', type=int, default=600,
                        help='Maximum steps per episode/test.')
    parser.add_argument('--test_steps', type=int, default=None,
                        help='Number of steps for testing simulation (defaults to max_steps).')
    parser.add_argument('--fixed_kp', type=float, default=0.5,
                        help='Fixed Kp value for comparison during testing.')
    parser.add_argument('--fixed_ki', type=float, default=30.0,
                        help='Fixed Ki value for comparison during testing.')
    parser.add_argument('--checkpoint_dir', type=str, default="buck_boost_sac_checkpoints",
                        help='Directory to save/load checkpoints.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for networks.')
    parser.add_argument('--buffer_capacity', type=int, default=150000, help='Replay buffer capacity.')
    parser.add_argument('--start_steps', type=int, default=2000, help='Number of random exploration steps before training.')
    parser.add_argument('--checkpoint_interval', type=int, default=25, help='Save checkpoint every N episodes.')
    # 添加环境参数
    parser.add_argument('--vref', type=float, default=15.0, help='Reference voltage.')
    parser.add_argument('--vin', type=float, default=8.0, help='Input voltage.')
    parser.add_argument('--dt', type=float, default=5e-6, help='Simulation time step.')


    args = parser.parse_args()

    # --- 初始化环境 ---
    # 使用命令行参数初始化环境
    env = BuckBoostEnv(Vref=args.vref, Vin=args.vin, dt=args.dt, max_steps=args.max_steps)

    # --- 初始化 Agent ---
    # 使用命令行参数初始化 Agent
    agent = SACAgent(env, gamma=0.99, tau=0.005, alpha=0.2, lr=args.lr,
                     hidden_dim=args.hidden_dim, buffer_capacity=args.buffer_capacity,
                     batch_size=args.batch_size, auto_entropy_tuning=True,
                     target_update_interval=1) # target_update_interval=1 is common

    # --- 执行逻辑 ---
    initial_episode = 0 # 用于继续训练
    final_model_path = None # 存储最终训练模型的路径

    if args.mode == 'continue' or (args.mode == 'train' and args.checkpoint_path and os.path.isfile(args.checkpoint_path)):
        if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
            if agent.load_checkpoint(args.checkpoint_path):
                 # 尝试从检查点文件名中提取回合数以继续计数 (可选)
                 try:
                     # 假设文件名格式为 "...ep<number>.pth"
                     base = os.path.basename(args.checkpoint_path)
                     num_str = base.split('ep')[-1].split('.pth')[0]
                     initial_episode = int(num_str)
                     print(f"Attempting to continue training from episode {initial_episode + 1}")
                 except ValueError:
                     print("Could not parse episode number from checkpoint filename. Starting episode count from loaded step.")
                     # 可以根据 agent.train_step_count 估算回合数，但从 0 开始更简单
                     initial_episode = 0 # 或者不修改，让 train 函数的循环从 0 开始，但总步数是连续的
            else:
                 print(f"Error: Failed to load checkpoint {args.checkpoint_path} for mode '{args.mode}'. Exiting.")
                 exit(1) # 无法加载检查点则退出
        elif args.mode == 'continue':
             print(f"Error: Checkpoint path '{args.checkpoint_path}' not found or not specified for 'continue' mode. Exiting.")
             exit(1) # continue 模式必须提供有效检查点

    if args.mode == 'train' or args.mode == 'continue':
        print(f"--- Starting/Continuing Training (up to {args.episodes} episodes) ---")
        rewards, final_model_path = train(env, agent, max_episodes=args.episodes, max_steps=args.max_steps,
                                          update_every=1, start_steps=args.start_steps,
                                          checkpoint_interval=args.checkpoint_interval,
                                          checkpoint_dir=args.checkpoint_dir,
                                          initial_episode=initial_episode) # 传递起始回合
        print("--- Training Finished ---")
        if rewards:
            plot_rewards(rewards)
        # 训练结束后，自动进行测试
        print("\n--- Proceeding to Test Phase after Training ---")
        test_steps = args.test_steps if args.test_steps is not None else args.max_steps
        test_agent(env, agent,
                   num_steps=test_steps,
                   checkpoint_path=final_model_path, # 测试最终训练好的模型
                   fixed_Kp=args.fixed_kp,
                   fixed_Ki=args.fixed_ki)

    elif args.mode == 'test':
        if not args.checkpoint_path or not os.path.isfile(args.checkpoint_path):
            print(f"Error: Checkpoint path '{args.checkpoint_path}' not found or not specified for 'test' mode. Exiting.")
            exit(1)
        print(f"--- Starting Testing Phase Only ---")
        test_steps = args.test_steps if args.test_steps is not None else args.max_steps
        # test_agent 会加载指定的 checkpoint_path
        test_agent(env, agent,
                   num_steps=test_steps,
                   checkpoint_path=args.checkpoint_path,
                   fixed_Kp=args.fixed_kp,
                   fixed_Ki=args.fixed_ki)

    else:
        print(f"Error: Unknown mode '{args.mode}'.")
        exit(1)

