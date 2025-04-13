import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym # Use Gymnasium (updated Gym)
from gymnasium import spaces

# --- 1. Buck-Boost Converter Environment ---

class BuckBoostEnv(gym.Env):
    """
    Simplified Buck-Boost Converter Environment for PI tuning using RL.

    State: [error, integral_error]
           error = Vref - Vout
    Action: [Kp, Ki] parameters for the PI controller.
            Action space is continuous and normalized to [-1, 1],
            which will be scaled to actual Kp, Ki ranges.
    Reward: Penalizes the absolute error: -abs(error)
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, V_in=12.0, L=1e-4, C=1e-4, R=10.0, V_ref=20.0,
                 dt=1e-5, sim_duration=0.02, render_mode=None):
        super().__init__()

        self.V_in = V_in
        self.L = L
        self.C = C
        self.R = R
        self.V_ref = V_ref
        self.dt = dt
        self.sim_steps = int(sim_duration / dt)
        self.current_step = 0

        # State variables
        self.Vc = 0.0  # Capacitor voltage (negative of output for buck-boost)
        self.IL = 0.0  # Inductor current
        self.integral_error = 0.0
        self.Vout = 0.0 # Actual output voltage Vout = -Vc

        # Action space: [Kp, Ki] - normalized
        # Define reasonable Kp, Ki ranges (example)
        self.kp_min, self.kp_max = 0.01, 1.0
        self.ki_min, self.ki_max = 0.1, 100.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [error, integral_error]
        # Define reasonable bounds (can be adjusted)
        low_obs = np.array([-V_ref * 2, -V_ref * sim_duration * 2], dtype=np.float32)
        high_obs = np.array([V_ref * 2, V_ref * sim_duration * 2], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # For rendering
        self.render_mode = render_mode
        self.history = {'t': [], 'Vout': [], 'Vref': [], 'Kp': [], 'Ki': []}

        # --- PI Controller state ---
        # These are set by the RL agent's action in each step
        self.Kp = self.kp_min
        self.Ki = self.ki_min


    def _get_obs(self):
        error = self.V_ref - self.Vout
        # Clip observation to avoid potential numerical issues if error explodes
        clipped_error = np.clip(error, self.observation_space.low[0], self.observation_space.high[0])
        clipped_integral = np.clip(self.integral_error, self.observation_space.low[1], self.observation_space.high[1])
        return np.array([clipped_error, clipped_integral], dtype=np.float32)

    def _get_info(self):
        # Provides extra info, like current Kp/Ki for debugging
        return {"Kp": self.Kp, "Ki": self.Ki, "Vout": self.Vout}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility

        # Reset state variables
        self.Vc = 0.0
        self.IL = 0.0
        self.integral_error = 0.0
        self.Vout = 0.0
        self.current_step = 0

        # Reset PI parameters to initial guess or minimums
        self.Kp = self.kp_min
        self.Ki = self.ki_min

        # Reset rendering history
        self.history = {'t': [], 'Vout': [], 'Vref': [], 'Kp': [], 'Ki': []}

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _scale_action(self, action):
        # Scale normalized action [-1, 1] to Kp, Ki ranges
        kp_norm = (action[0] + 1.0) / 2.0 # Scale to [0, 1]
        ki_norm = (action[1] + 1.0) / 2.0 # Scale to [0, 1]

        kp = self.kp_min + kp_norm * (self.kp_max - self.kp_min)
        ki = self.ki_min + ki_norm * (self.ki_max - self.ki_min)

        return kp, ki

    def step(self, action):
        # 1. Get Kp, Ki from the agent's action
        self.Kp, self.Ki = self._scale_action(action)

        # 2. Simulate one step of the Buck-Boost converter with PI control
        error = self.V_ref - self.Vout
        self.integral_error += error * self.dt

        # PI control output (duty cycle)
        duty_cycle = self.Kp * error + self.Ki * self.integral_error
        D = np.clip(duty_cycle, 0.01, 0.99) # Clamp duty cycle

        # Simple average model state update (Euler integration)
        # dVc/dt = (IL * (1 - D) - Vc / R) / C
        # dIL/dt = (V_in * D + Vc * (1 - D)) / L # More accurate form
        # dIL/dt = (V_in * D - Vc*(1-D))/L # simplified form ignoring Vc in Vl during D'Ton
        dIL_dt = (self.V_in * D + self.Vc * (1 - D)) / self.L
        dVc_dt = (self.IL * (1 - D) - self.Vc / self.R) / self.C # Vout = -Vc

        self.IL += dIL_dt * self.dt
        self.Vc += dVc_dt * self.dt
        self.Vout = -self.Vc # Buck-boost is inverting

        # 3. Calculate reward
        current_error = self.V_ref - self.Vout
        # Reward: negative absolute error (encourage error minimization)
        # Could add penalties for overshoot, oscillation, settling time etc.
        reward = -np.abs(current_error)
        # Small bonus for being very close?
        # if abs(current_error) < 0.1:
        #     reward += 0.1

        # 4. Check termination
        self.current_step += 1
        terminated = self.current_step >= self.sim_steps
        truncated = False # Could add truncation based on instability

        # 5. Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Store history for rendering
        if self.render_mode is not None or 'render_fps' in self.metadata:
             self.history['t'].append(self.current_step * self.dt)
             self.history['Vout'].append(self.Vout)
             self.history['Vref'].append(self.V_ref)
             self.history['Kp'].append(self.Kp)
             self.history['Ki'].append(self.Ki)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # Human rendering is handled during step/reset

    def _render_frame(self):
        # Simple text-based rendering for human mode
        if self.render_mode == "human":
             print(f"Step: {self.current_step}/{self.sim_steps} Vout: {self.Vout:.3f} Err: {self.V_ref - self.Vout:.3f} Kp: {self.Kp:.3f} Ki: {self.Ki:.2f}")
        elif self.render_mode == "rgb_array":
             # For more complex rendering, could return an image array
             # Requires libraries like Pygame or Matplotlib figure to array conversion
             pass

    def plot_history(self):
        if not self.history['t']:
             print("No history to plot.")
             return

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Vout and Vref
        axs[0].plot(self.history['t'], self.history['Vout'], label='Vout')
        axs[0].plot(self.history['t'], self.history['Vref'], label='Vref', linestyle='--')
        axs[0].set_ylabel('Voltage (V)')
        axs[0].set_title('Buck-Boost Step Response')
        axs[0].legend()
        axs[0].grid(True)

        # Plot Kp and Ki (on twin axes)
        ax2_kp = axs[1]
        ax2_ki = ax2_kp.twinx() # instantiate a second axes that shares the same x-axis

        color_kp = 'tab:red'
        ax2_kp.set_xlabel('Time (s)')
        ax2_kp.set_ylabel('Kp', color=color_kp)
        ax2_kp.plot(self.history['t'], self.history['Kp'], color=color_kp, label='Kp')
        ax2_kp.tick_params(axis='y', labelcolor=color_kp)
        ax2_kp.grid(True, axis='y', linestyle=':', color=color_kp)

        color_ki = 'tab:blue'
        ax2_ki.set_ylabel('Ki', color=color_ki) # we already handled the x-label with ax1
        ax2_ki.plot(self.history['t'], self.history['Ki'], color=color_ki, label='Ki')
        ax2_ki.tick_params(axis='y', labelcolor=color_ki)

        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axs[0].transAxes)
        # fig.tight_layout() # otherwise the right y-label is slightly clipped
        plt.title('PI Parameters During Episode') # Overwrites subplot title if uncommented
        plt.show()


    def close(self):
        # Clean up any resources if needed (e.g., rendering window)
        pass

# --- 2. Replay Buffer ---

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# --- 3. SAC Networks ---

class Actor(nn.Module):
    """ Policy Network π(a|s) """
    def __init__(self, state_dim, action_dim, hidden_dim, action_high, action_low):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_fc = nn.Linear(hidden_dim, action_dim)
        self.log_std_fc = nn.Linear(hidden_dim, action_dim)

        # For scaling the output action
        self.action_high = torch.tensor(action_high, dtype=torch.float32)
        self.action_low = torch.tensor(action_low, dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        # Log std constraints
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            z = mean # Use mean for deterministic actions
        else:
            z = normal.rsample() # Reparameterization trick

        # Apply Tanh squashing and scaling
        action_raw = torch.tanh(z)
        action = action_raw * self.action_scale + self.action_bias # Scale to env limits

        # Calculate log prob with correction for Tanh squashing
        # log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1 - action_raw.pow(2)) + 1e-6) # Correction for scaling
        # Simplified correction term log(1 - tanh(x)^2)
        log_prob = normal.log_prob(z) - torch.log(1 - action_raw.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True) # Sum across action dimension

        # Return scaled action [-1, 1] for SAC internal math, env scales it later
        return action_raw, log_prob

    def get_action_scaled(self, state, device, deterministic=False):
         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
         mean, log_std = self.forward(state_tensor)
         std = log_std.exp()
         normal = Normal(mean, std)
         if deterministic:
             z = mean
         else:
             z = normal.rsample()
         action_raw = torch.tanh(z) # Output is in [-1, 1]
         return action_raw.squeeze(0).detach().cpu().numpy()


class Critic(nn.Module):
    """ Q-Value Network Q(s, a) """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        # Q1 architecture
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        # Q1 path
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)

        # Q2 path
        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        return q1, q2

# --- 4. SAC Agent ---

class SACAgent:
    def __init__(self, state_dim, action_dim, action_high, action_low, hidden_dim=256,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, tau=0.005,
                 buffer_capacity=100000, batch_size=256, device='cpu', learn_alpha=True):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # Actor Network
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_high, action_low).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic Networks (Twin Q)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Initialize target same as main
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Temperature Alpha (Entropy Regularization)
        self.learn_alpha = learn_alpha
        if self.learn_alpha:
            # Target entropy: heuristic target entropy: -|A|
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            # Use log alpha for stability, optimize it
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(0.2, device=device) # Fixed alpha

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_raw, _ = self.actor.sample(state_tensor, deterministic)
        return action_raw.squeeze(0).detach().cpu().numpy() # Return [-1, 1] action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return # Not enough samples yet

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Move batch to device
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device) # Actions are [-1, 1]
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)


        # --- Update Critic ---
        with torch.no_grad():
            # Get next action and log prob from current policy
            next_action_raw, next_log_prob = self.actor.sample(next_state)
            # Get Q values from target critic networks
            q1_target_next, q2_target_next = self.critic_target(next_state, next_action_raw)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            # Calculate target Q value including entropy term
            target_q = reward + (1.0 - done) * self.gamma * (q_target_next - self.alpha * next_log_prob)

        # Get current Q estimates
        q1_current, q2_current = self.critic(state, action)

        # Calculate critic loss (MSE)
        critic_loss_q1 = F.mse_loss(q1_current, target_q)
        critic_loss_q2 = F.mse_loss(q2_current, target_q)
        critic_loss = critic_loss_q1 + critic_loss_q2

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Freeze critic gradients
        for p in self.critic.parameters():
            p.requires_grad = False

        # Get actions and log probs for current state from policy
        pi_action_raw, pi_log_prob = self.actor.sample(state)
        # Get Q values for these actions from the (now frozen) critic
        q1_pi, q2_pi = self.critic(state, pi_action_raw)
        q_pi = torch.min(q1_pi, q2_pi)

        # Calculate actor loss (maximize Q - alpha * log_prob)
        actor_loss = (self.alpha.detach() * pi_log_prob - q_pi).mean() # .detach() alpha here

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic gradients
        for p in self.critic.parameters():
            p.requires_grad = True

        # --- Update Alpha (Temperature) ---
        if self.learn_alpha:
            # Use actions/log_probs computed during actor update
            alpha_loss = -(self.log_alpha.exp() * (pi_log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp() # Update alpha value

        # --- Update Target Networks (Polyak Averaging) ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


    def save_models(self, path_prefix):
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        # Could save optimizers and alpha too if needed for resuming training

    def load_models(self, path_prefix):
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict()) # Ensure target starts same


# --- 5. Training Loop ---

if __name__ == "__main__":
    # Environment Setup
    V_REF_TARGET = 15.0 # Example target voltage
    env = BuckBoostEnv(V_ref=V_REF_TARGET)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high = env.action_space.high # For actor scaling
    action_low = env.action_space.low   # For actor scaling

    # Agent Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    agent = SACAgent(state_dim, action_dim, action_high, action_low, device=DEVICE, learn_alpha=True)

    # Training Hyperparameters
    MAX_EPISODES = 200
    MAX_STEPS_PER_EPISODE = env.sim_steps # Run full simulation length
    UPDATES_PER_STEP = 1 # How many times to call agent.update() per env step
    START_STEPS = 1000 # Number of random steps before starting training
    EVAL_FREQ = 20 # Evaluate every N episodes

    total_steps = 0
    episode_rewards = []

    print("Starting training...")
    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            if total_steps < START_STEPS:
                # Sample random action from the space initially to fill buffer
                action = env.action_space.sample()
            else:
                # Select action from policy
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            # Perform learning updates
            if total_steps > START_STEPS:
                 for _ in range(UPDATES_PER_STEP):
                     agent.update()

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode: {episode}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}, Final Kp: {info['Kp']:.3f}, Final Ki: {info['Ki']:.2f}, Alpha: {agent.alpha.item():.4f}")

        # Evaluate deterministic policy periodically
        if episode % EVAL_FREQ == 0:
            print(f"\n--- Evaluating at Episode {episode} ---")
            eval_env = BuckBoostEnv(V_ref=V_REF_TARGET, render_mode=None) # Separate env for eval
            eval_state, _ = eval_env.reset()
            eval_done = False
            eval_reward_sum = 0
            while not eval_done:
                eval_action = agent.select_action(eval_state, deterministic=True)
                eval_next_state, eval_reward, eval_term, eval_trunc, eval_info = eval_env.step(eval_action)
                eval_done = eval_term or eval_trunc
                eval_state = eval_next_state
                eval_reward_sum += eval_reward
            print(f"Evaluation Reward: {eval_reward_sum:.2f}")
            eval_env.plot_history() # Plot the step response of the evaluation run
            print("--- End Evaluation ---\n")

    print("Training finished.")

    # Save the final agent
    agent.save_models("buckboost_sac")
    print("Models saved.")

    # Plot average rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, MAX_EPISODES + 1), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Episode Rewards during Training")
    plt.grid(True)
    plt.show()

    env.close()