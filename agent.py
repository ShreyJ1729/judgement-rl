import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import random
from state_encoder import StateEncoder


class PolicyNetwork(nn.Module):
    """
    Policy network for PPO agent.
    Outputs action probabilities for both bidding and card playing.
    """

    def __init__(self, state_dim: int, max_action_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.max_action_dim = max_action_dim
        self.hidden_dim = hidden_dim

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Policy head (action probabilities)
        self.policy_head = nn.Linear(hidden_dim // 2, max_action_dim)

        # Value head (state value)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        Returns (action_logits, state_value)
        """
        shared_features = self.shared_layers(state)
        action_logits = self.policy_head(shared_features)
        state_value = self.value_head(shared_features)
        return action_logits, state_value

    def get_action_probs(
        self, state: torch.Tensor, legal_actions: List[int]
    ) -> torch.Tensor:
        """
        Get action probabilities for legal actions only.
        """
        action_logits, _ = self.forward(state)

        # Mask illegal actions
        mask = torch.zeros_like(action_logits)
        for action in legal_actions:
            if action < self.max_action_dim:
                mask[0, action] = 1.0

        # Apply mask and softmax
        masked_logits = action_logits * mask
        # Add large negative value to masked actions to make their probabilities near zero
        masked_logits = masked_logits + (mask - 1) * 1e8

        action_probs = F.softmax(masked_logits, dim=-1)
        return action_probs


class PPOMemory:
    """
    Memory buffer for storing PPO experiences.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.action_probs = deque(maxlen=max_size)
        self.state_values = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.legal_actions = deque(maxlen=max_size)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        action_prob: float,
        state_value: float,
        done: bool,
        legal_actions: List[int],
    ):
        """Add an experience to the memory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.action_probs.append(action_prob)
        self.state_values.append(state_value)
        self.dones.append(done)
        self.legal_actions.append(legal_actions)

    def clear(self):
        """Clear all experiences."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.action_probs.clear()
        self.state_values.clear()
        self.dones.clear()
        self.legal_actions.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for the Judgement card game.
    Implements self-play training.
    """

    def __init__(
        self,
        state_encoder: StateEncoder,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_dim: int = 256,
    ):

        self.state_encoder = state_encoder
        self.state_dim = state_encoder.get_state_dim()
        self.max_action_dim = state_encoder.get_action_dim(state_encoder.max_cards)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Initialize networks
        self.policy_net = PolicyNetwork(self.state_dim, self.max_action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Memory for experience collection
        self.memory = PPOMemory()

        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "total_losses": [],
        }

    def select_action(
        self, state: Dict[str, Any], legal_actions: List[int], epsilon: float = 0.0
    ) -> Tuple[int, float, float]:
        """
        Select an action using the current policy.
        Returns (action, action_prob, state_value)
        """
        # Encode state
        encoded_state = self.state_encoder.encode_state(state)
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0)

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action = random.choice(legal_actions)
            action_prob = 1.0 / len(legal_actions)
            with torch.no_grad():
                _, state_value = self.policy_net(state_tensor)
            return action, action_prob, state_value.item()

        # Get action probabilities from policy
        with torch.no_grad():
            action_probs = self.policy_net.get_action_probs(state_tensor, legal_actions)
            _, state_value = self.policy_net(state_tensor)

        # Sample action from probability distribution
        action_probs_np = action_probs.numpy().flatten()
        action = np.random.choice(len(action_probs_np), p=action_probs_np)

        # Get probability of selected action
        action_prob = action_probs_np[action]

        return action, action_prob, state_value.item()

    def collect_experience(self, env, num_episodes: int = 1, epsilon: float = 0.1):
        """
        Collect experience by playing games.
        Only collects experience from the agent's perspective (player 0).
        Returns episode data for monitoring.
        """
        episode_data = {}

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Get current player
                current_player = env.current_player

                # Get legal actions
                legal_actions = env.get_legal_actions(current_player)

                # Select action
                action, action_prob, state_value = self.select_action(
                    state, legal_actions, epsilon
                )

                # Take step in environment
                next_state, reward, done = env.step(current_player, action)

                # Only store experience if this is the agent's turn (player 0)
                # or if the game is done and we need to store the final reward
                if current_player == 0 or done:
                    encoded_state = self.state_encoder.encode_state(state)
                    encoded_next_state = self.state_encoder.encode_state(next_state)

                    # If the game is done, the reward should be the final reward for player 0
                    if done:
                        # Calculate the actual final reward for player 0
                        final_reward = env._calculate_reward(0)
                        reward = final_reward
                        print(f"Agent (Player 0) final reward: {final_reward}")

                    self.memory.add(
                        encoded_state,
                        action,
                        reward,
                        encoded_next_state,
                        action_prob,
                        state_value,
                        done,
                        legal_actions,
                    )

                    if current_player == 0:
                        episode_reward += reward

                state = next_state

            # Store episode statistics
            self.training_stats["episode_rewards"].append(episode_reward)

            # Collect comprehensive episode data for monitoring
            episode_data = {
                "episode_reward": episode_reward,
                "declarations": (
                    env.declarations.copy() if hasattr(env, "declarations") else []
                ),
                "tricks_won": (
                    env.tricks_won.copy() if hasattr(env, "tricks_won") else []
                ),
                "total_tricks": env.round_cards if hasattr(env, "round_cards") else 0,
                "round_count": 1,  # Each episode is one round
                "trump_suit": env.trump if hasattr(env, "trump") else None,
                "num_players": env.num_players if hasattr(env, "num_players") else 4,
            }

        return episode_data

    def compute_gae_returns(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> List[float]:
        """
        Compute Generalized Advantage Estimation (GAE) returns.
        """
        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])

        return returns

    def train(self, batch_size: int = 64, num_epochs: int = 4):
        """
        Train the agent using collected experience.
        Returns loss data for monitoring.
        """
        if len(self.memory) < batch_size:
            return {}

        # Convert memory to tensors
        states = torch.FloatTensor(np.array(list(self.memory.states)))
        actions = torch.LongTensor(list(self.memory.actions))
        rewards = torch.FloatTensor(list(self.memory.rewards))
        next_states = torch.FloatTensor(np.array(list(self.memory.next_states)))
        old_action_probs = torch.FloatTensor(list(self.memory.action_probs))
        old_values = torch.FloatTensor(list(self.memory.state_values))
        dones = torch.BoolTensor(list(self.memory.dones))
        legal_actions_list = list(self.memory.legal_actions)

        # Compute GAE returns
        returns = self.compute_gae_returns(
            list(self.memory.rewards),
            list(self.memory.state_values),
            list(self.memory.dones),
        )
        returns = torch.FloatTensor(returns)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training loop
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0
        avg_total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            # Create batches
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_action_probs = old_action_probs[batch_indices]
                batch_legal_actions = [legal_actions_list[i] for i in batch_indices]

                # Forward pass
                action_logits, values = self.policy_net(batch_states)

                # Compute action probabilities for legal actions
                action_probs = []
                for i, legal_actions in enumerate(batch_legal_actions):
                    mask = torch.zeros_like(action_logits[i])
                    for action in legal_actions:
                        if action < self.max_action_dim:
                            mask[action] = 1.0

                    masked_logits = action_logits[i] * mask
                    masked_logits = masked_logits + (mask - 1) * 1e8
                    probs = F.softmax(masked_logits, dim=-1)
                    action_probs.append(probs)

                action_probs = torch.stack(action_probs)

                # Get probabilities of taken actions
                action_probs_taken = action_probs.gather(
                    1, batch_actions.unsqueeze(1)
                ).squeeze(1)

                # Compute policy loss (PPO clipped objective)
                ratio = action_probs_taken / (batch_old_action_probs + 1e-8)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)

                # Compute entropy loss (for exploration)
                entropy = (
                    -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
                )
                entropy_loss = -entropy

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Store losses
                self.training_stats["policy_losses"].append(policy_loss.item())
                self.training_stats["value_losses"].append(value_loss.item())
                self.training_stats["entropy_losses"].append(entropy_loss.item())
                self.training_stats["total_losses"].append(total_loss.item())

                # Accumulate for averaging
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy_loss += entropy_loss.item()
                avg_total_loss += total_loss.item()
                num_batches += 1

        # Clear memory after training

        # Return average loss data for monitoring
        if num_batches > 0:
            loss_data = {
                "policy_loss": avg_policy_loss / num_batches,
                "value_loss": avg_value_loss / num_batches,
                "entropy_loss": avg_entropy_loss / num_batches,
                "total_loss": avg_total_loss / num_batches,
            }
        else:
            loss_data = {}

        self.memory.clear()
        return loss_data

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_stats": self.training_stats,
                "state_encoder_config": {
                    "num_players": self.state_encoder.num_players,
                    "max_cards": self.state_encoder.max_cards,
                },
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]

    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics."""
        return self.training_stats.copy()


class SelfPlayTrainer:
    """
    Trainer for self-play PPO training.
    """

    def __init__(
        self,
        env,
        state_encoder: StateEncoder,
        num_agents: int = 4,
        learning_rate: float = 3e-4,
    ):
        self.env = env
        self.state_encoder = state_encoder
        self.num_agents = num_agents
        self.learning_rate = learning_rate

        # Create agents (all start with same policy)
        self.agents = [
            PPOAgent(state_encoder, learning_rate) for _ in range(num_agents)
        ]
        self.current_best_agent = 0

        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "agent_performances": [],
            "policy_improvements": [],
        }

    def train_episode(self, epsilon: float = 0.1):
        """
        Train for one episode using self-play.
        """
        # Reset environment
        state = self.env.reset()
        episode_rewards = [0] * self.num_agents
        done = False

        while not done:
            current_player = self.env.current_player
            legal_actions = self.env.get_legal_actions(current_player)

            # Select action using current player's agent
            agent = self.agents[current_player]
            action, action_prob, state_value = agent.select_action(
                state, legal_actions, epsilon
            )

            # Take step
            next_state, reward, done = self.env.step(current_player, action)

            # Store experience for current player's agent
            encoded_state = self.state_encoder.encode_state(state)
            encoded_next_state = self.state_encoder.encode_state(next_state)

            # If the game is done, calculate the final reward for this player
            if done:
                final_reward = self.env._calculate_reward(current_player)
                reward = final_reward
                print(f"Player {current_player} final reward: {final_reward}")

            agent.memory.add(
                encoded_state,
                action,
                reward,
                encoded_next_state,
                action_prob,
                state_value,
                done,
                legal_actions,
            )

            episode_rewards[current_player] += reward
            state = next_state

        # Store episode statistics
        self.training_stats["episode_rewards"].append(sum(episode_rewards))
        self.training_stats["agent_performances"].append(episode_rewards)

        return episode_rewards

    def train(
        self,
        num_episodes: int,
        episodes_per_update: int = 10,
        epsilon: float = 0.1,
        batch_size: int = 64,
    ):
        """
        Train using self-play for multiple episodes.
        """
        for episode in range(num_episodes):
            # Train one episode
            episode_rewards = self.train_episode(epsilon)

            # Update policies every few episodes
            if (episode + 1) % episodes_per_update == 0:
                for agent in self.agents:
                    agent.train(batch_size=batch_size)

                # Update best agent based on recent performance
                recent_performances = self.training_stats["agent_performances"][
                    -episodes_per_update:
                ]
                avg_performances = [
                    np.mean([perf[i] for perf in recent_performances])
                    for i in range(self.num_agents)
                ]
                self.current_best_agent = np.argmax(avg_performances)

                # Copy best agent's policy to other agents (with some noise)
                best_agent = self.agents[self.current_best_agent]
                for i, agent in enumerate(self.agents):
                    if i != self.current_best_agent:
                        # Copy policy weights with small random noise
                        for param, best_param in zip(
                            agent.policy_net.parameters(),
                            best_agent.policy_net.parameters(),
                        ):
                            param.data = (
                                best_param.data + torch.randn_like(param.data) * 0.01
                            )

                print(
                    f"Episode {episode + 1}: Best agent {self.current_best_agent}, "
                    f"Avg reward: {np.mean(episode_rewards):.2f}"
                )

    def save_best_agent(self, filepath: str):
        """Save the best performing agent."""
        self.agents[self.current_best_agent].save_model(filepath)

    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics."""
        return self.training_stats.copy()
