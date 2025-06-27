# AI Player for the Judgement Card Game

Welcome to the "Judgement AI" project! This repository will guide you through building an AI player for the "Judgement" card game using reinforcement learning (RL). The AI will learn to bid accurately and play cards strategically to maximize its score based on the game's rules.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Step 1: Set Up the Environment](#step-1-set-up-the-environment)
  - [Step 2: Implement the Game Logic](#step-2-implement-the-game-logic)
  - [Step 3: Define the State Representation](#step-3-define-the-state-representation)
  - [Step 4: Define the Action Space](#step-4-define-the-action-space)
  - [Step 5: Design the Reward System](#step-5-design-the-reward-system)
  - [Step 6: Implement the RL Agent](#step-6-implement-the-rl-agent)
  - [Step 7: Train the Agent](#step-7-train-the-agent)
  - [Step 8: Evaluate the Agent](#step-8-evaluate-the-agent)
- [Real-time Monitoring](#real-time-monitoring)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Resources](#resources)

## Introduction

"Judgement" is a trick-taking card game where players bid on the number of tricks they aim to win in each round and play cards to meet their bids. The game features unique mechanics:

- **Rounds**: Card counts vary (e.g., 7, 6, 5, ..., 1, then 2, 3, ..., 7).
- **Trump Suits**: Rotate through No Trump, Spades, Diamonds, Clubs, Hearts.
- **Bidding**: Total bids must not equal the number of players.
- **Scoring**: Players score highly for meeting bids exactly, with penalties otherwise.

The goal of this project is to create an AI that learns optimal strategies for bidding and gameplay using RL with self-play training.

## Prerequisites

To get started, you'll need:

- **Python Knowledge**: Familiarity with Python programming.
- **Reinforcement Learning Basics**: Understanding of states, actions, and rewards.
- **Deep Learning Frameworks**: Experience with PyTorch is helpful.
- **Tools**: Git, a Python IDE (e.g., VSCode), and a virtual environment.

## Project Structure

Here's how the project is organized:

- `judgement_env.py`: Game environment and logic.
- `state_encoder.py`: State encoding and representation.
- `agent.py`: PPO agent implementation with self-play capabilities.
- `train.py`: Training script with self-play and real-time monitoring.
- `config.py`: Configuration file for all hyperparameters and training variables.
- `evaluate.py`: Standalone evaluation script for testing agents.
- `realtime_monitor.py`: Real-time training monitoring with live updating graphs.
- `demo_monitoring.py`: Demo script showing how to use the monitoring system.
- `heuristic_agent.py`: Simple heuristic-based agent for comparison.
- `demo_ppo.py`: Demonstration script for trained agents.
- `test_implementation.py`: Comprehensive testing suite.
- `models/`: Directory for saving trained models.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Step-by-Step Guide

### Step 1: Set Up the Environment

Let's set up your development environment.

**Clone the Repository:**

```bash
git clone https://github.com/your-repo/judgement-ai.git
cd judgement-ai
```

**Create a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

### Step 2: Implement the Game Logic

The game logic lives in `judgement_env.py`. You'll create a class to manage the game state and rules.

**Key Features:**

- Deal cards based on the round.
- Rotate trump suits.
- Handle bidding and trick-taking.

**Code Example:**

```python
import random

class JudgementEnv:
    def __init__(self, num_players=4, max_cards=7):
        self.num_players = num_players
        self.max_cards = max_cards
        self.trump_suits = ["No Trump", "Spades", "Diamonds", "Clubs", "Hearts"]
        self.reset()

    def reset(self):
        self.round_cards = random.randint(1, self.max_cards)  # Simplify for now
        self.trump = self.trump_suits[0]  # Rotate in full implementation
        self.deck = self._create_deck()
        self.hands = self._deal_cards()
        self.declarations = [0] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        return self._get_state(0)

    def _create_deck(self):
        suits = ["Spades", "Diamonds", "Clubs", "Hearts"]
        ranks = [str(i) for i in range(2, 11)] + ["J", "Q", "K", "A"]
        return [f"{rank} of {suit}" for suit in suits for rank in ranks]

    def _deal_cards(self):
        deck = self.deck.copy()
        random.shuffle(deck)
        return [deck[i::self.num_players][:self.round_cards] for i in range(self.num_players)]

    def _get_state(self, player_idx):
        state = {
            "hand": self.hands[player_idx],
            "trump": self.trump,
            "declarations": self.declarations.copy(),
            "current_trick": self.current_trick.copy(),
            "tricks_won": self.tricks_won.copy()
        }
        if self.round_cards == 1:
            state["others_cards"] = [self.hands[i][0] for i in range(self.num_players) if i != player_idx]
        return state
```

**Tasks:**

- Add a step method to handle bidding and card play.
- Implement trick resolution (e.g., highest trump or suit wins).

### Step 3: Define the State Representation

The state tells the AI what's happening in the game. Include:

- Player's hand.
- Trump suit.
- All players' bids.
- Current trick and trick history.
- Visible cards in the one-card round.

**Code Example:** See the `_get_state` method above. You'll need to encode this into a numerical format (e.g., one-hot encoding) for the RL agent later.

### Step 4: Define the Action Space

The AI has two decision types:

- **Bidding**: Choose a number from 0 to `round_cards`.
- **Playing**: Select a legal card from the hand.

**Code Example:**

```python
def step(self, player_idx, action):
    if len(self.declarations) < self.num_players:
        self.declarations[player_idx] = action
        return self._get_state((player_idx + 1) % self.num_players), 0, False
    else:
        card = self.hands[player_idx].pop(action)
        self.current_trick.append((player_idx, card))  # Add trick resolution logic here
        return self._get_state((player_idx + 1) % self.num_players), 0, False
```

**Tasks:**

- Enforce bidding rules (e.g., last player can't make total bids equal number of players).
- Filter legal card plays (e.g., must follow suit).

### Step 5: Design the Reward System

Rewards guide the AI's learning:

- **End of Round**: Score based on meeting the bid (e.g., 11n + 10 if exact, -10|n - actual| otherwise).
- **During Play**: Small rewards for winning tricks up to the bid.

**Code Example:**

```python
def _calculate_reward(self, player_idx):
    n = self.declarations[player_idx]
    actual = self.tricks_won[player_idx]
    if n == actual:
        return 11 * n + 10
    return -10 * abs(n - actual)
```

**Tasks:**

- Add intermediate rewards in step.
- Test the reward function with sample games.

### Step 6: Implement the RL Agent

We'll use a PPO (Proximal Policy Optimization) agent in `agent.py`.

**Code Example:**

```python
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            q_values = self(state)
            return q_values.argmax().item()
```

**Tasks:**

- Define `state_size` and `action_size` based on your state and action encodings.
- Add a replay buffer and training logic.

### Step 7: Train the Agent

Create a training script in `train.py` that implements PPO with self-play capabilities.

**Code Example:**

```python
import torch
import numpy as np
from judgement_env import JudgementEnv
from state_encoder import StateEncoder
from agent import PPOAgent, SelfPlayTrainer
from config import TrainingConfig

# Initialize environment and encoder
env = JudgementEnv(num_players=4, max_cards=7)
state_encoder = StateEncoder(num_players=4, max_cards=7)

# Initialize self-play trainer
trainer = SelfPlayTrainer(
    env=env,
    state_encoder=state_encoder,
    num_agents=4,
    learning_rate=3e-4
)

# Training configuration
config = TrainingConfig(
    num_episodes=1000,
    episodes_per_update=10,
    batch_size=64,
    epsilon=0.1
)

# Training loop
trainer.train(
    num_episodes=config.num_episodes,
    episodes_per_update=config.episodes_per_update,
    epsilon=config.epsilon,
    batch_size=config.batch_size,
)
```

**Tasks:**

- Implement experience collection with GAE (Generalized Advantage Estimation)
- Add policy and value loss calculations with PPO clipping
- Include entropy regularization for exploration
- Implement self-play training for better opponent modeling
- Add training statistics and visualization
- Save and load trained models

### Step 8: Evaluate the Agent

Test the AI's performance using comprehensive evaluation metrics.

**Code Example:**

```python
def evaluate_agent(agent, env, num_games=100):
    total_rewards = []
    total_tricks_won = []
    total_declarations_met = []

    for game in range(num_games):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            # Select action (no exploration during evaluation)
            action, _, _ = agent.select_action(state, legal_actions, epsilon=0.0)

            # Take step
            next_state, reward, done = env.step(current_player, action)
            episode_reward += reward
            state = next_state

        # Calculate additional metrics
        tricks_won = env.tricks_won[0]  # Assuming agent is player 0
        declaration = env.declarations[0]
        declaration_met = 1 if tricks_won == declaration else 0

        total_rewards.append(episode_reward)
        total_tricks_won.append(tricks_won)
        total_declarations_met.append(declaration_met)

    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_tricks_won": np.mean(total_tricks_won),
        "declaration_success_rate": np.mean(total_declarations_met),
    }
```

**Tasks:**

- Implement comprehensive evaluation metrics (rewards, trick success, declaration accuracy)
- Compare against random baseline and different training approaches
- Test with different player counts and game configurations
- Generate performance plots and statistics
- Create evaluation script for standalone testing

## Real-time Monitoring

The project includes a comprehensive real-time monitoring system that provides live updating graphs of training metrics. Key features:

- **Live Metrics**: Episode rewards, policy losses, value losses, bidding accuracy, and more
- **Multi-Agent Tracking**: Monitor multiple agents simultaneously during self-play
- **Interactive Controls**: Pause/resume monitoring and save metrics to JSON files
- **Custom Metrics**: Add your own metrics for specialized tracking

**Usage:**

```python
from realtime_monitor import create_monitor_and_callback

# Create monitor and callback
monitor, callback = create_monitor_and_callback(["Agent1", "Agent2"])
monitor.start_monitoring()

# During training, update metrics
callback.on_episode_end("Agent1", agent, env, episode_data)
callback.on_training_step("Agent1", agent, loss_data)

# Save metrics
monitor.save_metrics("training_metrics.json")
```

## Configuration

The project uses a centralized configuration system in `config.py` for easy hyperparameter management:

**Training Configuration:**

```python
from config import TrainingConfig

config = TrainingConfig(
    num_episodes=1000,
    batch_size=64,
    learning_rate=3e-4,
    num_agents=4,
    use_monitor=True
)
```

**Key Configuration Options:**

- Environment settings (players, cards)
- Agent hyperparameters (learning rate, network size)
- Training parameters (episodes, batch size, epsilon)
- Self-play settings (number of agents, policy noise)
- Monitoring options (update interval, max points)
- Model saving and evaluation intervals

## Contributing

Want to help? Here's how:

1. Fork the repo.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add feature'`).
4. Push (`git push origin feature/your-feature`).
5. Open a pull request.

## Resources

- [RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Judgement Rules](<https://en.wikipedia.org/wiki/Judgement_(card_game)>)

Happy coding!

TODO:
opponentmodel
