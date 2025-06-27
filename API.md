# API Documentation

This document provides comprehensive API documentation for the Judgement RL project.

## Table of Contents

- [Core Components](#core-components)
- [Environment](#environment)
- [Agents](#agents)
- [Configuration](#configuration)
- [Utilities](#utilities)
- [CLI Tools](#cli-tools)
- [Examples](#examples)

## Core Components

### JudgementEnv

The main game environment implementing the Judgement card game.

```python
from judgement_rl import JudgementEnv

env = JudgementEnv(
    num_players=4,
    max_cards=7,
    reward_config=RewardConfig()
)
```

#### Methods

##### `reset(seed=None)`

Reset the environment to initial state.

**Parameters:**

- `seed` (int, optional): Random seed for reproducibility

**Returns:**

- `tuple`: (observation, info)

**Example:**

```python
obs, info = env.reset(seed=42)
```

##### `step(action)`

Take an action in the environment.

**Parameters:**

- `action` (int): Action to take (bid or card index)

**Returns:**

- `tuple`: (observation, reward, terminated, truncated, info)

**Example:**

```python
obs, reward, done, truncated, info = env.step(action=3)
```

##### `get_legal_actions()`

Get list of legal actions for current state.

**Returns:**

- `list[int]`: List of legal action indices

**Example:**

```python
legal_actions = env.get_legal_actions()
```

##### `render()`

Render the current game state.

**Returns:**

- `str`: String representation of game state

**Example:**

```python
print(env.render())
```

### StateEncoder

Converts game state to numerical representation for neural networks.

```python
from judgement_rl import StateEncoder

encoder = StateEncoder(
    num_players=4,
    max_cards=7,
    hidden_dim=256
)
```

#### Methods

##### `encode_state(game_state)`

Encode game state to numerical representation.

**Parameters:**

- `game_state` (GameState): Current game state

**Returns:**

- `torch.Tensor`: Encoded state tensor

**Example:**

```python
encoded_state = encoder.encode_state(game_state)
```

##### `get_state_dim()`

Get the dimension of encoded state.

**Returns:**

- `int`: State dimension

**Example:**

```python
state_dim = encoder.get_state_dim()
```

### PPOAgent

Proximal Policy Optimization agent for learning Judgement strategies.

```python
from judgement_rl import PPOAgent

agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=256,
    learning_rate=1e-4,
    device="cpu"
)
```

#### Methods

##### `select_action(state, legal_actions, epsilon=0.0)`

Select action using current policy.

**Parameters:**

- `state` (torch.Tensor): Current state
- `legal_actions` (list[int]): List of legal actions
- `epsilon` (float): Exploration rate

**Returns:**

- `tuple`: (action, probability, value)

**Example:**

```python
action, prob, value = agent.select_action(state, legal_actions, epsilon=0.1)
```

##### `update_policy(experiences)`

Update policy using collected experiences.

**Parameters:**

- `experiences` (list): List of experience tuples

**Returns:**

- `dict`: Training metrics

**Example:**

```python
metrics = agent.update_policy(experiences)
```

##### `save_model(path)`

Save trained model to file.

**Parameters:**

- `path` (str): Path to save model

**Example:**

```python
agent.save_model("models/ppo_agent.pth")
```

##### `load_model(path)`

Load trained model from file.

**Parameters:**

- `path` (str): Path to load model from

**Example:**

```python
agent.load_model("models/ppo_agent.pth")
```

### SelfPlayTrainer

Trainer for self-play training of Judgement agents.

```python
from judgement_rl import SelfPlayTrainer

trainer = SelfPlayTrainer(
    env_config=EnvironmentConfig(),
    agent_config=AgentConfig(),
    training_config=TrainingConfig()
)
```

#### Methods

##### `train(num_episodes)`

Train agent using self-play.

**Parameters:**

- `num_episodes` (int): Number of training episodes

**Returns:**

- `dict`: Training statistics

**Example:**

```python
stats = trainer.train(num_episodes=1000)
```

##### `evaluate(num_games=100)`

Evaluate current agent performance.

**Parameters:**

- `num_games` (int): Number of evaluation games

**Returns:**

- `dict`: Evaluation metrics

**Example:**

```python
metrics = trainer.evaluate(num_games=100)
```

### HeuristicAgent

Rule-based agent for comparison and evaluation.

```python
from judgement_rl import HeuristicAgent

agent = HeuristicAgent(
    strategy="aggressive",  # or "conservative", "balanced"
    randomness=0.1
)
```

#### Methods

##### `select_action(game_state, legal_actions)`

Select action based on heuristic rules.

**Parameters:**

- `game_state` (GameState): Current game state
- `legal_actions` (list[int]): List of legal actions

**Returns:**

- `int`: Selected action

**Example:**

```python
action = agent.select_action(game_state, legal_actions)
```

## Environment

### GameState

Represents the current state of a Judgement game.

```python
from judgement_rl.environment import GameState

# GameState is typically created internally by JudgementEnv
# but can be accessed for analysis
```

#### Attributes

- `current_player` (int): Index of current player
- `phase` (str): Current game phase ("bidding" or "playing")
- `round_num` (int): Current round number
- `hands` (list): List of player hands
- `bids` (list): List of player bids
- `tricks` (list): List of played tricks
- `scores` (list): List of player scores
- `trump_suit` (int): Trump suit for current round

### RewardConfig

Configuration for reward calculation.

```python
from judgement_rl.config import RewardConfig

reward_config = RewardConfig(
    win_reward=100.0,
    lose_reward=-100.0,
    trick_reward=10.0,
    bid_accuracy_bonus=5.0,
    early_termination_penalty=-50.0
)
```

## Agents

### AgentConfig

Configuration for PPO agent training.

```python
from judgement_rl.config import AgentConfig

agent_config = AgentConfig(
    hidden_dim=256,
    learning_rate=1e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    update_epochs=4,
    batch_size=64
)
```

### TrainingConfig

Configuration for training process.

```python
from judgement_rl.config import TrainingConfig

training_config = TrainingConfig(
    num_episodes=1000,
    save_interval=100,
    eval_interval=50,
    log_interval=10,
    checkpoint_dir="checkpoints/",
    tensorboard_dir="runs/",
    wandb_project="judgement-rl"
)
```

## Configuration

### EnvironmentConfig

Configuration for game environment.

```python
from judgement_rl.config import EnvironmentConfig

env_config = EnvironmentConfig(
    num_players=4,
    max_cards=7,
    reward_config=RewardConfig(),
    seed=42
)
```

### MonitoringConfig

Configuration for training monitoring.

```python
from judgement_rl.config import MonitoringConfig

monitoring_config = MonitoringConfig(
    log_dir="logs/",
    update_interval=1.0,
    max_points=1000,
    track_rewards=True,
    track_losses=True,
    track_actions=True,
    track_epsilon=True
)
```

### EvaluationConfig

Configuration for model evaluation.

```python
from judgement_rl.config import EvaluationConfig

eval_config = EvaluationConfig(
    num_games=100,
    opponents=["random", "heuristic"],
    metrics=["win_rate", "average_score", "bid_accuracy"],
    save_results=True,
    results_dir="results/"
)
```

### GUIConfig

Configuration for GUI interface.

```python
from judgement_rl.config import GUIConfig

gui_config = GUIConfig(
    window_width=1200,
    window_height=800,
    card_width=80,
    card_height=120,
    animation_speed=0.5,
    show_probabilities=True,
    show_ai_thinking=True,
    auto_play=False
)
```

## Utilities

### Logging

#### setup_logger

Set up logging configuration.

```python
from judgement_rl.utils.logging import setup_logger

logger = setup_logger(
    name="judgement_rl",
    level="INFO",
    log_file="logs/training.log",
    use_tensorboard=True,
    tensorboard_dir="runs/"
)
```

#### TrainingLogger

Specialized logger for training metrics.

```python
from judgement_rl.utils.logging import TrainingLogger

train_logger = TrainingLogger(
    log_dir="logs/",
    experiment_name="ppo_training",
    use_wandb=True,
    wandb_project="judgement-rl"
)

# Log training metrics
train_logger.log_episode(episode=100, reward=50.0, loss=0.1)
train_logger.log_evaluation(win_rate=0.75, average_score=25.0)
```

## CLI Tools

### Training CLI

Train a Judgement RL agent from command line.

```bash
# Basic training
python -m judgement_rl.cli.train --episodes 1000 --mode single

# Self-play training
python -m judgement_rl.cli.train --episodes 1000 --mode self-play

# With custom configuration
python -m judgement_rl.cli.train \
    --episodes 1000 \
    --learning-rate 1e-4 \
    --hidden-dim 256 \
    --batch-size 64 \
    --save-interval 100 \
    --log-dir logs/
```

### Evaluation CLI

Evaluate trained models.

```bash
# Evaluate against random opponent
python -m judgement_rl.cli.evaluate \
    --model-path models/ppo_agent.pth \
    --num-games 100 \
    --opponent random

# Evaluate against heuristic opponent
python -m judgement_rl.cli.evaluate \
    --model-path models/ppo_agent.pth \
    --num-games 100 \
    --opponent heuristic

# Evaluate against another trained model
python -m judgement_rl.cli.evaluate \
    --model-path models/ppo_agent.pth \
    --opponent trained \
    --opponent-model-path models/opponent.pth
```

### GUI CLI

Launch the GUI interface.

```bash
# Basic GUI
python -m judgement_rl.cli.gui --model-path models/ppo_agent.pth

# With custom settings
python -m judgement_rl.cli.gui \
    --model-path models/ppo_agent.pth \
    --difficulty medium \
    --window-width 1200 \
    --window-height 800 \
    --show-probabilities \
    --show-ai-thinking
```

### Monitoring CLI

Monitor training progress in real-time.

```bash
# Monitor training
python -m judgement_rl.cli.monitor --log-dir logs/

# With specific experiment
python -m judgement_rl.cli.monitor \
    --log-dir logs/ \
    --experiment-name ppo_training \
    --update-interval 2.0 \
    --track-rewards \
    --track-losses
```

## Examples

### Basic Training Example

```python
from judgement_rl import JudgementEnv, PPOAgent, StateEncoder

# Create environment
env = JudgementEnv(num_players=4, max_cards=7)

# Create state encoder
encoder = StateEncoder(num_players=4, max_cards=7)

# Create agent
agent = PPOAgent(
    state_dim=encoder.get_state_dim(),
    action_dim=7,  # max_cards
    hidden_dim=256
)

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    done = False

    while not done:
        # Encode state
        state = encoder.encode_state(env.game_state)

        # Get legal actions
        legal_actions = env.get_legal_actions()

        # Select action
        action, prob, value = agent.select_action(state, legal_actions)

        # Take action
        obs, reward, done, truncated, info = env.step(action)

        # Store experience (simplified)
        # In practice, you'd use a proper experience buffer

    # Update policy periodically
    if episode % 10 == 0:
        # Update with collected experiences
        pass
```

### Self-Play Training Example

```python
from judgement_rl import SelfPlayTrainer

# Create trainer
trainer = SelfPlayTrainer()

# Train agent
stats = trainer.train(num_episodes=1000)

# Evaluate performance
metrics = trainer.evaluate(num_games=100)
print(f"Win rate: {metrics['win_rate']:.2f}")
print(f"Average score: {metrics['average_score']:.2f}")
```

### Evaluation Example

```python
from judgement_rl import PPOAgent, JudgementEnv, HeuristicAgent

# Load trained agent
agent = PPOAgent()
agent.load_model("models/ppo_agent.pth")

# Create environment
env = JudgementEnv(num_players=4, max_cards=7)

# Create opponent
opponent = HeuristicAgent(strategy="aggressive")

# Play games
wins = 0
total_games = 100

for game in range(total_games):
    obs, info = env.reset()
    done = False

    while not done:
        if env.current_player == 0:  # Our agent
            action, _, _ = agent.select_action(obs, env.get_legal_actions())
        else:  # Opponent
            action = opponent.select_action(env.game_state, env.get_legal_actions())

        obs, reward, done, truncated, info = env.step(action)

        if done and reward > 0:
            wins += 1

win_rate = wins / total_games
print(f"Win rate against heuristic: {win_rate:.2f}")
```

### Custom Reward Configuration

```python
from judgement_rl import JudgementEnv, RewardConfig

# Create custom reward configuration
reward_config = RewardConfig(
    win_reward=200.0,           # Higher win reward
    lose_reward=-100.0,         # Standard lose penalty
    trick_reward=15.0,          # Higher trick reward
    bid_accuracy_bonus=10.0,    # Higher bid accuracy bonus
    early_termination_penalty=-25.0  # Lower early termination penalty
)

# Create environment with custom rewards
env = JudgementEnv(
    num_players=4,
    max_cards=7,
    reward_config=reward_config
)
```

### Advanced Training with Monitoring

```python
from judgement_rl import SelfPlayTrainer, TrainingLogger
from judgement_rl.config import TrainingConfig

# Create training configuration
config = TrainingConfig(
    num_episodes=2000,
    save_interval=200,
    eval_interval=100,
    log_interval=20,
    checkpoint_dir="checkpoints/",
    tensorboard_dir="runs/",
    wandb_project="judgement-rl-experiment"
)

# Create trainer
trainer = SelfPlayTrainer(training_config=config)

# Create logger
logger = TrainingLogger(
    log_dir="logs/",
    experiment_name="advanced_training",
    use_wandb=True,
    wandb_project="judgement-rl"
)

# Training with logging
for episode in range(config.num_episodes):
    # Train one episode
    episode_stats = trainer.train_episode()

    # Log metrics
    logger.log_episode(
        episode=episode,
        reward=episode_stats['total_reward'],
        loss=episode_stats.get('loss', 0.0),
        win_rate=episode_stats.get('win_rate', 0.0)
    )

    # Periodic evaluation
    if episode % config.eval_interval == 0:
        eval_metrics = trainer.evaluate(num_games=50)
        logger.log_evaluation(
            episode=episode,
            win_rate=eval_metrics['win_rate'],
            average_score=eval_metrics['average_score'],
            bid_accuracy=eval_metrics['bid_accuracy']
        )
```

This API documentation provides comprehensive coverage of all public interfaces in the Judgement RL project. For more detailed information about specific components, refer to the individual module docstrings and the development guide.
