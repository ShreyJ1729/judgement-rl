#!/usr/bin/env python3
"""
Evaluation script for trained Judgement card game agents.
This implements Step 8 from the README.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, List, Optional
import time
import sys

# Add the src directory to the path so we can import from judgement_rl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent
from judgement_rl.agents.heuristic_agent import HeuristicAgent


def evaluate_agent(
    agent: PPOAgent, env: JudgementEnv, num_games: int = 100, agent_player_idx: int = 0
) -> Dict[str, float]:
    """
    Evaluate the agent's performance over multiple games.

    Args:
        agent: The agent to evaluate
        env: The game environment
        num_games: Number of games to play
        agent_player_idx: Which player position the agent plays as

    Returns:
        Dictionary containing evaluation metrics
    """
    total_rewards = []
    total_tricks_won = []
    total_declarations_met = []
    total_declarations = []
    total_actual_tricks = []
    game_durations = []

    print(f"Evaluating agent over {num_games} games...")

    for game in range(num_games):
        state = env.reset()
        episode_reward = 0
        done = False
        game_start = time.time()

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            # Select action (no exploration during evaluation)
            if current_player == agent_player_idx:
                action, _, _ = agent.select_action(state, legal_actions, epsilon=0.0)
            else:
                # Other players use random actions for now
                action = np.random.choice(legal_actions)

            # Take step
            next_state, reward, done = env.step(current_player, action)

            if current_player == agent_player_idx:
                episode_reward += reward

            state = next_state

        game_duration = time.time() - game_start
        game_durations.append(game_duration)

        # Calculate additional metrics
        tricks_won = env.tricks_won[agent_player_idx]
        declaration = env.declarations[agent_player_idx]
        declaration_met = 1 if tricks_won == declaration else 0

        total_rewards.append(episode_reward)
        total_tricks_won.append(tricks_won)
        total_declarations_met.append(declaration_met)
        total_declarations.append(declaration)
        total_actual_tricks.append(tricks_won)

        if (game + 1) % 20 == 0:
            print(f"Completed {game + 1}/{num_games} games...")

    return {
        "avg_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "avg_tricks_won": np.mean(total_tricks_won),
        "std_tricks_won": np.std(total_tricks_won),
        "declaration_success_rate": np.mean(total_declarations_met),
        "avg_declaration": np.mean(total_declarations),
        "avg_actual_tricks": np.mean(total_actual_tricks),
        "avg_game_duration": np.mean(game_durations),
        "total_rewards": total_rewards,
        "total_tricks_won": total_tricks_won,
        "total_declarations_met": total_declarations_met,
    }


def compare_agents(
    agents: Dict[str, PPOAgent], env: JudgementEnv, num_games: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple agents against each other.

    Args:
        agents: Dictionary mapping agent names to agent objects
        env: The game environment
        num_games: Number of games per agent

    Returns:
        Dictionary mapping agent names to evaluation results
    """
    results = {}

    for agent_name, agent in agents.items():
        print(f"\nEvaluating {agent_name}...")
        results[agent_name] = evaluate_agent(agent, env, num_games)

    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """
    Print a formatted comparison table of agent performances.
    """
    print("\n" + "=" * 80)
    print("AGENT PERFORMANCE COMPARISON")
    print("=" * 80)

    # Header
    print(
        f"{'Agent Name':<20} {'Avg Reward':<12} {'Success Rate':<12} {'Avg Tricks':<10} {'Avg Decl':<10}"
    )
    print("-" * 80)

    # Results
    for agent_name, metrics in results.items():
        print(
            f"{agent_name:<20} "
            f"{metrics['avg_reward']:<12.3f} "
            f"{metrics['declaration_success_rate']:<12.3f} "
            f"{metrics['avg_tricks_won']:<10.2f} "
            f"{metrics['avg_declaration']:<10.2f}"
        )

    print("=" * 80)


def plot_evaluation_results(
    results: Dict[str, Dict[str, float]], save_path: Optional[str] = None
):
    """
    Create plots comparing agent performances.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    agent_names = list(results.keys())

    # Average rewards
    avg_rewards = [results[name]["avg_reward"] for name in agent_names]
    axes[0, 0].bar(agent_names, avg_rewards)
    axes[0, 0].set_title("Average Rewards")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Declaration success rates
    success_rates = [results[name]["declaration_success_rate"] for name in agent_names]
    axes[0, 1].bar(agent_names, success_rates)
    axes[0, 1].set_title("Declaration Success Rate")
    axes[0, 1].set_ylabel("Success Rate")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # Average tricks won vs declared
    avg_tricks = [results[name]["avg_tricks_won"] for name in agent_names]
    avg_declarations = [results[name]["avg_declaration"] for name in agent_names]

    x = np.arange(len(agent_names))
    width = 0.35

    axes[1, 0].bar(x - width / 2, avg_tricks, width, label="Actual Tricks")
    axes[1, 0].bar(x + width / 2, avg_declarations, width, label="Declared Tricks")
    axes[1, 0].set_title("Tricks: Declared vs Actual")
    axes[1, 0].set_ylabel("Number of Tricks")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(agent_names, rotation=45)
    axes[1, 0].legend()

    # Reward distribution
    for i, agent_name in enumerate(agent_names):
        rewards = results[agent_name]["total_rewards"]
        axes[1, 1].hist(rewards, alpha=0.7, label=agent_name, bins=20)

    axes[1, 1].set_title("Reward Distribution")
    axes[1, 1].set_xlabel("Reward")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Evaluation plots saved to {save_path}")
    else:
        plt.show()


def evaluate_different_configurations(agent: PPOAgent, num_games: int = 50):
    """
    Evaluate agent performance with different game configurations.
    """
    print("\n" + "=" * 60)
    print("EVALUATING DIFFERENT GAME CONFIGURATIONS")
    print("=" * 60)

    configs = [
        {"num_players": 3, "max_cards": 7, "name": "3 Players, 7 Cards"},
        {"num_players": 4, "max_cards": 7, "name": "4 Players, 7 Cards"},
        {"num_players": 5, "max_cards": 7, "name": "5 Players, 7 Cards"},
        {"num_players": 4, "max_cards": 5, "name": "4 Players, 5 Cards"},
    ]

    config_results = {}

    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        env = JudgementEnv(
            num_players=config["num_players"], max_cards=config["max_cards"]
        )
        state_encoder = StateEncoder(
            num_players=config["num_players"], max_cards=config["max_cards"]
        )

        # Create a new agent with the correct state encoder
        test_agent = PPOAgent(state_encoder=state_encoder)
        test_agent.policy_net.load_state_dict(agent.policy_net.state_dict())

        results = evaluate_agent(test_agent, env, num_games)
        config_results[config["name"]] = results

    # Print configuration comparison
    print("\nConfiguration Performance Comparison:")
    print(f"{'Configuration':<25} {'Avg Reward':<12} {'Success Rate':<12}")
    print("-" * 60)
    for config_name, results in config_results.items():
        print(
            f"{config_name:<25} "
            f"{results['avg_reward']:<12.3f} "
            f"{results['declaration_success_rate']:<12.3f}"
        )


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(description="Evaluate trained Judgement agents")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument(
        "--num_games", type=int, default=100, help="Number of games to evaluate"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple agents"
    )
    parser.add_argument(
        "--configs", action="store_true", help="Test different game configurations"
    )
    parser.add_argument("--save_plots", type=str, help="Path to save evaluation plots")

    args = parser.parse_args()

    print("Judgement Card Game - Agent Evaluation")
    print("=" * 50)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize environment and encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    if args.compare:
        # Compare multiple agents
        agents = {}

        # Add random baseline
        random_agent = PPOAgent(state_encoder)
        agents["Random"] = random_agent

        # Add heuristic agent if available
        try:
            heuristic_agent = HeuristicAgent()
            agents["Heuristic"] = heuristic_agent
        except:
            print("Heuristic agent not available")

        # Add trained agents if models exist
        model_paths = [
            ("Single PPO", "models/single_ppo_agent.pth"),
            ("Self-Play PPO", "models/selfplay_best_agent.pth"),
        ]

        for agent_name, model_path in model_paths:
            if os.path.exists(model_path):
                agent = PPOAgent(state_encoder)
                agent.load_model(model_path)
                agents[agent_name] = agent
                print(f"Loaded {agent_name} from {model_path}")

        if len(agents) < 2:
            print("Need at least 2 agents to compare. Training a simple agent...")
            # Train a simple agent for comparison
            simple_agent = PPOAgent(state_encoder)
            simple_agent.collect_experience(env, num_episodes=10, epsilon=0.3)
            simple_agent.train(batch_size=32)
            agents["Simple PPO"] = simple_agent

        # Compare agents
        results = compare_agents(agents, env, args.num_games)
        print_comparison_table(results)

        if args.save_plots:
            plot_evaluation_results(results, args.save_plots)
        else:
            plot_evaluation_results(results)

    elif args.model_path:
        # Evaluate single agent
        if not os.path.exists(args.model_path):
            print(f"Model file not found: {args.model_path}")
            return

        agent = PPOAgent(state_encoder)
        agent.load_model(args.model_path)
        print(f"Loaded agent from {args.model_path}")

        results = evaluate_agent(agent, env, args.num_games)

        print("\nEvaluation Results:")
        print(
            f"Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}"
        )
        print(f"Declaration Success Rate: {results['declaration_success_rate']:.3f}")
        print(
            f"Average Tricks Won: {results['avg_tricks_won']:.2f} ± {results['std_tricks_won']:.2f}"
        )
        print(f"Average Declaration: {results['avg_declaration']:.2f}")
        print(f"Average Game Duration: {results['avg_game_duration']:.2f}s")

        if args.configs:
            evaluate_different_configurations(agent, args.num_games // 2)

    else:
        print("Please specify either --model_path or --compare")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
