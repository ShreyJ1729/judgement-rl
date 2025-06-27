#!/usr/bin/env python3
"""
Evaluation script for the best trained model against random players.
Runs 100 games of the self-play trained model against 3 random players.
"""

import torch
import numpy as np
import os
import sys
import time
import json
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path so we can import from judgement_rl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent


class RandomAgent:
    """Simple random agent for evaluation."""

    def __init__(self, player_idx: int):
        self.player_idx = player_idx

    def select_action(
        self, state: Dict[str, Any], legal_actions: List[int], epsilon: float = 0.0
    ) -> Tuple[int, float, float]:
        """Select a random action from legal actions."""
        action = np.random.choice(legal_actions)
        action_prob = 1.0 / len(legal_actions)
        return action, action_prob, 0.0


def find_best_model(models_dir: str = "../models") -> str:
    """Find the best trained model in the models directory."""
    model_files = []

    for file in os.listdir(models_dir):
        if file.endswith(".pth"):
            model_files.append(file)

    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    # Look for 'best' in filename first
    best_models = [f for f in model_files if "best" in f.lower()]
    if best_models:
        return os.path.join(models_dir, best_models[0])

    # Otherwise, take the most recent one
    model_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True
    )
    return os.path.join(models_dir, model_files[0])


def evaluate_model_against_random(
    model_path: str, num_games: int = 100, agent_player_idx: int = 0
) -> Dict[str, Any]:
    """
    Evaluate the trained model against 3 random players.

    Args:
        model_path: Path to the trained model
        num_games: Number of games to play
        agent_player_idx: Which player position the trained agent plays as

    Returns:
        Dictionary containing detailed evaluation results
    """
    print(f"Loading model from: {model_path}")

    # Initialize environment and state encoder
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(max_cards=7)

    # Load the trained agent
    trained_agent = PPOAgent(state_encoder)
    trained_agent.load_model(model_path)
    trained_agent.policy_net.eval()  # Set to evaluation mode

    # Initialize random agents
    random_agents = [RandomAgent(i) for i in range(4)]

    # Results storage
    results = {
        "model_path": model_path,
        "num_games": num_games,
        "agent_player_idx": agent_player_idx,
        "game_results": [],
        "player_stats": defaultdict(
            lambda: {
                "total_rewards": [],
                "total_tricks_won": [],
                "declarations_met": [],
                "total_declarations": [],
                "wins": 0,
                "game_durations": [],
            }
        ),
        "round_stats": defaultdict(
            lambda: {"tricks_won": [], "declarations": [], "success_rate": []}
        ),
    }

    print(
        f"Starting evaluation: {num_games} games, trained agent at position {agent_player_idx}"
    )
    print("=" * 80)

    for game in range(num_games):
        state = env.reset()
        episode_rewards = [0] * 4
        done = False
        game_start = time.time()

        # Track round-specific stats
        round_cards = env.round_cards
        trump = env.trump

        while not done:
            current_player = env.current_player
            legal_actions = env.get_legal_actions(current_player)

            # Select action based on player type
            if current_player == agent_player_idx:
                action, _, _ = trained_agent.select_action(
                    state, legal_actions, epsilon=0.0
                )
            else:
                action, _, _ = random_agents[current_player].select_action(
                    state, legal_actions
                )

            # Take step
            next_state, reward, done = env.step(current_player, action)
            episode_rewards[current_player] += reward
            state = next_state

        game_duration = time.time() - game_start

        # Calculate game results
        tricks_won = env.tricks_won.copy()
        declarations = env.declarations.copy()

        # Determine winner (player with most tricks won)
        winner = np.argmax(tricks_won)

        # Store results for each player
        for player_idx in range(4):
            player_stats = results["player_stats"][player_idx]
            player_stats["total_rewards"].append(episode_rewards[player_idx])
            player_stats["total_tricks_won"].append(tricks_won[player_idx])
            player_stats["total_declarations"].append(declarations[player_idx])
            player_stats["declarations_met"].append(
                1 if tricks_won[player_idx] == declarations[player_idx] else 0
            )
            player_stats["game_durations"].append(game_duration)

            if player_idx == winner:
                player_stats["wins"] += 1

        # Store round-specific stats
        round_key = f"{round_cards}_cards_{trump}"
        round_stats = results["round_stats"][round_key]
        round_stats["tricks_won"].extend(tricks_won)
        round_stats["declarations"].extend(declarations)
        round_stats["success_rate"].extend(
            [1 if t == d else 0 for t, d in zip(tricks_won, declarations)]
        )

        # Store game result
        game_result = {
            "game_id": game,
            "round_cards": round_cards,
            "trump": trump,
            "tricks_won": tricks_won.copy(),
            "declarations": declarations.copy(),
            "rewards": episode_rewards.copy(),
            "winner": winner,
            "duration": game_duration,
        }
        results["game_results"].append(game_result)

        # Progress update
        if (game + 1) % 20 == 0:
            print(f"Completed {game + 1}/{num_games} games...")

    # Calculate summary statistics
    results["summary"] = calculate_summary_stats(results)

    return results


def calculate_summary_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary statistics from the evaluation results."""
    summary = {}

    # Overall statistics
    total_games = results["num_games"]
    agent_idx = results["agent_player_idx"]

    # Trained agent statistics
    agent_stats = results["player_stats"][agent_idx]
    summary["trained_agent"] = {
        "win_rate": agent_stats["wins"] / total_games,
        "avg_reward": np.mean(agent_stats["total_rewards"]),
        "std_reward": np.std(agent_stats["total_rewards"]),
        "avg_tricks_won": np.mean(agent_stats["total_tricks_won"]),
        "std_tricks_won": np.std(agent_stats["total_tricks_won"]),
        "declaration_success_rate": np.mean(agent_stats["declarations_met"]),
        "avg_declaration": np.mean(agent_stats["total_declarations"]),
        "avg_game_duration": np.mean(agent_stats["game_durations"]),
        "total_wins": agent_stats["wins"],
    }

    # Random agents statistics (average across all random agents)
    random_stats = {
        "total_rewards": [],
        "total_tricks_won": [],
        "declarations_met": [],
        "total_declarations": [],
        "wins": 0,
        "game_durations": [],
    }

    for player_idx in range(4):
        if player_idx != agent_idx:
            player_stats = results["player_stats"][player_idx]
            random_stats["total_rewards"].extend(player_stats["total_rewards"])
            random_stats["total_tricks_won"].extend(player_stats["total_tricks_won"])
            random_stats["declarations_met"].extend(player_stats["declarations_met"])
            random_stats["total_declarations"].extend(
                player_stats["total_declarations"]
            )
            random_stats["wins"] += player_stats["wins"]
            random_stats["game_durations"].extend(player_stats["game_durations"])

    summary["random_agents"] = {
        "win_rate": random_stats["wins"] / (total_games * 3),  # 3 random agents
        "avg_reward": np.mean(random_stats["total_rewards"]),
        "std_reward": np.std(random_stats["total_rewards"]),
        "avg_tricks_won": np.mean(random_stats["total_tricks_won"]),
        "std_tricks_won": np.std(random_stats["total_tricks_won"]),
        "declaration_success_rate": np.mean(random_stats["declarations_met"]),
        "avg_declaration": np.mean(random_stats["total_declarations"]),
        "avg_game_duration": np.mean(random_stats["game_durations"]),
        "total_wins": random_stats["wins"],
    }

    # Round-specific statistics
    summary["round_stats"] = {}
    for round_key, round_data in results["round_stats"].items():
        summary["round_stats"][round_key] = {
            "avg_tricks_won": np.mean(round_data["tricks_won"]),
            "avg_declaration": np.mean(round_data["declarations"]),
            "success_rate": np.mean(round_data["success_rate"]),
            "num_games": len(round_data["tricks_won"]) // 4,  # 4 players per game
        }

    return summary


def print_detailed_results(results: Dict[str, Any]):
    """Print detailed evaluation results."""
    summary = results["summary"]
    agent_idx = results["agent_player_idx"]

    print("\n" + "=" * 100)
    print("DETAILED EVALUATION RESULTS")
    print("=" * 100)

    print(f"\nModel: {results['model_path']}")
    print(f"Games played: {results['num_games']}")
    print(f"Trained agent position: {agent_idx}")
    print(f"Evaluation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Trained agent performance
    print("\n" + "-" * 50)
    print("TRAINED AGENT PERFORMANCE")
    print("-" * 50)
    agent_stats = summary["trained_agent"]
    print(
        f"Win Rate: {agent_stats['win_rate']:.3f} ({agent_stats['total_wins']}/{results['num_games']})"
    )
    print(
        f"Average Reward: {agent_stats['avg_reward']:.3f} ± {agent_stats['std_reward']:.3f}"
    )
    print(
        f"Average Tricks Won: {agent_stats['avg_tricks_won']:.3f} ± {agent_stats['std_tricks_won']:.3f}"
    )
    print(f"Declaration Success Rate: {agent_stats['declaration_success_rate']:.3f}")
    print(f"Average Declaration: {agent_stats['avg_declaration']:.3f}")
    print(f"Average Game Duration: {agent_stats['avg_game_duration']:.3f}s")

    # Random agents performance
    print("\n" + "-" * 50)
    print("RANDOM AGENTS PERFORMANCE")
    print("-" * 50)
    random_stats = summary["random_agents"]
    print(
        f"Win Rate: {random_stats['win_rate']:.3f} ({random_stats['total_wins']}/{results['num_games'] * 3})"
    )
    print(
        f"Average Reward: {random_stats['avg_reward']:.3f} ± {random_stats['std_reward']:.3f}"
    )
    print(
        f"Average Tricks Won: {random_stats['avg_tricks_won']:.3f} ± {random_stats['std_tricks_won']:.3f}"
    )
    print(f"Declaration Success Rate: {random_stats['declaration_success_rate']:.3f}")
    print(f"Average Declaration: {random_stats['avg_declaration']:.3f}")
    print(f"Average Game Duration: {random_stats['avg_game_duration']:.3f}s")

    # Performance comparison
    print("\n" + "-" * 50)
    print("PERFORMANCE COMPARISON")
    print("-" * 50)
    win_rate_diff = agent_stats["win_rate"] - random_stats["win_rate"]
    reward_diff = agent_stats["avg_reward"] - random_stats["avg_reward"]
    tricks_diff = agent_stats["avg_tricks_won"] - random_stats["avg_tricks_won"]
    success_diff = (
        agent_stats["declaration_success_rate"]
        - random_stats["declaration_success_rate"]
    )

    print(f"Win Rate Difference: {win_rate_diff:+.3f}")
    print(f"Reward Difference: {reward_diff:+.3f}")
    print(f"Tricks Won Difference: {tricks_diff:+.3f}")
    print(f"Declaration Success Difference: {success_diff:+.3f}")

    # Round-specific performance
    print("\n" + "-" * 50)
    print("ROUND-SPECIFIC PERFORMANCE")
    print("-" * 50)
    for round_key, round_stats in summary["round_stats"].items():
        print(f"\n{round_key}:")
        print(f"  Games: {round_stats['num_games']}")
        print(f"  Avg Tricks Won: {round_stats['avg_tricks_won']:.3f}")
        print(f"  Avg Declaration: {round_stats['avg_declaration']:.3f}")
        print(f"  Success Rate: {round_stats['success_rate']:.3f}")

    print("\n" + "=" * 100)


def save_results(results: Dict[str, Any], output_dir: str = "../results"):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    json_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary as text
    txt_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write("JUDGEMENT RL MODEL EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        summary = results["summary"]
        agent_stats = summary["trained_agent"]
        random_stats = summary["random_agents"]

        f.write(f"Model: {results['model_path']}\n")
        f.write(f"Games: {results['num_games']}\n")
        f.write(f"Agent Position: {results['agent_player_idx']}\n\n")

        f.write("TRAINED AGENT:\n")
        f.write(f"  Win Rate: {agent_stats['win_rate']:.3f}\n")
        f.write(f"  Avg Reward: {agent_stats['avg_reward']:.3f}\n")
        f.write(
            f"  Declaration Success: {agent_stats['declaration_success_rate']:.3f}\n\n"
        )

        f.write("RANDOM AGENTS:\n")
        f.write(f"  Win Rate: {random_stats['win_rate']:.3f}\n")
        f.write(f"  Avg Reward: {random_stats['avg_reward']:.3f}\n")
        f.write(
            f"  Declaration Success: {random_stats['declaration_success_rate']:.3f}\n"
        )

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {txt_path}")

    return json_path, txt_path


def main():
    """Main evaluation function."""
    print("Judgement RL - Best Model Evaluation")
    print("=" * 50)

    # Find the best model
    try:
        model_path = find_best_model()
        print(f"Found best model: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Run evaluation
    results = evaluate_model_against_random(
        model_path=model_path, num_games=100, agent_player_idx=0
    )

    # Print results
    print_detailed_results(results)

    # Save results
    save_results(results)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
