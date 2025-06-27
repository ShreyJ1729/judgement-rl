#!/usr/bin/env python3
"""
Interactive Judgement Card Game

A simple command-line interface to play the Judgement card game.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from judgement_rl import JudgementEnv, HeuristicAgent
from judgement_rl.config import EnvironmentConfig, RewardConfig
import random


def print_game_state(env, player_hand):
    """Print the current game state."""
    print("\n" + "=" * 50)
    print(f"Round {env.game_state.round_num}")
    print(f"Current Player: {env.game_state.current_player}")
    print(f"Phase: {env.game_state.phase}")
    print(f"Trump Suit: {env.game_state.trump_suit}")
    print(f"Your Hand: {player_hand}")

    if env.game_state.phase == "bidding":
        print(f"Bids so far: {env.game_state.bids}")
    else:
        print(f"Bids: {env.game_state.bids}")
        print(f"Tricks: {env.game_state.tricks}")

    print(f"Scores: {env.game_state.scores}")
    print("=" * 50)


def get_player_action(legal_actions, phase):
    """Get action from player."""
    while True:
        try:
            if phase == "bidding":
                print(f"\nLegal bids: {legal_actions}")
                action = int(input("Enter your bid: "))
            else:
                print(f"\nYour cards: {list(enumerate(legal_actions))}")
                action = int(input("Enter card index to play: "))

            if action in legal_actions:
                return action
            else:
                print("Invalid action. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nGame interrupted. Goodbye!")
            sys.exit(0)


def play_game():
    """Play a complete game of Judgement."""
    print("Welcome to Judgement Card Game!")
    print("You are Player 0 (first player)")

    # Create environment
    env_config = EnvironmentConfig(
        num_players=4, max_cards=7, reward_config=RewardConfig()
    )
    env = JudgementEnv(config=env_config)

    # Create AI opponents
    opponents = [
        HeuristicAgent(strategy="balanced", randomness=0.1),
        HeuristicAgent(strategy="aggressive", randomness=0.1),
        HeuristicAgent(strategy="conservative", randomness=0.1),
    ]

    # Start game
    obs, info = env.reset(seed=random.randint(0, 1000))
    player_hand = env.game_state.hands[0]

    print_game_state(env, player_hand)

    while not env.game_state.is_terminal():
        current_player = env.game_state.current_player
        legal_actions = env.get_legal_actions()

        if current_player == 0:  # Human player
            action = get_player_action(legal_actions, env.game_state.phase)
            print(f"You chose: {action}")
        else:  # AI opponent
            opponent = opponents[current_player - 1]
            action = opponent.select_action(env.game_state, legal_actions)
            print(f"Player {current_player} chose: {action}")

        # Take action
        obs, reward, done, truncated, info = env.step(action)

        # Update player hand
        if current_player == 0:
            player_hand = env.game_state.hands[0]

        print_game_state(env, player_hand)

        if done:
            break

    # Game over
    print("\n" + "=" * 50)
    print("GAME OVER!")
    print(f"Final scores: {env.game_state.scores}")

    winner = env.game_state.scores.index(max(env.game_state.scores))
    if winner == 0:
        print("🎉 Congratulations! You won!")
    else:
        print(f"Player {winner} won the game.")
    print("=" * 50)


def main():
    """Main function."""
    try:
        while True:
            play_game()

            play_again = (
                input("\nWould you like to play again? (y/n): ").lower().strip()
            )
            if play_again not in ["y", "yes"]:
                print("Thanks for playing!")
                break

    except KeyboardInterrupt:
        print("\nGame interrupted. Goodbye!")


if __name__ == "__main__":
    main()
