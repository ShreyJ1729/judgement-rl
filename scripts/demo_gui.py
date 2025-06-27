#!/usr/bin/env python3
"""
Demo script that shows how the GUI interface works.
This script simulates a few turns of the game to demonstrate the functionality.
"""

import sys
import os
import time

# Add the src directory to the path so we can import from judgement_rl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from judgement_rl.environment.judgement_env import JudgementEnv
from judgement_rl.utils.state_encoder import StateEncoder
from judgement_rl.agents.agent import PPOAgent


def print_game_state(env, state, player_name="Player"):
    """Print the current game state in a readable format."""
    print(f"\n{'='*60}")
    print(f"{player_name}'s Turn")
    print(f"{'='*60}")
    print(f"Round: {env.round_number} | Cards: {env.round_cards} | Trump: {env.trump}")
    print(f"Phase: {env.phase}")
    print(f"Current Player: {env.current_player}")

    # Print hand
    if "hand" in state:
        print(f"\n{player_name}'s Hand:")
        for i, card in enumerate(state["hand"]):
            print(f"  {i}: {card}")

    # Print bidding
    print(f"\nBidding:")
    for i, bid in enumerate(env.declarations):
        player = "You" if i == 0 else f"AI {i}"
        bid_text = str(bid) if bid is not None else "-"
        print(f"  {player}: {bid_text}")

    # Print tricks won
    print(f"\nTricks Won:")
    for i, tricks in enumerate(env.tricks_won):
        player = "You" if i == 0 else f"AI {i}"
        print(f"  {player}: {tricks}")

    # Print current trick
    if env.current_trick:
        print(f"\nCurrent Trick:")
        for player, card in env.current_trick:
            player_name = "You" if player == 0 else f"AI {player}"
            print(f"  {player_name}: {card}")
    else:
        print(f"\nCurrent Trick: None")


def demo_game():
    """Run a demo of the game."""
    print("Judgement AI GUI Demo")
    print("This demo shows how the GUI interface would work")
    print("=" * 60)

    # Initialize components
    env = JudgementEnv(num_players=4, max_cards=7)
    state_encoder = StateEncoder(num_players=4, max_cards=7)

    # Try to load agent
    agent = None
    model_path = "models/selfplay_best_agent.pth"
    if os.path.exists(model_path):
        try:
            agent = PPOAgent(state_encoder)
            agent.load_model(model_path)
            print("✓ Loaded trained agent")
        except Exception as e:
            print(f"⚠ Failed to load agent: {e}")
            print("  Using random agent instead")
    else:
        print("⚠ No trained model found, using random agent")

    # Start game
    state = env.reset()
    print_game_state(env, state, "You")

    # Demo a few turns
    turn_count = 0
    max_turns = 10  # Limit demo length

    while turn_count < max_turns and env.phase == "bidding":
        if env.current_player == 0:
            # Human player's turn (simulated)
            legal_actions = env.get_legal_actions(0)
            print(f"\nYour legal actions: {legal_actions}")

            # Simulate a reasonable bid
            if env.phase == "bidding":
                # Choose a bid that doesn't make total = 4
                current_total = sum(d for d in env.declarations if d is not None)
                remaining_players = env.num_players - sum(
                    1 for d in env.declarations if d is not None
                )

                if remaining_players == 1:
                    # Last player - avoid making total = 4
                    if current_total == 4:
                        action = 0  # Must bid 0
                    else:
                        action = min(env.round_cards, 4 - current_total - 1)
                else:
                    # Not last player - bid reasonably
                    action = min(env.round_cards // 2, env.round_cards)

                print(f"Simulating your bid: {action}")
            else:
                action = legal_actions[0]  # Play first legal card
                print(f"Simulating your card play: {state['hand'][action]}")
        else:
            # AI player's turn
            legal_actions = env.get_legal_actions(env.current_player)

            if agent is not None:
                action, _, _ = agent.select_action(state, legal_actions, epsilon=0.0)
            else:
                action = legal_actions[0]  # Random choice

            if env.phase == "bidding":
                print(f"AI Player {env.current_player} bids: {action}")
            else:
                ai_hand = env.hands[env.current_player]
                print(f"AI Player {env.current_player} plays: {ai_hand[action]}")

        # Take action
        state, reward, done = env.step(env.current_player, action)

        # Update display
        player_name = "You" if env.current_player == 0 else f"AI {env.current_player}"
        print_game_state(env, state, player_name)

        if done:
            print(f"\nGame Over! Final reward: {reward}")
            break

        turn_count += 1
        time.sleep(1)  # Small delay for readability

    # Show final results
    print(f"\n{'='*60}")
    print("Demo Complete!")
    print(f"{'='*60}")
    print("In the actual GUI, you would:")
    print("1. See your cards as clickable buttons")
    print("2. Click on cards to play them")
    print("3. Click on numbers to make bids")
    print("4. Watch AI players make their moves automatically")
    print("5. See real-time updates of game state")
    print("\nTo try the full GUI:")
    print("  python play_against_ai.py")


if __name__ == "__main__":
    demo_game()
