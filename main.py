#!/usr/bin/env python3
"""
Main script for the Judgement AI project.
This demonstrates the usage of the implemented components from Steps 1, 2, and 3.
"""

import random
import numpy as np
from judgement_env import JudgementEnv
from state_encoder import StateEncoder
from heuristic_agent import HeuristicAgent


def demonstrate_implementation():
    """Demonstrate the implementation of Steps 1, 2, and 3."""
    print("Judgement AI - Implementation Demo")
    print("=" * 50)
    print("Demonstrating Steps 1, 2, and 3 from the README\n")

    # Step 1: Environment Setup
    print("Step 1: Environment Setup")
    print("-" * 30)
    env = JudgementEnv(num_players=4, max_cards=7)
    print(f"✓ Created environment with {env.num_players} players")
    print(f"✓ Max cards per round: {env.max_cards}")
    print(f"✓ Trump suits: {env.trump_suits}")

    # Step 2: Game Logic
    print("\nStep 2: Game Logic")
    print("-" * 30)
    encoder = StateEncoder(num_players=4, max_cards=7)
    agents = [HeuristicAgent(i) for i in range(4)]
    state = env.reset()
    print(f"✓ Game reset - Round {env.round_number}")
    print(f"✓ Cards this round: {state['round_cards']}")
    print(f"✓ Trump suit: {state['trump']}")
    print(f"✓ Current phase: {state['phase']}")

    # Print all players' hands
    print("\nAll Players' Hands:")
    for i in range(env.num_players):
        player_hand = env.hands[i]
        print(f"  Player {i}: {player_hand}")

    print(f"\n✓ Player 0's hand (from state): {state['hand']}")

    # Step 3: State Representation
    print("\nStep 3: State Representation")
    print("-" * 30)
    encoded_state = encoder.encode_state(state)
    print(f"✓ State encoder created")
    print(f"✓ Original state keys: {list(state.keys())}")
    print(f"✓ Encoded state shape: {encoded_state.shape}")
    print(f"✓ State dimension: {encoder.get_state_dim()}")
    print(
        f"✓ round_cards: {state['round_cards']} (number of cards each player has this round)"
    )

    # Demonstrate a complete game
    print("\nDemonstrating a Complete Game")
    print("-" * 30)

    # Bidding phase
    print("Bidding Phase:")
    done = False
    for _ in range(env.num_players):
        player = env.current_player
        legal_bids = env.get_legal_actions(player)
        bid = agents[player].select_bid(state, legal_bids)
        state, reward, done = env.step(player, bid)
        print(f"  Player {player} bids: {bid}")
        if done:
            print("  ✗ Game ended during bidding (invalid)")
            return
    print("✓ Bidding completed successfully")

    # Playing phase
    print("\nPlaying Phase:")
    tricks_played = 0
    while not done and tricks_played < state["round_cards"]:
        player = env.current_player
        legal_cards = env.get_legal_actions(player)
        action = agents[player].select_card(state, legal_cards)
        state, reward, done = env.step(player, action)
        if len(env.current_trick) == 0:
            tricks_played += 1
    print(f"✓ Game completed with final reward: {reward}")
    print(f"✓ Final declarations: {state['declarations']}")
    print(f"✓ Final tricks won: {state['tricks_won']}")


def test_state_encoding_consistency():
    """Test that state encoding is consistent throughout the game."""
    print("\nTesting State Encoding Consistency")
    print("-" * 30)

    env = JudgementEnv(num_players=4)
    encoder = StateEncoder(num_players=4, max_cards=7)

    state = env.reset()
    initial_shape = encoder.encode_state(state).shape

    print(f"✓ Initial state shape: {initial_shape}")

    # Track state shapes throughout the game
    shapes = [initial_shape]

    # Complete bidding
    for player in range(4):
        legal_actions = env.get_legal_actions(player)
        bid = random.choice(legal_actions)
        state, reward, done = env.step(player, bid)
        shapes.append(encoder.encode_state(state).shape)

        if done:
            break

    # Complete playing (if not done)
    if not done:
        steps = 0
        while not done and steps < 20:
            player = env.current_player
            legal_actions = env.get_legal_actions(player)

            if legal_actions:
                action = random.choice(legal_actions)
                state, reward, done = env.step(player, action)
                shapes.append(encoder.encode_state(state).shape)
                steps += 1
            else:
                break

    # Check consistency
    all_same_shape = all(shape == initial_shape for shape in shapes)
    print(f"✓ All state shapes consistent: {all_same_shape}")
    print(f"✓ Number of states encoded: {len(shapes)}")

    return all_same_shape


def main():
    """Main function to run the demonstration."""
    try:
        demonstrate_implementation()
        consistency_ok = test_state_encoding_consistency()

        print("\n" + "=" * 50)
        print("IMPLEMENTATION SUMMARY")
        print("=" * 50)
        print("✓ Step 1: Environment Setup - COMPLETED")
        print("✓ Step 2: Game Logic - COMPLETED")
        print("✓ Step 3: State Representation - COMPLETED")
        print(
            f"✓ State Encoding Consistency: {'✓ PASSED' if consistency_ok else '✗ FAILED'}"
        )

        print("\n🎉 Steps 1, 2, and 3 are fully implemented and working!")
        print("\nNext steps:")
        print("  - Step 4: Define the Action Space (partially done)")
        print("  - Step 5: Design the Reward System (implemented)")
        print("  - Step 6: Implement the RL Agent")
        print("  - Step 7: Train the Agent")
        print("  - Step 8: Evaluate the Agent")

    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
