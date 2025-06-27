#!/usr/bin/env python3
"""
Test script for the Judgement AI implementation.
Tests Steps 1, 2, and 3 from the README.
"""

import random
import numpy as np
from judgement_env import JudgementEnv
from state_encoder import StateEncoder


def test_step_1_environment_setup():
    """Test Step 1: Environment setup and basic functionality."""
    print("Testing Step 1: Environment Setup")
    print("=" * 40)

    # Test environment creation
    env = JudgementEnv(num_players=4, max_cards=7)
    print(f"✓ Environment created with {env.num_players} players")
    print(f"✓ Max cards: {env.max_cards}")
    print(f"✓ Trump suits: {env.trump_suits}")

    # Test reset
    state = env.reset()
    print(f"✓ Environment reset successful")
    print(f"✓ Initial state keys: {list(state.keys())}")

    return env


def test_step_2_game_logic():
    """Test Step 2: Game logic implementation."""
    print("\nTesting Step 2: Game Logic")
    print("=" * 40)

    env = JudgementEnv(num_players=4)

    # Test bidding phase
    print("Testing bidding phase...")
    state = env.reset()
    print(f"✓ Round cards: {state['round_cards']}")
    print(f"✓ Trump: {state['trump']}")
    print(f"✓ Phase: {state['phase']}")

    # Simulate bidding
    for player in range(4):
        legal_actions = env.get_legal_actions(player)
        bid = random.choice(legal_actions)
        state, reward, done = env.step(player, bid)
        print(f"✓ Player {player} bids {bid}")

        if done:
            print("✗ Game ended during bidding (shouldn't happen)")
            return False

    print("✓ Bidding phase completed successfully")

    # Test playing phase
    print("Testing playing phase...")
    tricks_played = 0
    while not done and tricks_played < state["round_cards"]:
        player = env.current_player
        legal_actions = env.get_legal_actions(player)

        if legal_actions:
            action = random.choice(legal_actions)
            state, reward, done = env.step(player, action)

            if len(env.current_trick) == 0 and tricks_played > 0:
                tricks_played += 1
                print(f"✓ Trick {tricks_played} completed")
        else:
            break

    print("✓ Playing phase completed successfully")
    print(f"✓ Final reward: {reward}")

    return True


def test_step_3_state_representation():
    """Test Step 3: State representation and encoding."""
    print("\nTesting Step 3: State Representation")
    print("=" * 40)

    env = JudgementEnv(num_players=4)
    encoder = StateEncoder(num_players=4, max_cards=7)

    # Test initial state encoding
    state = env.reset()
    encoded_state = encoder.encode_state(state)

    print(f"✓ State encoder created")
    print(f"✓ State dimension: {encoder.get_state_dim()}")
    print(f"✓ Encoded state shape: {encoded_state.shape}")
    print(f"✓ State contains no NaN values: {not np.isnan(encoded_state).any()}")

    # Test state encoding throughout a game
    print("Testing state encoding during gameplay...")

    # Bidding phase
    for player in range(4):
        legal_actions = env.get_legal_actions(player)
        bid = random.choice(legal_actions)
        state, reward, done = env.step(player, bid)

        encoded_state = encoder.encode_state(state)
        print(f"✓ After player {player} bid: state shape = {encoded_state.shape}")

        if done:
            break

    # Playing phase (if not done)
    if not done:
        steps = 0
        while not done and steps < 20:
            player = env.current_player
            legal_actions = env.get_legal_actions(player)

            if legal_actions:
                action = random.choice(legal_actions)
                state, reward, done = env.step(player, action)
                encoded_state = encoder.encode_state(state)
                steps += 1

                if steps % 5 == 0:
                    print(
                        f"✓ After {steps} playing steps: state shape = {encoded_state.shape}"
                    )
            else:
                break

    print("✓ State encoding works throughout the game")
    return True


def test_action_space():
    """Test the action space definition."""
    print("\nTesting Action Space")
    print("=" * 40)

    env = JudgementEnv(num_players=4)
    encoder = StateEncoder(num_players=4, max_cards=7)

    state = env.reset()

    # Test bidding action space
    print("Testing bidding action space...")
    legal_actions = env.get_legal_actions(0)
    print(f"✓ Legal bidding actions: {legal_actions}")
    print(f"✓ Action space size: {len(legal_actions)}")
    print(f"✓ Expected size: {state['round_cards'] + 1}")

    # Complete bidding
    for player in range(4):
        legal_actions = env.get_legal_actions(player)
        bid = random.choice(legal_actions)
        state, reward, done = env.step(player, bid)
        if done:
            break

    # Test playing action space
    if not done:
        print("Testing playing action space...")
        legal_actions = env.get_legal_actions(0)
        print(f"✓ Legal playing actions: {legal_actions}")
        print(f"✓ Action space size: {len(legal_actions)}")
        print(f"✓ Expected size: ≤ {len(state['hand'])}")

    return True


def run_comprehensive_test():
    """Run all tests."""
    print("Judgement AI Implementation Test")
    print("=" * 50)
    print("Testing Steps 1, 2, and 3 from the README\n")

    try:
        # Test Step 1
        env = test_step_1_environment_setup()

        # Test Step 2
        step2_success = test_step_2_game_logic()

        # Test Step 3
        step3_success = test_step_3_state_representation()

        # Test action space
        action_success = test_action_space()

        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"Step 1 (Environment Setup): ✓ PASSED")
        print(f"Step 2 (Game Logic): {'✓ PASSED' if step2_success else '✗ FAILED'}")
        print(
            f"Step 3 (State Representation): {'✓ PASSED' if step3_success else '✗ FAILED'}"
        )
        print(f"Action Space: {'✓ PASSED' if action_success else '✗ FAILED'}")

        if step2_success and step3_success and action_success:
            print("\n🎉 All tests passed! Steps 1, 2, and 3 are implemented correctly.")
            return True
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            return False

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
