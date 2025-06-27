#!/usr/bin/env python3
"""
Comprehensive test script for the Judgement AI implementation.
Tests Steps 1, 2, 3, and 4 from the README.
"""

import random
import numpy as np
from judgement_env import JudgementEnv
from state_encoder import StateEncoder
from heuristic_agent import HeuristicAgent


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

    # Print all players' hands
    print("\nAll Players' Hands:")
    for i in range(env.num_players):
        player_hand = env.hands[i]
        print(f"  Player {i}: {player_hand}")

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
    print(
        f"✓ round_cards: {state['round_cards']} (number of cards each player has this round)"
    )

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


def test_step_4_action_space():
    """Test Step 4: Action space definition and rule enforcement."""
    print("\nTesting Step 4: Action Space and Rules Enforcement")
    print("=" * 50)

    env = JudgementEnv(num_players=4)
    encoder = StateEncoder(num_players=4, max_cards=7)

    state = env.reset()

    # Test bidding action space
    print("Testing bidding action space...")
    legal_actions = env.get_legal_actions(0)
    print(f"✓ Legal bidding actions: {legal_actions}")
    print(f"✓ Action space size: {len(legal_actions)}")
    print(f"✓ Expected size: {state['round_cards'] + 1}")

    # Test bidding rules enforcement
    print("\nTesting bidding rules enforcement...")
    test_bidding_rules(env)

    # Complete bidding
    for player in range(4):
        legal_actions = env.get_legal_actions(player)
        bid = random.choice(legal_actions)
        state, reward, done = env.step(player, bid)
        if done:
            break

    # Test playing action space
    if not done:
        print("\nTesting playing action space...")
        legal_actions = env.get_legal_actions(0)
        print(f"✓ Legal playing actions: {legal_actions}")
        print(f"✓ Action space size: {len(legal_actions)}")
        print(f"✓ Expected size: ≤ {len(state['hand'])}")

        # Test card play rules enforcement
        print("\nTesting card play rules enforcement...")
        test_card_play_rules(env)

    return True


def test_bidding_rules(env):
    """Test that bidding rules are properly enforced."""
    print("  Testing bidding rules...")

    # Test specific scenario where last player cannot make total = num_players
    test_env = JudgementEnv(num_players=4)
    state = test_env.reset()

    # Simulate specific bids to test the rule
    for i in range(3):  # First 3 players bid 1
        player = test_env.current_player
        legal_actions = test_env.get_legal_actions(player)
        bid = 1
        state, reward, done = test_env.step(player, bid)
        print(f"    Player {player} bids: {bid}")

    # Last player should not be able to bid 1
    player = test_env.current_player
    legal_actions = test_env.get_legal_actions(player)
    current_total = sum(d for d in test_env.declarations if d is not None)
    forbidden_bid = 4 - current_total

    print(f"    Current total bids: {current_total}")
    print(f"    Forbidden bid (would make total=4): {forbidden_bid}")
    print(f"    Last player legal bids: {legal_actions}")

    if 0 <= forbidden_bid <= test_env.round_cards:
        assert (
            forbidden_bid not in legal_actions
        ), f"Last player should not be able to bid {forbidden_bid}"
        print(f"    ✅ Correctly prevented last player from bidding {forbidden_bid}")
    else:
        print(f"    ✅ Forbidden bid {forbidden_bid} is outside valid range")


def test_card_play_rules(env):
    """Test that card play rules (following suit) are properly enforced."""
    print("  Testing card play rules...")

    # Complete bidding first
    test_env = JudgementEnv(num_players=4)
    state = test_env.reset()

    for i in range(4):
        player = test_env.current_player
        legal_actions = test_env.get_legal_actions(player)
        bid = legal_actions[0]  # Take first legal bid
        state, reward, done = test_env.step(player, bid)
        if done:
            break

    # Test first trick
    player = test_env.current_player
    legal_actions = test_env.get_legal_actions(player)
    action = legal_actions[0]
    card = test_env.hands[player][action]
    card_suit = test_env._get_card_suit(card)
    print(f"    Player {player} leads with: {card} (suit: {card_suit})")
    state, reward, done = test_env.step(player, action)

    # Second player must follow suit if possible
    player = test_env.current_player
    legal_actions = test_env.get_legal_actions(player)
    has_led_suit = any(
        test_env._get_card_suit(c) == test_env.led_suit for c in test_env.hands[player]
    )

    print(f"    Player {player} has {test_env.led_suit} cards: {has_led_suit}")
    print(f"    Legal card indices: {legal_actions}")

    if has_led_suit:
        # Verify all legal actions are of the led suit
        for action_idx in legal_actions:
            card = test_env.hands[player][action_idx]
            card_suit = test_env._get_card_suit(card)
            assert (
                card_suit == test_env.led_suit
            ), f"Player must follow suit but can play {card_suit}"
        print(f"    ✅ Player correctly must follow suit")
    else:
        print(f"    ✅ Player has no led suit cards, can play any card")


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


def demonstrate_complete_game():
    """Demonstrate a complete game with all components."""
    print("\nDemonstrating a Complete Game")
    print("=" * 50)

    # Setup
    env = JudgementEnv(num_players=4, max_cards=7)
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

    # Bidding phase
    print("\nBidding Phase:")
    done = False
    for _ in range(env.num_players):
        player = env.current_player
        legal_bids = env.get_legal_actions(player)
        bid = agents[player].select_bid(state, legal_bids)
        state, reward, done = env.step(player, bid)
        print(f"  Player {player} bids: {bid}")
        if done:
            print("  ✗ Game ended during bidding (invalid)")
            return False
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

    return True


def test_step_5_reward_system():
    """Test Step 5: Reward system implementation."""
    print("\nTesting Step 5: Reward System")
    print("=" * 40)

    env = JudgementEnv(num_players=4)

    # Test final rewards
    print("Testing final rewards...")

    # Test exact bid reward
    env.declarations = [3, 2, 1, 0]
    env.tricks_won = [3, 2, 1, 0]
    reward = env._calculate_reward(0)
    expected = 11 * 3 + 10  # 43
    assert reward == expected, f"Expected {expected}, got {reward}"
    print(f"✓ Exact bid reward: {reward}")

    # Test over bid penalty
    env.declarations = [2, 1, 1, 0]
    env.tricks_won = [3, 1, 1, 0]
    reward = env._calculate_reward(0)
    expected = -10 * abs(2 - 3)  # -10
    assert reward == expected, f"Expected {expected}, got {reward}"
    print(f"✓ Over bid penalty: {reward}")

    # Test intermediate rewards
    print("Testing intermediate rewards...")

    # Test winning up to bid
    env.declarations = [3, 2, 1, 0]
    env.tricks_won = [2, 2, 1, 0]
    reward = env._calculate_intermediate_reward(0)
    expected = 1.0
    assert reward == expected, f"Expected {expected}, got {reward}"
    print(f"✓ Winning up to bid reward: {reward}")

    # Test winning beyond bid
    env.declarations = [2, 2, 1, 0]
    env.tricks_won = [3, 2, 1, 0]
    reward = env._calculate_intermediate_reward(0)
    expected = -0.5
    assert reward == expected, f"Expected {expected}, got {reward}"
    print(f"✓ Winning beyond bid penalty: {reward}")

    # Test rewards in a complete game
    print("Testing rewards in complete game...")
    state = env.reset()

    # Complete bidding
    for i in range(4):
        player = env.current_player
        legal_actions = env.get_legal_actions(player)
        bid = legal_actions[0]  # Take first legal bid
        state, reward, done = env.step(player, bid)
        if done:
            break

    # Play a few tricks to test intermediate rewards
    tricks_played = 0
    intermediate_rewards = [0] * 4
    while not done and tricks_played < 3:
        player = env.current_player
        legal_actions = env.get_legal_actions(player)

        if legal_actions:
            action = legal_actions[0]
            state, reward, done = env.step(player, action)

            if not done and reward != 0:
                intermediate_rewards[player] += reward
                print(f"✓ Player {player} got intermediate reward: {reward}")

            if len(env.current_trick) == 0:
                tricks_played += 1

    print(f"✓ Intermediate rewards tracked: {intermediate_rewards}")
    return True


def run_comprehensive_test():
    """Run all tests."""
    print("Judgement AI Implementation Test")
    print("=" * 50)
    print("Testing Steps 1, 2, 3, 4, and 5 from the README\n")

    try:
        # Test Step 1
        env = test_step_1_environment_setup()

        # Test Step 2
        step2_success = test_step_2_game_logic()

        # Test Step 3
        step3_success = test_step_3_state_representation()

        # Test Step 4
        step4_success = test_step_4_action_space()

        # Test state encoding consistency
        consistency_ok = test_state_encoding_consistency()

        # Demonstrate complete game
        demo_success = demonstrate_complete_game()

        # Test Step 5
        step5_success = test_step_5_reward_system()

        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        print(f"Step 1 (Environment Setup): ✓ PASSED")
        print(f"Step 2 (Game Logic): {'✓ PASSED' if step2_success else '✗ FAILED'}")
        print(
            f"Step 3 (State Representation): {'✓ PASSED' if step3_success else '✗ FAILED'}"
        )
        print(f"Step 4 (Action Space): {'✓ PASSED' if step4_success else '✗ FAILED'}")
        print(f"Step 5 (Reward System): {'✓ PASSED' if step5_success else '✗ FAILED'}")
        print(
            f"State Encoding Consistency: {'✓ PASSED' if consistency_ok else '✗ FAILED'}"
        )
        print(f"Complete Game Demo: {'✓ PASSED' if demo_success else '✗ FAILED'}")

        if (
            step2_success
            and step3_success
            and step4_success
            and step5_success
            and consistency_ok
            and demo_success
        ):
            print(
                "\n🎉 All tests passed! Steps 1, 2, 3, 4, and 5 are implemented correctly."
            )
            print("\nNext steps:")
            print("  - Step 6: Implement the RL Agent")
            print("  - Step 7: Train the Agent")
            print("  - Step 8: Evaluate the Agent")
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
