"""
Tests for the Judgement environment.

This module tests the game environment, including game logic, state management,
action validation, and reward calculation.
"""

import pytest
import numpy as np
from typing import Dict, Any

from judgement_rl.environment.judgement_env import JudgementEnv


class TestJudgementEnv:
    """Test suite for JudgementEnv class."""

    def test_initialization(self, env_config):
        """Test environment initialization."""
        env = JudgementEnv(
            num_players=env_config.num_players, max_cards=env_config.max_cards
        )

        assert env.num_players == env_config.num_players
        assert env.max_cards == env_config.max_cards
        assert len(env.trump_suits) == 5
        assert env.round_number == 0
        assert env.total_rounds == 12  # 7->2, 2->7 = 12 rounds

    def test_reset(self, environment):
        """Test environment reset functionality."""
        state = environment.reset()

        # Check state structure
        required_keys = {
            "hand",
            "trump",
            "declarations",
            "current_trick",
            "tricks_won",
            "round_cards",
            "phase",
            "current_player",
        }
        assert all(key in state for key in required_keys)

        # Check initial values
        assert environment.round_number == 1
        assert environment.phase == "bidding"
        assert len(environment.hands[0]) == environment.round_cards
        assert all(decl is None for decl in environment.declarations)
        assert all(tricks == 0 for tricks in environment.tricks_won)
        assert len(environment.current_trick) == 0

    def test_deck_creation(self, environment):
        """Test deck creation and dealing."""
        environment.reset()

        # Check deck size
        assert len(environment.deck) == 52

        # Check all hands have correct number of cards
        for hand in environment.hands:
            assert len(hand) == environment.round_cards

        # Check no duplicate cards
        all_cards = []
        for hand in environment.hands:
            all_cards.extend(hand)
        assert len(all_cards) == len(set(all_cards))

    def test_trump_rotation(self, environment):
        """Test trump suit rotation across rounds."""
        trumps_seen = []

        for _ in range(5):  # Test 5 rounds
            state = environment.reset()
            trumps_seen.append(state["trump"])

        # Should see different trump suits
        assert len(set(trumps_seen)) >= 3  # At least 3 different trumps

    def test_bidding_phase(self, environment):
        """Test bidding phase logic."""
        state = environment.reset()
        assert state["phase"] == "bidding"

        # Test legal actions
        legal_actions = environment.get_legal_actions(0)
        assert len(legal_actions) == environment.round_cards + 1
        assert all(0 <= action <= environment.round_cards for action in legal_actions)

        # Test bidding
        for player in range(environment.num_players):
            legal_actions = environment.get_legal_actions(player)
            bid = legal_actions[0]  # Take first legal action

            state, reward, done = environment.step(player, bid)

            if player < environment.num_players - 1:
                assert not done
                assert reward == 0
            else:
                # Last player - check if game moves to playing phase
                if sum(environment.declarations) != environment.num_players:
                    assert state["phase"] == "playing"

    def test_invalid_bidding(self, environment):
        """Test invalid bidding scenarios."""
        environment.reset()

        # Test invalid bid (too high)
        invalid_bid = environment.round_cards + 1
        state, reward, done = environment.step(0, invalid_bid)
        assert done
        assert reward < 0  # Should be penalized

    def test_bidding_total_rule(self, environment):
        """Test that total bids cannot equal number of players."""
        environment.reset()

        # Set up scenario where total would equal num_players
        bids = [1, 1, 1, 0]  # Total = 3, but we need to test last player

        for i in range(3):
            state, reward, done = environment.step(i, bids[i])
            assert not done

        # Last player tries to bid 1, making total = 4 (num_players)
        state, reward, done = environment.step(3, 1)
        assert done
        assert reward < 0  # Should be penalized

    def test_playing_phase(self, environment):
        """Test playing phase logic."""
        # Complete bidding first
        environment.reset()
        for player in range(environment.num_players):
            legal_actions = environment.get_legal_actions(player)
            bid = legal_actions[0]
            state, reward, done = environment.step(player, bid)
            if done:
                break

        if not done:  # If bidding completed successfully
            # Now in playing phase
            assert state["phase"] == "playing"

            # Test playing cards
            for _ in range(environment.round_cards):
                for player in range(environment.num_players):
                    if len(environment.hands[player]) > 0:
                        legal_actions = environment.get_legal_actions(player)
                        if legal_actions:
                            card_idx = legal_actions[0]
                            state, reward, done = environment.step(player, card_idx)

                            if done:
                                break
                if done:
                    break

    def test_card_legality(self, environment):
        """Test card play legality rules."""
        environment.reset()

        # Complete bidding
        for player in range(environment.num_players):
            legal_actions = environment.get_legal_actions(player)
            bid = legal_actions[0]
            state, reward, done = environment.step(player, bid)
            if done:
                break

        if not done:
            # Test following suit rule
            # Play first card to establish led suit
            first_player = environment.current_player
            legal_actions = environment.get_legal_actions(first_player)
            first_card_idx = legal_actions[0]
            first_card = environment.hands[first_player][first_card_idx]
            led_suit = environment._get_card_suit(first_card)

            state, reward, done = environment.step(first_player, first_card_idx)

            # Next player must follow suit if possible
            next_player = (first_player + 1) % environment.num_players
            if len(environment.hands[next_player]) > 0:
                # Check if they have cards of led suit
                has_led_suit = any(
                    environment._get_card_suit(card) == led_suit
                    for card in environment.hands[next_player]
                )

                if has_led_suit:
                    # Must play led suit
                    legal_actions = environment.get_legal_actions(next_player)
                    for action in legal_actions:
                        card = environment.hands[next_player][action]
                        if environment._get_card_suit(card) != led_suit:
                            # This should be illegal
                            assert not environment._is_legal_card(next_player, card)

    def test_trick_resolution(self, environment):
        """Test trick resolution logic."""
        environment.reset()

        # Complete bidding
        for player in range(environment.num_players):
            legal_actions = environment.get_legal_actions(player)
            bid = legal_actions[0]
            state, reward, done = environment.step(player, bid)
            if done:
                break

        if not done:
            # Play a complete trick
            for player in range(environment.num_players):
                legal_actions = environment.get_legal_actions(player)
                if legal_actions:
                    card_idx = legal_actions[0]
                    state, reward, done = environment.step(player, card_idx)

            # Check that trick was resolved
            assert len(environment.current_trick) == 0
            assert environment.led_suit is None
            assert environment.led_player is None

    def test_reward_calculation(self, environment):
        """Test reward calculation."""
        environment.reset()

        # Complete a full game
        done = False
        while not done:
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action = legal_actions[0]
                state, reward, done = environment.step(current_player, action)

        # Check final rewards
        assert isinstance(reward, (int, float))
        # Rewards should be reasonable (not extreme values)
        assert -1000 < reward < 1000

    def test_game_completion(self, environment):
        """Test that games complete properly."""
        environment.reset()

        # Play until completion
        done = False
        steps = 0
        max_steps = 100  # Prevent infinite loops

        while not done and steps < max_steps:
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action = legal_actions[0]
                state, reward, done = environment.step(current_player, action)
                steps += 1
            else:
                break

        # Game should complete
        assert done or steps < max_steps

        # All hands should be empty
        assert all(len(hand) == 0 for hand in environment.hands)

    def test_state_consistency(self, environment):
        """Test that state remains consistent throughout game."""
        state = environment.reset()

        # Track state through several steps
        for _ in range(10):
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action = legal_actions[0]
                next_state, reward, done = environment.step(current_player, action)

                # Check state consistency
                assert isinstance(next_state, dict)
                assert "hand" in next_state
                assert "trump" in next_state
                assert "phase" in next_state

                if done:
                    break

    def test_edge_cases(self, environment):
        """Test edge cases and error conditions."""
        # Test with different player counts
        env_2p = JudgementEnv(num_players=2, max_cards=7)
        state = env_2p.reset()
        assert env_2p.num_players == 2

        # Test with different max cards
        env_small = JudgementEnv(num_players=4, max_cards=3)
        state = env_small.reset()
        assert env_small.max_cards == 3
        assert env_small.round_cards <= 3

    def test_card_utilities(self, environment):
        """Test card utility functions."""
        # Test suit extraction
        assert environment._get_card_suit("A of Spades") == "Spades"
        assert environment._get_card_suit("10 of Hearts") == "Hearts"

        # Test rank extraction
        assert environment._get_card_rank("A of Spades") == "A"
        assert environment._get_card_rank("10 of Hearts") == "10"

    def test_legal_actions_consistency(self, environment):
        """Test that legal actions are consistent with game state."""
        state = environment.reset()

        for _ in range(5):  # Test a few steps
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            # Legal actions should be valid indices
            if environment.phase == "bidding":
                assert all(
                    0 <= action <= environment.round_cards for action in legal_actions
                )
            else:  # playing phase
                assert all(
                    0 <= action < len(environment.hands[current_player])
                    for action in legal_actions
                )

            if legal_actions:
                action = legal_actions[0]
                state, reward, done = environment.step(current_player, action)
                if done:
                    break


class TestEnvironmentIntegration:
    """Integration tests for environment with other components."""

    def test_environment_with_encoder(self, environment, state_encoder):
        """Test environment works with state encoder."""
        state = environment.reset()
        encoded_state = state_encoder.encode_state(state)

        assert isinstance(encoded_state, np.ndarray)
        assert encoded_state.shape[0] == state_encoder.get_state_dim()

    def test_environment_with_agent(self, environment, ppo_agent):
        """Test environment works with PPO agent."""
        state = environment.reset()

        for _ in range(5):  # Test a few steps
            current_player = environment.current_player
            legal_actions = environment.get_legal_actions(current_player)

            if legal_actions:
                action, prob, value = ppo_agent.select_action(state, legal_actions)
                state, reward, done = environment.step(current_player, action)

                if done:
                    break
