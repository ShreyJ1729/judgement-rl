import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class JudgementEnv:
    """
    Environment for the Judgement card game.
    Implements the game logic, state management, and reward system.
    """

    def __init__(self, num_players: int = 4, max_cards: int = 7):
        self.num_players = num_players
        self.max_cards = max_cards
        self.trump_suits = ["No Trump", "Spades", "Diamonds", "Clubs", "Hearts"]
        self.current_trump_index = 0
        # Precompute the round card counts for a full game, skipping 1-card round
        self.round_card_counts = [i for i in range(self.max_cards, 1, -1)] + [
            i for i in range(2, self.max_cards + 1)
        ]
        self.total_rounds = len(self.round_card_counts)
        self.round_number = 0  # Start at 0, will be 1 after first reset
        # Don't call reset() here - let the user call it explicitly

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment for a new game.
        Returns the initial state for player 0.
        """
        # Determine number of cards for this round using the precomputed list
        if self.round_number >= self.total_rounds:
            self.round_number = 0  # Restart the cycle if needed
        self.round_cards = self.round_card_counts[self.round_number]

        # Rotate trump suit
        self.trump = self.trump_suits[self.current_trump_index]
        self.current_trump_index = (self.current_trump_index + 1) % len(
            self.trump_suits
        )

        # Create and deal cards
        self.deck = self._create_deck()
        self.hands = self._deal_cards()

        # Initialize game state
        self.declarations = [None] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.current_trick = []
        self.led_suit = None
        self.phase = "bidding"  # "bidding" or "playing"

        # Set the first bidder for this round
        self.bidding_start_player = (self.round_number) % self.num_players
        self.current_player = self.bidding_start_player

        # Only increment round_number after using it
        self.round_number += 1
        return self._get_state(self.current_player)

    def _create_deck(self) -> List[str]:
        """Create a standard 52-card deck."""
        suits = ["Spades", "Diamonds", "Clubs", "Hearts"]
        ranks = [str(i) for i in range(2, 11)] + ["J", "Q", "K", "A"]
        return [f"{rank} of {suit}" for suit in suits for rank in ranks]

    def _deal_cards(self) -> List[List[str]]:
        """Deal cards to all players."""
        deck_copy = self.deck.copy()
        random.shuffle(deck_copy)
        hands = []
        for i in range(self.num_players):
            start_idx = i * self.round_cards
            end_idx = start_idx + self.round_cards
            hands.append(deck_copy[start_idx:end_idx])
        return hands

    def _get_state(self, player_idx: int) -> Dict[str, Any]:
        """
        Get the current state for a specific player.
        This is the state representation for Step 3.
        """
        state = {
            "hand": self.hands[player_idx].copy(),
            "trump": self.trump,
            "declarations": self.declarations.copy(),
            "current_trick": self.current_trick.copy(),
            "tricks_won": self.tricks_won.copy(),
            "round_cards": self.round_cards,
            "phase": self.phase,
            "current_player": self.current_player,
            "led_suit": self.led_suit,
        }

        # Add visible cards in the one-card round
        if self.round_cards == 1:
            state["others_cards"] = [
                self.hands[i][0] for i in range(self.num_players) if i != player_idx
            ]

        return state

    def step(self, player_idx: int, action: Any) -> Tuple[Dict[str, Any], float, bool]:
        """
        Take a step in the environment.
        Action can be either a bid (int) or card index (int).
        Returns (next_state, reward, done)
        """
        if self.phase == "bidding":
            return self._handle_bidding(player_idx, action)
        else:
            return self._handle_card_play(player_idx, action)

    def _handle_bidding(
        self, player_idx: int, bid: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Handle the bidding phase."""
        # Validate bid
        if not (0 <= bid <= self.round_cards):
            return self._get_state(player_idx), -100, True  # Invalid bid penalty

        self.declarations[player_idx] = bid

        # Check if all players have bid
        if all(decl is not None for decl in self.declarations):
            # Check if total bids equal number of players (invalid)
            total_bids = sum(self.declarations)
            if total_bids == self.num_players:
                return self._get_state(player_idx), -100, True  # Invalid total penalty

            # Move to playing phase
            self.phase = "playing"
            # Set current_player to the player with the highest bid (first to play)
            max_bid = max(self.declarations)
            # If multiple players have the same max bid, pick the first one
            self.current_player = self.declarations.index(max_bid)
            return self._get_state(self.current_player), 0, False
        else:
            # Move to next player in bidding order
            next_player = (
                player_idx - self.bidding_start_player + 1
            ) % self.num_players + self.bidding_start_player
            next_player = next_player % self.num_players
            self.current_player = next_player
            return self._get_state(next_player), 0, False

    def _handle_card_play(
        self, player_idx: int, card_idx: int
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Handle the card playing phase."""
        # Validate card index
        if not (0 <= card_idx < len(self.hands[player_idx])):
            return self._get_state(player_idx), -100, True  # Invalid card penalty

        # Get the card and check if it's legal
        card = self.hands[player_idx][card_idx]
        if not self._is_legal_card(player_idx, card):
            return self._get_state(player_idx), -100, True  # Illegal play penalty

        # Remove card from hand and add to current trick
        self.hands[player_idx].pop(card_idx)
        self.current_trick.append((player_idx, card))
        print(f"Player {player_idx} plays: {card}")

        # Set led suit if this is the first card
        if len(self.current_trick) == 1:
            self.led_suit = self._get_card_suit(card)

        # Check if trick is complete
        if len(self.current_trick) == self.num_players:
            # Resolve the trick
            winner = self._resolve_trick()
            self.tricks_won[winner] += 1
            print(f"Player {winner} won the trick!")
            self.current_trick = []
            self.led_suit = None

            # Check if round is complete
            if len(self.hands[0]) == 0:
                # Calculate final rewards
                rewards = [self._calculate_reward(i) for i in range(self.num_players)]
                return self._get_state(winner), rewards[player_idx], True

            # Start new trick: winner goes first
            self.current_player = winner
            print(f"Next trick starts with Player {winner}")
            return self._get_state(winner), 0, False
        else:
            # Move to next player in order (0, 1, 2, 3)
            next_player = (player_idx + 1) % self.num_players
            self.current_player = next_player
            return self._get_state(next_player), 0, False

    def _is_legal_card(self, player_idx: int, card: str) -> bool:
        """Check if a card play is legal (must follow suit if possible)."""
        if self.led_suit is None:
            return True  # First card of the trick

        card_suit = self._get_card_suit(card)
        if card_suit == self.led_suit:
            return True  # Following suit

        # Check if player has any cards of the led suit
        hand = self.hands[player_idx]
        has_led_suit = any(self._get_card_suit(c) == self.led_suit for c in hand)

        return not has_led_suit  # Legal if no led suit in hand

    def _get_card_suit(self, card: str) -> str:
        """Extract the suit from a card string."""
        return card.split(" of ")[1]

    def _get_card_rank(self, card: str) -> str:
        """Extract the rank from a card string."""
        return card.split(" of ")[0]

    def _resolve_trick(self) -> int:
        """Determine the winner of the current trick."""
        if not self.current_trick:
            return 0

        # Define rank values for comparison
        rank_values = {str(i): i for i in range(2, 11)}
        rank_values.update({"J": 11, "Q": 12, "K": 13, "A": 14})

        best_card = None
        best_player = 0
        best_rank_value = 0
        best_is_trump = False

        # Determine the effective trump for this trick
        trick_trump = self.trump
        if trick_trump == "No Trump":
            trick_trump = self.led_suit  # Led suit acts as trump

        for player_idx, card in self.current_trick:
            card_rank = self._get_card_rank(card)
            card_suit = self._get_card_suit(card)
            rank_value = rank_values[card_rank]
            is_trump = trick_trump is not None and card_suit == trick_trump

            # Check if this card is better than the current best
            is_better = False

            # Trump cards beat non-trump cards
            if is_trump and not best_is_trump:
                is_better = True
            # If both are trump, highest rank wins
            elif is_trump and best_is_trump:
                if rank_value > best_rank_value:
                    is_better = True
            # If both are non-trump, highest card of led suit wins
            elif not is_trump and not best_is_trump:
                if card_suit == self.led_suit:
                    if (
                        best_card is None
                        or self._get_card_suit(best_card) != self.led_suit
                    ):
                        is_better = True
                    elif rank_value > best_rank_value:
                        is_better = True

            if is_better:
                best_card = card
                best_player = player_idx
                best_rank_value = rank_value
                best_is_trump = is_trump

        return best_player

    def _calculate_reward(self, player_idx: int) -> float:
        """
        Calculate the reward for a player at the end of the round.
        This implements the reward system from Step 5.
        """
        n = self.declarations[player_idx]
        actual = self.tricks_won[player_idx]

        if n == actual:
            return 11 * n + 10  # Bonus for exact bid
        else:
            return -10 * abs(n - actual)  # Penalty for missing bid

    def get_legal_actions(self, player_idx: int) -> List[int]:
        """Get the list of legal actions for the current player."""
        if self.phase == "bidding":
            # All bids from 0 to round_cards are legal, except for last bidder
            legal_bids = list(range(self.round_cards + 1))
            # Last bidder restriction
            num_bids = sum(1 for d in self.declarations if d is not None)
            if num_bids == self.num_players - 1:
                forbidden = self.round_cards - sum(
                    d for d in self.declarations if d is not None
                )
                if 0 <= forbidden <= self.round_cards:
                    legal_bids.remove(forbidden)
            return legal_bids
        else:
            # Legal card indices (cards that follow suit if possible)
            legal_cards = []
            for i, card in enumerate(self.hands[player_idx]):
                if self._is_legal_card(player_idx, card):
                    legal_cards.append(i)
            return legal_cards

    def render(self):
        """Render the current game state for debugging."""
        print(f"\n=== Round {self.round_number} ===")
        print(f"Trump: {self.trump}")
        print(f"Cards per player: {self.round_cards}")
        print(f"Phase: {self.phase}")
        print(f"Declarations: {self.declarations}")
        print(f"Tricks won: {self.tricks_won}")
        print(f"Current trick: {self.current_trick}")
        print(f"Current player: {self.current_player}")
        for i, hand in enumerate(self.hands):
            print(f"Player {i} hand: {hand}")


# Example usage and testing
if __name__ == "__main__":
    # Test the environment
    env = JudgementEnv(num_players=4)

    print("Testing Judgement Environment")
    print("=" * 40)

    # Test a few rounds
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        state = env.reset()
        env.render()

        # Simulate bidding
        for player in range(4):
            legal_actions = env.get_legal_actions(player)
            bid = random.choice(legal_actions)
            state, reward, done = env.step(player, bid)
            print(f"Player {player} bids: {bid}")
            if done:
                print(f"Game ended with reward: {reward}")
                break

        if not done:
            # Simulate card playing
            while not done:
                player = env.current_player
                legal_actions = env.get_legal_actions(player)
                if legal_actions:
                    action = random.choice(legal_actions)
                    state, reward, done = env.step(player, action)
                    print(f"Player {player} plays card {action}")
                else:
                    break

        print(f"Final state: {state}")
        print(f"Final reward: {reward}")
