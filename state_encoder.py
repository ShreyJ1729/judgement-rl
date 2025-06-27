import numpy as np
from typing import Dict, List, Any, Tuple


class StateEncoder:
    """
    Encodes the game state into numerical representations for the RL agent.
    This implements the state representation from Step 3.
    """

    def __init__(self, num_players: int = 4, max_cards: int = 7):
        self.num_players = num_players
        self.max_cards = max_cards

        # Card encoding constants
        self.suits = ["Spades", "Diamonds", "Clubs", "Hearts"]
        self.ranks = [str(i) for i in range(2, 11)] + ["J", "Q", "K", "A"]
        self.trump_suits = ["No Trump", "Spades", "Diamonds", "Clubs", "Hearts"]

        # Create mappings
        self.suit_to_idx = {suit: idx for idx, suit in enumerate(self.suits)}
        self.rank_to_idx = {rank: idx for idx, rank in enumerate(self.ranks)}
        self.trump_to_idx = {trump: idx for idx, trump in enumerate(self.trump_suits)}

        # Calculate state dimensions
        self.card_dim = len(self.suits) + len(
            self.ranks
        )  # One-hot encoding for each card
        self.hand_dim = self.max_cards * self.card_dim
        self.state_dim = self._calculate_state_dim()

    def _calculate_state_dim(self) -> int:
        """Calculate the total dimension of the encoded state."""
        # Hand encoding
        hand_size = self.hand_dim

        # Trump suit (one-hot)
        trump_size = len(self.trump_suits)

        # Declarations (one-hot for each player)
        declarations_size = self.num_players * (self.max_cards + 1)

        # Tricks won (one-hot for each player)
        tricks_won_size = self.num_players * (self.max_cards + 1)

        # Current trick (max cards in trick * card encoding)
        current_trick_size = self.num_players * self.card_dim

        # Phase (one-hot: bidding/playing)
        phase_size = 2

        # Current player (one-hot)
        current_player_size = self.num_players

        # Led suit (one-hot + none)
        led_suit_size = len(self.suits) + 1

        # Led player (one-hot - always exists during playing)
        led_player_size = self.num_players

        # Others cards (for one-card rounds)
        others_cards_size = (self.num_players - 1) * self.card_dim

        return (
            hand_size
            + trump_size
            + declarations_size
            + tricks_won_size
            + current_trick_size
            + phase_size
            + current_player_size
            + led_suit_size
            + led_player_size
            + others_cards_size
        )

    def encode_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Encode the game state into a numerical vector.
        Returns a numpy array representing the state.
        """
        encoded = []

        # Encode hand
        encoded.extend(self._encode_hand(state["hand"]))

        # Encode trump suit
        encoded.extend(self._encode_trump(state["trump"]))

        # Encode declarations
        encoded.extend(self._encode_declarations(state["declarations"]))

        # Encode tricks won
        encoded.extend(self._encode_tricks_won(state["tricks_won"]))

        # Encode current trick
        encoded.extend(self._encode_current_trick(state["current_trick"]))

        # Encode phase
        encoded.extend(self._encode_phase(state["phase"]))

        # Encode current player
        encoded.extend(self._encode_current_player(state["current_player"]))

        # Encode led suit
        encoded.extend(self._encode_led_suit(state.get("led_suit")))

        # Encode led player
        encoded.extend(self._encode_led_player(state.get("led_player")))

        # Encode others cards (for one-card rounds)
        encoded.extend(self._encode_others_cards(state.get("others_cards", [])))

        return np.array(encoded, dtype=np.float32)

    def _encode_hand(self, hand: List[str]) -> List[float]:
        """Encode the player's hand."""
        encoding = [0.0] * self.hand_dim

        for i, card in enumerate(hand):
            if i < self.max_cards:
                card_encoding = self._encode_card(card)
                start_idx = i * self.card_dim
                encoding[start_idx : start_idx + self.card_dim] = card_encoding

        return encoding

    def _encode_card(self, card: str) -> List[float]:
        """Encode a single card."""
        if not card:
            return [0.0] * self.card_dim

        rank, suit = card.split(" of ")

        # One-hot encoding for suit
        suit_encoding = [0.0] * len(self.suits)
        if suit in self.suit_to_idx:
            suit_encoding[self.suit_to_idx[suit]] = 1.0

        # One-hot encoding for rank
        rank_encoding = [0.0] * len(self.ranks)
        if rank in self.rank_to_idx:
            rank_encoding[self.rank_to_idx[rank]] = 1.0

        return suit_encoding + rank_encoding

    def _encode_trump(self, trump: str) -> List[float]:
        """Encode the trump suit."""
        encoding = [0.0] * len(self.trump_suits)
        if trump in self.trump_to_idx:
            encoding[self.trump_to_idx[trump]] = 1.0
        return encoding

    def _encode_declarations(self, declarations: List[int]) -> List[float]:
        """Encode player declarations."""
        encoding = []
        for decl in declarations:
            if decl is None:
                # No declaration yet
                decl_encoding = [0.0] * (self.max_cards + 1)
            else:
                # One-hot encoding for the declaration
                decl_encoding = [0.0] * (self.max_cards + 1)
                if 0 <= decl <= self.max_cards:
                    decl_encoding[decl] = 1.0
            encoding.extend(decl_encoding)
        return encoding

    def _encode_tricks_won(self, tricks_won: List[int]) -> List[float]:
        """Encode tricks won by each player."""
        encoding = []
        for tricks in tricks_won:
            tricks_encoding = [0.0] * (self.max_cards + 1)
            if 0 <= tricks <= self.max_cards:
                tricks_encoding[tricks] = 1.0
            encoding.extend(tricks_encoding)
        return encoding

    def _encode_current_trick(
        self, current_trick: List[Tuple[int, str]]
    ) -> List[float]:
        """Encode the current trick."""
        encoding = [0.0] * (self.num_players * self.card_dim)

        for i, (player_idx, card) in enumerate(current_trick):
            if i < self.num_players:
                card_encoding = self._encode_card(card)
                start_idx = player_idx * self.card_dim
                encoding[start_idx : start_idx + self.card_dim] = card_encoding

        return encoding

    def _encode_phase(self, phase: str) -> List[float]:
        """Encode the game phase."""
        encoding = [0.0, 0.0]  # [bidding, playing]
        if phase == "bidding":
            encoding[0] = 1.0
        elif phase == "playing":
            encoding[1] = 1.0
        return encoding

    def _encode_current_player(self, current_player: int) -> List[float]:
        """Encode the current player."""
        encoding = [0.0] * self.num_players
        if 0 <= current_player < self.num_players:
            encoding[current_player] = 1.0
        return encoding

    def _encode_led_suit(self, led_suit: str) -> List[float]:
        """Encode the led suit."""
        encoding = [0.0] * (len(self.suits) + 1)  # +1 for no led suit
        if led_suit is None:
            encoding[-1] = 1.0  # No led suit
        elif led_suit in self.suit_to_idx:
            encoding[self.suit_to_idx[led_suit]] = 1.0
        return encoding

    def _encode_led_player(self, led_player: int) -> List[float]:
        """Encode the led player."""
        encoding = [0.0] * self.num_players
        if led_player is not None and 0 <= led_player < self.num_players:
            encoding[led_player] = 1.0
        # If led_player is None (during bidding), all zeros is fine
        return encoding

    def _encode_others_cards(self, others_cards: List[str]) -> List[float]:
        """Encode other players' cards (for one-card rounds)."""
        encoding = [0.0] * ((self.num_players - 1) * self.card_dim)

        for i, card in enumerate(others_cards):
            if i < self.num_players - 1:
                card_encoding = self._encode_card(card)
                start_idx = i * self.card_dim
                encoding[start_idx : start_idx + self.card_dim] = card_encoding

        return encoding

    def get_state_dim(self) -> int:
        """Get the dimension of the encoded state."""
        return self.state_dim

    def get_action_dim(self, round_cards: int) -> int:
        """Get the dimension of the action space for a given round."""
        # During bidding: 0 to round_cards
        # During playing: 0 to round_cards - 1 (card indices)
        return round_cards + 1


# Example usage
if __name__ == "__main__":
    from judgement_env import JudgementEnv

    # Test the state encoder
    env = JudgementEnv(num_players=4)
    encoder = StateEncoder(num_players=4, max_cards=7)

    print("Testing State Encoder")
    print("=" * 30)

    # Get initial state
    state = env.reset()
    print(f"Original state keys: {list(state.keys())}")

    # Encode state
    encoded_state = encoder.encode_state(state)
    print(f"Encoded state shape: {encoded_state.shape}")
    print(f"State dimension: {encoder.get_state_dim()}")
    print(f"Action dimension: {encoder.get_action_dim(state['round_cards'])}")

    # Test a few steps
    for step in range(3):
        legal_actions = env.get_legal_actions(env.current_player)
        action = legal_actions[0]  # Take first legal action
        state, reward, done = env.step(env.current_player, action)

        encoded_state = encoder.encode_state(state)
        print(
            f"Step {step + 1}: State shape = {encoded_state.shape}, Reward = {reward}"
        )

        if done:
            break
