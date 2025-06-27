import random


class HeuristicAgent:
    def __init__(self, player_idx):
        self.player_idx = player_idx

    def select_bid(self, state, legal_bids):
        # Simple heuristic: bid number of high cards (J/Q/K/A or trump)
        hand = state["hand"]
        trump = state["trump"]
        high_ranks = {"J", "Q", "K", "A"}
        count = 0
        for card in hand:
            rank, suit = card.split(" of ")
            if rank in high_ranks or (trump != "No Trump" and suit == trump):
                count += 1
        # Clamp to legal bids
        if count in legal_bids:
            return count
        return random.choice(legal_bids)

    def select_card(self, state, legal_cards):
        # Play the lowest legal card (by rank)
        hand = state["hand"]
        rank_order = {str(i): i for i in range(2, 11)}
        rank_order.update({"J": 11, "Q": 12, "K": 13, "A": 14})
        min_idx = min(
            legal_cards, key=lambda idx: rank_order[hand[idx].split(" of ")[0]]
        )
        return min_idx
