#!/usr/bin/env python3
"""
Debug script to test the new trick completion logic.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from judgement_rl.environment.judgement_env import JudgementEnv


def debug_trick_completion():
    """Debug the new trick completion logic step by step."""
    env = JudgementEnv(num_players=4, max_cards=7)
    state = env.reset()

    print("Initial state:")
    print(f"Phase: {env.phase}")
    print(f"Current player: {env.current_player}")
    print(f"Current trick: {env.current_trick}")

    # Skip bidding phase for this test
    env.phase = "playing"
    env.current_player = 0

    print("\nAfter setting phase to playing:")
    print(f"Phase: {env.phase}")
    print(f"Current player: {env.current_player}")
    print(f"Current trick: {env.current_trick}")

    # Simulate playing 4 cards to complete a trick
    for i in range(4):
        print(f"\n--- Player {i} playing card ---")
        print(f"Current trick BEFORE action: {env.current_trick}")
        print(f"Length of current trick: {len(env.current_trick)}")

        # Get legal actions
        legal_actions = env.get_legal_actions(i)
        print(f"Legal actions for player {i}: {legal_actions}")

        # Take first legal action
        if legal_actions:
            action = legal_actions[0]
            print(f"Taking action: {action}")

            # Capture trick state BEFORE the action
            current_trick_before = env.current_trick.copy()
            print(f"Captured trick before action: {current_trick_before}")
            print(f"Length of captured trick: {len(current_trick_before)}")

            # NEW LOGIC: If this is the 4th card, capture the complete trick
            if env.phase == "playing" and len(current_trick_before) == 3:
                # Get the card that will be played
                card = env.hands[i][action]
                # Create the complete trick by adding the current card
                complete_trick = current_trick_before + [(i, card)]
                print(
                    f"*** NEW LOGIC: Complete trick with 4th card: {complete_trick} ***"
                )
            else:
                complete_trick = current_trick_before

            next_state, reward, done = env.step(i, action)
            print(f"Action taken: {action}")
            print(f"Current trick AFTER action: {env.current_trick}")
            print(f"Length of current trick after: {len(env.current_trick)}")
            print(f"Done: {done}")

            # Check if this would trigger the delay logic
            if len(current_trick_before) == 3 and len(env.current_trick) == 0:
                print("*** This would trigger the delay logic! ***")
                print(f"Stored trick for display: {complete_trick}")
                print(f"Length of stored trick: {len(complete_trick)}")
            elif len(current_trick_before) == 3 and len(env.current_trick) == 4:
                print("*** This would trigger the 4-card delay logic! ***")
                print(f"Stored trick for display: {complete_trick}")
                print(f"Length of stored trick: {len(complete_trick)}")

            if done:
                print("Game is done!")
                break
        else:
            print(f"No legal actions for player {i}")
            break


if __name__ == "__main__":
    debug_trick_completion()
