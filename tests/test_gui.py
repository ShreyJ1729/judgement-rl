#!/usr/bin/env python3
"""
Test script to verify GUI components work correctly.
This script tests the imports and basic functionality without launching the full GUI.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from judgement_env import JudgementEnv

        print("✓ JudgementEnv imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import JudgementEnv: {e}")
        return False

    try:
        from state_encoder import StateEncoder

        print("✓ StateEncoder imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import StateEncoder: {e}")
        return False

    try:
        from agent import PPOAgent

        print("✓ PPOAgent imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PPOAgent: {e}")
        return False

    try:
        import tkinter as tk
        from tkinter import ttk

        print("✓ tkinter imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import tkinter: {e}")
        return False

    return True


def test_environment():
    """Test that the game environment works correctly."""
    print("\nTesting game environment...")

    try:
        from judgement_env import JudgementEnv
        from state_encoder import StateEncoder

        # Initialize environment
        env = JudgementEnv(num_players=4, max_cards=7)
        state_encoder = StateEncoder(num_players=4, max_cards=7)

        # Test reset
        state = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Round cards: {env.round_cards}")
        print(f"  - Trump: {env.trump}")
        print(f"  - Player hand size: {len(state['hand'])}")

        # Test legal actions
        legal_actions = env.get_legal_actions(0)
        print(f"✓ Legal actions generated: {len(legal_actions)} actions")

        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_agent_loading():
    """Test that agent loading works."""
    print("\nTesting agent loading...")

    try:
        from agent import PPOAgent
        from state_encoder import StateEncoder

        state_encoder = StateEncoder(num_players=4, max_cards=7)
        agent = PPOAgent(state_encoder)

        # Try to load the model
        model_path = "models/selfplay_best_agent.pth"
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"✓ Agent loaded successfully from {model_path}")
            return True
        else:
            print(f"⚠ No model found at {model_path}")
            print("  This is okay - the GUI will work with random AI")
            return True

    except Exception as e:
        print(f"✗ Agent loading test failed: {e}")
        return False


def test_gui_components():
    """Test that GUI components can be created."""
    print("\nTesting GUI components...")

    try:
        import tkinter as tk
        from tkinter import ttk

        # Create a minimal root window
        root = tk.Tk()
        root.withdraw()  # Hide the window

        # Test basic GUI creation
        frame = ttk.Frame(root)
        label = ttk.Label(frame, text="Test")
        button = ttk.Button(frame, text="Test")

        print("✓ Basic GUI components created successfully")

        # Clean up
        root.destroy()
        return True

    except Exception as e:
        print(f"✗ GUI component test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Judgement AI GUI Test Suite")
    print("=" * 40)

    tests = [test_imports, test_environment, test_agent_loading, test_gui_components]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! The GUI should work correctly.")
        print("\nTo run the GUI:")
        print("  python play_against_ai.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
