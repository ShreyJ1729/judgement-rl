#!/usr/bin/env python3
"""
Test script to verify card image generation works correctly.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_card_images():
    """Test card image generation."""
    print("Testing card image generation...")

    try:
        import tkinter as tk

        # Create a minimal root window for tkinter
        root = tk.Tk()
        root.withdraw()  # Hide the window

        from gui_interface import CardImageManager

        # Create card manager
        card_manager = CardImageManager()

        # Test a few cards
        test_cards = [
            "A of Hearts",
            "K of Spades",
            "Q of Diamonds",
            "J of Clubs",
            "10 of Hearts",
        ]

        print("✓ CardImageManager created successfully")

        for card in test_cards:
            image = card_manager.get_card_image(card)
            if image:
                print(f"✓ Generated image for {card}")
            else:
                print(f"✗ Failed to generate image for {card}")
                root.destroy()
                return False

        # Test card back image
        back_image = card_manager.get_card_back_image()
        if back_image:
            print("✓ Generated card back image")
        else:
            print("✗ Failed to generate card back image")
            root.destroy()
            return False

        # Clean up
        root.destroy()

        print("✓ All card images generated successfully!")
        return True

    except Exception as e:
        print(f"✗ Card image test failed: {e}")
        return False


def main():
    """Run the test."""
    print("Card Image Test")
    print("=" * 30)

    if test_card_images():
        print("\n✓ Card image generation works correctly!")
        print("The GUI should display cards as images instead of text.")
    else:
        print("\n✗ Card image generation failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
