#!/usr/bin/env python3
"""
Simple launcher script for the Judgement AI GUI interface.
Run this script to play against the trained AI agent.
"""

import sys
import os

# Add the src directory to the path so we can import from judgement_rl
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from gui_interface import main

    print("Starting Judgement AI GUI...")
    print("You will be playing against the trained AI agent.")
    print("Make sure you have a trained model in the 'models/' directory.")
    print()

    # Start the GUI
    main()

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all required packages are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting GUI: {e}")
    sys.exit(1)
