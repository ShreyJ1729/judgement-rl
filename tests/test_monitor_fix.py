#!/usr/bin/env python3
"""
Test script to verify the monitoring fix works correctly.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_monitor_stop():
    """Test that the monitor can be stopped without errors."""
    print("Testing monitor stop functionality...")

    try:
        from realtime_monitor import create_monitor_and_callback

        # Create monitor and callback
        monitor, callback = create_monitor_and_callback(["TestAgent"])

        # Test stopping without starting (should not crash)
        print("Testing stop without start...")
        monitor.stop_monitoring()
        print("✓ Stop without start works")

        # Test starting and stopping
        print("Testing start and stop...")
        monitor.start_monitoring()
        monitor.stop_monitoring()
        print("✓ Start and stop works")

        # Test stopping again (should not crash)
        print("Testing stop again...")
        monitor.stop_monitoring()
        print("✓ Multiple stops work")

        return True

    except Exception as e:
        print(f"✗ Monitor test failed: {e}")
        return False


def main():
    """Run the test."""
    print("Monitor Fix Test")
    print("=" * 30)

    if test_monitor_stop():
        print("\n✓ Monitor fix works correctly!")
        print("The training script should no longer crash at the end.")
    else:
        print("\n✗ Monitor fix failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
