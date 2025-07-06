#!/usr/bin/env python3
"""
Simple test runner for the agent-dashboard project.
Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py test_model.py     # Run specific test file
    python run_tests.py -v                # Run with verbose output
"""

import sys
import subprocess
import os


def run_tests(test_file=None, verbose=False):
    """Run pytest with the specified options."""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_file:
        cmd.append(test_file)
    else:
        # Run all test files
        cmd.extend(["test_model.py", "test_controller.py", "test_integration.py"])
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest pytest-asyncio")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for agent-dashboard")
    parser.add_argument("test_file", nargs="?", help="Specific test file to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🧪 Running tests for agent-dashboard...")
    success = run_tests(args.test_file, args.verbose)
    
    sys.exit(0 if success else 1) 