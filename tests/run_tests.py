#!/usr/bin/env python3
"""
Simple test runner for the agent-dashboard project.
Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py tests/test_model.py     # Run specific test file
    python tests/run_tests.py -v                # Run with verbose output
"""

import sys
import subprocess
import os

def run_tests(test_file=None, verbose=False):
    """Run pytest with the specified options."""
    # LINTER FIX: The command needs to be run from the project root for paths to work.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = ["uv", "run", "python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_file:
        cmd.append(os.path.join("tests", test_file))
    else:
        # LINTER FIX: Updated the list of test files to match our refactored suite.
        # Removed test_integration.py and added test_primitives.py and test_agent_selection.py
        test_files = [
            "tests/test_primitives.py",
            "tests/test_model.py",
            "tests/test_controller.py",
            "tests/test_agent_selection.py"
        ]
        cmd.extend(test_files)
    
    try:
        # Run the command from the project root directory.
        result = subprocess.run(cmd, check=True, cwd=project_root)
        print(f"\nAll tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ uv or pytest not found. Please ensure they are installed and in your PATH.")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for agent-dashboard")
    parser.add_argument("test_file", nargs="?", help="Specific test file to run (e.g., test_model.py)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("Running tests for agent-dashboard...")
    success = run_tests(args.test_file, args.verbose)
    
    sys.exit(0 if success else 1)