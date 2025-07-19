#!/usr/bin/env python3
"""
Test script for agent selection functionality.
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent_registry import get_agent, list_available_agents, AGENT_REGISTRY

def test_agent_registry():
    """Test the agent registry functionality."""
    print("Testing Agent Registry...")
    
    # Test listing available agents
    available_agents = list_available_agents()
    print(f"Available agents: {available_agents}")
    assert len(available_agents) >= 2, "Should have at least 2 agents"
    
    # Test getting valid agents
    minimal_agent = get_agent("minimal")
    coding_agent = get_agent("coding")
    print("✓ Successfully retrieved minimal and coding agents")
    
    # Test getting invalid agent
    try:
        get_agent("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"✓ Correctly raised KeyError for invalid agent: {e}")
    
    print("All agent registry tests passed!")

# Remove outdated test for agent characteristics (instances/names)

if __name__ == "__main__":
    test_agent_registry()
    print("\n🎉 All tests passed! Agent selection system is working correctly.") 