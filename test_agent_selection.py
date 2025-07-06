#!/usr/bin/env python3
"""
Test script for agent selection functionality.
"""

import asyncio
from agent_definitions import get_agent, list_available_agents, AGENT_REGISTRY

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

def test_agent_characteristics():
    """Test that agents have different characteristics."""
    print("\nTesting Agent Characteristics...")
    
    minimal_agent = get_agent("minimal")
    coding_agent = get_agent("coding")
    
    # Check that they're different instances
    assert minimal_agent != coding_agent, "Agents should be different instances"
    
    # Check that they have different names
    assert minimal_agent.name != coding_agent.name, "Agents should have different names"
    
    print("✓ Agents have different characteristics")
    print(f"  Minimal agent: {minimal_agent.name}")
    print(f"  Coding agent: {coding_agent.name}")
    
    print("All agent characteristics tests passed!")

if __name__ == "__main__":
    test_agent_registry()
    test_agent_characteristics()
    print("\n🎉 All tests passed! Agent selection system is working correctly.") 