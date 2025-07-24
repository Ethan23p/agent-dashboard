# tests/test_agent_selection.py
import pytest
from agent_registry import get_agent, list_available_agents, AGENT_REGISTRY, DEFAULT_AGENT
from mcp_agent.core.fastagent import FastAgent

def test_list_available_agents():
    """Ensures list_available_agents returns the correct names."""
    available_agents = list_available_agents()
    assert set(available_agents) == set(AGENT_REGISTRY.keys())
    assert len(available_agents) >= 2 # We have at least minimal and coding

def test_get_specific_agent():
    """Tests successful retrieval of a specific, configured agent."""
    agent = get_agent("coding")
    assert agent is not None
    assert isinstance(agent, FastAgent)
    
    agent_name = list(agent.agents.keys())[0]
    assert agent_name == "coding"

def test_get_default_agent():
    """Tests retrieval of the default agent and verifies its name."""
    agent = get_agent() # No name provided
    assert agent is not None
    assert isinstance(agent, FastAgent)
    
    agent_name = list(agent.agents.keys())[0]
    assert agent_name == DEFAULT_AGENT

def test_get_nonexistent_agent_raises_keyerror():
    """Tests that requesting a non-existent agent raises a KeyError."""
    with pytest.raises(KeyError) as exc_info:
        get_agent("nonexistent_agent")
    assert "not found" in str(exc_info.value)

def test_agent_characteristics_are_distinct():
    """Tests that different agents have distinct properties."""
    minimal_agent = get_agent("minimal")
    coding_agent = get_agent("coding")

    # The .agents property holds the configuration provided to the decorator.
    # Let's access the config dictionaries directly by their known names for clarity.
    minimal_config = minimal_agent.agents["minimal"]
    coding_config = coding_agent.agents["coding"]

    # --- FIX: Use dictionary key access for simple values ---
    assert minimal_config['instruction'] != coding_config['instruction']

    # --- And attribute access for the RequestParams object ---
    assert minimal_config['request_params'].maxTokens != coding_config['request_params'].maxTokens