import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from model import Model
from textual_view import AgentDashboardApp

@pytest.fixture
def mock_model():
    """Provides a mock Model object with a spec for type hinting."""
    # The 'spec' argument makes the mock type-aware for Pylance.
    return MagicMock(spec=Model)

@pytest.fixture
def mock_app():
    """Provides a mock Textual App object with a spec and mocked run_worker."""
    mock = MagicMock(spec=AgentDashboardApp)
    # We still need to mock the implementation of run_worker.
    mock.run_worker = AsyncMock(side_effect=lambda coro, **kwargs: asyncio.create_task(coro))
    return mock