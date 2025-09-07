# Gemini Project Context: agent-dashboard

## Project Overview

This project is a terminal-based dashboard for interacting with `fast-agent` agents. It uses the `textual` library to create a modern, responsive terminal user interface (TUI). The application follows a Model-View-Controller (MVC) architecture to ensure a clean separation of concerns between the application's state, its presentation, and its business logic.

### Key Technologies

*   **Python:** The core programming language.
*   **Textual:** A TUI framework for building sophisticated terminal applications.
*   **fast-agent-mcp:** The framework for building and running AI agents.
*   **Rich:** A Python library for rich text and beautiful formatting in the terminal.
*   **pytest:** The framework for running tests.

### Architecture

The application is structured around a Model-View-Controller (MVC) pattern:

*   **Model (`src/model.py`):** Manages the application's state, including sessions and interactions. It is the single source of truth and notifies listeners (the view) of any changes.
*   **View (`src/textual_view.py`):** The user interface, built with `textual`. It is responsible for rendering the application's state and capturing user input. It is stateless and observes the model for changes.
*   **Controller (`src/controller.py`):** Contains the application's business logic. It processes user input from the view, interacts with the `fast-agent` agents, and updates the model.

## Building and Running

### Installation

To install the required dependencies, run the following command:

```bash
pip install -e .[dev]
```

### Running the Application

The application can be started by running the `main.py` script:

```bash
python src/main.py
```

You can also specify an agent to use with the `--agent` or `-a` flag:

```bash
python src/main.py --agent <agent_name>
```

### Running Tests

The project uses `pytest` for testing. To run the test suite, use the following command:

```bash
pytest
```

## Development Conventions

*   **MVC Architecture:** Adhere to the strict separation of concerns defined by the MVC pattern.
*   **State Management:** The `Model` is the only stateful component. The View and Controller must remain stateless.
*   **Asynchronous:** The application is built on `asyncio`.
*   **Configuration:** All agent configuration is centralized in `fastagent.config.yaml`.
*   **Commands:** User commands are prefixed with `/` and handled by the `Controller`.
*   **Logging:** The application logs to `agent-dashboard.log`.
