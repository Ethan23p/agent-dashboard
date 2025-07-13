# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized state management in Model class
- Unified vocabulary using "Interaction" terminology
- Separated command logic into dedicated `commands.py` module
- Renamed `agent_definitions.py` to `agent_registry.py` for clarity
- Fixed duplicate user message display issue
- Improved ExitCommand exception handling for graceful shutdown

### Changed
- Refactored Controller to be stateless
- Moved conversation history management to Model
- Updated all imports to use new module structure
- Reorganized project structure with `src/`, `paperwork/`, and `tests/` directories
- Moved all Python code to `src/` directory
- Grouped documentation and project files in `paperwork/` directory

## [0.1.0] - 2024-12-19

### Added
- Textual-based TUI overhaul with better modularization
- Agent switching functionality with `/switch` command
- Multiple agent support (minimal, coding, interpreter)
- Retry mechanism with exponential backoff for agent calls
- Comprehensive testing suite with unit and integration tests
- Model-View-Controller (MVC) architecture
- Asynchronous core with non-blocking UI
- Stateful conversation history with save/load functionality
- MCP server integration (filesystem, fetch, sequential-thinking)

### Changed
- Migrated from basic CLI to Textual-based interface
- Improved error handling and resilience
- Enhanced separation of concerns across codebase

### Fixed
- Visual bugs in Textual interface
- Error handling for transient failures
- Code organization and modularization

## [Initial Development] - 2024-12-18

### Added
- Minimum Viable Implementation with MVC structure
- Basic agent framework integration
- Initial client and agent framework setup
- Core functionality for agent interactions

---

*Note: This changelog is based on recent commit history. For a complete history, see the git log.* 