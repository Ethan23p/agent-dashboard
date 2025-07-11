# secure_filesystem_server.py
# Needs to be validated; not sure this is the correct implementation.
import os
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP
import typer

# Initialize the FastMCP server
mcp = FastMCP("secure-filesystem")

def is_path_safe(base_dirs: List[Path], target_path: Path) -> bool:
    """Ensure the target path is within one of the allowed base directories."""
    resolved_path = target_path.resolve()
    for base in base_dirs:
        try:
            resolved_path.relative_to(base.resolve())
            return True
        except ValueError:
            continue
    return False

@mcp.tool()
def read_file(path: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Reads the complete contents of a single file."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_file():
        return f"Error: Path is not a file: {path}"
    return target_path.read_text(encoding="utf-8")

@mcp.tool()
def list_directory(path: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Lists the contents of a directory."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_dir():
        return f"Error: Path is not a directory: {path}"
    
    contents = []
    for item in target_path.iterdir():
        prefix = "[DIR]" if item.is_dir() else "[FILE]"
        contents.append(f"{prefix} {item.name}")
    return "\n".join(contents)

@mcp.tool()
def search_files(path: str, pattern: str, allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Recursively searches for files matching a pattern in a directory."""
    target_path = Path(path)
    if not is_path_safe(allowed_dirs, target_path):
        return f"Error: Access denied. Path is outside of allowed directories."
    if not target_path.is_dir():
        return f"Error: Path is not a directory: {path}"

    matches = [str(p) for p in target_path.rglob(pattern)]
    return "\n".join(matches) if matches else "No matching files found."


@mcp.tool()
def list_allowed_directories(allowed_dirs: List[Path] = typer.Option(...)) -> str:
    """Lists all directories the server is allowed to access."""
    return "This server has read-only access to the following directories:\n" + "\n".join([str(d.resolve()) for d in allowed_dirs])


# The `run` function provided by FastMCP will automatically handle CLI arguments.
# Any arguments defined here (like `allowed_dirs`) that are not tools themselves
# will be passed to all tool functions that require them.
@mcp.run(transport="stdio")
def main(allowed_dirs: List[Path] = typer.Argument(..., help="List of directories to allow read access to.")):
    """
    A read-only filesystem MCP server.
    This server will run until the client disconnects.
    """
    # The typer decorator and mcp.run handle the server lifecycle.
    pass