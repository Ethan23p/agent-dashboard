# create_context.py
import sys
from pathlib import Path

# --- Configuration ---
# The root directory to start scanning from. "." means the current directory.
ROOT_DIR = Path(".")

# A dedicated directory to store context-related files, keeping the project root clean.
# This entire directory will be ignored by the script.
CONTEXT_DIR = Path("_context")

# The output file will now be placed inside the CONTEXT_DIR.
# This file will be overwritten on each run to ensure it's always up-to-date.
OUTPUT_FILE = CONTEXT_DIR / "project_contents.md"

# List of patterns to ignore. This includes directories, files, and extensions.
IGNORE_PATTERNS = [
    str(CONTEXT_DIR),       # Ignore the entire context directory itself.
    ".venv",                # Ignore the virtual environment folder.
    "__pycache__",          # Ignore Python cache folders.
    ".git",                 # Ignore git directory.
    ".vscode",              # Ignore VSCode settings.
    "*.pyc",                # Ignore Python compiled files.
    "*.lock",               # Ignore lock files (like uv.lock).
    "*.jsonl",              # Ignore jsonl log files.
    "create_context.py",    # Ignore this script itself by name.
]
# --- End Configuration ---

def is_ignored(path: Path) -> bool:
    """
    Check if a given path should be ignored based on IGNORE_PATTERNS.
    This is more robust, checking full directory names and glob patterns.
    """
    # 1. Check if any directory in the path is an exact match in our ignore list.
    # This is safer than checking loose parts. e.g., ignores '.venv/' but not 'my.venvs.txt'
    for part in path.parts:
        if part in IGNORE_PATTERNS:
            return True

    # 2. Check if the file/dir name matches any glob pattern in the list.
    for pattern in IGNORE_PATTERNS:
        if path.match(pattern):
            return True

    return False

def main():
    """Main function to generate the context file."""
    try:
        # Ensure the context directory exists.
        CONTEXT_DIR.mkdir(exist_ok=True)

        print(f"Starting to process directory: '{ROOT_DIR.resolve()}'")
        print(f"Output will be saved to: '{OUTPUT_FILE.resolve()}'")

        # Get a list of all files recursively.
        all_files = sorted(list(ROOT_DIR.rglob("*")))

        # Filter out ignored files and all directories.
        files_to_process = [
            f for f in all_files if f.is_file() and not is_ignored(f)
        ]

        if not files_to_process:
            print("No files to process after filtering. Check IGNORE_PATTERNS.")
            return

        with OUTPUT_FILE.open("w", encoding="utf-8") as f_out:
            f_out.write(f"# Contents of the '{ROOT_DIR.resolve().name}' project\n\n")

            for file_path in files_to_process:
                relative_path = file_path.relative_to(ROOT_DIR).as_posix() # Use posix for consistency
                print(f"  - Processing: {relative_path}")

                f_out.write(f"--- START OF FILE {relative_path} ---\n\n")

                language = file_path.suffix[1:] if file_path.suffix else ""
                f_out.write(f"```{language}\n")

                try:
                    content = file_path.read_text(encoding="utf-8")
                    f_out.write(content)
                except UnicodeDecodeError:
                    f_out.write("[Error: Could not decode file content, likely a binary file.]")
                except Exception as e:
                    f_out.write(f"[Error: Could not read file: {e}]")

                f_out.write(f"\n```\n\n--- END OF FILE {relative_path} ---\n\n")

        print(f"\nSuccess! All content has been written to '{OUTPUT_FILE}'")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()