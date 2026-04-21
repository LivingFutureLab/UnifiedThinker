#!/usr/bin/env python3

"""
Description: Execute a bash command in the terminal, with Python version compatibility.

Parameters:
  --cmd (string, required): The bash command to execute. For example: --cmd 'python my_script.py'
"""

import argparse
import subprocess
import sys

BLOCKED_BASH_COMMANDS = ["ipython", "jupyter", "pip", "nohup"]


def run_command(cmd):
    try:
        # Try to use the new parameters (Python 3.7+)
        return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    except TypeError:
        # Fallback for Python 3.5 and 3.6:
        return subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Execute a bash command.")
    parser.add_argument(
        "--cmd",
        help="The command (and optional arguments) to execute. For example: --cmd 'python my_script.py'",
    )
    parser.add_argument(
        "--command",
        help="The command (and optional arguments) to execute. For example: --command 'python my_script.py'",
    )
    args = parser.parse_args()

    # 优先使用 --cmd，如果没有则使用 --command
    cmd = args.cmd if args.cmd else args.command
    if not cmd:
        parser.error("Either --cmd or --command must be provided")

    # Check if the command is blocked
    first_token = cmd.strip().split()[0]
    if first_token in BLOCKED_BASH_COMMANDS:
        print(f"Bash command '{first_token}' is not allowed. " "Please use a different command or tool.")
        sys.exit(1)

    result = run_command(cmd)

    if result.returncode != 0:
        print(f"Error executing command:\n")
        print("[STDOUT]\n")
        print(result.stdout.strip(), "\n")
        print("[STDERR]\n")
        print(result.stderr.strip())
        sys.exit(result.returncode)

    print("[STDOUT]\n")
    print(result.stdout.strip(), "\n")
    print("[STDERR]\n")
    print(result.stderr.strip())


if __name__ == "__main__":
    main()
