# feda_project/utils/debugging.py
"""Utility functions for the FEDA project, such as debugging helpers."""

# Global Debug Flag
DEBUG_MODE = False  # Set to True for verbose output, False for cleaner results

def print_debug(message: str):
    """Prints a message if DEBUG_MODE is enabled.

    Args:
        message (str): The message to print.
    """
    if DEBUG_MODE:
        print(message)