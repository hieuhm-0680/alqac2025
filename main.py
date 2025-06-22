#!/usr/bin/env python3
"""Main entry point for ALQAC 2025."""

from src.cli.main import cli
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


if __name__ == "__main__":
    cli()
