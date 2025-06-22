"""ALQAC 2025 Competition Package.

This package provides tools and models for the Automated Legal Question Answering
Competition 2025, including legal document retrieval and question answering systems.
"""

__version__ = "0.1.0"
__author__ = "UIT legalGrep"

from src.utils.config import Settings
from src.models.schemas import QuestionType, TaskType

__all__ = ["Settings", "QuestionType", "TaskType"]
