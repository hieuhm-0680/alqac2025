"""Data loading and preprocessing utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger
from src.models.schemas import (
    Document,
    QuestionType,
    TrainingQuestion,
    TestQuestion,
)

logger = get_logger(__name__)


class DataLoader:
    """Data loader for ALQAC 2025 competition data."""
    pass