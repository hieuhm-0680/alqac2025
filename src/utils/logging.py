import logging
import sys
from pathlib import Path
from typing import Optional
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors based on log level."""
    LEVEL_COLORS = {
        logging.DEBUG: Fore.YELLOW,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.MAGENTA,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    if format_string is None:
        format_string = "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False  # Ngăn logger này gửi log lên root_logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Formatter for console (with color)
    color_formatter = ColorFormatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

logger = setup_logging("alqac25", "DEBUG")
