# =========================================================
# File: utils/logger.py
# Project: flight_anomaly_ai_core
# =========================================================

from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


# =========================================================
# Define Project Paths
# =========================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

LOGS_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOGS_DIR / "flight_anomaly_ai_core.log"


# =========================================================
# Create Logs Directory
# =========================================================

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# Logger Configuration
# =========================================================

LOGGER_NAME = "flight_anomaly_ai_core"

LOG_LEVEL = logging.INFO

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =========================================================
# Create Logger
# =========================================================

logger = logging.getLogger(
    LOGGER_NAME
)

logger.setLevel(LOG_LEVEL)

logger.propagate = False


# =========================================================
# Prevent Duplicate Handlers
# =========================================================

if not logger.handlers:

    # =====================================================
    # Console Handler
    # =====================================================

    console_handler = logging.StreamHandler()

    console_handler.setLevel(LOG_LEVEL)

    console_formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    console_handler.setFormatter(
        console_formatter
    )

    # =====================================================
    # Rotating File Handler
    # =====================================================

    file_handler = RotatingFileHandler(

        LOG_FILE,

        maxBytes=5 * 1024 * 1024,  # 5 MB

        backupCount=5,

        encoding="utf-8"
    )

    file_handler.setLevel(LOG_LEVEL)

    file_formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    file_handler.setFormatter(
        file_formatter
    )

    # =====================================================
    # Add Handlers
    # =====================================================

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)


# =========================================================
# Helper Logging Functions
# =========================================================

def log_info(message):
    """
    Log INFO level messages.
    """

    logger.info(message)


def log_warning(message):
    """
    Log WARNING level messages.
    """

    logger.warning(message)


def log_error(message):
    """
    Log ERROR level messages.
    """

    logger.error(message)


def log_debug(message):
    """
    Log DEBUG level messages.
    """

    logger.debug(message)


def log_critical(message):
    """
    Log CRITICAL level messages.
    """

    logger.critical(message)


# =========================================================
# Example Usage
# =========================================================

if __name__ == "__main__":

    log_info(
        "Flight telemetry pipeline initialized."
    )

    log_warning(
        "Telemetry oscillation threshold exceeded."
    )

    log_error(
        "Anomaly scoring model failed to load."
    )

    log_debug(
        "Debugging telemetry feature pipeline."
    )

    log_critical(
        "Critical instability detected in flight stream."
    )

    print("\n========================================")
    print(" Logger Test Completed ")
    print("========================================")

    print(f"\nLog File:")
    print(LOG_FILE)  # logger.py
