
"""
This module configures logging for the application.
"""

import logging
import os
import sys
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logger with both file and console handlers
logger = logging.getLogger("research_agent")
logger.setLevel(logging.INFO)

# Create formatters
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

# File handler - create a new log file for each day
today = datetime.now().strftime("%Y-%m-%d")
file_handler = logging.FileHandler(f"logs/research_agent_{today}.log")
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)  # More detailed logging to file

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Don't propagate to root logger
logger.propagate = False

def set_log_level(level_name):
    """
    Set the logging level dynamically.
    
    Args:
        level_name: The name of the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        logger.warning(f"Invalid log level: {level_name}")
        return False
    
    logger.setLevel(level)
    console_handler.setLevel(level)
    logger.info(f"Log level set to {level_name}")
    return True