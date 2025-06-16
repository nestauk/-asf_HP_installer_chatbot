"""This module provides a utility function to set up a logger for the HPInstallerChatbotAPI.
It configures the logger to output logs to stdout with a specific format.
The logger can be used throughout the application to log messages at various levels.
It is recommended to use this logger for all logging needs in the application.
Example usage:
    from api.app.utils.logging import get_logger
    logger = get_logger()
    logger.info("This is an info message.")
    logger.error("This is an error message.")
"""

import sys
import logging


def get_logger():
    """Get a logger instance for the HPInstallerChatbotAPI.

    This function sets up a logger with the name "HPInstallerChatbotAPI",
    sets its logging level to INFO, and configures it to output logs to stdout.
    The log format includes the timestamp, logger name, log level, and message.
    The logger can be used throughout the application to log messages at various levels.
    It is useful for debugging and tracking the application's behavior.
    It is recommended to use this logger for all logging needs in the application.
    This function does not take any parameters and returns a logger instance.
    It is typically called at the start of the application to set up logging.
    Example usage:
        logger = get_logger()
        logger.info("This is an info message.")
        logger.error("This is an error message.")

    Returns:
        logging.Logger: A configured logger instance for the application.
    """
    logger = logging.getLogger("HPInstallerChatbotAPI")
    logger.setLevel(logging.INFO)
    logging.StreamHandler(sys.stdout)
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logger


logger = get_logger()
