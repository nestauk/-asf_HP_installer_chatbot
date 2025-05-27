import sys
import logging


def get_logger():
    logger = logging.getLogger("HPInstallerChatbotAPI")
    logger.setLevel(logging.INFO)
    logging.StreamHandler(sys.stdout)
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logger


logger = get_logger()
