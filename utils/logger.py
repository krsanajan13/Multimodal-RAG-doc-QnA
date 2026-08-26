"""
utils/logger.py

Centralized logger factory.
"""
import logging
import sys


def get_logger(name: str = None, level: int = logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        formatter = logging.Formatter(fmt)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger


# usage:
# from utils.logger import get_logger
# logger = get_logger("my-module")
