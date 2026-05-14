from __future__ import annotations

import logging


def get_logger(name: str = "novalm_hybrid") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger(name)
