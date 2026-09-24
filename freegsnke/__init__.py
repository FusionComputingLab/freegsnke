"""
FreeGSNKE package.

Provides tools for solving Grad-Shafranov equilibria and related
numerical methods for plasma physics modelling.
"""

import importlib.metadata

__version__ = importlib.metadata.version("freegsnke")
__author__ = "The FreeGSNKE Developers"

from .logging import (
    disable_default_handler,
    enable_default_handler,
    get_logger,
    set_log_level,
    setup_logging,
)

# Set up sensible default console logging to stdout at INFO level
enable_default_handler()
