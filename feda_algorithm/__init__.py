# feda_project/feda_algorithm/__init__.py

"""
FEDA Algorithm Package
----------------------

This package contains implementations of various Estimation of Distribution
Algorithms (EDAs).

Available Optimizers:
- RF_MIMIC: A Forest-guided EDA (FEDA).
- MIMIC_O2: MIMIC using a dependency tree from mutual information.
- MIMIC_MY: A simpler MIMIC variant using only marginal probabilities.
"""

from .rf_mimic import RF_MIMIC
from .mimic_o2 import MIMIC_O2
from .mimic_my import MIMIC_MY

__all__ = [
    "RF_MIMIC",
    "MIMIC_O2",
    "MIMIC_MY"
]
