"""
DWOSM: Deep Water Oil Spill Model
=================================

Numerical framework for simulating deep water oil spill transport and fate.

"""

__version__ = "1.0.0"
__author__ = "Zhaoyang Yang"

from .DWOSM_API import DWOSM
from .NFM import Blowout
from .FFM import FFM_API
from .SPM import Model_parcel

__all__ = ["DWOSM", "Blowout", "FFM_API", "Model_parcel"]