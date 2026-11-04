"""Temporary shim. The decode loop lives in spec_mtp.structure.algorithm.decode.loop.

This module object is that loop module, so existing imports and monkeypatches
of spec_mtp.decode see the same names.
"""

import sys

import spec_mtp.structure.algorithm.decode.loop as _loop

sys.modules[__name__] = _loop
