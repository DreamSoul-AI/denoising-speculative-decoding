"""Temporary shim. The decode loop lives in spec_mtp.structure.algorithm.decode.loop.

Explicit imports keep the public names visible to mypy. This module object is
then that loop module, so existing imports and monkeypatches of spec_mtp.decode
see the same names.
"""

import sys

from spec_mtp.structure.algorithm.decode.loop import (
    DecodeFailure,
    SpecDecoder,
    _fork_generator,
    ar_generate,
    generate,
    select_token,
    tree_probs,
)
import spec_mtp.structure.algorithm.decode.loop as _loop

__all__ = [
    'DecodeFailure',
    'SpecDecoder',
    '_fork_generator',
    'ar_generate',
    'generate',
    'select_token',
    'tree_probs',
]

sys.modules[__name__] = _loop
