"""Future paired Study only; active8B/norm admission files stay frozen."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import gpu_admission  # noqa: E402
from norm_admission import apply_capacity, CAPACITY  # noqa: E402
from resource_admission import wait_for_admission as original_wait  # noqa: E402
import resource_admission  # noqa: E402


def observe(initial=False):
    record = gpu_admission.observe(initial)
    result = apply_capacity(record, json.loads(CAPACITY.read_text(encoding='utf-8')), initial)
    result['process_classification_policy'] = record['policy']
    return result


def wait_for_admission(assets, initial=False):
    with patch.object(resource_admission, 'observe', observe):
        return original_wait(assets, initial=initial)
