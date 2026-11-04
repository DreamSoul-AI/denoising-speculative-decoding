"""Exercise the exact Actions branch gate without GitHub or project execution."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.mark.parametrize('base,head,head_repo,allowed', [
    ('dev', 'refactor/spec-rpipe', 'DreamSoul-AI/denoising-speculative-decoding', True),
    ('dev', 'fix/example', 'contributor/fork', True),
    ('dev', 'main', 'DreamSoul-AI/denoising-speculative-decoding', False),
    ('dev', 'dev', 'DreamSoul-AI/denoising-speculative-decoding', False),
    ('main', 'dev', 'DreamSoul-AI/denoising-speculative-decoding', True),
    ('main', 'refactor/spec-rpipe', 'DreamSoul-AI/denoising-speculative-decoding', False),
    ('main', 'dev', 'contributor/fork', False),
])
def test_branch_gate(base, head, head_repo, allowed):
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / '.github/workflows/branch-flow.yml').read_text(encoding='utf-8'))
    step = workflow['jobs']['branch-flow']['steps'][0]
    script = step['run'].split("python - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    env = dict(os.environ, BASE_BRANCH=base, HEAD_BRANCH=head, HEAD_REPOSITORY=head_repo,
               BASE_REPOSITORY='DreamSoul-AI/denoising-speculative-decoding')
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is allowed, result.stdout + result.stderr
