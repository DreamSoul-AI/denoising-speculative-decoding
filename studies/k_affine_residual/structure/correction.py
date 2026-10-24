"""Centered float64 ridge and immutable full/bias-only correction parameters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

ARMS = ('uncorrected', 'bias', 'full')
PROTOCOL = 'postcommit-final-norm-v1'
RIDGE = .001


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def matrix(value, name: str) -> torch.Tensor:
    def reject_bool(item):
        if isinstance(item, bool):
            raise ValueError(f'{name} must not contain bool')
        if isinstance(item, (list, tuple)):
            for child in item:
                reject_bool(child)
    reject_bool(value)
    try:
        raw = torch.as_tensor(value)
        if raw.dtype == torch.bool or raw.is_complex():
            raise ValueError(f'{name} must be real, not bool/complex')
        tensor = torch.as_tensor(value, device='cpu', dtype=torch.float64).detach()
    except (TypeError, RuntimeError) as error:
        raise ValueError(f'{name} must be a numeric matrix') from error
    if tensor.ndim != 2 or not all(tensor.shape) or not torch.isfinite(tensor).all():
        raise ValueError(f'{name} must be a nonempty finite matrix')
    return tensor


def require_sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a SHA256')


@dataclass(frozen=True)
class FrozenCorrection:
    # Immutable JSON avoids the writable-tensor loophole in a frozen dataclass.
    payload: str
    sha256: str

    def __post_init__(self):
        if hashlib.sha256(self.payload.encode()).hexdigest() != self.sha256:
            raise ValueError('frozen correction SHA mismatch')
        body = json.loads(self.payload)
        if body.get('protocol') != PROTOCOL or body.get('ridge') != RIDGE:
            raise ValueError('unsupported correction protocol')
        require_sha(body['model_space'], 'model_space')
        require_sha(body['fit_data_sha'], 'fit_data_sha')
        documents = body['fit_documents']
        if not documents or len({row[0] for row in documents}) != len(documents):
            raise ValueError('fit document IDs must be nonempty and unique')
        for name, sha in documents:
            if not isinstance(name, str) or not name:
                raise ValueError('fit document ID must be a nonempty string')
            require_sha(sha, 'fit document hash')
        a, b, b0 = (matrix(body[name], name) for name in ('a', 'b', 'b0'))
        if b.shape != b0.shape or b.shape != (1, a.shape[0]):
            raise ValueError('coefficient dimensions disagree')

    def body(self) -> dict:
        return json.loads(self.payload)

    def record(self) -> dict:
        return {'payload': self.payload, 'sha256': self.sha256}


def fit_correction(x, residual, *, model_space: str, fit_documents, fit_data_sha: str) -> FrozenCorrection:
    x, residual = matrix(x, 'x'), matrix(residual, 'residual')
    if x.shape[0] != residual.shape[0]:
        raise ValueError('fit row counts disagree')
    xm, rm = x.mean(0), residual.mean(0)
    xc, rc = x - xm, residual - rm
    a = torch.linalg.solve(xc.T @ xc + RIDGE * torch.eye(x.shape[1], dtype=torch.float64), xc.T @ rc).T
    body = {
        'protocol': PROTOCOL, 'ridge': RIDGE, 'model_space': model_space,
        'fit_documents': sorted(tuple(row) for row in fit_documents), 'fit_data_sha': fit_data_sha,
        'a': a.tolist(), 'b': (rm - a @ xm).unsqueeze(0).tolist(),
        'b0': rm.unsqueeze(0).tolist(),
    }
    payload = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return FrozenCorrection(payload, hashlib.sha256(payload.encode()).hexdigest())


def apply_correction(frozen: FrozenCorrection, x, z, arm: str, *, model_space: str) -> torch.Tensor:
    if arm not in ARMS:
        raise ValueError(f'unknown correction arm: {arm}')
    body = frozen.body()
    if model_space != body['model_space']:
        raise ValueError('model/representation space changed')
    x, z = matrix(x, 'x'), matrix(z, 'z')
    a, b, b0 = (matrix(body[name], name) for name in ('a', 'b', 'b0'))
    if x.shape[0] != z.shape[0] or x.shape[1] != a.shape[1] or z.shape[1] != a.shape[0]:
        raise ValueError('apply dimensions disagree')
    result = z if arm == 'uncorrected' else z + b0 if arm == 'bias' else z + x @ a.T + b
    if not torch.isfinite(result).all():
        raise ValueError('correction overflow')
    return result
