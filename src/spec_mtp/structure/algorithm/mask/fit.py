"""Pure K-fit math for mask correction."""

import torch


def fit_mask_correction(m_hat, z, target, kind):
    """Least squares for m = m_hat + K(z - m_hat), or plus a bias when affine."""
    if kind not in ('scalar', 'diagonal', 'affine'):
        raise ValueError('fit kind must be scalar, diagonal or affine')
    delta = (z - m_hat).double()
    error = (target - m_hat).double()
    if delta.ndim != 2 or error.shape != delta.shape or delta.shape[0] < 2:
        raise ValueError('correction fit needs at least two rows')
    if not torch.isfinite(delta).all() or not torch.isfinite(error).all():
        raise ValueError('correction fit inputs must be finite')
    if kind == 'scalar':
        num = (error * delta).sum()
        den = (delta * delta).sum().clamp_min(1e-12)
        return {'k': float(num / den), 'num': float(num), 'den': float(den)}
    if kind == 'diagonal':
        num = (error * delta).sum(dim=0)
        den = (delta * delta).sum(dim=0).clamp_min(1e-12)
        return {'k': (num / den).tolist()}
    design = torch.cat([delta, torch.ones(delta.shape[0], 1, dtype=delta.dtype, device=delta.device)], dim=1)
    solution = torch.linalg.lstsq(design, error).solution
    gram = design.T @ design
    rhs = design.T @ error
    return {'A': solution[:-1].T.tolist(), 'b': solution[-1].tolist(),
            'gram': gram.tolist(), 'rhs': rhs.tolist()}


RIDGE_GRID = tuple(10.0 ** exponent for exponent in range(-4, 5))


def fit_affine_ridge(m_hat, z, target, ridge):
    """Affine least squares with ridge on A only. The intercept column is unpenalized."""
    delta = (z - m_hat).double()
    error = (target - m_hat).double()
    if delta.ndim != 2 or error.shape != delta.shape or delta.shape[0] < 2:
        raise ValueError('correction fit needs at least two rows')
    if not torch.isfinite(delta).all() or not torch.isfinite(error).all():
        raise ValueError('correction fit inputs must be finite')
    ridge = float(ridge)
    if ridge <= 0.0:
        raise ValueError('affine ridge must be positive')
    design = torch.cat([delta, torch.ones(delta.shape[0], 1, dtype=delta.dtype, device=delta.device)], dim=1)
    gram = design.T @ design
    rhs = design.T @ error
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    penalty[-1, -1] = 0.0
    solution = torch.linalg.solve(gram + ridge * penalty, rhs)
    return {'A': solution[:-1].T.tolist(), 'b': solution[-1].tolist(),
            'gram': gram.tolist(), 'rhs': rhs.tolist(), 'ridge': ridge}


def select_affine_ridge(blocks, seed, folds=5):
    """Pick the ridge on prompt blocks. The score is replay MSE inside those blocks only."""
    count = len(blocks)
    if count < folds or folds < 2:
        raise ValueError('ridge selection needs at least one prompt per fold')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(int(seed))).tolist()
    sizes = [count // folds] * folds
    for index in range(count % folds):
        sizes[index] += 1
    chunks, cursor = [], 0
    for size in sizes:
        chunks.append(order[cursor:cursor + size])
        cursor += size
    cv = []
    for ridge in RIDGE_GRID:
        squared = rows = 0.0
        for held in chunks:
            train = [blocks[index] for index in range(count) if index not in held]
            val = [blocks[index] for index in held]
            fit = fit_affine_ridge(*_stack_blocks(train), ridge)
            squared += _affine_residual(*_stack_blocks(val), fit)
            rows += sum(block[0].shape[0] for block in val)
        cv.append({'ridge': ridge, 'mse': squared / rows})
    chosen = min(cv, key=lambda item: (item['mse'], -item['ridge']))
    fit = fit_affine_ridge(*_stack_blocks(blocks), chosen['ridge'])
    fit['cv'] = cv
    fit['rows'] = int(sum(block[0].shape[0] for block in blocks))
    fit['prompts'] = count
    return fit


def _stack_blocks(blocks) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.cat([block[0] for block in blocks], dim=0),
        torch.cat([block[1] for block in blocks], dim=0),
        torch.cat([block[2] for block in blocks], dim=0),
    )


def centered_r2(error, prediction, center):
    """1 - ||e - pred||^2 / ||e - center||^2. `center` is the training mean."""
    error = error.float()
    prediction = prediction.float()
    center = center.float().reshape(1, -1)
    total = (error - center).pow(2).sum()
    if float(total) <= 0.0:
        raise ValueError('centered R^2 needs nonzero target energy')
    return float(1.0 - (error - prediction).pow(2).sum() / total)


def fit_lowrank_delta(delta, error, rank, ridge, iters=8):
    """Low-rank map error ≈ (delta @ V) @ U^T + b. Ridge penalizes U and V, not b."""
    delta = delta.float().contiguous()
    error = error.float().contiguous()
    rank = int(rank)
    ridge = float(ridge)
    if delta.ndim != 2 or error.shape != delta.shape or delta.shape[0] < 2:
        raise ValueError('low-rank delta fit needs at least two aligned rows')
    if rank < 1 or rank >= delta.shape[1] or ridge <= 0.0 or int(iters) < 1:
        raise ValueError('low-rank delta fit needs a positive rank below the width and a positive ridge')
    if not torch.isfinite(delta).all() or not torch.isfinite(error).all():
        raise ValueError('low-rank delta fit inputs must be finite')
    left, right, bias = _als_lowrank(delta, error, rank, ridge, int(iters))
    return {'U': left, 'V': right, 'b': bias, 'rank': rank, 'ridge': float(ridge)}


def apply_lowrank_delta(delta, fit):
    delta = delta.float()
    factor_in = fit['V'].to(device=delta.device, dtype=torch.float32)
    factor_out = fit['U'].to(device=delta.device, dtype=torch.float32)
    bias = fit['b'].to(device=delta.device, dtype=torch.float32)
    return (delta @ factor_in) @ factor_out.T + bias


def fit_lowrank_ridge(m_hat, z, target, rank, ridge, iters=8):
    """Low-rank map m = m_hat + U V^T (z - m_hat) + b. Ridge penalizes U and V, not b."""
    delta = (z - m_hat).float()
    error = (target - m_hat).float()
    rank = int(rank)
    ridge = float(ridge)
    if delta.ndim != 2 or error.shape != delta.shape or delta.shape[0] < 2:
        raise ValueError('correction fit needs at least two rows')
    if rank < 1 or rank >= delta.shape[1] or ridge <= 0.0 or int(iters) < 1:
        raise ValueError('low-rank fit needs a positive rank below the hidden size and a positive ridge')
    if not torch.isfinite(delta).all() or not torch.isfinite(error).all():
        raise ValueError('correction fit inputs must be finite')
    left, right, bias = _als_lowrank(delta, error, rank, ridge, int(iters))
    return {'U': left.cpu().tolist(), 'V': right.cpu().tolist(), 'b': bias.cpu().tolist(),
            'rank': rank, 'ridge': ridge, 'delta': delta.cpu(), 'error': error.cpu()}


def select_lowrank_ridge(blocks, rank, seed, folds=5, iters=8, insample=False):
    """Pick the ridge on prompt blocks. Eval blocks are not an argument."""
    if insample:
        scores = []
        for ridge in RIDGE_GRID:
            fit = fit_lowrank_ridge(*_stack_blocks(blocks), rank, ridge, iters)
            scores.append({'ridge': ridge, 'mse': lowrank_replay_mse(*_stack_blocks(blocks), fit)})
        chosen = min(scores, key=lambda item: (item['mse'], -item['ridge']))
        fit = fit_lowrank_ridge(*_stack_blocks(blocks), rank, chosen['ridge'], iters)
        fit['cv'] = scores
        fit['rows'] = int(sum(block[0].shape[0] for block in blocks))
        fit['prompts'] = len(blocks)
        return fit
    count = len(blocks)
    if count < folds or folds < 2:
        raise ValueError('ridge selection needs at least one prompt per fold')
    order = torch.randperm(count, generator=torch.Generator().manual_seed(int(seed))).tolist()
    sizes = [count // folds] * folds
    for index in range(count % folds):
        sizes[index] += 1
    chunks, cursor = [], 0
    for size in sizes:
        chunks.append(order[cursor:cursor + size])
        cursor += size
    cv = []
    for ridge in RIDGE_GRID:
        squared = rows = 0.0
        for held in chunks:
            train = [blocks[index] for index in range(count) if index not in held]
            val = [blocks[index] for index in held]
            fit = fit_lowrank_ridge(*_stack_blocks(train), rank, ridge, iters)
            squared += _lowrank_residual(*_stack_blocks(val), fit)
            rows += sum(block[0].shape[0] for block in val)
        cv.append({'ridge': ridge, 'mse': squared / rows})
    chosen = min(cv, key=lambda item: (item['mse'], -item['ridge']))
    fit = fit_lowrank_ridge(*_stack_blocks(blocks), rank, chosen['ridge'], iters)
    fit['cv'] = cv
    fit['rows'] = int(sum(block[0].shape[0] for block in blocks))
    fit['prompts'] = count
    return fit


def lowrank_replay_mse(m_hat, z, target, fit):
    return _lowrank_residual(m_hat, z, target, fit) / max(int(target.numel()), 1)


def _lowrank_residual(m_hat, z, target, fit):
    delta = (z - m_hat).float()
    left = torch.tensor(fit['V'], dtype=torch.float32, device=delta.device)
    right = torch.tensor(fit['U'], dtype=torch.float32, device=delta.device)
    bias = torch.tensor(fit['b'], dtype=torch.float32, device=delta.device)
    prediction = m_hat.float() + (delta @ left) @ right.T + bias
    return float((prediction - target.float()).pow(2).sum())


def _als_lowrank(delta, error, rank, ridge, iters, v_init=None):
    if v_init is None:
        _unused, _values, vh = torch.linalg.svd(delta, full_matrices=False)
        right = vh[:rank].T.contiguous()
    else:
        right = v_init.to(device=delta.device, dtype=delta.dtype)
    left = torch.zeros(delta.shape[1], rank, dtype=delta.dtype, device=delta.device)
    bias = torch.zeros(delta.shape[1], dtype=delta.dtype, device=delta.device)
    for _step in range(int(iters)):
        left, bias = _solve_lowrank_output(delta, error, right, ridge)
        right = _solve_lowrank_input(delta, error, left, bias, ridge)
    return left, right, bias


def _solve_lowrank_output(delta, error, right, ridge):
    features = torch.cat([delta @ right, torch.ones(delta.shape[0], 1, dtype=delta.dtype, device=delta.device)], dim=1)
    gram = features.T @ features
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    penalty[-1, -1] = 0.0
    solution = torch.linalg.solve(gram + float(ridge) * penalty, features.T @ error)
    return solution[:-1].T.contiguous(), solution[-1]


def _solve_lowrank_input(delta, error, left, bias, ridge):
    target = (error - bias) @ left
    gram_u = left.T @ left
    target = target @ torch.linalg.inv(gram_u + 1e-8 * torch.eye(left.shape[1], dtype=gram_u.dtype, device=gram_u.device))
    gram = delta.T @ delta
    penalty = float(ridge) * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + penalty, delta.T @ target)


def _affine_residual(m_hat, z, target, fit):
    delta = z.double() - m_hat.double()
    weight = torch.as_tensor(fit['A'], dtype=delta.dtype, device=delta.device)
    bias = torch.as_tensor(fit['b'], dtype=delta.dtype, device=delta.device)
    error = m_hat.double() + delta @ weight.T + bias - target.double()
    return float(error.pow(2).sum())


def calibrate_diagonal_gain(m_hat, z, target, q_ratio=0.01):
    """Diagonal K and its noise prior. P is the least-squares variance, not 1."""
    delta = (z - m_hat).double()
    error = (target - m_hat).double()
    if delta.ndim != 2 or error.shape != delta.shape or delta.shape[0] < 2:
        raise ValueError('gain calibration needs at least two rows')
    if not torch.isfinite(delta).all() or not torch.isfinite(error).all():
        raise ValueError('gain calibration inputs must be finite')
    q_ratio = float(q_ratio)
    if q_ratio <= 0.0:
        raise ValueError('gain calibration q_ratio must be positive')
    den = (delta * delta).sum(dim=0).clamp_min(1e-8)
    gain = (error * delta).sum(dim=0) / den
    residual = error - gain * delta
    noise = residual.pow(2).mean(dim=0).clamp_min(1e-12)
    variance = noise / den
    return {'k': gain.float().tolist(), 'p': variance.float().tolist(),
            'r': noise.float().tolist(), 'q': (variance * q_ratio).float().tolist(),
            'rows': int(delta.shape[0])}
