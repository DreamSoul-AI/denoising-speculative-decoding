"""Method-name registry for mask providers."""

from spec_mtp.structure.algorithm.mask.providers import (
    AlignedResidualKalman,
    BonusSlotMaskProvider,
    EMAVelocityMaskProvider,
    ESPMaskProvider,
    HybridGainKalman,
    KalmanMaskProvider,
    ObservationBlendMaskProvider,
    OnlineVerifyKalman,
    OutputHiddenHybridKalman,
    OutputHiddenMaskProvider,
    VerifyCorrectionMaskProvider,
)


def build_mask_provider(spec_cfg, embedding_table, generator=None):
    observation = spec_cfg.get('observation')
    bonus = spec_cfg.get('bonus_slot')
    correction = spec_cfg.get('correction')
    if correction is not None and (observation is not None or bonus is not None or str(spec_cfg.get('method', 'esp')).lower() == 'kalman'):
        raise ValueError('correction does not combine with observation, bonus_slot or kalman')
    if str(spec_cfg.get('method', 'esp')).lower() == 'kalman' and (observation is not None or bonus is not None):
        raise ValueError('kalman does not combine with observation or bonus_slot')
    merge = spec_cfg.get('kalman_merge')
    align = spec_cfg.get('kalman_align')
    hybrid = spec_cfg.get('hybrid_gain')
    output_kalman = spec_cfg.get('output_kalman')
    if merge is not None and align is not None:
        raise ValueError('kalman_merge does not combine with kalman_align')
    if merge is not None and (observation is not None or bonus is not None or correction is not None):
        raise ValueError('kalman_merge does not combine with observation, bonus_slot or correction')
    if align is not None and (observation is not None or bonus is not None or correction is not None):
        raise ValueError('kalman_align does not combine with observation, bonus_slot or correction')
    if hybrid is not None and (observation is not None or bonus is not None or correction is not None or merge is not None or align is not None):
        raise ValueError('hybrid_gain does not combine with the other mask corrections')
    if output_kalman is not None and (observation is not None or bonus is not None or correction is not None or merge is not None or align is not None or hybrid is not None):
        raise ValueError('output_kalman does not combine with the other mask corrections')
    if output_kalman is not None and str(spec_cfg.get('method', 'esp')).lower() != 'output_hidden':
        raise ValueError('output_kalman wraps an output hidden trajectory')
    if correction is not None:
        if not isinstance(correction, dict) or correction.get('kind') not in ('scalar', 'diagonal', 'affine', 'lowrank', 'reject_token'):
            raise ValueError('spec.correction kind must be scalar, diagonal, affine, lowrank or reject_token')
        if not set(correction) <= {'kind', 'online', 'k', 'num', 'den', 'A', 'b', 'gram', 'rhs', 'ridge', 'U', 'V', 'rank', 'delta', 'error'}:
            raise ValueError('spec.correction has an unknown field')
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'correction'}
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return VerifyCorrectionMaskProvider(base, correction['kind'], correction, online=bool(correction.get('online', False)))
    if observation is not None and bonus is not None:
        raise ValueError('bonus_slot does not combine with observation')
    if bonus is not None:
        if not isinstance(bonus, dict) or set(bonus) != {'mode'} or bonus.get('mode') not in ('hard', 'topk'):
            raise ValueError('bonus_slot accepts only mode hard or topk')
        widths = list((spec_cfg.get('tree') or {}).get('branches') or [])
        if bonus['mode'] == 'topk' and (not widths or any(not isinstance(w, int) or isinstance(w, bool) or w <= 0 for w in widths)):
            raise ValueError('topk bonus slot needs positive integer tree branch widths')
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'bonus_slot'}
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return BonusSlotMaskProvider(base, embedding_table, bonus['mode'], widths)
    if merge is not None:
        if str(spec_cfg.get('method', 'esp')).lower() != 'esp':
            raise ValueError('kalman_merge wraps an ESP trajectory')
        if not isinstance(merge, dict) or not set(merge) <= {'part', 'forget', 'p0', 'q0', 'r0', 'r_min', 'q_min'}:
            raise ValueError('spec.kalman_merge has an unknown field')
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'kalman_merge'}
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return OnlineVerifyKalman(
            base, part=merge.get('part', 'full'), forget=merge.get('forget', 0.95),
            p0=merge.get('p0', 1.0), q0=merge.get('q0', 0.01), r0=merge.get('r0', 1.0),
            r_min=merge.get('r_min', 1e-6), q_min=merge.get('q_min', 1e-8))
    if align is not None:
        if str(spec_cfg.get('method', 'esp')).lower() != 'esp':
            raise ValueError('kalman_align wraps an ESP trajectory')
        if not isinstance(align, dict) or not set(align) <= {'part', 'forget', 'p0', 'q0', 'r0', 'r_min', 'q_min'}:
            raise ValueError('spec.kalman_align has an unknown field')
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'kalman_align'}
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return AlignedResidualKalman(
            base, part=align.get('part', 'full'), forget=align.get('forget', 0.95),
            p0=align.get('p0', 1.0), q0=align.get('q0', 0.01), r0=align.get('r0', 1.0),
            r_min=align.get('r_min', 1e-3), q_min=align.get('q_min', 1e-6))
    if hybrid is not None:
        if str(spec_cfg.get('method', 'esp')).lower() != 'esp':
            raise ValueError('hybrid_gain wraps an ESP trajectory')
        if not isinstance(hybrid, dict) or hybrid.get('mode') not in (
                'hybrid', 'scratch', 'frozen', 'trajectory', 'opened', 'branch', 'replace', 'collect'):
            raise ValueError(
                'spec.hybrid_gain mode must be hybrid, scratch, frozen, trajectory, opened, branch, replace or collect')
        allowed = {'mode', 'k', 'p', 'r', 'q', 'boost', 'observe'}
        if not set(hybrid) <= allowed:
            raise ValueError('spec.hybrid_gain has an unknown field')
        if hybrid.get('observe', 'first') not in ('first', 'last'):
            raise ValueError('spec.hybrid_gain observe must be first or last')
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'hybrid_gain'}
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return HybridGainKalman(base, hybrid['mode'], hybrid)
    if observation is not None:
        base_cfg = {key: value for key, value in dict(spec_cfg).items() if key != 'observation'}
        if 'gain' not in observation or not set(observation) <= {'gain', 'condition', 'form'}:
            raise ValueError('spec.observation accepts only gain, condition and form')
        base = build_mask_provider(base_cfg, embedding_table, generator)
        return ObservationBlendMaskProvider(base, embedding_table, observation['gain'],
                                            observation.get('condition', 'none'), observation.get('form', 'mean'))
    method = str(spec_cfg.get('method', 'esp')).lower()
    num_masks = int(spec_cfg.get('num_masks', 1))
    if method == 'esp':
        mask_cfg = spec_cfg.get('mask', {})
        return ESPMaskProvider(
            num_masks=num_masks,
            embedding_table=embedding_table,
            init=mask_cfg.get('init', 'mean_prompt'),
            lam=mask_cfg.get('lam', 0.1),
            update=mask_cfg.get('update', True),
            offset_scale=mask_cfg.get('offset_scale', 5.0),
            generator=generator,
        )
    if method == 'kalman':
        cfg = dict(spec_cfg.get('kalman') or {})
        if not set(cfg) <= {'mode', 'forget', 'r_min', 'q_min', 'q_scale'}:
            raise ValueError('spec.kalman accepts only mode, forget, r_min, q_min and q_scale')
        return KalmanMaskProvider(
            num_masks=num_masks, embedding_table=embedding_table, generator=generator,
            mode=cfg.get('mode', 'online'), forget=cfg.get('forget', 0.95),
            r_min=cfg.get('r_min', 1e-6), q_min=cfg.get('q_min', 1e-8), q_scale=cfg.get('q_scale', 0.01))
    if method == 'ema_velocity':
        ema_cfg = spec_cfg.get('ema', {})
        return EMAVelocityMaskProvider(
            num_masks=num_masks,
            embedding_table=embedding_table,
            beta=ema_cfg.get('beta', 0.9),
            step_scale=ema_cfg.get('step_scale', 1.0),
            normalize=ema_cfg.get('normalize', False),
            extrapolate_ema=ema_cfg.get('extrapolate_ema', False),
            update=ema_cfg.get('update', True),
            level=ema_cfg.get('level', 'last'),
            lam=ema_cfg.get('lam', 0.1),
            generator=generator,
        )
    if method == 'output_hidden':
        provider = OutputHiddenMaskProvider(num_masks, embedding_table, **dict(spec_cfg.get('output', {})))
        if output_kalman is None:
            return provider
        if not isinstance(output_kalman, dict) or output_kalman.get('mode') not in ('hybrid', 'scratch', 'frozen', 'trajectory', 'collect'):
            raise ValueError('spec.output_kalman mode must be hybrid, scratch, frozen, trajectory or collect')
        if not set(output_kalman) <= {'mode', 'k', 'p', 'r', 'q', 'boost'}:
            raise ValueError('spec.output_kalman has an unknown field')
        return OutputHiddenHybridKalman(provider, output_kalman['mode'], output_kalman)
    raise ValueError(f'Unsupported spec.method: {method}')
