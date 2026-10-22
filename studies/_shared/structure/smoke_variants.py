"""CPU smoke spec blocks (match configs/smoke/tiny_llama/run_0/*.yaml)."""

VARIANTS = {
    'esp_static': {
        'method': 'esp',
        'num_masks': 2,
        'impl': 'efficient',
        'tree': {'mode': 'static', 'branches': [7, 2], 'pruning': True},
        'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
    },
    'esp_dynamic': {
        'method': 'esp',
        'num_masks': 2,
        'block_complexity': 30,
        'impl': 'naive',
        'tree': {'mode': 'dynamic', 'pruning': True},
        'mask': {'init': 'mean_prompt', 'lam': 0.1, 'update': True},
    },
    'ema_velocity': {
        'method': 'ema_velocity',
        'num_masks': 2,
        'impl': 'efficient',
        'tree': {'mode': 'static', 'branches': [7, 2], 'pruning': True},
        'ema': {
            'beta': 0.9,
            'step_scale': 1.0,
            'normalize': False,
            'extrapolate_ema': False,
            'update': True,
        },
    },
}
