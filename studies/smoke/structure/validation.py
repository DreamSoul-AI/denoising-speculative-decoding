"""Pure smoke observation contracts and rendering of the preserved CPU counts."""
from __future__ import annotations
import hashlib
import json
from spec_mtp.structure.artifact.observation import _check_argmax, _check_decode_stats
from studies._shared.structure.smoke_variants import VARIANTS

PROTOCOL = 'smoke-cuda-engineering-v1'
IMPLEMENTATION_COMMIT = '934dac3fd3b1662664ebf71d8317b49183e22f0b'
TEMPERATURES = (0.0, 0.7, 1.0)
MODEL_CONFIG = {'vocab_size':256,'hidden_size':64,'intermediate_size':128,'num_layers':2,
                'num_heads':4,'num_kv_heads':2,'max_position_embeddings':1024}


def cell_observation(prompt, output, ar, stats, temperature):
    if output[:len(prompt)] != prompt or ar[:len(prompt)] != prompt:
        raise ValueError('smoke output does not preserve its input prefix')
    _check_decode_stats(stats, output[len(prompt):])
    checked = _check_argmax({'input_ids':prompt,'output_ids':output,'stats':stats}) if temperature == 0 else None
    return {'input_ids':prompt,'tokens':output,'ar_tokens':ar,'exact_match':output==ar,'trace':stats['trace'],
        'argmax_checked_tokens':checked,
        **{key:stats[key] for key in ('accepted','new_tokens','committed','decode_calls',
                                     'attempt_by_depth','accept_by_depth','hit_context_limit')}}


def summarize_cells(cells):
    expected = {f'{variant}|T={temperature}|p={i}' for variant in VARIANTS for temperature in TEMPERATURES for i in range(4)}
    if (not isinstance(cells,dict) or set(cells) != expected
            or any(not isinstance(cell,dict) or cell.get('new_tokens') != 32
                   or type(cell.get('exact_match')) is not bool for cell in cells.values())):
        raise ValueError('smoke requires the complete frozen 36-cell/32-token matrix')
    greedy = [cell for name,cell in cells.items() if '|T=0.0|' in name]
    if len(greedy) != 12 or any(cell.get('argmax_checked_tokens') != 32 for cell in greedy):
        raise ValueError('smoke T0 requires complete committed-token argmax coverage')
    return {'cells':36,'T0_argmax_checked_tokens':sum(cell['argmax_checked_tokens'] for cell in greedy),
        'T0_AR_exact_cells':sum(cell['exact_match'] for cell in greedy),
        'sampling_AR_exact_cells':sum(cell['exact_match'] for name,cell in cells.items() if '|T=0.0|' not in name),
        'sampling_distribution_validated':False,
        'cells_sha256':hashlib.sha256(json.dumps(cells,sort_keys=True).encode()).hexdigest()}


def compare_reports(previous, current):
    if (previous.get('protocol') != PROTOCOL or current.get('protocol') != PROTOCOL
            or previous.get('identity') != current['identity']):
        raise ValueError('incompatible smoke identity: old CPU baselines and different CUDA/RNG protocols cannot compare')
    if (previous.get('schema') != 1 or previous.get('summary') != summarize_cells(previous.get('cells'))):
        raise ValueError('comparison requires a complete schema1 smoke report with reconciled summary')
    model = current.get('model_digest')
    if (not isinstance(model,str) or len(model)!=64 or any(c not in '0123456789abcdef' for c in model)
            or previous.get('model_digest') != model):
        raise ValueError('incompatible smoke actual model identity')
    if previous.get('cells') != current['cells']:
        raise ValueError('compatible smoke observations differ')


def historical_cpu_figure(reference):
    from matplotlib.figure import Figure
    rows = reference['cpu']
    variants = ['esp_static','esp_dynamic','ema_velocity']
    if (reference['implementation_commit'] != IMPLEMENTATION_COMMIT or [row['variant'] for row in rows] != variants
            or any(row['exact_match_rate'] != 1 or row['prompts'] != 4 or row['seed'] != 123
                or row['device'] != 'cpu' or row['dtype'] != 'torch.float32' or row['backend'] != 'eager' for row in rows)
            or [row['accepted'] for row in rows] != [28,31,8] or [row['accept_d1'] for row in rows] != [27,31,8]):
        raise ValueError('historical smoke figure requires its original CPU/934dac3 saved-count contract')
    fig = Figure(figsize=(8,4.8),layout='constrained');ax=fig.subplots()
    x=list(range(len(rows)))
    for offset,values,label,color in ((-.19,[r['accepted'] for r in rows],'All-depth accepted drafts','#2868a0'),
                                     (.19,[r['accept_d1'] for r in rows],'Depth-1 accepted drafts','#81b29a')):
        bars=ax.bar([value+offset for value in x],values,width=.36,label=label,color=color)
        ax.bar_label(bars,padding=3)
    ax.set_xticks(x,variants);ax.set_ylabel('Draft tokens accepted (4 prompts)');ax.set_ylim(0,39)
    ax.set_title('Historical CPU tiny validation — exact match = 1 for every variant')
    ax.legend(frameon=False,loc='upper right',fontsize=9);ax.spines[['top','right']].set_visible(False)
    ax.text(0,-.20,'Source: delivery_validation.json / 934dac3 / n=1 seed per variant\n'
        'FP32, eager, seed=123, T=0, 32-token budget; counts before budget/EOS truncation',transform=ax.transAxes,fontsize=8)
    return fig
