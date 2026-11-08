"""Original exact-AR protocol and the fixed BC10 auxiliary role."""
from studies.esp_table1_bf16_full_argmax.structure.readout import count_summary as summarize

__all__ = ['summarize', 'ordered_cells', 'CELLS']

CELLS = ((10, 1, 'auxiliary', None), (30, 1, 'primary', 1.63), (60, 1, 'primary', 1.74), (60, 2, 'control', 1.74))


def ordered_cells(observed):
    result=[]
    for bc,k,role,target in CELLS:
        cell=dict(observed.get((bc,k),{}),BC=bc,num_masks=k,role=role,target=target)
        cell['pending']=(bc,k) not in observed
        for metric in ('pooled_tau','prompt_mean_tau','category_mean_pooled_tau'):
            value=cell.get(metric)
            cell[metric+'_target_delta']=value-target if value is not None and target is not None else None
        result.append(cell)
    return result
