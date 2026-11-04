import importlib.util
import sys
import pytest
from test_window_observer import ROOT


@pytest.mark.parametrize('study',['output_space_4b_recheck_v2','output_space_4b_holdout','output_space_4b_input_control',
                                'qwen3_4b_bf16_five_methods'])
@pytest.mark.parametrize('phase',['collect','summarize','write','process'])
def test_4b_native_phase_entry_is_function_not_module(study,phase):
    directory=ROOT/'studies'/study/'flow'/phase
    name=f'phase4b_{study}_{phase}'
    spec=importlib.util.spec_from_file_location(name,directory/'__init__.py',submodule_search_locations=[str(directory)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
        assert callable(module.run)
        assert module.run.__module__.startswith('studies._shared.flow.')
    finally:
        sys.modules.pop(name,None)
        sys.modules.pop(name+'.run',None)
