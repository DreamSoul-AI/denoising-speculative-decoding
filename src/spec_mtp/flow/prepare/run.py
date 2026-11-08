"""prepare: read Config, build Control, land Structure; do not modify Config."""

from __future__ import annotations

import copy

from spec_mtp.flow.context import FlowContext
from spec_mtp.flow.study import run_study_phase
from spec_mtp.structure.algorithm.factory import AlgorithmFactory
from spec_mtp.structure.algorithm.tracker import make_tracker
from spec_mtp.structure.data.factory import DataFactory
from spec_mtp.structure.model.factory import ModelFactory
from spec_mtp.structure.system.factory import SystemFactory
from spec_mtp.structure.system.runtime import apply_runtime
from spec_mtp.structure.artifact.asset import ensure_assets
from spec_mtp.structure.artifact.config.io import load_config
from spec_mtp.structure.artifact.provenance import check_frozen
from spec_mtp.structure.control.control import control_from_config
from spec_mtp.structure.control.contract import validate_control
from spec_mtp.structure.algorithm.config import AlgorithmConfig
from spec_mtp.structure.data.config import DataConfig
from spec_mtp.structure.model.config import ModelConfig
from spec_mtp.structure.system.config import SystemConfig
from spec_mtp.structure.make.expand import load_study_yaml
from spec_mtp.structure.make.recipe import RecipeContext, apply_recipe
from spec_mtp.structure.origin import apply_model_origin, normalize_origin


def run(ctx: FlowContext) -> None:
    cfg = load_config(ctx.layout.config_path)
    ctx.config = cfg
    ctx.control = control_from_config(cfg)
    validate_control(ctx.control)
    ensure_assets(ctx.layout)
    system_config = SystemConfig.from_mapping(ctx.control.system)
    ctx.state['runtime'] = apply_runtime(ctx.control.seed, system_config)
    ctx.state['seed'] = ctx.control.seed

    system = SystemFactory.build(
        system_config,
        ctx.layout.assets_dir,
        logger=ctx.state.get('logger'),
    )
    system.meta.update(ctx.state['runtime'])
    ctx.state['system'] = system
    ctx.state['logger'] = system.logger
    study = load_study_yaml(ctx.study_dir) if (ctx.study_dir / 'study.yaml').is_file() else {}
    check_frozen(ctx.study_dir, study)
    run_study_phase(ctx.study_dir, 'prepare', ctx, study, entry='before')
    ctx.state['recipe'] = apply_recipe(
        ctx.study_dir,
        study,
        RecipeContext(
            study_dir=ctx.study_dir,
            run_id=str(ctx.control.id),
            seed=int(ctx.control.seed),
            config=copy.deepcopy(cfg),
            shared_data_dir=ctx.layout.shared_data_dir,
            shared_model_dir=ctx.layout.shared_model_dir,
            assets_dir=ctx.layout.assets_dir,
        ),
    )
    origin = cfg.get('origin')
    if origin not in (None, ''):
        apply_model_origin(normalize_origin(origin))
    data = DataFactory.build(
        DataConfig.from_mapping(ctx.control.data),
        ctx.layout.shared_data_dir,
        seed=ctx.control.seed,
        origin=origin,
    )
    model = ModelFactory.build(
        ModelConfig.from_mapping(ctx.control.model),
        ctx.layout.shared_model_dir,
        data_meta=getattr(data, 'meta', None),
        origin=origin,
    )
    if model.module is not None:
        model.module = system.place_module(model.module)
    algorithm = AlgorithmFactory.build(AlgorithmConfig.from_mapping(ctx.control.algorithm))
    tracker = make_tracker(
        ctx.layout.assets_dir,
        AlgorithmConfig.from_mapping(ctx.control.algorithm),
    )

    ctx.state['system'] = system
    ctx.state['data'] = data
    ctx.state['model'] = model
    ctx.state['algorithm'] = algorithm
    ctx.state['tracker'] = tracker
    ctx.state['logger'] = system.logger
    ctx.state.setdefault('observations', [])
    system.logger.info(f'prepared control={ctx.control.id}')
