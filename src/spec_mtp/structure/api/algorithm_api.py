"""algorithm_api: Algorithm, AlgorithmTracker, AlgorithmFactory.build, AlgorithmConfig."""

from pathlib import Path

from spec_mtp.structure.algorithm import (
    Algorithm,
    AlgorithmConfig,
    AlgorithmFactory,
    AlgorithmRegistry,
    AlgorithmTracker,
)
from spec_mtp.structure.algorithm.metric import MetricBundle, resolve_metric_names
from spec_mtp.structure.algorithm.resume import sibling_train_dependency


def build(algorithm_config: AlgorithmConfig, **kwargs):
    return AlgorithmFactory.build(algorithm_config, **kwargs)


def make_tracker(assets_dir: Path | str, algorithm_config: AlgorithmConfig | None = None) -> AlgorithmTracker:
    names = None
    glue = None
    if algorithm_config is not None:
        names = resolve_metric_names(algorithm_config.setting('metric'))
        if any('GLUE' in items for items in names.values()):
            glue = str(algorithm_config.setting('glue_subset', 'cola') or 'cola')
    bundle = MetricBundle(names, glue_subset=glue) if glue else MetricBundle(names)
    return AlgorithmTracker(assets_dir, metrics=bundle)
