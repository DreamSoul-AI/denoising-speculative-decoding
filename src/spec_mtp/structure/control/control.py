"""Control object: holds RunConfig for one Run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spec_mtp.structure.control.run_config import RunConfig, run_config_to_mapping


@dataclass
class Control:
    """Structure-side handle for one Run's resolved RunConfig."""

    run: RunConfig
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.run.id

    @property
    def seed(self) -> int | None:
        return self.run.seed

    @property
    def experiment(self) -> str | None:
        return self.run.experiment

    @property
    def data(self) -> dict[str, Any]:
        return self.run.data.to_mapping()

    @property
    def model(self) -> dict[str, Any]:
        return self.run.model.to_mapping()

    @property
    def algorithm(self) -> dict[str, Any]:
        return self.run.algorithm.to_mapping()

    @property
    def system(self) -> dict[str, Any]:
        return self.run.system.to_mapping()

    def to_dict(self) -> dict[str, Any]:
        return run_config_to_mapping(self.run)


def control_from_config(cfg: dict[str, Any]) -> Control:
    """Build Control from Artifact Config mapping with its content-hash id."""
    run = RunConfig.from_mapping(cfg, assign_id=True)
    return Control(run=run, raw=dict(cfg))
