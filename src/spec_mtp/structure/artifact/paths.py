"""Canonical artifact names and explicit derived-output boundaries."""

from pathlib import Path

CONFIG_NAME = 'config.yaml'
RESULT_NAME = 'result.json'
INDEX_NAME = 'index.json'
PROCESS_NAME = 'process.json'
DERIVED_NAME = 'derived.json'
ASSETS_DIRNAME = 'assets'
SHARED_DIRNAME = 'shared'
DOCS_DIRNAME = 'docs'
RUNS_DIRNAME = 'runs'


def validate_output_root(output_root: Path | str, *inputs: Path | str) -> Path:
    """Keep derived output separate from explicit inputs and source checkouts."""
    output = Path(output_root).resolve()
    roots = {Path(value).resolve() for value in inputs}
    roots.add(Path(__file__).resolve().parents[2])
    for source in tuple(roots):
        for candidate in (source, *source.parents):
            if (candidate / '.git').exists() or (candidate / 'pyproject.toml').is_file():
                roots.add(candidate)
                break
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError('output root must be separate from the inputs and source roots')
    return output
