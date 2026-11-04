from typing import Any

import csv
import os


class Logger:
    """Simple CSV + stdout logger for experiment metrics."""

    def __init__(self, save_dir: str | None, filename: str = 'metrics.csv') -> None:
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename) if save_dir else None
        self._header_written = False

    def log(self, metrics: dict[str, Any], step: int | None = None, prefix: str = '') -> None:
        row: dict[str, Any] = {}
        if step is not None:
            row['step'] = step
        for k, v in metrics.items():
            key = f'{prefix}/{k}' if prefix else k
            row[key] = f'{v:.4f}' if isinstance(v, float) else v

        print('  '.join([f'{k}={v}' for k, v in row.items()]))

        if self.filepath:
            write_header = not self._header_written
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                    self._header_written = True
                writer.writerow(row)
