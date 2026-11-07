from typing import Any

import csv
import os


class Logger:
    """Simple CSV + stdout logger for experiment metrics."""

    def __init__(self, save_dir: str | None, filename: str = 'metrics.csv') -> None:
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename) if save_dir else None
        self._fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any], step: int | None = None, prefix: str = '') -> None:
        row: dict[str, Any] = {}
        if step is not None:
            row['step'] = step
        for k, v in metrics.items():
            key = f'{prefix}/{k}' if prefix else k
            row[key] = f'{v:.4f}' if isinstance(v, float) else v

        print('  '.join([f'{k}={v}' for k, v in row.items()]))

        if self.filepath:
            nonempty = os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0
            if self._fieldnames is not None and not nonempty:
                raise ValueError('CSV output disappeared or was emptied')
            fieldnames = list(row)
            if nonempty:
                with open(self.filepath, 'rb') as handle:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b'\n':
                        raise ValueError('CSV output has an unterminated record')
                with open(self.filepath, encoding='utf-8', newline='') as handle:
                    fieldnames = next(csv.reader(handle, strict=True), [])
            if (not fieldnames or any(not isinstance(name, str) or not name.strip() for name in fieldnames)
                    or len(set(fieldnames)) != len(fieldnames)):
                raise ValueError('CSV header must contain unique nonempty column names')
            if self._fieldnames is not None and fieldnames != self._fieldnames:
                raise ValueError('CSV header order changed during logging')
            if set(fieldnames) != set(row):
                raise ValueError('CSV row schema differs from its header')
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not nonempty:
                    writer.writeheader()
                writer.writerow(row)
            self._fieldnames = fieldnames
