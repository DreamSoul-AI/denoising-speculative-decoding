"""Per-prompt CSV rows."""

import csv

from studies._shared.structure.csv_logger import Logger


def test_appends_rows_under_one_header(tmp_path):
    logger = Logger(tmp_path, filename='spec_metrics.csv')
    logger.log({'accepted': 3, 'tau': 1.5})
    logger.log({'accepted': 1, 'tau': 1.0})
    with (tmp_path / 'spec_metrics.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {'accepted': '3', 'tau': '1.5000'},
        {'accepted': '1', 'tau': '1.0000'},
    ]
