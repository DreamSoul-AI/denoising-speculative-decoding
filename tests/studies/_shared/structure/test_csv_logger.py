"""Per-prompt CSV rows."""

import csv

import pytest

from studies._shared.structure import csv_logger
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


@pytest.mark.parametrize('new_instance', [False, True])
def test_csv_rows_follow_one_frozen_header(tmp_path, new_instance):
    logger = Logger(tmp_path)
    logger.log({'accepted': 3, 'tau': 1.5})
    previous = (tmp_path/'metrics.csv').read_bytes()
    if new_instance:
        logger = Logger(tmp_path)
    logger.log({'tau': 2., 'accepted': 4})
    path = tmp_path/'metrics.csv'
    assert path.read_bytes().startswith(previous)
    with path.open(encoding='utf-8', newline='') as handle:
        assert list(csv.DictReader(handle)) == [{'accepted': '3', 'tau': '1.5000'}, {'accepted': '4', 'tau': '2.0000'}]


@pytest.mark.parametrize('damage', ['missing', 'extra', 'empty_first'])
def test_csv_schema_admission_preserves_previous_bytes(tmp_path, damage):
    logger = Logger(tmp_path)
    path = tmp_path/'metrics.csv'
    if damage == 'empty_first':
        path.touch(); row = {}
    else:
        logger.log({'a': 1, 'b': 2})
        row = {'a': 3} if damage == 'missing' else {'a': 3, 'b': 4, 'c': 5}
    previous = path.read_bytes()
    with pytest.raises(ValueError): logger.log(row)
    assert path.read_bytes() == previous


@pytest.mark.parametrize('damage', ['empty_header', 'empty_column', 'duplicate', 'quote', 'encoding', 'schema',
                                   'no_newline', 'live_order', 'live_empty', 'live_removed'])
def test_invalid_existing_csv_header_is_not_repaired(tmp_path, damage):
    logger = Logger(tmp_path)
    path = tmp_path/'metrics.csv'
    if damage.startswith('live_'):
        logger.log({'a': 1, 'b': 2})
    data = {'empty_header': b'\r\n', 'empty_column': b'a,\r\n', 'duplicate': b'a,a\r\n',
        'quote': b'a,"b\r\n', 'encoding': b'\xff,b\r\n', 'schema': b'other,b\r\n1,2\r\n',
        'no_newline': b'a,b\r\n1,2', 'live_order': b'b,a\r\n2,1\r\n', 'live_empty': b''}
    if damage == 'live_removed': path.unlink()
    else: path.write_bytes(data[damage])
    previous = path.read_bytes() if path.exists() else None
    with pytest.raises((ValueError, csv.Error)): logger.log({'a': 3, 'b': 4})
    assert path.read_bytes() == previous if previous is not None else not path.exists()


def test_empty_existing_csv_receives_one_header(tmp_path):
    path = tmp_path/'metrics.csv'; path.touch()
    logger = Logger(tmp_path)
    logger.log({'a': 1}); logger.log({'a': 2})
    with path.open(encoding='utf-8', newline='') as handle:
        assert list(csv.reader(handle)) == [['a'], ['1'], ['2']]


def test_step_prefix_and_utf8_csv_round_trip(tmp_path):
    logger = Logger(tmp_path)
    logger.log({'name': '中文,"quote"\nline', 'unknown': None, 'flag': False, 'count': 3, 'value': 1.23456},
               step=0, prefix='test')
    path = tmp_path/'metrics.csv'
    assert '中文' in path.read_bytes().decode('utf-8')
    with path.open(encoding='utf-8', newline='') as handle:
        assert list(csv.DictReader(handle)) == [{'step': '0', 'test/name': '中文,"quote"\nline',
            'test/unknown': '', 'test/flag': 'False', 'test/count': '3', 'test/value': '1.2346'}]


@pytest.mark.parametrize('save_dir', [None, ''])
def test_stdout_only_logger_keeps_dynamic_rows(save_dir, monkeypatch, capsys):
    def no_io(*args, **kwargs): pytest.fail('stdout-only logger performed file IO')
    monkeypatch.setattr(csv_logger.os, 'makedirs', no_io)
    monkeypatch.setattr(csv_logger, 'open', no_io, raising=False)
    logger = Logger(save_dir)
    logger.log({'a': 1.23456}, step=0, prefix='test')
    logger.log({'b': False, 'missing': None})
    text = capsys.readouterr().out
    assert 'step=0' in text and 'test/a=1.2346' in text and 'b=False' in text and 'missing=None' in text


@pytest.mark.parametrize('failure', ['mkdir', 'open', 'write'])
def test_csv_io_errors_propagate(tmp_path, monkeypatch, failure):
    def deny(*args, **kwargs): raise PermissionError('controlled CSV IO failure')
    if failure == 'mkdir': monkeypatch.setattr(csv_logger.os, 'makedirs', deny)
    elif failure == 'open': monkeypatch.setattr(csv_logger, 'open', deny, raising=False)
    else:
        original = csv_logger.csv.DictWriter
        def writer(*args, **kwargs):
            result = original(*args, **kwargs); result.writerow = deny
            return result
        monkeypatch.setattr(csv_logger.csv, 'DictWriter', writer)
    with pytest.raises(PermissionError, match='controlled CSV IO failure'):
        Logger(tmp_path).log({'a': 1})
