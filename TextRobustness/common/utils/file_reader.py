"""
File Read module, support csv and json.

https://github.com/fastnlp/fastNLP/blob/master/fastNLP/io/file_reader.py

============================================

"""

import json
import csv


def read_csv(path, encoding='utf-8', headers=None, sep=',', dropna=True):
    """ Construct a generator to read csv items.

    Args:
        path: file path
        encoding: file's encoding, default: utf-8
        headers:  file's headers, if None, make file's first line as headers. default: None
        sep: separator for each column. default: ','
        dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True

    Returns:
        generator, every time yield (line number, csv item)
    """
    with open(path, 'r', encoding=encoding) as csv_file:
        f = csv.reader(csv_file, delimiter=sep)
        start_idx = 0
        if headers is None:
            headers = next(f)
            start_idx += 1
        elif not isinstance(headers, (list, tuple)):
            raise TypeError("headers should be list or tuple, not {0}.".format(type(headers)))

        for line_idx, line in enumerate(f, start_idx):
            contents = line
            if len(contents) != len(headers):
                if dropna:
                    continue
                else:
                    if "" in headers:
                        raise ValueError(("Line {0} has {1} parts, while header has {2} parts.\n" +
                                          "Please check the empty parts or unnecessary '{3}'s  in header.")
                                         .format(line_idx, len(contents), len(headers), sep))
                    else:
                        raise ValueError("Line {0} has {1} parts, while header has {2} parts."
                                         .format(line_idx, len(contents), len(headers)))
            _dict = {}
            for header, content in zip(headers, contents):
                _dict[header] = content

            yield line_idx, _dict


def read_json(path, encoding='utf-8', fields=None, dropna=True):
    """ Construct a generator to read json items.

    Args:
        path: file path
        encoding: file's encoding, default: utf-8
        fields: json object's fields that needed, if None, all fields are needed. default: None
        dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True

    Returns:
        generator, every time yield (line number, json item)
    """
    if fields:
        fields = set(fields)
    with open(path, 'r', encoding=encoding) as f:
        for line_idx, line in enumerate(f):
            data = json.loads(line)
            if fields is None:
                yield line_idx, data
                continue
            _res = {}
            for k, v in data.items():
                if k in fields:
                    _res[k] = v
            if len(_res) < len(fields):
                if dropna:
                    continue
                else:
                    raise ValueError('invalid instance at line: {0}'.format(line_idx))
            yield line_idx, _res
