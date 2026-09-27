"""Canonicalize explicit positive coordinate signs for Foundation mesh readers.

Only leading '+' signs on finite ASCII vector components are removed. Exponent
signs, negative signs, precision, and every other byte are preserved. Repairs
retain the original file and an audit record; unrelated syntax is not repaired.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile

_NUMBER = rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_VECTOR = re.compile(rb"\s*\(\s*(" + _NUMBER + rb")\s+(" + _NUMBER
                     + rb")\s+(" + _NUMBER + rb")\s*\)\s*")
_PLUS = re.compile(rb"(?<=[(\s])\+(?=[\d.])")


def normalize_point_signs(path):
    """Return number of repaired point records; leave clean/binary files alone."""
    path = Path(path)
    edits = {}
    header = bytearray()
    in_header = True
    with path.open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            if in_header:
                header.extend(line)
                if b'}' not in line:
                    if len(header) > 65536:
                        raise ValueError(f'Unrecognized points header: {path}')
                    continue
                in_header = False
                if re.search(rb'\bformat\s+binary\s*;', header):
                    return 0
                if not re.search(rb'\bformat\s+ascii\s*;', header):
                    raise ValueError(f'Unrecognized points format: {path}')
                continue
            if b'+' not in line or not _PLUS.search(line):
                continue
            if not _VECTOR.fullmatch(line):
                raise ValueError(f'Refusing to repair non-vector syntax: {path}:{line_number}')
            edits[line_number] = (line, _PLUS.sub(b'', line))
    if not edits:
        return 0

    # Keep the exact offending bytes for diagnosis, including across repeat runs.
    with tempfile.NamedTemporaryFile(prefix=path.name + '.before-sign-fix-',
                                     dir=path.parent, delete=False) as original:
        backup = Path(original.name)
    shutil.copy2(path, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + '.normalized-',
                                         dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            with backup.open('rb') as stream:
                for line_number, line in enumerate(stream, 1):
                    output.write(edits.get(line_number, (line, line))[1])
        shutil.copymode(path, temporary)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    record = {'path': str(path), 'backup': str(backup), 'repairs': [
        {'line': n, 'before': before.decode().strip(), 'after': after.decode().strip()}
        for n, (before, after) in edits.items()
    ]}
    with path.with_name('point-sign-repairs.jsonl').open('a') as audit:
        audit.write(json.dumps(record) + '\n')
    return len(edits)
