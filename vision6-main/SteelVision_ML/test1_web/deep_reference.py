"""Read saved notebook outputs without executing any notebook cell."""
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
import base64
import hashlib
import json


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in {'td', 'th'}:
            self.cell = []

    def handle_data(self, value):
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in {'td', 'th'} and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _scalar(value):
    try:
        return float(value) if any(c in value for c in '.eE') else int(value)
    except ValueError:
        return value


@lru_cache(maxsize=2)
def _load(filename, version):
    raw = Path(filename).read_bytes()
    notebook = json.loads(raw.decode('utf-8-sig'))
    tables, images, texts = [], [], []
    for cell in notebook.get('cells', []):
        for output in cell.get('outputs', []):
            data = output.get('data', {})
            if 'text/html' in data:
                parser = _TableParser()
                parser.feed(''.join(data['text/html']))
                if len(parser.rows) > 1:
                    columns = parser.rows[0][1:]  # pandas display index
                    tables.append([dict(zip(columns, map(_scalar, row[1:])))
                                   for row in parser.rows[1:] if len(row) == len(columns)+1])
            if 'image/png' in data:
                images.append(base64.b64decode(''.join(data['image/png'])))
            if output.get('output_type') == 'stream':
                texts.append(''.join(output.get('text', [])))
    return dict(sha256=hashlib.sha256(raw).hexdigest(), tables=tables, images=images, texts=texts)


def load_reference(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return _load(str(path), (stat.st_mtime_ns, stat.st_size))


def table_with(reference, *columns):
    if reference:
        return next((rows for rows in reference['tables']
                     if rows and set(columns).issubset(rows[0])), [])
    return []
