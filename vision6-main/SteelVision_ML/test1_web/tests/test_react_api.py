import base64
import io
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from PIL import Image
import pytest

import server


@pytest.fixture(scope='module')
def api():
    instance = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance.server_port
    instance.shutdown()
    instance.server_close()


def request(port, method, path, raw=None):
    connection = HTTPConnection('127.0.0.1', port, timeout=30)
    connection.request(method, path, body=raw, headers={'Content-Type': 'image/png'} if raw else {})
    response = connection.getresponse()
    content = response.read()
    status = response.status
    connection.close()
    return status, content


def test_react_entry_and_model_catalog(api):
    status, content = request(api, 'GET', '/')
    assert status == 200 and b'type="module"' in content
    status, content = request(api, 'GET', '/api/metadata')
    catalog = json.loads(content)
    assert status == 200
    assert {m['id'] for m in catalog['models']} == {'rf_current','resnet18','resnet18_tta','frozen_cnn','baseline_rf'}
    assert len(catalog['deep_comparison']) == 4
    assert catalog['revision']
    assert [e['filename'] for e in catalog['examples']] == ['ceccb1eef.jpg','00c88fed0.jpg','d5d024e03.jpg','1fc012f23.jpg']


def test_corrupt_upload_and_localization_shape_are_errors(api):
    status, content = request(api, 'POST', '/api/inspect?model=rf_current', b'not-an-image')
    assert status == 400 and json.loads(content)['error']
    buffer = io.BytesIO()
    Image.new('RGB', (100,100)).save(buffer, format='PNG')
    status, content = request(api, 'POST', '/api/localize', buffer.getvalue())
    assert status == 400 and '1600' in json.loads(content)['error']
    assert request(api, 'GET', '/api/reference/not_a_saved_file.png')[0] == 404
    assert request(api, 'GET', '/../../settings.json')[0] == 404


def test_real_preprocessing_and_reference_png_roundtrip(api):
    raw = (server.ROOT/'artifacts/verification/defect/input.png').read_bytes()
    status, content = request(api, 'POST', '/api/preprocess', raw)
    result = json.loads(content)
    assert status == 200
    assert sum(result['histogram']['original']) == 1600*256
    assert sum(result['histogram']['processed']) == 1600*256
    assert 0 <= result['edge_density'] <= 1
    processed = base64.b64decode(result['processed_url'].split(',',1)[1])
    with Image.open(io.BytesIO(processed)) as image:
        assert image.size == (1600,256)
    status, content = request(api, 'GET', '/api/reference/cell_06_0.png')
    assert status == 200
    with Image.open(io.BytesIO(content)) as image:
        image.verify()


@pytest.mark.parametrize('filename', ['steel-hero-desktop.mp4', 'steel-hero-mobile.mp4'])
def test_video_ranges_and_head_support_browser_playback(api, filename):
    path = server.DIST/'landing'/filename
    size = path.stat().st_size
    connection = HTTPConnection('127.0.0.1', api, timeout=15)
    connection.request('GET', '/landing/'+filename, headers={'Range': 'bytes=0-511'})
    response = connection.getresponse()
    assert response.status == 206
    assert response.getheader('Content-Type') == 'video/mp4'
    assert response.getheader('Content-Range') == f'bytes 0-511/{size}'
    with path.open('rb') as stream:
        assert response.read() == stream.read(512)
    connection.close()
    connection = HTTPConnection('127.0.0.1', api, timeout=15)
    connection.request('GET', '/landing/'+filename, headers={'Range': 'bytes=-64'})
    response = connection.getresponse()
    assert response.status == 206
    with path.open('rb') as stream:
        stream.seek(-64, 2)
        assert response.read() == stream.read()
    connection.close()
    connection = HTTPConnection('127.0.0.1', api, timeout=15)
    connection.request('GET', '/landing/'+filename, headers={'Range': f'bytes={size}-'})
    response = connection.getresponse()
    assert response.status == 416
    assert response.getheader('Content-Range') == f'bytes */{size}'
    assert response.read() == b''
    connection.close()
    connection = HTTPConnection('127.0.0.1', api, timeout=15)
    connection.request('HEAD', '/landing/'+filename)
    response = connection.getresponse()
    assert response.status == 200
    assert int(response.getheader('Content-Length')) == size
    assert response.getheader('Accept-Ranges') == 'bytes'
    assert response.read() == b''
    connection.close()
