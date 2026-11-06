import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

import server


@pytest.fixture(scope='module')
def mobile_api():
    instance = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
    instance.allowed_origins = ('http://localhost:8081',)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance.server_port
    instance.shutdown()
    instance.server_close()


def send(port, method, path, headers=None):
    connection = HTTPConnection('127.0.0.1', port, timeout=15)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_phone_access_is_explicit_and_existing_web_stays_local():
    assert server.parse_arguments([]).host == '127.0.0.1'
    args = server.parse_arguments(['--host', '0.0.0.0', '--port', '8504', '--allow-origin', 'http://localhost:8081'])
    assert (args.host, args.port, args.allow_origin) == ('0.0.0.0', 8504, ['http://localhost:8081'])


def test_expo_web_binary_upload_preflight(mobile_api):
    status, headers, body = send(mobile_api, 'OPTIONS', '/api/inspect', {
        'Origin': 'http://localhost:8081',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'content-type',
    })
    assert status == 204 and body == b''
    assert headers['Access-Control-Allow-Origin'] == 'http://localhost:8081'
    assert headers['Access-Control-Allow-Headers'] == 'Content-Type, Accept'
    status, headers, body = send(mobile_api, 'GET', '/api/health', {'Origin': 'http://localhost:8081'})
    assert status == 200 and headers['Access-Control-Allow-Origin'] == 'http://localhost:8081'
    assert json.loads(body)['network_access'] is False


def test_other_browser_origins_and_headers_are_not_enabled(mobile_api):
    status, headers, _ = send(mobile_api, 'OPTIONS', '/api/inspect', {'Origin': 'https://unrelated.example'})
    assert status == 403 and 'Access-Control-Allow-Origin' not in headers
    status, headers, _ = send(mobile_api, 'GET', '/api/health', {'Origin': 'https://unrelated.example'})
    assert status == 200 and 'Access-Control-Allow-Origin' not in headers
    status, _, _ = send(mobile_api, 'OPTIONS', '/api/inspect', {
        'Origin': 'http://localhost:8081', 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'authorization',
    })
    assert status == 403
