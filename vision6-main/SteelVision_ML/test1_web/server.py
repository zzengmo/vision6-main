"""Local React + Python server. Serves built assets and the existing model adapters."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote, quote
import argparse
import base64
import csv
import io
import json
import mimetypes
import sys
import hashlib
import threading
import webbrowser
import re

import cv2
import numpy as np
from PIL import Image

import backend
import deep_adapter as deep
from deep_reference import table_with
from preview.preprocessing import preprocess
from preview.edge_contour import auto_canny
from preview.roi import analyze_roi

ROOT = Path(__file__).resolve().parent
DIST = ROOT/'frontend/dist'
EXAMPLES = [dict(class_id=1, filename='ceccb1eef.jpg', coverage_pct=1.4),
            dict(class_id=2, filename='00c88fed0.jpg', coverage_pct=1.1),
            dict(class_id=3, filename='d5d024e03.jpg', coverage_pct=7.1),
            dict(class_id=4, filename='1fc012f23.jpg', coverage_pct=11.0)]
POLICIES = {'checkpoint': '원래 저장 기준', 'RF_same_FP_budget': '누락 우선 · 검증 RF 오탐 상한',
            'recall95': '검증 Recall 95% 기준', 'RF_FP_budget75pct': '검증 오탐 상한 75%'}


def png_bytes(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format='PNG')
    return buffer.getvalue()


def data_url(raw, mime='image/png'):
    return 'data:'+mime+';base64,'+base64.b64encode(raw).decode('ascii')


def image_url(rgb):
    return data_url(png_bytes(rgb))


def csv_url(rows):
    buffer = io.StringIO()
    columns = list(rows[0]) if rows else ['x', 'y', 'width', 'height', 'mean_score']
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return data_url(buffer.getvalue().encode('utf-8-sig'), 'text/csv;charset=utf-8')


def example_path(filename):
    for directory in [ROOT/'data/examples', ROOT/'data/train_images']:
        candidate = directory/filename
        if candidate.is_file():
            return candidate
    return None


def metadata():
    rf, seg, state = backend.rf_status(), backend.seg_status(), deep.status()
    reference = deep.reference()
    models = [dict(id='rf_current', label='Random Forest · 현재 PC', family='머신러닝',
                   ready=bool(rf), input='회색조 200×32 · 특징 39개',
                   policies=[dict(id='checkpoint', label='검증에서 선택한 원래 기준', threshold=rf['threshold'] if rf else None)])]
    for model_id, (_, label) in deep.MODEL_FILES.items():
        policies = state.get('model_policies', {}).get(model_id, {})
        models.append(dict(id=model_id, label=label, family='머신러닝' if model_id == 'baseline_rf' else '딥러닝',
                           ready=state['runtime_ready'] and bool(policies),
                           input='회색조 200×32 · 특징 39개' if model_id == 'baseline_rf' else '회색조 800×128 · ImageNet 정규화',
                           policies=[dict(id=k, label=POLICIES.get(k, k), threshold=v) for k, v in policies.items()]))
    versions = [deep.file_version(p) for p in [ROOT/'artifacts/rf/model.joblib', ROOT/'artifacts/segmentation/models/unet_best.pt',
                ROOT/'artifacts/deep_learning/runtime.json', *deep.paths().values(),
                *(state['models']/item['filename'] for item in state['inventory'])] if isinstance(p, Path)]
    revision = hashlib.sha256(repr(versions).encode()).hexdigest()
    return dict(models=models, rf=rf, segmentation=seg, revision=revision,
                checkpoint=state['checkpoint'], deep_ready=state['runtime_ready'],
                deep_comparison=table_with(reference, '운영 방식', '누락 FN', '오탐 FP'),
                deep_candidates=table_with(reference, 'candidate', 'policy', 'split', 'fn', 'fp'),
                deep_class_alarm=table_with(reference, '클래스', '평가 양성 이미지', 'DL 경보 recall'),
                deep_figures=[f'/api/deep-figures/{i}' for i in range(len(reference['images']))] if reference else [],
                reference_figures=backend.reference_outputs()['images'],
                examples=[dict(item, available=bool(example_path(item['filename']))) for item in EXAMPLES],
                report_url='/api/report', limits=dict(bytes=30*1024*1024, pixels=25_000_000, unet_size=[1600, 256]))


def inspect_image(raw, model_id, policy):
    original = deep.decode_input(raw)
    if model_id == 'rf_current':
        if policy != 'checkpoint':
            raise ValueError('현재 PC RF는 저장된 원래 판정 기준만 지원합니다.')
        source = backend.inspect_quality(raw)
        result = dict(model_id=model_id, run_id='current_pc_rf', decision='defect' if source['defect'] else 'normal',
                      score=source['score'], threshold=source['threshold'], elapsed_seconds=source['seconds'],
                      classification_unit='image', weights_sha256=backend.rf_status()['weights_sha256'], policy='checkpoint',
                      class_names=[], detections=[], masks=None, coordinate_frame='original_image')
    else:
        result = deep.inspect(raw, model_id, policy)
    result['original_url'] = image_url(original)
    result['csv_url'] = csv_url([{key: result.get(key) for key in
        ['model_id', 'run_id', 'decision', 'score', 'threshold', 'policy', 'elapsed_seconds', 'weights_sha256', 'classification_unit']}])
    return result


def localize_image(raw):
    original = deep.decode_input(raw)
    if original.size != (1600, 256):
        raise ValueError('U-Net 위치 분석은 EXIF 방향 처리 후 1600×256인 원본 이미지만 지원합니다.')
    result = backend.inspect_location(raw)
    return dict(boxes=result['boxes'], mask_pixels=result['mask_pixels'], elapsed_seconds=result['wall_seconds'],
                threshold=result['threshold'], coordinate_frame=result['coordinate_frame'],
                original_url=image_url(original), overlay_url=data_url(result['overlay.png']),
                mask_url=data_url(result['mask.png']), candidates_url=data_url(result['box_candidates.png']),
                csv_url=data_url(result['boxes.csv'], 'text/csv;charset=utf-8'))


def preprocess_image(raw, options):
    if not isinstance(options, dict):
        raise ValueError('전처리 옵션 형식이 올바르지 않습니다.')
    original = np.asarray(deep.decode_input(raw))
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    processed, steps = preprocess(gray, {key: bool(options.get(key, False)) for key in
                                      ['clahe', 'bilateral', 'sharpen', 'opening']}, return_steps=True)
    return dict(original_url=image_url(original), processed_url=image_url(processed),
                canny_url=image_url(auto_canny(processed)),
                roi_url=image_url(cv2.cvtColor(analyze_roi(processed)['overlay'], cv2.COLOR_BGR2RGB)) if options.get('grid') else None,
                brightness=float(processed.mean()), contrast=float(processed.std()),
                edge_density=float((auto_canny(processed)>0).mean()),
                histogram=dict(original=np.histogram(gray, 32, (0, 256))[0].tolist(),
                               processed=np.histogram(processed, 32, (0, 256))[0].tolist()),
                steps=[dict(name=name, url=image_url(image)) for name, image in steps.items()])


class Handler(BaseHTTPRequestHandler):
    def end_headers(self):
        origin = self.headers.get('Origin')
        if origin and origin in getattr(self.server, 'allowed_origins', ()):
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        super().end_headers()

    def do_OPTIONS(self):
        origin = self.headers.get('Origin')
        if not urlsplit(self.path).path.startswith('/api/') or origin not in getattr(self.server, 'allowed_origins', ()):
            return self.json(403, dict(error='허용되지 않은 앱 주소입니다.'))
        if self.headers.get('Access-Control-Request-Method', 'GET') not in {'GET', 'POST'}:
            return self.json(405, dict(error='지원하지 않는 요청 방식입니다.'))
        requested = {name.strip().lower() for name in self.headers.get('Access-Control-Request-Headers', '').split(',') if name.strip()}
        if not requested.issubset({'content-type', 'accept'}):
            return self.json(403, dict(error='지원하지 않는 요청 헤더입니다.'))
        self.send_response(204)
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Accept')
        self.send_header('Access-Control-Max-Age', '600')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def send(self, code, content, mime):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store' if self.path.startswith('/api/') else 'no-cache')
        self.end_headers()
        if self.command == 'HEAD':
            return
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json(self, code, value):
        self.send(code, json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8'), 'application/json; charset=utf-8')

    def send_file(self, path):
        """Stream static assets; byte ranges let browsers seek without loading the whole video."""
        with path.open('rb') as stream:
            size = path.stat().st_size
            start, end, partial = 0, size-1, False
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
                valid = bool(match and size and any(match.groups()))
                if valid:
                    left, right = match.groups()
                    if left:
                        start, end = int(left), min(int(right), size-1) if right else size-1
                    else:
                        suffix = int(right)
                        valid = suffix > 0
                        start, end = max(0, size-suffix), size-1
                    valid = valid and 0 <= start <= end < size
                if not valid:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                partial = True
            length = max(0, end-start+1)
            self.send_response(206 if partial else 200)
            self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(length))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if partial:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command == 'HEAD':
                return
            stream.seek(start)
            try:
                remaining = length
                while remaining:
                    chunk = stream.read(min(256*1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        try:
            if path == '/api/health':
                host = self.server.server_address[0]
                return self.json(200, dict(ok=True, app='steelvision-react', version='video-home-v2',
                                           listen_host=host, port=self.server.server_port,
                                           network_access=host not in {'127.0.0.1', '::1', 'localhost'}))
            if path == '/_stcore/health':
                return self.send(200, b'ok', 'text/plain')
            if path == '/api/metadata':
                return self.json(200, metadata())
            if path.startswith('/api/deep-figures/'):
                index = int(path.rsplit('/', 1)[1])
                reference = deep.reference()
                if not reference or not 0 <= index < len(reference['images']):
                    raise FileNotFoundError('저장된 그림이 없습니다.')
                return self.send(200, reference['images'][index], 'image/png')
            if path.startswith('/api/reference/'):
                filename = path.rsplit('/', 1)[1]
                allowed = {name for rows in backend.reference_outputs()['images'].values() for name in rows}
                if filename not in allowed:
                    raise FileNotFoundError('저장된 그림이 없습니다.')
                return self.send(200, (ROOT/'reference/assets'/filename).read_bytes(), 'image/png')
            if path.startswith('/api/samples/'):
                label = path.rsplit('/', 1)[1]
                if label not in {'normal', 'defect'}:
                    raise FileNotFoundError('검증 예시가 없습니다.')
                return self.send(200, (ROOT/'artifacts/verification'/label/'input.png').read_bytes(), 'image/png')
            if path.startswith('/api/examples/'):
                filename = path.rsplit('/', 1)[1]
                source = example_path(filename) if filename in {item['filename'] for item in EXAMPLES} else None
                if source is None:
                    raise FileNotFoundError('해당 원본 JPG가 로컬에 없습니다. 노트북 저장 그림과 원본 파일은 별개입니다.')
                return self.send(200, source.read_bytes(), 'image/jpeg')
            if path == '/api/report':
                source = next((ROOT/'reports').glob('*.html'))
                self.send_response(302)
                self.send_header('Location', '/reports/'+quote(source.name))
                self.end_headers()
                return
            if path.startswith('/reports/'):
                source = (ROOT/path.lstrip('/')).resolve()
                if not source.is_relative_to((ROOT/'reports').resolve()) or not source.is_file():
                    raise FileNotFoundError('보고서 파일이 없습니다.')
                content = source.read_bytes()
                if source.suffix == '.html':
                    content = content.replace('../../../../02_작업상세.md'.encode(), '/api/document/details'.encode())
                return self.send(200, content, mimetypes.guess_type(str(source))[0] or 'application/octet-stream')
            if path == '/api/document/details':
                return self.send(200, (ROOT.parents[2]/'02_작업상세.md').read_bytes(), 'text/plain; charset=utf-8')
            if path.startswith('/api/'):
                raise FileNotFoundError('존재하지 않는 API입니다.')
            target = (DIST/path.lstrip('/')).resolve()
            if not target.is_relative_to(DIST.resolve()):
                raise FileNotFoundError('허용되지 않는 경로입니다.')
            if not target.is_file():
                target = DIST/'index.html' if '.' not in Path(path).name else target
            self.send_file(target)
        except (ValueError, FileNotFoundError, StopIteration) as exc:
            self.json(404, dict(error=str(exc) or '파일을 찾을 수 없습니다.'))
        except Exception as exc:
            self.json(500, dict(error='자료를 읽지 못했습니다: '+str(exc)))

    def do_POST(self):
        parsed = urlsplit(self.path)
        if parsed.path not in {'/api/inspect', '/api/localize', '/api/preprocess'}:
            return self.json(404, dict(error='존재하지 않는 API입니다.'))
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 30*1024*1024:
                raise ValueError('JPG/PNG 파일을 30MB 이하로 업로드하세요.')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('업로드가 완료되지 않았습니다.')
            params = parse_qs(parsed.query)
            if parsed.path == '/api/inspect':
                result = inspect_image(raw, params.get('model', ['rf_current'])[0], params.get('policy', ['checkpoint'])[0])
            elif parsed.path == '/api/localize':
                result = localize_image(raw)
            else:
                result = preprocess_image(raw, json.loads(params.get('options', ['{}'])[0]))
            self.json(200, result)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            self.json(400, dict(error=str(exc)))
        except Exception as exc:
            self.json(500, dict(error='분석 중 오류가 발생했습니다: '+str(exc)))


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8502)
    parser.add_argument('--host', default='127.0.0.1', help='Use 0.0.0.0 for a phone on the same local network.')
    parser.add_argument('--allow-origin', action='append', default=[], help='Optional Expo web preview origin; native apps do not need CORS.')
    parser.add_argument('--open-browser', action='store_true')
    return parser.parse_args(argv)


def main():
    args = parse_arguments()
    if not (DIST/'index.html').is_file():
        sys.exit('React 화면 빌드가 없습니다. frontend 폴더에서 npm run build를 실행하세요.')
    state = deep.status()
    if state['source_exists'] and state['worker_exists'] and not state['runtime_ready']:
        try:
            deep.audit_checkpoint()
        except (ValueError, OSError, RuntimeError) as exc:
            print('딥러닝 준비 안내: '+str(exc), flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.allowed_origins = tuple(args.allow_origin)
    print(f'SteelVision ML: http://{args.host}:{args.port}', flush=True)
    if args.open_browser:
        threading.Timer(.7, lambda: webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
