"""Adapters for the isolated test1 web. No imports from the old ML backend."""
from pathlib import Path
from functools import lru_cache
from time import perf_counter
import hashlib
import io
import json
import os
import subprocess
import tempfile
import zipfile
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from native_features import extract_features

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT/'artifacts'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def settings():
    return read_json(ROOT/'settings.json')


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else (ROOT/p).resolve()


def python_path(worker=False):
    config = settings()
    name = 'worker_python' if worker else 'ui_python'
    candidates = [ROOT/('.venv-model/Scripts/python.exe' if worker else '.venv/Scripts/python.exe'),
                  resolve(config[name])]
    return next((p for p in candidates if p.is_file()), candidates[0])


def validate_upload(raw, localization=False):
    if len(raw) > 30*1024*1024:
        raise ValueError('파일은 30MB 이하로 업로드하세요.')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {'JPEG', 'PNG'}:
                raise ValueError('JPG/PNG만 지원합니다.')
            if image.width*image.height > 25_000_000:
                raise ValueError('최대 2,500만 픽셀까지 지원합니다.')
            if localization and image.size != (1600, 256):
                raise ValueError('test1 위치 모델은 원본 1600×256 이미지만 지원합니다. 임의 크기로 변형하지 않습니다.')
            return np.asarray(image.convert('RGB')).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('이미지 파일을 읽을 수 없습니다.') from exc


class SourceImages:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None
        names = self.zip.namelist() if self.zip else [str(p.relative_to(self.path)) for p in self.path.rglob('*.jpg')]
        self.index = {Path(n).name: n for n in names if Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png'}}
        if not self.index:
            raise ValueError('원본 이미지가 없습니다. settings.json의 images 경로를 확인하세요.')

    def read(self, image_id):
        name = self.index[image_id]
        return self.zip.read(name) if self.zip else (self.path/name).read_bytes()

    def close(self):
        if self.zip:
            self.zip.close()


def rf_status():
    path = ARTIFACTS/'rf/metadata.json'
    if not path.is_file() or not (ARTIFACTS/'rf/model.joblib').is_file():
        return None
    return read_json(path)


def seg_status():
    path = ARTIFACTS/'segmentation/web_metadata.json'
    if not path.is_file() or not (ARTIFACTS/'segmentation/models/unet_best.pt').is_file():
        return None
    return read_json(path)


@lru_cache(maxsize=2)
def load_rf(version):
    path = ARTIFACTS/'rf/model.joblib'
    metadata = rf_status()
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['weights_sha256']:
        raise ValueError('RF 가중치가 평가 당시와 다릅니다. 별도 준비 명령으로 다시 확인하세요.')
    return joblib.load(path)


def inspect_quality(raw):
    original = validate_upload(raw)
    metadata = rf_status()
    if metadata is None:
        raise FileNotFoundError('Random Forest가 없습니다. 준비 명령을 먼저 실행하세요.')
    start = perf_counter()
    model = load_rf((ARTIFACTS/'rf/model.joblib').stat().st_mtime_ns)
    features = extract_features(io.BytesIO(raw))
    score = float(model.predict_proba(features[None, :])[0, list(model.classes_).index(1)])
    return dict(defect=bool(score >= metadata['threshold']), score=score,
                threshold=metadata['threshold'], seconds=perf_counter()-start,
                original=original, model='Random Forest', classification_unit='image')


def inspect_location(raw):
    validate_upload(raw, localization=True)
    if seg_status() is None:
        raise FileNotFoundError('U-Net 가중치가 없습니다. 별도 학습/가중치 준비가 필요합니다.')
    python = python_path(worker=True)
    if not python.is_file():
        raise FileNotFoundError('모델 실행 Python 경로를 settings.json에서 확인하세요.')
    temporary = ARTIFACTS/'uploads'
    temporary.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with tempfile.TemporaryDirectory(dir=temporary) as folder:
        folder = Path(folder)
        source = folder/'input.png'
        source.write_bytes(raw)
        env = dict(os.environ, PYTHONUTF8='1', MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),
                   YOLO_CONFIG_DIR=str(ROOT/'.cache/ultralytics'))
        try:
            completed = subprocess.run([str(python), str(ROOT/'run_test1.py'), 'predict-unet',
                                    '--image', str(source), '--output', str(folder/'output')],
                                   cwd=ROOT, env=env, capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=180,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('U-Net 분석이 180초를 초과했습니다. 실행 환경을 확인하세요.') from exc
        if completed.returncode:
            raise RuntimeError((completed.stderr or completed.stdout)[-1800:])
        output = folder/'output'
        result = read_json(output/'summary.json')
        for filename in ['overlay.png', 'mask.png', 'box_candidates.png', 'boxes.csv']:
            result[filename] = (output/filename).read_bytes()
        result['overlay'] = np.asarray(Image.open(io.BytesIO(result['overlay.png'])).convert('RGB')).copy()
        result['wall_seconds'] = perf_counter()-start
        return result


def reference_outputs():
    return read_json(ROOT/'reference/outputs.json')
