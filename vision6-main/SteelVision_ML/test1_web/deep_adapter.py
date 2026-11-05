"""Lightweight UI adapter. PyTorch stays in the separate model process."""
from functools import lru_cache
from pathlib import Path
import hashlib
import io
import json
import os
import subprocess
import tempfile

from PIL import Image, ImageOps, UnidentifiedImageError

from backend import ROOT, ARTIFACTS, python_path, resolve, settings
from deep_reference import load_reference


MODEL_FILES = {
    'resnet18': ('best_resnet18.pt', 'ResNet18 미세조정'),
    'resnet18_tta': ('best_resnet18.pt', 'ResNet18 + 좌우반전 평균 (TTA)'),
    'frozen_cnn': ('frozen_cnn_logistic.joblib', '고정 CNN 특징 + Logistic Regression'),
    'baseline_rf': ('baseline_random_forest.joblib', '딥러닝 실험 RF 기준선'),
}


def paths():
    config = settings()
    def locate(key, default):
        configured = resolve(config.get(key, default))
        if configured.exists():
            return configured
        # Accept files beside the baseline or at the workspace root after relocation.
        for parent in [ROOT.parent, ROOT.parents[2]]:
            candidate = parent/configured.name
            if candidate.exists():
                return candidate
        return configured
    return {
        'models': resolve(config.get('deep_models', 'artifacts/deep_learning/models')),
        'notebook': locate('deep_notebook', 'reference/test1_deep_learning_comparison.ipynb'),
        'source': locate('deep_source', 'reference/test1_deep_learning.py'),
    }


def file_version(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=24)
def _sha256(version):
    return hashlib.sha256(Path(version[0]).read_bytes()).hexdigest()


def file_sha256(path):
    version = file_version(path)
    return _sha256(version) if version else None


def reference():
    try:
        return load_reference(paths()['notebook'])
    except (OSError, ValueError, KeyError):
        return None


def status():
    locations = paths()
    folder = locations['models']
    audit_path = ARTIFACTS/'deep_learning/checkpoint.json'
    try:
        audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.is_file() else {}
    except (OSError, ValueError):
        audit = {}
    weights = folder/'best_resnet18.pt'
    matches = bool(weights.is_file() and audit.get('weights_sha256') == file_sha256(weights))
    runtime_path = ARTIFACTS/'deep_learning/runtime.json'
    try:
        runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
        ready = bool(matches and runtime.get('source_verified') and runtime.get('source_sha256') == file_sha256(locations['source'])
                     and all(file_sha256(ROOT/name) == expected for name, expected in runtime['files'].items()))
    except (OSError, ValueError, KeyError):
        runtime, ready = {}, False
    return dict(models=folder, source=locations['source'], source_exists=locations['source'].is_file(),
                inventory=[dict(filename=name, exists=(folder/name).is_file(),
                                bytes=(folder/name).stat().st_size if (folder/name).is_file() else None)
                           for name in ['best_resnet18.pt', 'initialized_resnet18.pt',
                                        'frozen_cnn_logistic.joblib', 'baseline_random_forest.joblib']],
                checkpoint=audit if matches else None, weights_match=matches,
                worker_exists=python_path(worker=True).is_file(), runtime_ready=ready,
                model_policies=runtime.get('models', {}) if ready else {})


def decode_input(raw):
    if len(raw) > 30*1024*1024:
        raise ValueError('파일은 30MB 이하로 업로드하세요.')
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {'JPEG', 'PNG'}:
                raise ValueError('JPG/PNG만 지원합니다.')
            if image.width*image.height > 25_000_000:
                raise ValueError('최대 2,500만 픽셀까지 지원합니다.')
            return ImageOps.exif_transpose(image).convert('RGB').copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('이미지 파일을 읽을 수 없습니다.') from exc


def run_worker(command, *arguments):
    python = python_path(worker=True)
    if not python.is_file():
        raise FileNotFoundError('딥러닝 실행 환경이 없습니다. Setup_Test1_Model.cmd를 실행하세요.')
    try:
        process = subprocess.run(
            [str(python), str(ROOT/'deep_worker.py'), command, *map(str, arguments)],
            cwd=ROOT, env=dict(os.environ, PYTHONUTF8='1'), capture_output=True, text=True,
            encoding='utf-8', errors='replace', timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('딥러닝 처리가 180초를 초과했습니다.') from exc
    if process.returncode:
        raise RuntimeError((process.stderr or process.stdout)[-2000:])


def audit_checkpoint():
    run_worker('audit')
    return status()


def inspect(raw, model_id='resnet18', policy='checkpoint'):
    decode_input(raw)
    if model_id not in MODEL_FILES:
        raise ValueError('지원하지 않는 모델입니다.')
    upload_dir = ARTIFACTS/'uploads'
    upload_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=upload_dir) as temporary:
        temporary = Path(temporary)
        (temporary/'input.image').write_bytes(raw)
        run_worker('predict', '--image', temporary/'input.image', '--output', temporary/'result.json',
                   '--model', model_id, '--policy', policy)
        return json.loads((temporary/'result.json').read_text(encoding='utf-8'))


def result_key(raw, model_id, policy):
    locations = paths()
    return (hashlib.sha256(raw).hexdigest(), model_id, policy,
            tuple(file_version(locations['models']/name) for name in
                  ['best_resnet18.pt', 'initialized_resnet18.pt',
                   'frozen_cnn_logistic.joblib', 'baseline_random_forest.joblib']),
            file_version(locations['source']), file_version(locations['notebook']),
            file_version(ROOT/'deep_worker.py'))
