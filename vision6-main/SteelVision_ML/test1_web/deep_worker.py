"""Inference-only entry point in the existing PyTorch environment. Never trains."""
import argparse
import hashlib
import json
import importlib.util
import io
import re
import sys
from time import perf_counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def configuration():
    config = json.loads((ROOT/'settings.json').read_text(encoding='utf-8'))
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else (ROOT/path).resolve()
    return config, resolve


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_module():
    config, resolve = configuration()
    source = resolve(config['deep_source'])
    if not source.is_file():
        raise ValueError('원본 딥러닝 구현 파일이 없습니다.')
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('steelvision_native_deep', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_tensor(raw, module):
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(raw)) as source:
        if source.format not in {'JPEG', 'PNG'} or source.width*source.height > 25_000_000:
            raise ValueError('JPG/PNG, 최대 2,500만 픽셀만 지원합니다.')
        original_size = ImageOps.exif_transpose(source).size
    # Use the original OpenCV grayscale decode and INTER_AREA resize, without a PIL round trip.
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None or (gray.shape[1], gray.shape[0]) != original_size:
        raise ValueError('원본 방향과 딥러닝 입력 좌표가 일치하지 않습니다.')
    resized = cv2.resize(gray, module.SIZE, interpolation=cv2.INTER_AREA)
    return module.batch_input(torch.from_numpy(resized.copy())[None, None], 'cpu', augment=False)


def loaded_network(module, model_dir, frozen=False):
    import torch
    path = model_dir/('initialized_resnet18.pt' if frozen else 'best_resnet18.pt')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = module.StripResNet()
    model.load_state_dict(saved if frozen else saved['state_dict'], strict=True)
    return model.eval()


def audit():
    import torch
    config, resolve = configuration()
    path = resolve(config.get('deep_models', 'artifacts/deep_learning/models'))/'best_resnet18.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not {'state_dict', 'config', 'validation'}.issubset(checkpoint):
        raise ValueError('지원하는 ResNet18 체크포인트 형식이 아닙니다.')
    # Retain model settings, omit the originating machine's private absolute paths.
    metadata = {key: value for key, value in checkpoint['config'].items()
                if key not in {'data_dir', 'pretrained_source'}}
    result = dict(weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  epoch=checkpoint.get('epoch'), config=metadata,
                  validation=checkpoint['validation'],
                  tensor_shapes={key: list(value.shape) for key, value in checkpoint['state_dict'].items()
                                 if 'head' in key or key == 'backbone.conv1.weight'})
    output = ROOT/'artifacts/deep_learning/checkpoint.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # Validate only inference entry points. prepare/finetune are never invoked.
    import joblib
    import numpy as np
    from PIL import Image
    module = native_module()
    torch.set_num_threads(4)
    model_dir = path.parent
    network = loaded_network(module, model_dir)
    buffer = io.BytesIO()
    Image.fromarray(np.arange(256*1600, dtype=np.uint32).reshape(256, 1600).astype(np.uint8)).save(buffer, format='PNG')
    x = input_tensor(buffer.getvalue(), module)
    with torch.inference_mode():
        if not torch.isfinite(network(x)[0]).all():
            raise ValueError('신경망 출력이 유효하지 않습니다.')
    details = resolve(config['deep_training_details'])
    text = details.read_text(encoding='utf-8')
    policies = {}
    for policy in ['RF_same_FP_budget', 'recall95', 'RF_FP_budget75pct']:
        match = re.search(r'^[ \t]*[ABC]\.\s*'+re.escape(policy)+r'[^\n]*\n.*?threshold\s*=\s*([0-9.]+)', text, re.M | re.S)
        if not match:
            raise ValueError('학습 설명에 원래 TTA 임계값이 없습니다: '+policy)
        policies[policy] = float(match.group(1))
    from deep_reference import load_reference
    reference = load_reference(resolve(config['deep_notebook']))
    frozen_threshold = None
    for text_output in reference['texts']:
        match = re.search(r"사전학습 특징 기준선:.*?'threshold':\s*([0-9.]+)", text_output)
        if match:
            frozen_threshold = float(match.group(1))
    if frozen_threshold is None:
        raise ValueError('고정 CNN의 원래 검증 임계값이 없습니다.')
    loaded_network(module, model_dir, frozen=True)
    classifier = joblib.load(model_dir/'frozen_cnn_logistic.joblib')
    if classifier.n_features_in_ != 1024:
        raise ValueError('고정 CNN 특징 차원이 일치하지 않습니다.')
    bundle = joblib.load(model_dir/'baseline_random_forest.joblib')
    if bundle['model'].n_features_in_ != 39:
        raise ValueError('기준선 RF 특징 차원이 일치하지 않습니다.')
    files = [resolve(config['deep_source']), resolve(config['deep_source']).parent/'test1_deep_metrics.py', details,
             model_dir/'best_resnet18.pt', model_dir/'initialized_resnet18.pt',
             model_dir/'frozen_cnn_logistic.joblib', model_dir/'baseline_random_forest.joblib']
    signature = {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else Path('..')/p.relative_to(ROOT.parent)): digest(p) for p in files}
    runtime = dict(files=signature, input_size=list(module.SIZE), source_verified=True,
        source_sha256=digest(resolve(config['deep_source'])),
        models={'resnet18': {'checkpoint': checkpoint['validation']['threshold']},
                'resnet18_tta': policies, 'frozen_cnn': {'RF_same_FP_budget': frozen_threshold},
                'baseline_rf': {'checkpoint': float(bundle['threshold'])}},
        policy_source='provided training details (full precision), original checkpoint and notebook stream',
        device='cpu', localization_supported=False)
    (output.parent/'runtime.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding='utf-8')


def predict(args):
    import torch
    import joblib
    import numpy as np
    config, resolve = configuration()
    runtime = json.loads((ROOT/'artifacts/deep_learning/runtime.json').read_text(encoding='utf-8'))
    for relative, expected in runtime['files'].items():
        if digest(ROOT/relative) != expected:
            raise ValueError('원본 코드 또는 가중치가 확인 당시와 다릅니다. 모델 상태를 다시 확인하세요.')
    if args.model not in runtime['models'] or args.policy not in runtime['models'][args.model]:
        raise ValueError('선택한 모델에 저장된 판정 기준이 없습니다.')
    raw = Path(args.image).read_bytes()
    if len(raw) > 30*1024*1024:
        raise ValueError('파일은 30MB 이하여야 합니다.')
    module = native_module()
    model_dir = resolve(config['deep_models'])
    start = perf_counter()
    torch.set_num_threads(4)
    if args.model == 'baseline_rf':
        from native_features import extract_features
        bundle = joblib.load(model_dir/'baseline_random_forest.joblib')
        model = bundle['model']
        score = float(model.predict_proba(extract_features(io.BytesIO(raw))[None])[0, list(model.classes_).index(1)])
        weights = model_dir/'baseline_random_forest.joblib'
    else:
        x = input_tensor(raw, module)
        frozen = args.model == 'frozen_cnn'
        network = loaded_network(module, model_dir, frozen)
        with torch.inference_mode():
            if frozen:
                classifier = joblib.load(model_dir/'frozen_cnn_logistic.joblib')
                score = float(classifier.predict_proba(network.embedding(x).numpy())[0, list(classifier.classes_).index(1)])
            else:
                value = network(x)[0].sigmoid()
                if args.model == 'resnet18_tta':
                    value = (value + network(torch.flip(x, [-1]))[0].sigmoid())*.5
                score = float(value.item())
        weights = model_dir/('initialized_resnet18.pt' if frozen else 'best_resnet18.pt')
    if not np.isfinite(score):
        raise ValueError('모델 점수가 유효하지 않습니다.')
    threshold = runtime['models'][args.model][args.policy]
    result = dict(model_id=args.model, run_id='provided_deep_epoch5', decision='defect' if score >= threshold else 'normal',
                  score=score, threshold=threshold, policy=args.policy, classification_unit='image',
                  elapsed_seconds=perf_counter()-start, weights_sha256=digest(weights), source_sha256=digest(resolve(config['deep_source'])),
                  class_names=[], detections=[], masks=None, overlay_rgb=None, coordinate_frame='original_image', device='cpu')
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit', 'predict'])
    parser.add_argument('--image')
    parser.add_argument('--output')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--policy', default='checkpoint')
    args = parser.parse_args()
    audit() if args.command == 'audit' else predict(args)


if __name__ == '__main__':
    main()
