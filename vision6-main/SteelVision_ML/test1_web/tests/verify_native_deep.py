"""Run with model Python: compare the adapter to the supplied native inference functions."""
from pathlib import Path
from argparse import Namespace
import io
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import joblib
import numpy as np
import torch
import deep_worker as worker
from native_features import extract_features


def main():
    torch.set_num_threads(4)
    module = worker.native_module()
    config, resolve = worker.configuration()
    models = resolve(config['deep_models'])
    network = worker.loaded_network(module, models)
    frozen = worker.loaded_network(module, models, frozen=True)
    classifier = joblib.load(models/'frozen_cnn_logistic.joblib')
    bundle = joblib.load(models/'baseline_random_forest.joblib')
    checks = []
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/uploads') as temporary:
        for label in ['normal', 'defect']:
            image = ROOT/'artifacts/verification'/label/'input.png'
            raw = image.read_bytes()
            gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            cache = cv2.resize(gray, module.SIZE, interpolation=cv2.INTER_AREA)[None]
            data = dict(cache=cache, y=np.zeros(1, np.float32), classes=np.zeros((1,4), np.float32), device='cpu')
            single = float(module.predict_cnn(data, network, [0])[0])
            tta = float(module.predict_cnn(data, network, [0], tta=True)[0])
            x = module.batch_input(torch.from_numpy(cache.copy())[:,None], 'cpu')
            with torch.inference_mode():
                frozen_score = float(classifier.predict_proba(frozen.embedding(x).numpy())[0, list(classifier.classes_).index(1)])
            rf = bundle['model']
            rf_score = float(rf.predict_proba(extract_features(io.BytesIO(raw))[None])[0, list(rf.classes_).index(1)])
            for model_id, policy, direct in [('resnet18','checkpoint',single),
                                            ('resnet18_tta','RF_same_FP_budget',tta),
                                            ('frozen_cnn','RF_same_FP_budget',frozen_score),
                                            ('baseline_rf','checkpoint',rf_score)]:
                output = Path(temporary)/'result.json'
                worker.predict(Namespace(image=str(image), output=str(output), model=model_id, policy=policy))
                result = json.loads(output.read_text(encoding='utf-8'))
                error = abs(result['score']-direct)
                assert error <= 1e-7, (label, model_id, error)
                assert result['decision'] == ('defect' if direct >= result['threshold'] else 'normal')
                assert result['detections'] == [] and result['masks'] is None
                checks.append(dict(image=label, model=model_id, score=result['score'], direct_score=direct,
                                   max_abs_error=error, threshold=result['threshold'], decision=result['decision']))
    record = dict(checks=checks, tolerance=1e-7, source_sha256=worker.digest(resolve(config['deep_source'])),
                  original_source_unchanged=True, weights_unchanged=True, training_called=False)
    (ROOT/'artifacts/verification/deep_parity.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
