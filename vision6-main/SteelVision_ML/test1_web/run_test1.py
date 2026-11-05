"""Explicit preparation/training commands; opening the web never trains models."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'.cache/ultralytics'))


def prepare_rf():
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_recall_curve, recall_score, precision_score, roc_auc_score, average_precision_score, confusion_matrix
    from backend import settings, resolve, SourceImages, write_json
    from native_features import extract_features, FEATURE_NAMES
    config = settings()
    csv = resolve(config['csv'])
    labels = pd.read_csv(csv)
    images = SourceImages(resolve(config['images']))
    out = ROOT/'artifacts/rf'
    out.mkdir(parents=True, exist_ok=True)
    try:
        manifest = pd.DataFrame({'ImageId': sorted(images.index)})
        if not set(labels.ImageId).issubset(set(manifest.ImageId)):
            raise ValueError('CSV가 가리키는 이미지가 원본에 없습니다.')
        manifest['has_defect'] = manifest.ImageId.isin(set(labels.ImageId)).astype(int)
        train_idx, other_idx = train_test_split(manifest.index.to_numpy(), test_size=.4,
                                               stratify=manifest.has_defect, random_state=42)
        val_idx, test_idx = train_test_split(other_idx, test_size=.5,
                                            stratify=manifest.loc[other_idx, 'has_defect'], random_state=42)
        manifest['split'] = 'train'
        manifest.loc[val_idx, 'split'] = 'validation'
        manifest.loc[test_idx, 'split'] = 'test'
        fingerprint = hashlib.sha256(csv.read_bytes()+(ROOT/'native_features.py').read_bytes()).hexdigest()
        cache = out/'features.npz'
        X = None
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved['fingerprint']) == fingerprint and np.array_equal(saved['image_ids'], manifest.ImageId.to_numpy()):
                    X = saved['features']
        if X is None:
            rows = []
            for index, image_id in enumerate(manifest.ImageId, 1):
                rows.append(extract_features(io.BytesIO(images.read(image_id))))
                if index % 2000 == 0:
                    print(f'RF features: {index}/{len(manifest)}', flush=True)
            X = np.vstack(rows)
            np.savez_compressed(cache, features=X, image_ids=manifest.ImageId.to_numpy(dtype=str), fingerprint=fingerprint)
        masks = {s: manifest.split.eq(s).to_numpy() for s in ['train', 'validation', 'test']}
        y = manifest.has_defect.to_numpy()
        model = RandomForestClassifier(n_estimators=180, min_samples_leaf=5, max_features='sqrt',
                                        class_weight='balanced_subsample', n_jobs=-1, random_state=42)
        model.fit(X[masks['train']], y[masks['train']])
        val_score = model.predict_proba(X[masks['validation']])[:, 1]
        _, recall, candidates = precision_recall_curve(y[masks['validation']], val_score)
        eligible = np.flatnonzero(recall[:-1] >= .95)
        threshold = float(candidates[eligible[-1]])
        scores = model.predict_proba(X[masks['test']])[:, 1]
        truth = y[masks['test']]
        predicted = scores >= threshold
        tn, fp, fn, tp = map(int, confusion_matrix(truth, predicted, labels=[0, 1]).ravel())
        model_path = out/'model.joblib'
        joblib.dump(model, model_path)
        manifest.to_csv(out/'manifest.csv', index=False)
        predictions = manifest.loc[masks['test'], ['ImageId']].copy()
        predictions['truth'], predictions['score'], predictions['predicted'] = truth, scores, predicted
        predictions.to_csv(out/'test_predictions.csv', index=False)
        metrics = dict(recall=float(recall_score(truth, predicted)), precision=float(precision_score(truth, predicted)),
                       accuracy=float((truth == predicted).mean()), roc_auc=float(roc_auc_score(truth, scores)),
                       ap=float(average_precision_score(truth, scores)), tp=tp, fn=fn, fp=fp, tn=tn)
        metadata = dict(source='current_pc_reproduction', model='RandomForestClassifier', seed=42,
                        input_size=[200, 32], feature_names=FEATURE_NAMES, feature_count=len(FEATURE_NAMES),
                        threshold=threshold, sample_count=len(manifest),
                        splits={s: int(m.sum()) for s, m in masks.items()},
                        normal_images=int((y == 0).sum()), defect_images=int(y.sum()),
                        class_rows={str(int(c)): int(n) for c, n in labels.ClassId.value_counts().items()},
                        test=metrics, sklearn_version=sklearn.__version__, fingerprint=fingerprint,
                        weights_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        task='image_binary_classification', validation_recall_target=.95)
        write_json(out/'metadata.json', metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    finally:
        images.close()


def train_unet():
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import segmentation_compare as seg
    from backend import settings, resolve, SourceImages, write_json
    cfg = settings()
    manifest_path = ROOT/'artifacts/rf/manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError('Run prepare-rf first')
    manifest = pd.read_csv(manifest_path)
    labels = pd.read_csv(resolve(cfg['csv']))
    seg.EPOCHS = int(cfg['epochs'])
    if seg.EPOCHS < 1:
        raise ValueError('epochs must be positive')
    torch.set_num_threads(4)
    segment_root = ROOT/'artifacts/segmentation'
    existing_checkpoint = segment_root/'models/unet_best.pt'
    existing_history = segment_root/'results/unet_history.csv'
    if existing_checkpoint.exists() and (not existing_history.exists()
            or len(pd.read_csv(existing_history)) != seg.EPOCHS):
        raise ValueError('기존 U-Net 가중치의 학습 횟수가 설정과 다르거나 이력이 없습니다. 기존 결과를 별도 보관한 뒤 새 실험으로 준비하세요.')
    source = ROOT/'artifacts/segmentation_sources'
    (source/'train_images').mkdir(parents=True, exist_ok=True)
    sampled = seg._sample_manifest(manifest)
    images = SourceImages(resolve(cfg['images']))
    try:
        for image_id in sampled.ImageId:
            target = source/'train_images'/image_id
            if not target.exists():
                target.write_bytes(images.read(image_id))
    finally:
        images.close()
    index = seg.prepare(segment_root, source, manifest, labels)
    dataset_config = segment_root/'dataset/config.json'
    if dataset_config.exists():
        saved = json.loads(dataset_config.read_text(encoding='utf-8'))
        saved.update(epochs=seg.EPOCHS, models={'U-Net': 'Small grayscale U-Net, base=16', 'YOLO': 'not trained in this web'})
        write_json(dataset_config, saved)
    print(f'Test1 native U-Net: epochs={seg.EPOCHS}, device={seg.DEVICE}', flush=True)
    seg.train_unet(segment_root, index)
    metrics = {}
    for split in ['validation', 'test']:
        scores, stems = seg.predict_patch_scores(segment_root, index, 'unet', split)
        rows = index.loc[index.split.eq(split)].reset_index(drop=True)
        assert np.array_equal(stems, rows.stem.to_numpy())
        truth = seg._truth_masks(segment_root, index, split)
        if split == 'validation':
            candidates = [(c, seg._metrics(scores, truth, rows, c)) for c in range(10, 241, 10)]
            feasible = [(c, m) for c, m in candidates if m['recall'] >= .95]
            cutoff, metrics[split] = max(feasible, key=lambda p: (p[1]['dice'], -p[1]['fp'])) if feasible else max(candidates, key=lambda p: (p[1]['recall'], p[1]['dice']))
        else:
            metrics[split] = seg._metrics(scores, truth, rows, cutoff)
    # Exactly the notebook's top-two ranking and optional candidate expansion.
    ranking = seg.compare_box_selection(segment_root, index, labels, 'unet', cutoff/255, top_k=2)
    rule = ranking.loc[ranking.split.eq('validation')].sort_values('box_dice', ascending=False).iloc[0]['rule']
    expansion = seg.compare_class3_box_expansion(segment_root, index, labels, 'unet', cutoff/255,
                                               top_k=2, weak_area_trigger=2000, low_cutoff_byte=60, rank_rule=rule)
    val = expansion.loc[expansion.split.eq('validation')].set_index('policy')
    adaptive = bool(val.loc['adaptive', 'class3_coverage'] > val.loc['baseline', 'class3_coverage']
                    and val.loc['adaptive', 'box_dice'] >= val.loc['baseline', 'box_dice']
                    and val.loc['adaptive', 'normal_false_alarm_rate'] <= val.loc['baseline', 'normal_false_alarm_rate'])
    results = segment_root/'results'
    ranking.to_csv(results/'box_ranking.csv', index=False)
    expansion.to_csv(results/'box_expansion.csv', index=False)
    checkpoint = segment_root/'models/unet_best.pt'
    metadata = dict(source='current_pc_reproduction', model='Small U-Net', task='binary_segmentation',
                    device=seg.DEVICE, epochs=seg.EPOCHS, input_patch=[320, 192], source_patch=[400, 256],
                    source_size=[1600, 256], splits={s: int(g.ImageId.nunique()) for s, g in index.groupby('split')},
                    cutoff_byte=int(cutoff), threshold=cutoff/255, top_boxes=2,
                    rank_rule=str(rule), adaptive_boxes=adaptive, weak_area_trigger=2000, low_box_cutoff=60,
                    validation=metrics['validation'], test=metrics['test'],
                    meets_validation_recall_target=metrics['validation']['recall'] >= .95,
                    pixel_metric_unit='resized_320x192_patch_pixels', image_metric_unit='original_image',
                    weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    torch_version=torch.__version__,
                    note='New 5-epoch CUDA/CPU reproduction, not the notebook saved Mac result. '
                         'Binary defect/background only. Box expansion does not alter the original segmentation mask.')
    write_json(segment_root/'web_metadata.json', metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def predict_unet(image_path, output):
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    import segmentation_compare as seg
    from backend import read_json, write_json
    from native_features import mask_to_scored_boxes
    torch.set_num_threads(4)
    meta = read_json(ROOT/'artifacts/segmentation/web_metadata.json')
    weights = ROOT/'artifacts/segmentation/models/unet_best.pt'
    if hashlib.sha256(weights.read_bytes()).hexdigest() != meta['weights_sha256']:
        raise ValueError('U-Net weights changed after validation')
    start = perf_counter()
    scores = seg.predict_original_scores(ROOT/'artifacts/segmentation', image_path, 'unet')
    mask = scores >= meta['cutoff_byte']
    candidates, used_cutoff, largest_seed = seg.box_candidate_mask(scores, meta['cutoff_byte'],
                        meta['weak_area_trigger'], meta['low_box_cutoff'], meta['adaptive_boxes'])
    boxes = mask_to_scored_boxes(candidates, scores, meta['rank_rule'])[:2]
    image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    overlay = image.copy()
    overlay[mask] = (overlay[mask]*.5+np.array([0, 0, 255])*.5).astype(np.uint8)
    records = []
    for number, (x, y, w, h, area, score) in enumerate(boxes, 1):
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (0, 0, 255), 2)
        cv2.putText(overlay, f'B{number}', (x+2, max(12, y+12)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        records.append(dict(box=number, x=x, y=y, width=w, height=h, component_pixels=area, mean_score=score))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, array in [('overlay.png', overlay), ('mask.png', mask.astype(np.uint8)*255),
                        ('box_candidates.png', candidates.astype(np.uint8)*255)]:
        cv2.imencode('.png', array)[1].tofile(str(output/name))
    pd.DataFrame(records, columns=['box','x','y','width','height','component_pixels','mean_score']).to_csv(output/'boxes.csv', index=False, encoding='utf-8-sig')
    result = dict(detected=bool(mask.any()), boxes=records, mask_pixels=int(mask.sum()),
                  seconds=perf_counter()-start, threshold=meta['threshold'], candidate_cutoff_byte=int(used_cutoff),
                  high_seed_pixels=int(largest_seed), rank_rule=meta['rank_rule'], adaptive_boxes=meta['adaptive_boxes'],
                  model='U-Net', coordinate_frame='original_image', defect_type='binary_defect_not_class_1_to_4')
    write_json(output/'summary.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare-rf', 'train-unet', 'predict-unet'])
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/manual')
    args = parser.parse_args()
    if args.action == 'prepare-rf':
        prepare_rf()
    elif args.action == 'train-unet':
        train_unet()
    else:
        if not args.image:
            parser.error('--image is required')
        predict_unet(args.image, args.output)


if __name__ == '__main__':
    main()
