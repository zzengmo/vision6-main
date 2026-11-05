from pathlib import Path
import ast
import hashlib
import io
import json
import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from backend import ROOT, inspect_quality, rf_status, load_rf, validate_upload
from native_features import extract_features, mask_to_scored_boxes


def test_all_six_pages_for_current_and_reference():
    app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    pages = ['프로젝트 소개', '데이터 EDA', '이미지 전처리 체험', '자동 품질 검수',
             '정밀 결함 위치 (U-Net)', '모델 성능 비교']
    for experiment in ['현재 PC 재현 결과', '노트북 저장 결과 (참고)']:
        app.sidebar.selectbox[0].set_value(experiment).run(timeout=30)
        for page in pages:
            app.sidebar.radio[0].set_value(page).run(timeout=30)
            assert not app.exception, (experiment, page)


def test_feature_and_rf_prediction_match_direct_notebook_function():
    notebook = json.loads((ROOT/'reference/test1.ipynb').read_text(encoding='utf-8-sig'))
    module = ast.parse(''.join(notebook['cells'][12]['source']))
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_features')
    direct = {'Image': Image, 'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<notebook extract_features>', 'exec'), direct)
    image = np.random.default_rng(5).integers(0, 256, (256, 1600, 3), dtype=np.uint8)
    buffer = io.BytesIO(); Image.fromarray(image).save(buffer, format='JPEG')
    raw = buffer.getvalue()
    expected_features = direct['extract_features'](io.BytesIO(raw))
    assert np.array_equal(expected_features, extract_features(io.BytesIO(raw)))
    assert expected_features.shape == (39,)
    if rf_status():
        model = load_rf((ROOT/'artifacts/rf/model.joblib').stat().st_mtime_ns)
        score = model.predict_proba(expected_features[None, :])[0, list(model.classes_).index(1)]
        result = inspect_quality(raw)
        assert result['score'] == pytest.approx(score, abs=1e-12)
        assert result['defect'] == bool(score >= rf_status()['threshold'])


def test_upload_and_strict_native_localization_shape():
    with pytest.raises(ValueError):
        validate_upload(b'invalid')
    buffer = io.BytesIO(); Image.new('RGB', (800, 256)).save(buffer, format='PNG')
    assert validate_upload(buffer.getvalue()).shape == (256, 800, 3)
    with pytest.raises(ValueError, match='1600'):
        validate_upload(buffer.getvalue(), localization=True)


def test_top_two_boxes_follow_notebook_sum_ranking():
    scores = np.zeros((256, 1600), np.uint8)
    scores[10:20, 20:30] = 200
    scores[40:60, 80:110] = 180
    scores[100:105, 1000:1010] = 250
    boxes = mask_to_scored_boxes(scores > 0, scores, 'sum')[:2]
    assert len(boxes) == 2
    assert boxes[0][:4] == (80, 40, 30, 20)
    assert boxes[1][:4] == (20, 10, 10, 10)


def test_release_assets_survive_authorized_cleanup_unchanged():
    hashes = json.loads((ROOT/'artifacts/verification/release_assets.json').read_text(encoding='utf-8-sig'))
    assert len(hashes) == 20
    for filename, expected in hashes.items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected, filename
