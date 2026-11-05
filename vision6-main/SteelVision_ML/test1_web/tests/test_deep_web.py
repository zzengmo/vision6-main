import io
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import backend
import deep_adapter as deep
from deep_reference import table_with


def test_saved_deep_comparison_is_read_without_execution():
    reference = deep.reference()
    if reference is None:
        pytest.skip('Optional source notebook is not provided')
    rows = table_with(reference, '운영 방식', '누락 FN', '오탐 FP')
    assert [(row['누락 FN'], row['오탐 FP']) for row in rows] == [(59, 272), (4, 214), (57, 39), (5, 186)]
    assert len(table_with(reference, 'candidate', 'policy', 'split', 'fn', 'fp')) == 30
    assert len(reference['images']) == 10
    for raw in reference['images']:
        with Image.open(io.BytesIO(raw)) as image:
            assert image.format == 'PNG'
            image.verify()


def test_all_menus_and_deep_choices_work_without_optional_files(monkeypatch, tmp_path):
    monkeypatch.setattr(deep, 'paths', lambda: {
        'models': tmp_path/'models', 'notebook': tmp_path/'missing.ipynb', 'source': tmp_path/'missing.py'})
    monkeypatch.setattr(backend, 'rf_status', lambda: None)
    monkeypatch.setattr(backend, 'seg_status', lambda: None)
    app = AppTest.from_file(str(backend.ROOT/'app.py')).run(timeout=30)
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run(timeout=30)
        assert not app.exception, page
    app.sidebar.radio[0].set_value('자동 품질 검수').run(timeout=30)
    for selection in app.selectbox(key='quality_model').options[1:]:
        app.selectbox(key='quality_model').set_value(selection).run(timeout=30)
        assert not app.exception, selection
        assert next(button for button in app.button if button.label == '딥러닝 검수 실행').disabled
    assert not deep.status()['runtime_ready']


def test_results_invalidate_when_weights_or_source_change(monkeypatch, tmp_path):
    folder = tmp_path/'models'
    folder.mkdir()
    weights = folder/'best_resnet18.pt'
    weights.write_bytes(b'old-version')
    source = tmp_path/'source.py'
    monkeypatch.setattr(deep, 'paths', lambda: {
        'models': folder, 'notebook': tmp_path/'notebook.ipynb', 'source': source})
    first = deep.result_key(b'image-a', 'resnet18', 'checkpoint')
    assert first != deep.result_key(b'image-b', 'resnet18', 'checkpoint')
    assert first != deep.result_key(b'image-a', 'resnet18_tta', 'checkpoint')
    assert first != deep.result_key(b'image-a', 'resnet18', 'recall95')
    weights.write_bytes(b'new-weights-version')
    second = deep.result_key(b'image-a', 'resnet18', 'checkpoint')
    assert first != second
    source.write_text('# source supplied', encoding='utf-8')
    assert second != deep.result_key(b'image-a', 'resnet18', 'checkpoint')


def test_upload_rejects_damage_and_preserves_exif_orientation():
    with pytest.raises(ValueError, match='읽을 수 없습니다'):
        deep.decode_input(b'not-a-png')
    image = Image.new('RGB', (20, 10), (200, 10, 30))
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', exif=exif)
    decoded = deep.decode_input(buffer.getvalue())
    assert decoded.mode == 'RGB'
    assert decoded.size == (10, 20)
