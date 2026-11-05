"""Separate Streamlit front end for the team's test1.ipynb baseline."""
from pathlib import Path
import hashlib
import io
import json
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from backend import ROOT, ARTIFACTS, rf_status, seg_status, inspect_quality, inspect_location, validate_upload, reference_outputs
from preview.preprocessing import preprocess
from preview.edge_contour import auto_canny
from preview.roi import analyze_roi
from deep_ui import MODEL_LABELS, clear_quality_result, render_overview as deep_overview, render_inspection as deep_inspection, render_performance as deep_performance


def reference_gallery(cells):
    saved = reference_outputs()
    st.caption('다음 그림은 test1.ipynb에 저장된 팀원 실행 결과입니다. 현재 PC의 새 평가 결과와 구분합니다.')
    for cell in cells:
        for filename in saved['images'].get(str(cell), []):
            st.image(str(ROOT/'reference/assets'/filename), width='stretch')


def reference_text(cells):
    saved = reference_outputs()
    for cell in cells:
        text = saved['text'].get(str(cell))
        if text:
            st.code(text, language=None)


def csv_download(label, frame, name):
    st.download_button(label, frame.to_csv(index=False).encode('utf-8-sig'), name, 'text/csv')


def main():
    st.set_page_config(page_title='SteelVision ML', page_icon='🔎', layout='wide')
    st.title('SteelVision ML')
    st.caption('Random Forest·ResNet18 품질 검수와 U-Net 결함 위치 분석')
    experiment = st.sidebar.selectbox('실험 결과', ['현재 PC 재현 결과', '노트북 저장 결과 (참고)'])
    page = st.sidebar.radio('메뉴', ['프로젝트 소개', '데이터 EDA', '이미지 전처리 체험',
                                    '자동 품질 검수', '정밀 결함 위치 (U-Net)', '모델 성능 비교'])
    reference = experiment.startswith('노트북')
    rf, segmentation = rf_status(), seg_status()
    st.sidebar.caption('Streamlit 참고 화면 · 기본 포트 8503 · 최종 React 웹은 8502')
    if reference:
        st.sidebar.info('저장 결과 조회 모드입니다. 실제 업로드 추론은 현재 PC 재현 결과를 선택하세요.')
    if page == '프로젝트 소개':
        st.subheader('이미지 판별과 결함 위치를 각각 확인합니다')
        st.markdown('**정상/불량:** 전체 이미지 → 회색조 200×32 → 밝기·기울기·5구간 통계 39개 → Random Forest')
        st.markdown('**위치:** 원본 1600×256 → 400×256 패치 4개 → 320×192 → U-Net → 원본 좌표 복원 → 상위 박스 2개')
        st.info('위치 모델은 결함/배경 이진 분할입니다. Class 1~4의 결함 종류를 예측하지 않습니다. 두 개의 박스는 표시 제한이며 실제 결함 개수를 뜻하지 않습니다.')
        st.write('원본 이미지 ID 기준 학습 60% / 검증 20% / 테스트 20%를 사용합니다. RF와 위치 모델은 판정 목적과 평가 단위가 다릅니다.')
        if rf:
            st.write(f"RF: {rf['sample_count']:,}장 · 특징 {rf['feature_count']}개 · 임계값 {rf['threshold']:.4f}")
        else:
            st.info('RF 준비 전입니다. 별도 Prepare_Test1.cmd 또는 run_test1.py prepare-rf를 실행하세요.')
        if segmentation:
            counts = segmentation['splits']
            st.write(f"U-Net: {segmentation['epochs']}회 학습 · 학습 {counts['train']} / 검증 {counts['validation']} / 테스트 {counts['test']}장 · {segmentation['device']}")
        else:
            st.info('U-Net 가중치 준비 전입니다. 웹을 여는 것만으로 학습하지 않습니다.')
        with st.expander('노트북 저장 결과의 모델 선택'):
            reference_text([35])
        st.caption('노트북 설명의 5 epoch를 재현 설정으로 사용합니다. 별도 원본 Python 파일의 기본 12 epoch와 구분하며, 저장된 Mac 결과와 새 PC 측정값이 같다고 보장하지 않습니다.')
        deep_overview()
    elif page == '데이터 EDA':
        if rf and not reference:
            columns = st.columns(3)
            for column, label, value in zip(columns, ['원본 이미지', '정상 이미지', '불량 이미지'],
                                             [rf['sample_count'], rf['normal_images'], rf['defect_images']]):
                column.metric(label, f'{value:,}')
            st.subheader('클래스별 CSV 라벨 행 수')
            st.bar_chart(pd.DataFrame({'라벨 행 수': rf['class_rows']}))
            st.caption('한 이미지에 여러 클래스가 존재할 수 있습니다. 이 분포는 모델의 4종 분류 결과가 아닙니다.')
            manifest = pd.read_csv(ARTIFACTS/'rf/manifest.csv')
            table = manifest.groupby('split').has_defect.agg(이미지수='size', 불량='sum')
            table['정상'] = table['이미지수']-table['불량']
            st.dataframe(table, width='stretch')
        reference_gallery([4, 6, 8, 10])
    elif page == '이미지 전처리 체험':
        st.caption('아래 옵션은 화면 체험용입니다. RF와 U-Net의 원래 추론 전처리는 변경하지 않습니다.')
        uploaded = st.file_uploader('철강 이미지 업로드', type=['jpg', 'jpeg', 'png'])
        options = {key: st.sidebar.checkbox(label, True) for key, label in
                   [('clahe', 'CLAHE'), ('bilateral', 'Bilateral'), ('sharpen', 'Adaptive Sharpen'), ('opening', 'Opening')]}
        canny = st.sidebar.checkbox('Canny 표시', False)
        roi_grid = st.sidebar.checkbox('ROI Grid 표시', False)
        if uploaded:
            try:
                rgb = validate_upload(uploaded.getvalue())
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                processed, steps = preprocess(gray, options, return_steps=True)
                left, right = st.columns(2)
                left.image(rgb, caption='원본', width='stretch')
                right.image(processed, caption='전처리', width='stretch')
                st.metric('에지 밀도', f'{(auto_canny(processed)>0).mean():.3f}')
                st.bar_chart(pd.DataFrame({'원본': np.histogram(gray, 32, (0, 256))[0],
                                           '전처리': np.histogram(processed, 32, (0, 256))[0]}))
                with st.expander('단계별 결과'):
                    for label, image in steps.items():
                        st.image(image, caption=label, width='stretch')
                if canny:
                    st.image(auto_canny(processed), caption='Canny', width='stretch')
                if roi_grid:
                    roi = analyze_roi(processed)
                    st.image(roi['overlay'], channels='BGR', width='stretch')
                    st.dataframe(pd.DataFrame(roi['density']))
                st.download_button('전처리 이미지 다운로드', cv2.imencode('.png', processed)[1].tobytes(), 'preprocessed.png', 'image/png')
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    elif page == '자동 품질 검수':
        selection = st.selectbox('모델', ['Random Forest · Pillow 200×32 · 특징 39개', *MODEL_LABELS],
                                 key='quality_model', on_change=clear_quality_result)
        if selection in MODEL_LABELS:
            deep_inspection(MODEL_LABELS[selection], reference_mode=reference)
            return
        st.caption('test1의 이미지 단위 이진 판별입니다. 결함 종류나 위치는 이 RF 모델이 예측하지 않습니다.')
        st.number_input('최소 불량 패치 수', min_value=1, value=1, disabled=True,
                        help='이미지 전체를 판별하는 모델이므로 패치 개수로 집계하지 않습니다.')
        if reference:
            st.info('실제 검수는 사이드바에서 현재 PC 재현 결과를 선택하세요.')
            reference_text([16])
            return
        if rf is None:
            st.info('RF 모델을 준비한 후 사용할 수 있습니다.')
            return
        uploaded = st.file_uploader('검수할 이미지', type=['jpg', 'jpeg', 'png'])
        key = (hashlib.sha256(uploaded.getvalue()).hexdigest(), rf['weights_sha256'], rf['threshold']) if uploaded else None
        if uploaded and st.button('검수 실행', type='primary'):
            try:
                with st.spinner('이미지 검수 중…'):
                    result = inspect_quality(uploaded.getvalue())
                st.session_state['rf_result'], st.session_state['rf_key'] = result, key
            except (ValueError, OSError, RuntimeError) as exc:
                st.error(str(exc))
        if key and st.session_state.get('rf_key') == key:
            result = st.session_state['rf_result']
            columns = st.columns(3)
            columns[0].metric('품질 판정', '불량 의심' if result['defect'] else '정상')
            columns[1].metric('불량 점수', f"{result['score']:.4f}")
            columns[2].metric('처리 시간', f"{result['seconds']:.3f}초")
            st.caption(f"검증 선택 임계값 {result['threshold']:.4f} · 이미지 단위 · 점수는 보정된 확률이 아닙니다.")
            st.image(result['original'], caption='검사 원본 (RF는 위치 박스를 생성하지 않습니다)', width='stretch')
            frame = pd.DataFrame([dict(모델=result['model'], 불량=result['defect'], 점수=result['score'], 임계값=result['threshold'])])
            st.dataframe(frame, width='stretch')
            csv_download('판정 CSV 다운로드', frame, 'test1_quality.csv')
            buffer = io.BytesIO(); Image.fromarray(result['original']).save(buffer, format='PNG')
            st.download_button('검사 원본 PNG 다운로드', buffer.getvalue(), 'test1_input.png', 'image/png')
            st.info('결함 위치는 정밀 결함 위치 (U-Net) 메뉴에서 별도로 확인하세요.')
    elif page == '정밀 결함 위치 (U-Net)':
        st.subheader('U-Net 마스크와 상위 박스 2개')
        st.caption('1600×256 원본의 결함/배경을 예측합니다. 박스 후보 확장은 시각화용이며 원래 분할 마스크나 RF 판정을 바꾸지 않습니다.')
        if reference:
            reference_text([35, 40])
            reference_gallery([42])
            return
        if segmentation is None:
            st.info('U-Net 가중치를 준비한 후 사용할 수 있습니다. Prepare_Test1.cmd에서 새 폴더에만 학습합니다.')
            return
        metrics = segmentation['test']
        columns = st.columns(3)
        for column, label, value in zip(columns, ['테스트 Dice', '테스트 IoU', '테스트 이미지 리콜'],
                                         [metrics['dice'], metrics['iou'], metrics['recall']]):
            column.metric(label, f'{value:.1%}')
        st.caption(f"픽셀 지표: 320×192 축소 패치 기준 · 이미지 리콜: 원본 사진 기준 · 테스트 FN {metrics['fn']} / FP {metrics['fp']}")
        st.caption(f"고정 마스크 임계값 {segmentation['threshold']:.4f} · 박스 순위 {segmentation['rank_rule']} · 후보 확장 {'사용' if segmentation['adaptive_boxes'] else '미사용'}")
        if not segmentation['meets_validation_recall_target']:
            st.warning('이번 재현 모델은 검증 이미지 리콜 95% 목표에 미달했습니다. 결과는 실험용으로 확인하세요.')
        uploaded = st.file_uploader('결함 위치를 찾을 이미지 (1600×256 JPG/PNG)', type=['jpg', 'jpeg', 'png'])
        key = (hashlib.sha256(uploaded.getvalue()).hexdigest(), segmentation['weights_sha256'],
               segmentation['cutoff_byte'], segmentation['rank_rule'], segmentation['adaptive_boxes'],
               segmentation['weak_area_trigger'], segmentation['low_box_cutoff']) if uploaded else None
        if uploaded and st.button('결함 위치 찾기', type='primary'):
            try:
                with st.spinner('U-Net 위치 분석 중…'):
                    result = inspect_location(uploaded.getvalue())
                st.session_state['unet_result'], st.session_state['unet_key'] = result, key
            except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
                st.error(str(exc))
        if key and st.session_state.get('unet_key') == key:
            result = st.session_state['unet_result']
            left, right = st.columns(2)
            left.image(validate_upload(uploaded.getvalue()), caption='원본', width='stretch')
            right.image(result['overlay'], caption='U-Net 마스크와 최종 박스', width='stretch')
            columns = st.columns(3)
            columns[0].metric('표시 박스', len(result['boxes']))
            columns[1].metric('마스크 픽셀', f"{result['mask_pixels']:,}")
            columns[2].metric('전체 처리 시간', f"{result['wall_seconds']:.2f}초")
            if result['boxes']:
                st.dataframe(pd.DataFrame(result['boxes']), width='stretch')
            else:
                st.info('선택 기준에서 탐지 영역이 없습니다. 정상 보장이나 RF 판정의 대체 결과는 아닙니다.')
            st.caption('빨간 영역은 원래 임계값의 이진 예측 마스크입니다. 박스는 검증에서 선택한 후보 영역에서 최대 2개만 표시합니다. 점수는 구성 픽셀의 평균값이며 결함 종류 확률이 아닙니다.')
            for label, filename, mime in [('위치 표시 PNG 다운로드', 'overlay.png', 'image/png'),
                                           ('분할 마스크 PNG 다운로드', 'mask.png', 'image/png'),
                                           ('박스 후보 마스크 다운로드', 'box_candidates.png', 'image/png'),
                                           ('좌표 CSV 다운로드', 'boxes.csv', 'text/csv')]:
                st.download_button(label, result[filename], 'test1_'+filename, mime)
    else:
        current, saved, deep_tab = st.tabs(['현재 PC 측정', '노트북 저장 결과 (참고)', '딥러닝 비교 · 저장 결과'])
        with current:
            if rf:
                st.subheader('Random Forest · 이미지 단위 이진 판별')
                st.dataframe(pd.DataFrame([rf['test']]), width='stretch')
                st.caption(f"테스트 {rf['splits']['test']:,}장 · 분류 임계값은 검증에서 선택 · 실제 sklearn {rf['sklearn_version']}")
            else:
                st.info('현재 PC의 RF 평가가 없습니다.')
            if segmentation:
                st.subheader('U-Net · 이진 위치 분할')
                st.dataframe(pd.DataFrame([dict(분할=name, **values) for name, values in
                                           [('validation', segmentation['validation']), ('test', segmentation['test'])]]), width='stretch')
                st.caption('Dice/IoU는 축소 패치 픽셀 지표이며, RF 이미지 리콜과 같은 지표가 아닙니다. YOLO는 이 새 웹에서 학습하지 않았습니다.')
                with st.expander('상위 박스 선택·확장 평가'):
                    st.dataframe(pd.read_csv(ARTIFACTS/'segmentation/results/box_ranking.csv'), width='stretch')
                    st.dataframe(pd.read_csv(ARTIFACTS/'segmentation/results/box_expansion.csv'), width='stretch')
            else:
                st.info('현재 PC의 U-Net 평가가 없습니다.')
        with saved:
            st.info('팀원 Mac에서 저장한 출력입니다. 현재 PC의 가중치나 속도를 보장하는 값이 아닙니다.')
            reference_text([16, 24, 35])
            reference_gallery([25, 35])
        with deep_tab:
            deep_performance()


if __name__ == '__main__':
    main()
