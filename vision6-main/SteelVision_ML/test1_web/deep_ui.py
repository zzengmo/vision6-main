"""Deep learning sections within the existing six-menu Streamlit dashboard."""
import io

import pandas as pd
from PIL import Image
import streamlit as st

import deep_adapter as deep
from deep_reference import table_with


MODEL_LABELS = {
    'ResNet18 미세조정 · 딥러닝': 'resnet18',
    'ResNet18 + 좌우반전 평균 (TTA) · 노트북 선택': 'resnet18_tta',
    '고정 CNN 특징 + Logistic Regression': 'frozen_cnn',
    '딥러닝 실험 RF 기준선 · 제공 가중치': 'baseline_rf',
}
POLICIES = {
    '체크포인트 원래 기준': 'checkpoint',
    '누락 우선 · RF와 같은 검증 오탐 상한': 'RF_same_FP_budget',
    '재검사 부담 감소 · 검증 Recall 95%': 'recall95',
    '검증 오탐 상한 75%': 'RF_FP_budget75pct',
}


def clear_quality_result():
    for key in ['rf_result', 'rf_key', 'deep_result', 'deep_key']:
        st.session_state.pop(key, None)


def render_overview():
    with st.expander('추가된 딥러닝 모델 · ResNet18', expanded=True):
        status = deep.status()
        st.write('전체 이미지 → 회색조 800×128 → ResNet18 평균·최대 풀링 1,024개 특징 → 정상/불량 점수')
        st.caption('노트북 최종 선택은 미세조정 ResNet18의 원본·좌우반전 점수 평균(TTA)입니다. '
                   '분류 모델이며 결함 박스·픽셀 마스크를 출력하지 않습니다. 위치는 U-Net에서 별도로 확인합니다.')
        if status['checkpoint']:
            checkpoint = status['checkpoint']
            st.write(f"제공 가중치: best_resnet18.pt · 검증 선택 epoch {checkpoint['epoch']} · "
                     f"단일 추론의 저장 임계값 {checkpoint['validation']['threshold']:.9f}")
            st.caption('이 임계값은 TTA 운영점과 다릅니다. 모델·운영점별 원래 기준을 구분합니다.')
        else:
            st.info('일치하는 체크포인트 메타데이터가 없습니다. 자동 품질 검수에서 모델 상태를 확인하세요.')
        st.dataframe(pd.DataFrame(status['inventory']).rename(
            columns={'filename': '파일', 'exists': '파일 존재', 'bytes': '크기 (바이트)'}), hide_index=True)
        st.caption('initialized_resnet18.pt는 고정 CNN 특징의 기반 가중치입니다. 단독 검수 모델로 제공하지 않습니다.')
        if not status['source_exists']:
            st.warning('원본 test1_deep_learning.py가 없어 실제 딥러닝 추론은 준비 중입니다. '
                       '노트북 저장 성능은 모델 성능 비교의 딥러닝 탭에서 확인할 수 있습니다.')


def render_inspection(model_id, reference_mode=False):
    st.caption('이미지 단위 정상/불량 판별입니다. 보조 Class 1~4 학습 헤드를 결함 종류 판정이나 위치 예측으로 표시하지 않습니다.')
    st.number_input('최소 불량 패치 수', min_value=1, value=1, disabled=True,
                    help='전체 이미지 판별 모델이므로 불량 패치를 집계하지 않습니다.')
    status = deep.status()
    available_policies = status.get('model_policies', {}).get(model_id, {})
    policy_labels = [label for label, policy in POLICIES.items() if policy in available_policies]
    if not policy_labels:
        policy_labels = ['체크포인트 원래 기준'] if model_id in {'resnet18', 'baseline_rf'} else ['누락 우선 · RF와 같은 검증 오탐 상한']
    policy_label = st.selectbox('판정 기준', policy_labels, key='deep_policy', on_change=clear_quality_result)
    policy = POLICIES[policy_label]
    st.caption('운영점은 검증에서 고정한 기준입니다. 검증 Recall 95%는 새 업로드나 실제 생산 데이터의 성능 보장이 아닙니다.')
    if reference_mode:
        st.info('실제 업로드 검수는 사이드바의 현재 PC 재현 결과를 선택하세요.')
        return
    if st.button('모델 상태 새로 확인'):
        try:
            with st.spinner('가중치 구조와 저장 임계값 확인 중…'):
                status = deep.audit_checkpoint()
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))
    uploaded = st.file_uploader('딥러닝으로 검수할 이미지', type=['jpg', 'jpeg', 'png'], key='deep_upload',
                                on_change=clear_quality_result)
    ready = status['weights_match'] and status['runtime_ready'] and status['worker_exists']
    if not status['weights_match']:
        st.warning('ResNet18 가중치가 없거나 확인 당시와 다릅니다. 모델 상태를 다시 확인하세요.')
    if not status['source_exists']:
        st.warning('원본 test1_deep_learning.py의 전처리·forward 연결이 필요합니다. 실제 추론은 아직 사용할 수 없습니다.')
    elif not status['runtime_ready']:
        st.warning('원본 모듈은 발견했지만 추론 어댑터 검증 전입니다. 모델 등록·검증 후 사용할 수 있습니다.')
    if not status['worker_exists']:
        st.warning('딥러닝 실행 Python을 settings.json의 worker_python에서 확인하세요.')
    raw = uploaded.getvalue() if uploaded else None
    key = deep.result_key(raw, model_id, policy) if raw else None
    if st.button('딥러닝 검수 실행', type='primary', disabled=not (raw and ready)):
        clear_quality_result()
        try:
            with st.spinner('선택한 모델로 품질 검수 중…'):
                result = deep.inspect(raw, model_id, policy)
            st.session_state['deep_result'], st.session_state['deep_key'] = result, key
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(str(exc))
    if not key or st.session_state.get('deep_key') != key:
        if raw:
            try:
                st.image(deep.decode_input(raw), caption='업로드 원본 · 아직 검사 결과가 없습니다.', width='stretch')
            except ValueError as exc:
                st.error(str(exc))
        return
    result = st.session_state['deep_result']
    columns = st.columns(3)
    columns[0].metric('품질 판정', '불량 의심' if result['decision'] == 'defect' else '정상')
    columns[1].metric('불량 점수', f"{result['score']:.6f}")
    columns[2].metric('처리 시간', f"{result['elapsed_seconds']:.3f}초")
    st.caption(f"{result['model_id']} · 임계값 {result['threshold']:.9f} · 이미지 단위 · 점수는 보정된 확률이 아닙니다.")
    original = deep.decode_input(raw)
    st.image(original, caption='검사 원본 · 이미지 분류 모델은 박스나 마스크를 생성하지 않습니다.', width='stretch')
    frame = pd.DataFrame([{k: result.get(k) for k in
                          ['model_id', 'run_id', 'decision', 'score', 'threshold', 'policy',
                           'weights_sha256', 'elapsed_seconds', 'classification_unit']}])
    st.dataframe(frame, hide_index=True, width='stretch')
    st.download_button('딥러닝 판정 CSV 다운로드', frame.to_csv(index=False).encode('utf-8-sig'),
                       'deep_quality.csv', 'text/csv')
    buffer = io.BytesIO()
    original.save(buffer, format='PNG')
    st.download_button('딥러닝 검사 원본 PNG 다운로드', buffer.getvalue(), 'deep_input.png', 'image/png')
    st.download_button('마스크 PNG · 분류 모델 미지원', b'', 'unavailable.png', disabled=True)
    st.info('정밀 위치는 U-Net 메뉴에서 별도로 분석하세요. 서로 다른 모델이므로 판정이 다를 수 있습니다.')


def render_performance():
    st.subheader('딥러닝 실험 · 노트북 저장 결과')
    st.info('팀원이 실행해 노트북에 저장한 후속 비교입니다. 현재 PC에서 다시 평가한 수치가 아닙니다. '
            '기존에 관찰한 홀드아웃 2,514장을 사용했으며 새로운 맹검 평가가 아닙니다.')
    reference = deep.reference()
    status = deep.status()
    if not reference:
        st.info('딥러닝 노트북 저장 결과가 없습니다. settings.json의 deep_notebook 경로를 확인하세요.')
        return
    st.caption('원본 이미지 분할: 학습 7,540 / 검증 2,514 / 기존 홀드아웃 2,514장. '
               '이미지 경보 Recall·Precision·F1이며 픽셀 위치 성능이 아닙니다.')
    summary = table_with(reference, '운영 방식', '누락 FN', '오탐 FP')
    if summary:
        frame = pd.DataFrame(summary)
        st.dataframe(frame, hide_index=True, width='stretch')
        st.bar_chart(frame.set_index('운영 방식')[['누락 FN', '오탐 FP']])
        st.download_button('딥러닝 비교 CSV 다운로드', frame.to_csv(index=False).encode('utf-8-sig'),
                           'deep_notebook_comparison.csv', 'text/csv')
    st.warning('저장된 노트북 표에는 가중치 SHA-256 연결 정보가 없어, 제공 가중치와 같은 평가인지 확정할 수 없습니다. '
               '현재 가중치의 홀드아웃 재평가는 별도로 필요합니다.')
    if status['checkpoint']:
        checkpoint = status['checkpoint']
        with st.expander('현재 가중치에 포함된 검증 결과 · TTA 없는 단일 추론'):
            st.dataframe(pd.DataFrame([checkpoint['validation']]), hide_index=True, width='stretch')
            st.caption(f"best_resnet18.pt · epoch {checkpoint['epoch']} · SHA-256 {checkpoint['weights_sha256']}")
    validation = table_with(reference, 'candidate', 'split', 'threshold', 'fn', 'fp')
    holdout = table_with(reference, 'candidate', 'policy', 'split', 'fn', 'fp')
    with st.expander('검증 후보와 기존 홀드아웃 평가 구분'):
        st.write('검증 후보 선택')
        st.dataframe(pd.DataFrame(validation), hide_index=True, width='stretch')
        st.write('선택 이후 기존 홀드아웃 평가')
        st.dataframe(pd.DataFrame(holdout), hide_index=True, width='stretch')
    classes = table_with(reference, '클래스', '평가 양성 이미지', 'DL 경보 recall')
    with st.expander('클래스별 경보 Recall · 유형 분류 정확도 아님'):
        st.dataframe(pd.DataFrame(classes), hide_index=True, width='stretch')
        st.caption('각 유형의 정답 이미지에서 이진 불량 경보가 울린 비율입니다. Class 1~4 예측의 정확도가 아닙니다.')
    figure_names = ['학습·검증 곡선', '검증 후보별 FN·FP', '홀드아웃 후보별 FN·FP',
                    'RF·딥러닝 혼동행렬', '선택 모델 운영점', '클래스별 경보 Recall',
                    '검증 PR 곡선', 'RF 대비 판정 변화 사례', 'Grad-CAM 설명', '재검사 권고·추가 검출']
    with st.expander('노트북에 저장된 분석 그림'):
        for index, image in enumerate(reference['images']):
            caption = figure_names[index] if index < len(figure_names) else '저장된 분석 그림'
            st.image(image, caption=caption+' · 노트북 저장 결과', width='stretch')
        st.caption('Grad-CAM은 출력 기여 영역의 설명 그림이며 결함 마스크나 정확한 위치 예측이 아닙니다.')
    st.caption(f"참고 노트북 SHA-256: {reference['sha256']}")
