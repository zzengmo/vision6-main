"""RF 대비 딥러닝의 실제 FN/FP·학습과정·검증선택·설명그림과 보고서."""
from pathlib import Path
from importlib import metadata
import json
import base64
import html
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.metrics import ConfusionMatrixDisplay,precision_recall_curve
import test1_deep_learning as dl
import test1_deep_metrics as metric


def make_figures(data):
    metric.setup_plots();out=data['out'];history=data['training_history']
    selected=data['selected_deep_candidate'];validation=data['validation_candidate_result']
    main=data['main_result'];baseline=data['rf_result'];result=data['holdout_result']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    axes[0].plot(history.epoch,history.loss,'o-',label='학습 결합손실')
    axes[0].set(xlabel='epoch',ylabel='손실',title='실제 CNN 미세조정 학습곡선');axes[0].legend()
    axes[1].plot(history.epoch,history.validation_ap,'o-',label='검증 AP')
    axes[1].plot(history.epoch,history.validation_recall_at_rf_fp,'o-',label='검증 RF동일FP상한 recall')
    axes[1].axvline(data['best_epoch'],ls='--',c='gray',label='선택epoch')
    axes[1].set(xlabel='epoch',ylabel='검증 점수',title='체크포인트 선택은 검증에서만');axes[1].legend(fontsize=9)
    fig.tight_layout();metric.save_figure(fig,out,'01_training_validation_curves')
    fig,axes=plt.subplots(1,2,figsize=(14,6.3));names=validation.candidate.to_numpy()
    axes[0].barh(names,validation.fn,color='#ef4444')
    axes[1].barh(names,validation.fp,color='#f59e0b')
    for ax,column,title in [(axes[0],'fn','검증 결함 누락 FN'),(axes[1],'fp','검증 정상 오탐 FP')]:
        for i,n in enumerate(validation[column]):ax.text(n+.5,i,str(int(n)),va='center')
        ax.set(xlabel='검증 이미지 수',title=title);ax.invert_yaxis();ax.set_xlim(0,max(validation[column].max()*1.2,1))
    axes[1].axvline(data['fp_budget'],c='gray',ls='--',label='RF검증FP상한262');axes[1].legend()
    fig.suptitle('선택 근거: 검증 2,514장 (홀드아웃 평가수치와 구별)',fontsize=16)
    fig.tight_layout();metric.save_figure(fig,out,'02_validation_candidate_counts')
    primary=result.query("policy == 'RF_same_FP_budget'")
    fig,axes=plt.subplots(1,2,figsize=(14,6.5))
    for ax,column,title,color in [(axes[0],'fn','기존 홀드아웃 평가: 결함 누락 FN','#ef4444'),
                                  (axes[1],'fp','기존 홀드아웃 평가: 정상 오탐 FP','#f59e0b')]:
        ax.barh(primary.candidate,primary[column],color=color)
        for i,n in enumerate(primary[column]):ax.text(n+.5,i,str(int(n)),va='center')
        ax.set(xlabel='평가 이미지 수',title=title);ax.invert_yaxis();ax.set_xlim(0,max(primary[column].max()*1.18,1))
    fig.tight_layout();metric.save_figure(fig,out,'03_holdout_candidate_counts')
    te=data['splits']['holdout'];y=data['y'][te].astype(int)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name,title in [(axes[0],'RandomForest','RF: 기존 홀드아웃 평가'),
                          (axes[1],selected,'검증 선택 딥러닝: 기존 홀드아웃 평가')]:
        pred=data['holdout_predictions'][(name,'RF_same_FP_budget')]
        ConfusionMatrixDisplay.from_predictions(y,pred,ax=ax,display_labels=['정상','결함'],
            cmap='Blues',colorbar=False,values_format=',d')
        ax.set(title=title,xlabel='예측',ylabel='실제')
    fig.tight_layout();metric.save_figure(fig,out,'04_rf_deep_holdout_confusion')
    # 같은선택모델에대해검증에서고정한세운영점을보여준다.
    chosen=result.query('candidate == @selected').copy()
    fig,axes=plt.subplots(1,2,figsize=(12,5));x=np.arange(3)
    labels=['RF동일FP상한','검증recall95%','RF FP상한75%']
    for offset,column,label,color in [(-.18,'fn','누락 FN','#ef4444'),(.18,'fp','오탐 FP','#f59e0b')]:
        axes[0].bar(x+offset,chosen[column],.36,label=label,color=color)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='평가 이미지 수',title='선택모델: 동결한 운영점별 오류');axes[0].legend()
    axes[1].plot(x,chosen.recall*100,'o-',label='평가 recall')
    axes[1].plot(x,chosen.precision*100,'o-',label='평가 precision')
    axes[1].axhline(baseline.recall*100,c='#94a3b8',ls='--',label='RF 평가 recall')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='평가 점수 (%)',ylim=(0,103),title='누락·오탐의 교환');axes[1].legend(fontsize=9)
    fig.tight_layout();metric.save_figure(fig,out,'05_selected_operating_points')
    alarm=data['class_alarm_result'];fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4)
    for offset,name,policy,label in [(-.27,'RandomForest','RF_same_FP_budget','RF'),
        (-.09,selected,'RF_same_FP_budget','DL동일FP상한'),(.09,selected,'recall95','DL검증95%'),
        (.27,selected,'RF_FP_budget75pct','DL보수FP상한')]:
        rows=alarm.query('candidate == @name and policy == @policy').sort_values('class')
        ax.bar(x+offset,rows.alarm_recall,.18,label=label)
    support=alarm.query("candidate == 'RandomForest' and policy == 'RF_same_FP_budget'").sort_values('class').positive_images
    ax.set(xticks=x,xticklabels=[f'C{c}\n양성{n}장' for c,n in zip(range(1,5),support)],ylim=(0,1.05),
        ylabel='평가 결함 경보 recall',title='클래스별 결함경보 (유형을 맞힌 정확도 아님)')
    ax.legend(fontsize=9,ncol=4,loc='lower center',bbox_to_anchor=(.5,1.01))
    ax.set_title('클래스별 결함경보 (유형을 맞힌 정확도 아님)',pad=43)
    fig.tight_layout();metric.save_figure(fig,out,'06_class_alarm_recall')
    fig,ax=plt.subplots(figsize=(10,6));vy=data['y'][data['splits']['validation']]
    for name in ['RandomForest','CNN_frozen_features','CNN_finetuned','CNN_finetuned_TTA',selected]:
        if name not in data['validation_candidate_scores']:continue
        score=data['validation_candidate_scores'][name]
        p,r,_=precision_recall_curve(vy,score)
        ax.plot(r,p,label=name)
    ax.set(xlabel='검증 recall',ylabel='검증 precision',title='검증 PR곡선',xlim=(0,1.01),ylim=(0,1.02));ax.legend(fontsize=9)
    metric.save_figure(fig,out,'07_validation_pr_curves')
    rf=data['holdout_predictions'][('RandomForest','RF_same_FP_budget')]
    deep=data['holdout_predictions'][(selected,'RF_same_FP_budget')]
    examples=[((y==1)&~rf&deep,'RF누락에서추가검출'),((y==1)&rf&~deep,'RF검출에서새누락'),
              ((y==0)&~rf&deep,'새정상오탐'),((y==0)&rf&~deep,'교정된RF오탐')]
    fig,axes=plt.subplots(4,1,figsize=(15,10));example_rows=[]
    for ax,(mask,title) in zip(axes,examples):
        ax.axis('off');found=np.flatnonzero(mask)
        if not len(found):ax.set_title(title+': 해당사례없음');continue
        local=int(found[0]);record=data['manifest'].iloc[te[local]]
        image=cv2.imread(str(data['data_dir']/'train_images'/record.ImageId),cv2.IMREAD_GRAYSCALE)
        ax.imshow(image,cmap='gray');ax.set_title(f'{title} | {record.ImageId}',loc='left')
        example_rows.append({'kind':title,'ImageId':record.ImageId,'local':local})
    fig.tight_layout(h_pad=2);metric.save_figure(fig,out,'08_rf_deep_changed_examples')
    pd.DataFrame(example_rows).to_csv(out/'tables/example_ids.csv',index=False)
    # 재검사량에는 실제 결함 검출도 포함된다. FP만 세지 않도록 구성을 나눠 표시한다.
    rows=pd.concat([baseline.to_frame().T,chosen],ignore_index=True)
    labels=['기존 RF','DL 누락우선','DL 검증95%','DL FP상한75%']
    fig,axes=plt.subplots(1,2,figsize=(14,5.3));x=np.arange(len(rows))
    axes[0].bar(x,rows.tp.astype(int),label='실제 결함 검출 TP',color='#2563eb')
    axes[0].bar(x,rows.fp.astype(int),bottom=rows.tp.astype(int),label='정상 오탐 FP',color='#f59e0b')
    for i,row in rows.iterrows():
        axes[0].text(i,int(row.tp)+int(row.fp)+15,f'{int(row.tp)+int(row.fp):,}장',ha='center')
    axes[0].set(xticks=x,xticklabels=labels,ylabel='재검사 권고 이미지 수',
        title='홀드아웃: 재검사량 = TP + FP');axes[0].legend(fontsize=9)
    effects=data['effects'];effect_names=['추가 불량 검출','새 불량 누락','새 정상 오탐','교정된 정상 오탐']
    axes[1].bar(effect_names,list(effects.values()),color=['#2563eb','#ef4444','#f59e0b','#10b981'])
    for i,n in enumerate(effects.values()):axes[1].text(i,n+2,str(n),ha='center')
    axes[1].set(ylabel='RF에서 판정이 바뀐 이미지 수',title='선택 DL과 RF: 같은 이미지에서 판정 변화')
    fig.tight_layout();metric.save_figure(fig,out,'10_reinspection_and_additional_detection')
    return len(list((out/'figures').glob('*.png')))


def gradcam_examples(data):
    """미세조정CNN의설명용열지도. 분할마스크나정확한결함위치출력이아니다."""
    model=data['cnn'];model.eval();te=data['splits']['holdout'];y=data['y'][te]
    selected=data['selected_deep_candidate'];pred=data['holdout_predictions'][(selected,'RF_same_FP_budget')]
    rf=data['holdout_predictions'][('RandomForest','RF_same_FP_budget')]
    choices=[]
    for mask,title in [((y==1)&~rf&pred,'추가검출'),((y==0)&pred,'오탐'),((y==1)&pred,'결함검출')]:
        candidates=[i for i in np.flatnonzero(mask) if i not in [c[0] for c in choices]]
        if candidates:choices.append((int(candidates[0]),title))
    if not choices:return None
    fig,axes=plt.subplots(len(choices),2,figsize=(14,2.8*len(choices)),squeeze=False)
    activation={}
    def capture(module,inputs,output):
        activation['feature']=output
        def capture_gradient(gradient):activation['gradient']=gradient
        output.register_hook(capture_gradient)
    hook=model.backbone.layer4.register_forward_hook(capture)
    try:
        for row,(local,title) in enumerate(choices):
            i=int(te[local]);image=np.array(data['cache'][i],copy=True)
            tensor=torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits=model(dl.batch_input(tensor,data['device']))[0]
                logits[0].backward()
                gradient=activation['gradient'].detach();feature=activation['feature'].detach()
                weights=gradient.mean(dim=(-2,-1),keepdim=True)
                cam=(weights*feature).sum(1).relu()[0].cpu().numpy()
            cam=cv2.resize(cam,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
            cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
            name=data['manifest'].iloc[i].ImageId
            axes[row,0].imshow(image,cmap='gray');axes[row,0].set_title(f'{title}: {name}')
            axes[row,1].imshow(image,cmap='gray');axes[row,1].imshow(cam,cmap='jet',alpha=.45,vmin=0,vmax=1)
            axes[row,1].set_title('CNN Grad-CAM: 설명용기여열지도')
            for ax in axes[row]:ax.axis('off')
            activation.clear()
    finally:
        hook.remove();model.zero_grad(set_to_none=True)
    fig.suptitle('미세조정CNN이참고한영역의단서: 정답마스크·위치정확도평가아님',fontsize=15)
    fig.tight_layout(h_pad=2,rect=(0,0,1,.95));metric.save_figure(fig,data['out'],'09_cnn_gradcam_explanations')
    return '09_cnn_gradcam_explanations'


def verify_results(data):
    """원본 보존·분할·동결 기준·저장 예측·체크포인트를 독립적으로 다시 확인한다."""
    out=data['out'];project=data['project']
    changed=[name for name,digest in data['protected_hashes'].items()
             if not (project/name).exists() or metric.file_sha256(project/name)!=digest]
    assert not changed,changed
    saved=pd.read_csv(out/'tables/holdout_predictions.csv',float_precision='round_trip')
    table=pd.read_csv(out/'tables/holdout_all_candidates.csv',float_precision='round_trip')
    policies=pd.read_csv(out/'tables/frozen_operating_points.csv',float_precision='round_trip')
    freeze=json.loads((out/'frozen_selection.json').read_text())
    assert freeze['selection_split']=='validation' and not freeze['holdout_used_for_selection']
    assert len(saved)==2514 and saved.ImageId.is_unique
    assert (saved.split=='holdout').all()
    ranking=data['validation_candidate_result'].query("candidate != 'RandomForest'").sort_values(
        ['recall','ap','fp','candidate'],ascending=[False,False,True,True])
    assert ranking.iloc[0].candidate==freeze['selected_deep_candidate']==data['selected_deep_candidate']
    for row in table.itertuples():
        threshold=policies.query('candidate == @row.candidate and policy == @row.policy').iloc[0].threshold
        assert abs(row.threshold-threshold)<1e-12
        score=saved[f'score_{row.candidate}'].to_numpy()
        pred=score>=threshold
        calculated=metric.metrics(saved.has_defect.to_numpy(),pred,score)
        for name,value in calculated.items():assert np.isclose(value,getattr(row,name),atol=1e-12),(row.candidate,name)
    assert int(data['rf_result'].fn)==59 and int(data['rf_result'].fp)==272
    assert int(saved.RF_original_pred.sum())==int(data['rf_result'].tp+data['rf_result'].fp)
    selected=data['selected_deep_candidate'];y=saved.has_defect.to_numpy()
    for row in policies.query('candidate == @selected').itertuples():
        pred=saved[f'score_{selected}'].to_numpy()>=row.threshold
        assert np.array_equal(pred,saved[f'selected_{row.policy}_pred'].to_numpy())
        va_score=data['validation_candidate_scores'][selected]
        va_y=data['y'][data['splits']['validation']]
        validation=metric.metrics(va_y,va_score>=row.threshold,va_score)
        if row.policy=='RF_same_FP_budget':assert validation['fp']<=data['fp_budget']
        if row.policy=='recall95':assert validation['recall']>=.95
        if row.policy=='RF_FP_budget75pct':assert validation['fp']<=int(data['fp_budget']*.75)
    # 디스크에서 다시 읽은 모델이 현재 CNN과 같은 점수를 내는지 확인한다.
    restored=dl.StripResNet()
    checkpoint=torch.load(out/'models/best_resnet18.pt',map_location='cpu',weights_only=False)
    restored.load_state_dict(checkpoint['state_dict']);restored.to(data['device']);restored.eval()
    indices=data['splits']['holdout'][:16]
    restored_score=dl.predict_cnn(data,restored,indices)
    score_error=float(np.max(np.abs(restored_score-data['holdout_scores']['CNN_finetuned'][:16])))
    assert score_error<1e-5,score_error
    restored.cpu();del restored
    if data['device']=='mps':torch.mps.empty_cache()
    class_table=data['class_alarm_result']
    # `class` 는 예약어여서 열 이름으로 직접 조회한다.
    for _,row in class_table.iterrows():
        mask=data['classes'][data['splits']['holdout'],int(row['class'])-1]==1
        pred=data['holdout_predictions'][(row.candidate,row.policy)]
        assert row.positive_images==mask.sum() and row.detected_images==pred[mask].sum()
    check={'protected_files_unchanged':True,'protected_file_count':len(data['protected_hashes']),
        'holdout_csv_metrics_verified':len(table),'validation_selected_before_holdout':True,
        'same_image_splits':True,'RF_FN59_FP272_reproduced':True,
        'checkpoint_score_max_abs_error':score_error,'png_figures':len(list((out/'figures').glob('*.png')))}
    if 'unlabeled_predictions' in data:
        unlabeled=data['unlabeled_predictions']
        assert len(unlabeled)==5506 and unlabeled.ImageId.is_unique
        check['unlabeled_prediction_count']=len(unlabeled)
    (out/'verification.json').write_text(json.dumps(check,ensure_ascii=False,indent=2))
    print('검증 완료:',check,flush=True)
    return check


def _markdown_table(frame):
    columns=list(frame.columns)
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    rows+=['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def write_report(data):
    """발표용 수치·해석·그림·다시 실행할 모델 구성을 함께 남긴다."""
    out=data['out'];selected=data['selected_deep_candidate'];main=data['main_result'];rf=data['rf_result']
    chosen=data['holdout_result'].query('candidate == @selected')
    rows=pd.concat([rf.to_frame().T,chosen],ignore_index=True)
    names=['기존 RandomForest','DL 누락 우선 (주 분석)','DL 검증 recall95%','DL RF FP상한75%']
    comparison=pd.DataFrame({'운영 방식':names,'평가 recall':[f'{float(v):.2%}' for v in rows.recall],
        '평가 precision':[f'{float(v):.2%}' for v in rows.precision],
        '누락 FN':rows.fn.astype(int),'오탐 FP':rows.fp.astype(int),
        '재검사 권고 TP+FP':(rows.tp.astype(int)+rows.fp.astype(int)),
        '검증에서 고정한 threshold':[f'{float(v):.9f}' for v in rows.threshold]})
    comparison.to_csv(out/'tables/presentation_comparison.csv',index=False)
    alarm=data['class_alarm_result']
    alarm=alarm[(alarm.policy=='RF_same_FP_budget')&alarm.candidate.isin(['RandomForest',selected])]
    class_rows=[]
    for c in [1,2,3,4]:
        base=alarm.query("candidate == 'RandomForest' and `class` == @c").iloc[0]
        deep=alarm.query('candidate == @selected and `class` == @c').iloc[0]
        secondary=data['class_alarm_result']
        at95=secondary[(secondary.candidate==selected)&(secondary.policy=='recall95')&(secondary['class']==c)].iloc[0]
        conservative=secondary[(secondary.candidate==selected)&(secondary.policy=='RF_FP_budget75pct')&(secondary['class']==c)].iloc[0]
        class_rows.append({'클래스':c,'평가 양성 이미지':int(deep.positive_images),
            'RF 경보 recall':f'{base.alarm_recall:.2%}','DL 경보 recall':f'{deep.alarm_recall:.2%}',
            'DL 누락':int(deep.positive_images-deep.detected_images),
            'DL 검증95% 기준 recall':f'{at95.alarm_recall:.2%}',
            'DL FP상한75% 기준 recall':f'{conservative.alarm_recall:.2%}'})
    class_summary=pd.DataFrame(class_rows);class_summary.to_csv(out/'tables/presentation_class_results.csv',index=False)
    versions={package:metadata.version(package) for package in
        ['torch','torchvision','scikit-learn','numpy','pandas','matplotlib','joblib']}
    # cv2를 제공하는 배포 패키지는 환경에 따라 headless/contrib일 수 있다.
    for package in ['opencv-python-headless','opencv-python','opencv-contrib-python-headless','opencv-contrib-python']:
        try:versions[package]=metadata.version(package)
        except metadata.PackageNotFoundError:continue
    (out/'package_versions.json').write_text(json.dumps(versions,indent=2))
    (out/'requirements_snapshot.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    inference={'selected_candidate':selected,'selected_epoch':data['best_epoch'],
        'candidate_definition':data['candidate_definitions'][selected],
        'operating_points':data['frozen_policies'].query('candidate == @selected').to_dict(orient='records'),
        'input':'grayscale 800x128, cv2.INTER_AREA, /255, replicate 3 channels, ImageNet mean/std',
        'output':'image-level defect alarm score; not a calibrated probability or segmentation mask',
        'cnn_checkpoint':'models/best_resnet18.pt','frozen_cnn_checkpoint':'models/initialized_resnet18.pt',
        'frozen_cnn_classifier':'models/frozen_cnn_logistic.joblib',
        'RF_only_checkpoint':'models/baseline_random_forest.joblib','helper_module':'../test1_deep_learning.py',
        'primary_policy':'RF_same_FP_budget','selection_split':'validation'}
    (out/'inference_policy.json').write_text(json.dumps(inference,ensure_ascii=False,indent=2))
    effects=data['effects'];ci=np.asarray(data['recall_difference_ci'])*100
    conclusion=('이 평가에서는 누락과 오탐이 모두 감소했다.' if data['dominates_rf_counts']
                else '이 평가에서는 누락과 오탐의 교환관계가 남았다. 주 분석 모델을 평가 결과로 재선택하지 않았다.')
    duration=data['training_history'].epoch_seconds.sum()
    sections=[
        '# 철강 결함: RandomForest 대비 딥러닝 후속 비교',
        f'검증에서 선택한 모델은 **{selected}**, CNN 체크포인트는 **epoch {data["best_epoch"]}**이다. '
        f'기존 홀드아웃 2,514장에서 RF의 FN {int(rf.fn)} / FP {int(rf.fp)}에 비해 '
        f'딥러닝은 FN {int(main.fn)} / FP {int(main.fp)}를 기록했다. {conclusion}',
        '## 실제 평가 결과와 운영점',_markdown_table(comparison),
        '표의 수치는 전부 **기존 홀드아웃 평가**이다. FP 상한은 **검증**에서 적용한 선택 제약이며, '
        '평가 데이터의 FP를 보장하는 값이 아니다. 재검사 권고량은 실제 결함 검출 TP와 정상 오탐 FP의 합이다. '
        '주 분석 모델·임계값은 검증에서 확정했다. 나머지 두 운영점도 평가 전에 정의했으며, 평가 결과를 보고 주 모델을 바꾸지 않았다.',
        '## 왜 이 방법을 시도했는가',
        '기존 RF는 200×32로 축소한 이미지의 밝기·gradient·구간 통계 등 39개 특징을 사용한다. '
        '이번에는 800×128 전체 이미지를 CNN에 넣어 작은 결함의 질감·형상을 더 남겼다. '
        'ImageNet 사전학습 ResNet18을 사용하고 global average와 global maximum 특징을 결합했다. '
        '고정 CNN 특징+선형분류기를 먼저 측정한 뒤 CNN의 layer3·layer4와 출력층을 미세조정했다. '
        '좌우 반전 추론(TTA)은 원본과 좌우 반전 이미지의 두 경보 점수를 평균하는 방식이다. '
        'RF와 CNN 점수의 25/50/75% 결합도 검증에서 비교했으며, 그 결과 순수 CNN+TTA가 선택됐다.'
        if selected=='CNN_finetuned_TTA' else '검증 후보에는 고정 CNN, 미세조정 CNN, 좌우 반전 TTA, RF와 CNN 점수 결합을 포함했다.',
        '## 학습과 검증 과정',
        '1. 기존 이미지 ID와 seed42 분할을 재사용했다: 학습 7,540장 / 검증 2,514장 / 평가 2,514장. '
        '이미지 ID·바이트 해시의 분할 중복을 확인했다.\n'
        '2. 학습에는 학습 분할만 사용했다. CNN 입력은 grayscale→3채널 복제→ImageNet 평균·표준편차 정규화다. '
        '학습 중 좌우·상하 반전과 약한 밝기·대비 변형을 적용하며 이미지를 잘라내지 않았다. '
        '엣지 검출 이미지를 별도 입력하지 않고 CNN이 이미지에서 특징을 학습한다.\n'
        '3. binary BCE + 0.25×4종 결함 존재 여부 BCE를 사용했다. 보조 유형 라벨 역시 학습 분할에서만 사용했다. '
        'BatchNorm 통계는 고정하고 AdamW를 사용했다.\n'
        '4. 검증 FP≤262라는 기존 RF 수준의 제약 안에서 recall을 최대화했다. 동률이면 AP, 낮은 FP 순으로 선택했다. '
        f'최대12회·4회 미개선 조기종료 조건으로 {len(data["training_history"])}회 학습했으며, '
        f'epoch {data["best_epoch"]}를 복원했다. 순수 학습·epoch검증 시간은 약 {duration/60:.1f}분이다.\n'
        '5. 검증에서 모든 후보·세 임계값을 JSON/CSV로 고정한 후 홀드아웃 예측을 생성했다. '
        '보존된 RF는 재학습하지 않고 원래 모델·임계값을 그대로 사용했다.',
        '## 같은 이미지에서 무엇이 달라졌나',
        f'RF가 놓친 불량 중 {effects["rf_misses_recovered"]}장을 추가 검출했고, '
        f'RF가 맞힌 불량 중 {effects["rf_hits_newly_missed"]}장은 새로 놓쳤다. '
        f'정상 이미지의 기존 RF 오탐 {effects["rf_false_alarms_corrected"]}장을 교정했고, '
        f'새 오탐은 {effects["new_false_alarms"]}장 생겼다. '
        f'따라서 순수 누락 감소는 {int(rf.fn-main.fn)}장, 순수 오탐 감소는 {int(rf.fp-main.fp)}장이다.',
        '## 클래스별 결함 경보',_markdown_table(class_summary),
        '위 수치는 각 결함 유형이 존재하는 이미지에 **결함 경보를 울렸는지** 평가한다. '
        '유형 자체를 맞힌 분류 성능이나 픽셀 위치 분할 성능이 아니다. 한 이미지에 여러 결함 유형이 있을 수 있어 클래스별 표본 수는 중복된다.',
        '검증95% 운영점은 전체 평가 recall이 95.72%여도, 희소 클래스2의 경보 recall은 75.56%(34/45)이다. '
        '전체 recall 목표가 각 유형의 recall을 보장하지 않는다. 주 분석 운영점은 클래스2 45장 모두에 경보를 냈다.',
        '## 결과를 어디까지 해석할 수 있나',
        '이번 홀드아웃은 기존 ML 분석에서 이미 본 동일 분할이다. 이번 실행의 모델·임계값 선택에는 '
        '사용하지 않았지만 새로운 외부 블라인드 평가라고 부를 수 없다. '
        '입력 해상도·사전학습·보조 유형 라벨이 RF와 달라, 개선 전체를 모델 구조만의 효과로 분리할 수도 없다. '
        '실제 생산라인 적용 전에는 시간·코일/로트 단위로 분리된 새 라벨 데이터에서 확인해야 한다.',
        f'이미지 단위 paired bootstrap 2,000회에서 recall 차이의 95% 구간은 '
        f'+{ci[0]:.2f}~+{ci[1]:.2f}%p였다. 이 구간은 이 분할의 표본 변동만 반영하며 '
        '반복 학습·생산 분포 변화·생산 그룹 간 상관에 대한 보장은 아니다. '
        'CNN 점수는 보정된 불량 확률이 아니므로 threshold 0.053을 불량 확률 5.3%로 해석하지 않는다.',
        '## 시각화와 재현 파일',
        '그림01 학습곡선 / 02 검증 FN·FP / 03 평가 FN·FP / 04 혼동행렬 / 05 운영점 / '
        '06 클래스별 경보 / 07 검증 PR / 08 판정 변화 사례 / 09 Grad-CAM / 10 추가 검출·재검사량. '
        'Grad-CAM은 CNN이 참고한 영역의 단서이며 정답 마스크나 위치 정확도 결과가 아니다.',
        '`inference_policy.json`에 선택 모델·전처리·운영점이, `models/best_resnet18.pt`에 CNN 가중치가 있다. '
        '`models/baseline_random_forest.joblib`은 기존 RF만 분리한 복사본이다. '
        '전체 ML 모델 묶음은 Torch 커널에서 불러오지 않아 native OpenMP 충돌을 피한다. '
        '새 노트북은 완료된 9회 학습 기록과 최적 체크포인트를 기본으로 복원한다. '
        '새로 학습하려면 학습 셀의 force_retrain=True를 사용한다. '
        '설치된 패키지 버전은 `requirements_snapshot.txt`, 무결성 확인은 `verification.json`에 있다.',
        '라벨 없는 test_images 5,506장의 `unlabeled_test_predictions.csv`는 재검사 경보 예측이다. '
        '이 파일에서 성능을 평가하지 않았고, 픽셀 마스크/RLE 제출파일을 생성하지 않았다.',
        f'라벨 없는 이미지의 재검사 권고는 RF {int(data["unlabeled_predictions"].RF_original_pred.sum()):,}장, '
        f'CNN 주 분석 {int(data["unlabeled_predictions"].selected_RF_same_FP_budget_pred.sum()):,}장, '
        f'CNN 검증95% 운영점 {int(data["unlabeled_predictions"].selected_recall95_pred.sum()):,}장이다. '
        '라벨이 없으므로 이 개수로 실제 불량 비율이나 FN·FP를 판단할 수 없다.',
        '## 발표에서 사용할 설명',
        f'“RF의 통계 특징에서 CNN의 이미지 특징 학습으로 확장하고, 검증에서 오탐 상한을 정해 '
        f'불량 누락을 줄이는 방향으로 선택했습니다. 같은 평가 이미지에서 누락은 {int(rf.fn)}장에서 '
        f'{int(main.fn)}장, 오탐은 {int(rf.fp)}장에서 {int(main.fp)}장으로 줄었습니다. '
        '다만 기존 평가 분할의 후속 비교이므로, 새 생산 데이터에서도 같은지 추가 검증이 필요합니다.”',
        '## 참고한 공식 자료',
        '- [Torchvision ResNet18 모델·사전학습 가중치](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)\n'
        '- [PyTorch 전이학습 튜토리얼](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)\n'
        '- [PyTorch Apple MPS](https://docs.pytorch.org/docs/main/notes/mps.html)']
    markdown='\n\n'.join(sections)+'\n'
    (out/'REPORT.md').write_text(markdown)
    # HTML은 그래프를 내부에 포함해 단일 파일로 열거나 공유할 수 있다.
    document=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>철강 결함 딥러닝 비교</title>',
        '<style>body{font:16px/1.75 -apple-system,BlinkMacSystemFont,sans-serif;color:#172033;max-width:1180px;margin:40px auto;padding:0 24px;background:white}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}img{width:100%;height:auto}pre{white-space:pre-wrap;font:inherit}h1,h2{line-height:1.4}@media print{figure{break-inside:avoid}body{margin:0}}</style>',
        '<h1>철강 결함: RandomForest 대비 딥러닝</h1>',
        f'<p>검증 선택 모델: <b>{html.escape(selected)}</b> · 최적 epoch {data["best_epoch"]} · 동일 평가 2,514장</p>']
    for section in sections[1:]:
        if section.startswith('## '):document.append('<h2>'+html.escape(section[3:])+'</h2>')
        elif section.startswith('| '):
            frame=class_summary if '클래스' in section.splitlines()[0] else comparison
            document.append(frame.to_html(index=False,escape=True))
        else:
            rendered=html.escape(section)
            rendered=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',rendered)
            rendered=re.sub(r'`([^`]+)`',r'<code>\1</code>',rendered)
            rendered=re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',rendered)
            document.append('<p>'+rendered.replace('\n','<br>')+'</p>')
    for path in sorted((out/'figures').glob('*.png')):
        encoded=base64.b64encode(path.read_bytes()).decode()
        document.append(f'<figure><figcaption>{html.escape(path.stem)}</figcaption><img alt="{html.escape(path.stem)}" src="data:image/png;base64,{encoded}"></figure>')
    document.append('</html>');(out/'REPORT.html').write_text('\n'.join(document))
    print('발표용 보고서:',out/'REPORT.html',flush=True)
    return comparison,class_summary
