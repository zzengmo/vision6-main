"""RF와 같은 이미지분할에서 사전학습 CNN·미세조정·RF 결합을 검증한다.

원본 이미지를200x32의39통계로 줄이는 대신800x128전체시야를 보존한다.
모든 학습은 호출한 Jupyter커널 안에서 수행하며 별도 학습 프로세스를 만들지 않는다.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import json
import random
import copy
import hashlib
import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import test1_deep_metrics as mc

SEED=42
SIZE=(800,128)
BATCH=12
EPOCHS=12
PATIENCE=4
TTA_MODES=['none','horizontal']
FUSION_WEIGHTS=[.25,.5,.75]


def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def select_device():
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def synchronize(device):
    if device=='mps':torch.mps.synchronize()
    elif device=='cuda':torch.cuda.synchronize()


def prepare(project,data_dir,out):
    """기존 분할·RF검증 예측을 읽고, 새로운 학습 전에 목표와 조건을 기록한다."""
    project,data_dir,out=Path(project),Path(data_dir),Path(out)
    for name in ['tables','figures','models','pretrained']:(out/name).mkdir(parents=True,exist_ok=True)
    protected=[project/n for n in ['test1.ipynb','test1_top2_boxes.ipynb',
        'test1_ml_model_comparison.ipynb','test1_ml_comparison.py','test1_ml_report.py','segmentation_compare.py']]
    protected += [p for p in (project/'model_comparison_outputs').rglob('*') if p.is_file() and not p.name.startswith('~$')]
    hashes={str(p.relative_to(project)):mc.file_sha256(p) for p in protected}
    (out/'source_sha256_before.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
    previous=project/'model_comparison_outputs'
    manifest=pd.read_csv(previous/'tables/image_split_manifest.csv')
    labels=pd.read_csv(data_dir/'train.csv')
    assert np.array_equal(manifest.has_defect.to_numpy(),manifest.ImageId.isin(set(labels.ImageId)).astype(int).to_numpy())
    classes=pd.crosstab(labels.ImageId,labels.ClassId).reindex(index=manifest.ImageId,
        columns=[1,2,3,4],fill_value=0).gt(0).to_numpy(dtype=np.float32)
    splits={name:np.flatnonzero(manifest.split.to_numpy()==name) for name in ['train','validation','holdout']}
    for a,b in [('train','validation'),('train','holdout'),('validation','holdout')]:
        assert set(manifest.iloc[splits[a]].ImageId).isdisjoint(manifest.iloc[splits[b]].ImageId)
        assert set(manifest.iloc[splits[a]].sha256).isdisjoint(manifest.iloc[splits[b]].sha256)
    with np.load(previous/'image_features39.npz') as saved:
        assert np.array_equal(saved['image_ids'],manifest.ImageId.to_numpy())
        X39=saved['features']
    # 전체묶음의다른OpenMP기반모델을Torch와동시에역직렬화하지않는다.
    # 이RF는기존ML커널에서원본모델을재학습없이복사한파일이다.
    bundle=joblib.load(out/'models/baseline_random_forest.joblib')
    rf=bundle['model'];rf_cut=bundle['threshold']
    y=manifest.has_defect.to_numpy(dtype=np.float32)
    va=splits['validation'];rf_va=mc.score_model(rf,X39[va])
    rf_metrics=mc.metrics(y[va],rf_va>=rf_cut,rf_va)
    fp_budget=rf_metrics['fp']
    assert fp_budget==262
    config={'seed':SEED,'input_width_height':list(SIZE),'batch_size':BATCH,'max_epochs':EPOCHS,
        'patience':PATIENCE,'backbone':'ImageNet-pretrained ResNet18',
        'pool':'global average + global maximum (1024 features)',
        'finetune':'layer3,layer4,binary head,auxiliary4class head; freeze BatchNorm statistics',
        'augmentation':'horizontal/vertical flips, mild brightness/contrast; no random crop',
        'objective':'binary BCE + .25 weighted multilabel BCE',
        'validation_selection':'maximum recall at FP<=RF validation FP; ties AP then fewer FP',
        'validation_fp_budget':fp_budget,'secondary_policy':'validation recall>=.95 highest cutoff',
        'fusion_weights_cnn':FUSION_WEIGHTS,'tta_modes':TTA_MODES,
        'holdout_status':'previously seen in original analyses; follow-up comparison',
        'device':select_device(),'data_dir':str(data_dir),
        'pretrained_source':'https://download.pytorch.org/models/resnet18-f37072fd.pth'}
    (out/'experiment_plan.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    # 그레이스케일uint8캐시는1.29GB. 정규화는배치에서만 수행해메모리를절약한다.
    cache_path=out/'image_cache800x128.npy';shape=(len(manifest),SIZE[1],SIZE[0])
    config_path=out/'image_cache_metadata.json'
    expected={'shape':list(shape),'image_order_sha256':hashlib.sha256(
        '\n'.join(manifest.ImageId).encode()).hexdigest(),'interpolation':'cv2.INTER_AREA grayscale'}
    cache_ready=cache_path.exists() and config_path.exists() and json.loads(config_path.read_text())==expected
    if not cache_ready:
        cache=np.lib.format.open_memmap(cache_path,mode='w+',dtype=np.uint8,shape=shape)
        start=perf_counter();cv2.setNumThreads(1)
        def one(name):
            path=data_dir/'train_images'/name
            raw=path.read_bytes()
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(256,1600):raise ValueError(f'이미지읽기/크기 오류:{name}')
            return cv2.resize(image,SIZE,interpolation=cv2.INTER_AREA),hashlib.sha256(raw).hexdigest()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,(image,digest) in enumerate(pool.map(one,manifest.ImageId)):
                assert digest==manifest.sha256.iloc[i]
                cache[i]=image
                if (i+1)%2500==0 or i+1==len(manifest):
                    print(f'800x128 입력준비 {i+1:,}/{len(manifest):,} | {perf_counter()-start:.1f}초',flush=True)
        cache.flush();del cache
        config_path.write_text(json.dumps(expected,indent=2))
    cache=np.load(cache_path,mmap_mode='r')
    assert cache.shape==shape
    manifest.to_csv(out/'tables/image_split_manifest.csv',index=False)
    return {'project':project,'data_dir':data_dir,'out':out,'manifest':manifest,'labels':labels,
        'y':y,'classes':classes,'splits':splits,'X39':X39,'rf':rf,'rf_cutoff':rf_cut,
        'rf_validation_score':rf_va,'rf_validation_metrics':rf_metrics,'fp_budget':fp_budget,
        'cache':cache,'device':config['device'],'config':config,'protected_hashes':hashes}


class ImageDataset(Dataset):
    """이미지의부분을잘라정답을잃지않고 전체스트립을반환한다. worker프로세스는0개다."""
    def __init__(self,data,indices):
        self.cache=data['cache'];self.indices=np.asarray(indices)
        self.y=data['y'];self.classes=data['classes']
    def __len__(self):return len(self.indices)
    def __getitem__(self,position):
        i=int(self.indices[position])
        return (torch.from_numpy(np.array(self.cache[i],copy=True)).unsqueeze(0),
                torch.tensor(self.y[i],dtype=torch.float32),torch.from_numpy(self.classes[i].copy()))


def loader(data,indices,shuffle=False,batch_size=BATCH):
    return DataLoader(ImageDataset(data,indices),batch_size=batch_size,shuffle=shuffle,
        num_workers=0)


def batch_input(images,device,augment=False):
    x=images.to(device,dtype=torch.float32).div_(255.)
    if augment:
        b=len(x)
        horizontal=torch.rand((b,1,1,1),device=device)<.5
        vertical=torch.rand((b,1,1,1),device=device)<.5
        x=torch.where(horizontal,torch.flip(x,[-1]),x)
        x=torch.where(vertical,torch.flip(x,[-2]),x)
        gain=.85+.30*torch.rand((b,1,1,1),device=device)
        bias=-.04+.08*torch.rand((b,1,1,1),device=device)
        x=(x*gain+bias).clamp_(0,1)
    x=x.expand(-1,3,-1,-1).contiguous()
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    return (x-mean)/std


class StripResNet(nn.Module):
    """국소결함이평균에희석되지않도록 평균과최댓값을함께사용하는ResNet18."""
    def __init__(self,weight_path=None):
        super().__init__();self.backbone=resnet18(weights=None)
        if weight_path is not None:
            state=torch.load(weight_path,map_location='cpu',weights_only=True)
            self.backbone.load_state_dict(state)
        self.backbone.fc=nn.Identity()
        self.dropout=nn.Dropout(.10)
        self.binary_head=nn.Linear(1024,1)
        self.class_head=nn.Linear(1024,4)
    def embedding(self,x):
        net=self.backbone
        x=net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x=net.layer1(x);x=net.layer2(x);x=net.layer3(x);x=net.layer4(x)
        flat=x.flatten(2)
        return torch.cat([flat.mean(-1),flat.amax(-1)],dim=1)
    def forward(self,x):
        z=self.dropout(self.embedding(x))
        return self.binary_head(z).squeeze(1),self.class_head(z)
    def enable_finetune(self):
        for p in self.parameters():p.requires_grad=False
        for module in [self.backbone.layer3,self.backbone.layer4,self.binary_head,self.class_head]:
            for p in module.parameters():p.requires_grad=True
    def training_mode(self):
        self.train()
        # 소배치에서BN추정이흔들리지않게통계는사전학습값으로고정한다.
        for module in self.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()


def operating_curve(y,score):
    y=np.asarray(y,dtype=int);score=np.asarray(score,dtype=float)
    assert np.isfinite(score).all()
    order=np.argsort(-score,kind='stable');s=score[order]
    last=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    tp=np.cumsum(y[order])[last];fp=(last+1)-tp;fn=y.sum()-tp;tn=(1-y).sum()-fp
    return pd.DataFrame({'threshold':s[last],'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'recall':tp/max(int(y.sum()),1),'precision':tp/np.maximum(tp+fp,1)})


def at_fp_budget(y,score,budget):
    curve=operating_curve(y,score);valid=curve[curve.fp<=budget]
    if valid.empty:
        threshold=float(np.nextafter(np.max(score),np.inf))
    else:threshold=float(valid.sort_values(['recall','fp','threshold'],ascending=[False,True,False]).iloc[0].threshold)
    return {'threshold':threshold,**mc.metrics(y,np.asarray(score)>=threshold,score)}


def selection_tuple(row):
    return (float(row['recall']),float(row['ap']),-int(row['fp']))


@torch.inference_mode()
def predict_cnn(data,model,indices,tta=False):
    model.eval();values=[];device=data['device']
    for images,_,_ in loader(data,indices,batch_size=16):
        x=batch_input(images,device)
        probability=model(x)[0].sigmoid()
        if tta:probability=(probability+model(torch.flip(x,[-1]))[0].sigmoid())*.5
        values.append(probability.cpu().numpy())
    return np.concatenate(values)


def benchmark_and_features(data):
    seed_all();weight=data['out']/'pretrained/resnet18-f37072fd.pth'
    if not weight.exists():raise FileNotFoundError('공식 사전학습 가중치를 먼저 내려받아야 합니다.')
    assert mc.file_sha256(weight).startswith('f37072fd')
    model=StripResNet(weight).to(data['device']);model.eval()
    for p in model.parameters():p.requires_grad=False
    tr,va=data['splits']['train'],data['splits']['validation']
    # 학습·검증만전달한다. 홀드아웃은모델선택후처리한다.
    indices=np.r_[tr,va]
    start=perf_counter();features=np.zeros((len(data['manifest']),1024),np.float32)
    processed=0
    with torch.inference_mode():
        for images,_,_ in loader(data,indices,batch_size=16):
            x=batch_input(images,data['device'])
            embedding=model.embedding(x).cpu().numpy()
            features[indices[processed:processed+len(images)]]=embedding
            processed+=len(images)
            if processed%256<16 or processed==len(indices):
                print(f'사전학습 특징추출 {processed:,}/{len(indices):,} | {perf_counter()-start:.1f}초',flush=True)
    np.save(data['out']/'frozen_train_validation_embeddings.npy',features[indices])
    classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',
        max_iter=2500,random_state=SEED))
    classifier.fit(features[tr],data['y'][tr].astype(int))
    va_score=classifier.predict_proba(features[va])[:,1]
    frozen_row=at_fp_budget(data['y'][va],va_score,data['fp_budget'])
    # 표준화된선형모델을raw특징의torch.Linear계수로정확하게변환해미세조정의시작점으로쓴다.
    scaler,linear=classifier[0],classifier[1]
    coefficient=linear.coef_[0]/scaler.scale_
    intercept=linear.intercept_[0]-np.dot(coefficient,scaler.mean_)
    with torch.no_grad():
        model.binary_head.weight.copy_(torch.tensor(coefficient,dtype=torch.float32,device=data['device'])[None,:])
        model.binary_head.bias.copy_(torch.tensor([intercept],dtype=torch.float32,device=data['device']))
    data.update({'frozen_classifier':classifier,'frozen_validation_score':va_score,
                 'frozen_validation_metrics':frozen_row,'frozen_embeddings':features,'cnn':model})
    joblib.dump(classifier,data['out']/'models/frozen_cnn_logistic.joblib')
    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},data['out']/'models/initialized_resnet18.pt')
    print('사전학습 특징 기준선:',frozen_row,flush=True)
    return frozen_row


def restore_completed_training(data):
    """완료된 조기종료/최대epoch 기록만 복원한다. 부분학습을 완료로 오인하지 않는다."""
    history_path=data['out']/'tables/training_history.csv'
    checkpoint_path=data['out']/'models/best_resnet18.pt'
    if not (history_path.exists() and checkpoint_path.exists()):return None
    history=pd.read_csv(history_path)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    epoch=int(checkpoint['epoch']);last=int(history.epoch.max())
    if last<EPOCHS and last-epoch<PATIENCE:return None
    assert epoch in set(history.epoch)
    model=data['cnn'];model.enable_finetune()
    model.load_state_dict(checkpoint['state_dict']);model.to(data['device']);model.eval()
    data['training_history']=history;data['best_epoch']=epoch
    data['best_validation_score']=np.load(data['out']/'best_validation_cnn_score.npy')
    (data['out']/'training_completed.json').write_text(json.dumps({
        'last_epoch':last,'selected_epoch':epoch,'selection_split':'validation',
        'termination':'max_epochs' if last>=EPOCHS else 'early_stopping'},indent=2))
    print(f'완료된 학습 복원: {last}epoch 기록 / 검증 선택epoch {epoch}',flush=True)
    return history


def finetune(data,force_retrain=False):
    """고정계획12epoch·patience4. validation동일FP예산의recall/AP로체크포인트선택."""
    if not force_retrain:
        restored=restore_completed_training(data)
        if restored is not None:return restored
    seed_all();model=data['cnn'];model.enable_finetune()
    tr,va=data['splits']['train'],data['splits']['validation'];device=data['device']
    optimizer=torch.optim.AdamW([
        {'params':list(model.backbone.layer3.parameters())+list(model.backbone.layer4.parameters()),'lr':1e-4},
        {'params':list(model.binary_head.parameters())+list(model.class_head.parameters()),'lr':5e-4}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)
    binary_weight=torch.tensor((data['y'][tr]==0).sum()/(data['y'][tr]==1).sum(),device=device,dtype=torch.float32)
    class_pos=data['classes'][tr].sum(0)
    class_weight=torch.tensor(np.minimum((len(tr)-class_pos)/np.maximum(class_pos,1),20),device=device)
    best_key=(-1.,-1.,-999999);best_epoch=0;stale=0;history=[];start=perf_counter()
    for epoch in range(1,EPOCHS+1):
        epoch_start=perf_counter();model.training_mode();total=0.;n=0
        for step,(images,truth,classes) in enumerate(loader(data,tr,shuffle=True),1):
            x=batch_input(images,device,augment=True);truth=truth.to(device);classes=classes.to(device)
            optimizer.zero_grad(set_to_none=True)
            binary,types=model(x)
            loss=F.binary_cross_entropy_with_logits(binary,truth,pos_weight=binary_weight)
            loss=loss+.25*F.binary_cross_entropy_with_logits(types,classes,pos_weight=class_weight)
            loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            optimizer.step();total+=float(loss.detach().cpu())*len(images);n+=len(images)
            if step%50==0:print(f'epoch{epoch:02d} 학습 {n:,}/{len(tr):,} | loss {total/n:.4f}',flush=True)
        scheduler.step()
        val_score=predict_cnn(data,model,va)
        operating=at_fp_budget(data['y'][va],val_score,data['fp_budget'])
        cutoff95=mc.cutoff_at_recall(data['y'][va],val_score)
        at95=mc.metrics(data['y'][va],val_score>=cutoff95,val_score)
        row={'epoch':epoch,'loss':total/n,'epoch_seconds':perf_counter()-epoch_start,
            'validation_recall_at_rf_fp':operating['recall'],'validation_precision_at_rf_fp':operating['precision'],
            'validation_fp':operating['fp'],'validation_fn':operating['fn'],'validation_ap':operating['ap'],
            'validation95_precision':at95['precision'],'validation95_fp':at95['fp']}
        history.append(row);pd.DataFrame(history).to_csv(data['out']/'tables/training_history.csv',index=False)
        key=selection_tuple(operating)
        if key>best_key:
            best_key=key;best_epoch=epoch;stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':state,'epoch':epoch,'validation':operating,'config':data['config']},
                       data['out']/'models/best_resnet18.pt')
            np.save(data['out']/'best_validation_cnn_score.npy',val_score)
        else:stale+=1
        print(f'epoch{epoch:02d}/{EPOCHS}: loss={row["loss"]:.4f}, val AP={operating["ap"]:.4f}, '
            f'RF동일FP예산 recall={operating["recall"]:.4f}, FN={operating["fn"]}, FP={operating["fp"]}, '
            f'{row["epoch_seconds"]:.1f}초 | best={best_epoch}',flush=True)
        if stale>=PATIENCE:
            print(f'검증기준 {PATIENCE}epoch 개선 없어 조기종료',flush=True);break
    checkpoint=torch.load(data['out']/'models/best_resnet18.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict']);model.to(device);model.eval()
    data['training_history']=pd.DataFrame(history);data['best_epoch']=best_epoch
    data['best_validation_score']=np.load(data['out']/'best_validation_cnn_score.npy')
    (data['out']/'training_completed.json').write_text(json.dumps({
        'last_epoch':int(history[-1]['epoch']),'selected_epoch':best_epoch,
        'selection_split':'validation','termination':'early_stopping' if stale>=PATIENCE else 'max_epochs'},indent=2))
    print(f'미세조정완료: {perf_counter()-start:.1f}초, 선택epoch{best_epoch}',flush=True)
    return data['training_history']


@torch.inference_mode()
def predict_horizontal(data,model,indices):
    model.eval();values=[]
    for images,_,_ in loader(data,indices,batch_size=16):
        x=batch_input(images,data['device'])
        values.append(model(torch.flip(x,[-1]))[0].sigmoid().cpu().numpy())
    return np.concatenate(values)


def select_validation_candidates(data):
    """검증만으로CNN/TTA/RF결합과세운영점을동결한다. 홀드아웃은아직읽지않는다."""
    va=data['splits']['validation'];y=data['y'][va]
    cnn=data['best_validation_score']
    horizontal=predict_horizontal(data,data['cnn'],va)
    tta=(cnn+horizontal)*.5
    scores={'RandomForest':data['rf_validation_score'],
        'CNN_frozen_features':data['frozen_validation_score'],
        'CNN_finetuned':cnn,'CNN_finetuned_TTA':tta}
    definitions={name:{'kind':name,'cnn_weight':1.} for name in scores}
    for base,name_prefix in [('CNN_frozen_features','RF_frozenCNN'),('CNN_finetuned_TTA','RF_finetunedCNN')]:
        for alpha in FUSION_WEIGHTS:
            name=f'{name_prefix}_{int(alpha*100)}pct'
            scores[name]=alpha*scores[base]+(1-alpha)*scores['RandomForest']
            definitions[name]={'kind':'fusion','cnn_base':base,'cnn_weight':alpha}
    rows=[];policies=[]
    for name,score in scores.items():
        if name=='RandomForest':
            main={'threshold':data['rf_cutoff'],**data['rf_validation_metrics']}
        else:main=at_fp_budget(y,score,data['fp_budget'])
        rows.append({'candidate':name,'split':'validation','fp_budget':data['fp_budget'],**main})
        policies.append({'candidate':name,'policy':'RF_same_FP_budget','threshold':main['threshold']})
        cutoff=mc.cutoff_at_recall(y,score)
        policies.append({'candidate':name,'policy':'recall95','threshold':cutoff})
        # 안전마진을둔운영점도결과를보기전에고정한다. 실제FP보장은아니다.
        conservative=at_fp_budget(y,score,int(data['fp_budget']*.75))
        policies.append({'candidate':name,'policy':'RF_FP_budget75pct','threshold':conservative['threshold']})
    table=pd.DataFrame(rows)
    ranking=table.query("candidate != 'RandomForest'").sort_values(
        ['recall','ap','fp','candidate'],ascending=[False,False,True,True])
    selected=ranking.iloc[0].candidate
    data.update({'validation_candidate_scores':scores,'candidate_definitions':definitions,
        'validation_candidate_result':table,'selected_deep_candidate':selected,
        'frozen_policies':pd.DataFrame(policies)})
    table.to_csv(data['out']/'tables/validation_candidate_comparison.csv',index=False)
    data['frozen_policies'].to_csv(data['out']/'tables/frozen_operating_points.csv',index=False)
    np.savez_compressed(data['out']/'validation_candidate_scores.npz',**scores)
    freeze={'selected_deep_candidate':selected,'selected_epoch':data['best_epoch'],
        'selection_split':'validation','rf_validation_fp_budget':data['fp_budget'],
        'operating_policies':data['frozen_policies'].to_dict(orient='records'),
        'candidate_definitions':definitions,'holdout_used_for_selection':False,
        'note':'previously seen holdout will be a follow-up comparison'}
    (data['out']/'frozen_selection.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2))
    print('전체후보와세운영점동결. 선택딥러닝후보:',selected,flush=True)
    return table,data['frozen_policies']


@torch.inference_mode()
def frozen_holdout(data,indices):
    model=StripResNet()
    state=torch.load(data['out']/'models/initialized_resnet18.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state);model.to(data['device']);model.eval()
    features=[]
    for images,_,_ in loader(data,indices,batch_size=16):
        features.append(model.embedding(batch_input(images,data['device'])).cpu().numpy())
    features=np.vstack(features)
    score=data['frozen_classifier'].predict_proba(features)[:,1]
    model.cpu();del model
    if data['device']=='mps':torch.mps.empty_cache()
    return score


def evaluate_holdout(data):
    """고정된선택으로한번평가하며평가수치로재선택하지않는다. 단위는원본이미지다."""
    te=data['splits']['holdout'];y=data['y'][te].astype(int)
    rf=mc.score_model(data['rf'],data['X39'][te])
    assert mc.metrics(y,rf>=data['rf_cutoff'])['fn']==59
    print('동결후홀드아웃추론: 고정CNN특징',flush=True)
    fixed=frozen_holdout(data,te)
    print('동결후홀드아웃추론: 미세조정CNN/좌우반전',flush=True)
    cnn=predict_cnn(data,data['cnn'],te)
    tta=(cnn+predict_horizontal(data,data['cnn'],te))*.5
    scores={'RandomForest':rf,'CNN_frozen_features':fixed,'CNN_finetuned':cnn,'CNN_finetuned_TTA':tta}
    for name,spec in data['candidate_definitions'].items():
        if spec['kind']=='fusion':
            alpha=spec['cnn_weight'];scores[name]=alpha*scores[spec['cnn_base']]+(1-alpha)*rf
    rows=[];predictions={};classes=[]
    for row in data['frozen_policies'].itertuples():
        score=scores[row.candidate];pred=score>=row.threshold
        rows.append({'candidate':row.candidate,'policy':row.policy,'split':'holdout',
                     'threshold':row.threshold,**mc.metrics(y,pred,score)})
        predictions[(row.candidate,row.policy)]=pred
        for c in range(4):
            mask=data['classes'][te,c]==1
            classes.append({'candidate':row.candidate,'policy':row.policy,'class':c+1,
                'positive_images':int(mask.sum()),'detected_images':int(pred[mask].sum()),
                'alarm_recall':float(pred[mask].mean())})
    result=pd.DataFrame(rows);class_result=pd.DataFrame(classes)
    result.to_csv(data['out']/'tables/holdout_all_candidates.csv',index=False)
    class_result.to_csv(data['out']/'tables/holdout_class_alarm_recall.csv',index=False)
    chosen=data['selected_deep_candidate']
    main=result.query("candidate == @chosen and policy == 'RF_same_FP_budget'").iloc[0]
    baseline=result.query("candidate == 'RandomForest' and policy == 'RF_same_FP_budget'").iloc[0]
    assert int(baseline.fn)==59 and int(baseline.fp)==272
    best_pred=predictions[(chosen,'RF_same_FP_budget')]
    rf_pred=predictions[('RandomForest','RF_same_FP_budget')]
    effect={'rf_misses_recovered':int(((y==1)&~rf_pred&best_pred).sum()),
        'rf_hits_newly_missed':int(((y==1)&rf_pred&~best_pred).sum()),
        'new_false_alarms':int(((y==0)&~rf_pred&best_pred).sum()),
        'rf_false_alarms_corrected':int(((y==0)&rf_pred&~best_pred).sum())}
    positive=y==1;delta=best_pred[positive].astype(int)-rf_pred[positive].astype(int)
    rng=np.random.default_rng(SEED)
    ci=np.quantile(rng.choice(delta,(2000,len(delta)),replace=True).mean(1),[.025,.975]).tolist()
    dominates=bool(main.fn<=baseline.fn and main.fp<=baseline.fp and
                   (main.fn<baseline.fn or main.fp<baseline.fp))
    saved=data['manifest'].iloc[te].copy()
    for name,score in scores.items():saved[f'score_{name}']=score
    saved['RF_original_pred']=rf_pred.astype(int)
    for policy in ['RF_same_FP_budget','recall95','RF_FP_budget75pct']:
        saved[f'selected_{policy}_pred']=predictions[(chosen,policy)].astype(int)
    saved.to_csv(data['out']/'tables/holdout_predictions.csv',index=False)
    data.update({'holdout_scores':scores,'holdout_predictions':predictions,'holdout_result':result,
        'class_alarm_result':class_result,'effects':effect,'recall_difference_ci':ci,
        'main_result':main,'rf_result':baseline,'dominates_rf_counts':dominates})
    output={'selected_by_validation':chosen,'best_epoch':data['best_epoch'],
        'RF_holdout':baseline.to_dict(),'deep_holdout_primary':main.to_dict(),
        'deep_other_operating_points':result.query('candidate == @chosen').to_dict(orient='records'),
        'effects':effect,'paired_recall_difference_95ci':ci,'both_FN_FP_improved':dominates,
        'new_blind_evaluation':False,'holdout_used_for_model_selection':False}
    (data['out']/'metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
    return result,class_result


def predict_unlabeled(data):
    """라벨 없는 5,506장은 동결 모델로만 예측한다. 정확도·RLE를 만들지 않는다."""
    source=data['project']/'model_comparison_outputs/unlabeled_test_features39.npz'
    with np.load(source) as saved:
        names=saved['image_ids'].astype(str);features=saved['features']
    assert set(names)=={p.name for p in (data['data_dir']/'test_images').glob('*.jpg')}
    shape=(len(names),SIZE[1],SIZE[0]);path=data['out']/'unlabeled_cache800x128.npy'
    meta=data['out']/'unlabeled_cache_metadata.json'
    expected={'shape':list(shape),'order_sha256':hashlib.sha256('\n'.join(names).encode()).hexdigest()}
    if not (path.exists() and meta.exists() and json.loads(meta.read_text())==expected):
        cache=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=shape)
        def one(name):
            image=cv2.imread(str(data['data_dir']/'test_images'/name),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(256,1600):raise ValueError(f'이미지 오류:{name}')
            return cv2.resize(image,SIZE,interpolation=cv2.INTER_AREA)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i,image in enumerate(pool.map(one,names)):
                cache[i]=image
                if (i+1)%1500==0:print(f'무라벨 이미지 준비 {i+1:,}/{len(names):,}',flush=True)
        cache.flush();del cache;meta.write_text(json.dumps(expected,indent=2))
    inference=dict(data,cache=np.load(path,mmap_mode='r'),y=np.zeros(len(names),np.float32),
                   classes=np.zeros((len(names),4),np.float32))
    indices=np.arange(len(names));selected=data['selected_deep_candidate']
    spec=data['candidate_definitions'][selected];base=spec.get('cnn_base',selected)
    rf=mc.score_model(data['rf'],features)
    if base=='CNN_frozen_features':deep=frozen_holdout(inference,indices)
    else:
        deep=predict_cnn(inference,data['cnn'],indices)
        if base=='CNN_finetuned_TTA':deep=(deep+predict_horizontal(inference,data['cnn'],indices))*.5
    score=spec['cnn_weight']*deep+(1-spec['cnn_weight'])*rf if spec['kind']=='fusion' else deep
    assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
    table=pd.DataFrame({'ImageId':names,'selected_score':score,'RF_score':rf,
                        'RF_original_pred':(rf>=data['rf_cutoff']).astype(int)})
    for row in data['frozen_policies'].query('candidate == @selected').itertuples():
        table[f'selected_{row.policy}_pred']=(score>=row.threshold).astype(int)
    table.to_csv(data['out']/'unlabeled_test_predictions.csv',index=False)
    data['unlabeled_predictions']=table
    print('무라벨 test_images 예측저장:',len(table),'장. 성능평가나 픽셀분할 제출파일은 아님.',flush=True)
    return table
