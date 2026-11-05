"""Torch 학습커널에서 다른부스팅 라이브러리를 불러오지 않는 공통지표·그림 함수."""
from pathlib import Path
from hashlib import sha256
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from sklearn.metrics import confusion_matrix,average_precision_score,roc_auc_score,precision_recall_curve


def file_sha256(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def score_model(model,X):return model.predict_proba(X)[:,list(model.classes_).index(1)]


def metrics(y,pred,score=None):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    row={'recall':float(tp/max(tp+fn,1)),'precision':float(tp/max(tp+fp,1)),
         'f1':float(2*tp/max(2*tp+fp+fn,1)),'accuracy':float((tp+tn)/len(y)),
         'fpr':float(fp/max(fp+tn,1)),'flagged_rate':float((tp+fp)/len(y)),
         'tp':int(tp),'fn':int(fn),'fp':int(fp),'tn':int(tn)}
    if score is not None:
        row['ap']=float(average_precision_score(y,score));row['roc_auc']=float(roc_auc_score(y,score))
    return row


def cutoff_at_recall(y,score,target=.95):
    _,recall,thresholds=precision_recall_curve(y,score)
    return float(thresholds[np.flatnonzero(recall[:-1]>=target)[-1]])


def setup_plots():
    plt.style.use('default')
    for path in ['/System/Library/Fonts/AppleSDGothicNeo.ttc','/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                 '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name();break
    plt.rcParams.update({'axes.unicode_minus':False,'figure.dpi':110,'savefig.dpi':200,
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False})


def save_figure(fig,out,name):
    fig.patch.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white');ax.tick_params(colors='#172033')
        for spine in ax.spines.values():spine.set_color('#94a3b8')
    for text in fig.findobj(match=Text):text.set_color('#172033')
    for ax in fig.axes:
        for artist in ax.images:
            values=np.asarray(artist.get_array())
            if artist.get_cmap().name!='Blues' or values.ndim!=2 or values.size>100:continue
            for label in ax.texts:
                x,y=label.get_position();r,c=int(round(y)),int(round(x))
                if 0<=r<values.shape[0] and 0<=c<values.shape[1]:
                    label.set_color('white' if artist.norm(values[r,c])>.55 else '#172033')
    for ext,backend in [('png','agg'),('svg','svg')]:
        fig.savefig(Path(out)/'figures'/f'{name}.{ext}',bbox_inches='tight',facecolor='white',backend=backend)
    plt.close(fig)
