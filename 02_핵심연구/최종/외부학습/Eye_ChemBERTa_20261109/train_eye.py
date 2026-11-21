from pathlib import Path
import json,hashlib,platform,sys
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,matthews_corrcoef,roc_auc_score,balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
BASE=OUT.parent.parent/'외부 검증/Eye_cosmetic_validation_20261109'
def save(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
assert all(sha(Path(p))==v for p,v in manifest.items())
df=pd.read_csv(BASE/'training_reference/input_snapshot.csv',low_memory=False)
link=pd.read_csv(BASE/'training_reference/component_linkage.csv')
X=np.load(OUT/'chemberta_wmean.npy').astype(np.float64);W=np.load(BASE/'training_reference/wmean.npy')
y=df.binary_label.to_numpy(dtype=int)
assert np.array_equal(y,df.final_label.isin([1,2]).astype(int))
assert df.final_label.isin([1,2,3,4]).all()
for i,g in link.groupby('row_index'): assert g.formulation_id.eq(df.iloc[i].formulation_id).all()
parent=list(range(len(df)))
def root(i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]];i=parent[i]
    return i
def unite_values(values):
    seen={}
    for i,key in enumerate(values):
        if pd.isna(key) or str(key)=='': continue
        if key in seen: parent[root(i)]=root(seen[key])
        else: seen[key]=i
for col in ['split_group_id','comp_group_id','structure_group_id']:
    unite_values(df[col])
unite_values([hashlib.sha256(np.nan_to_num(v,nan=0).tobytes()).hexdigest() for v in W])
# Same encoded mixture must not cross the holdout boundary either.
unite_values([hashlib.sha256(v.tobytes()).hexdigest() if np.isfinite(v).all() else None for v in X])
groups=np.array([root(i) for i in range(len(df))])
eligible=np.isfinite(X).all(1)
pool=np.flatnonzero(eligible & df.evaluation_role.eq('existing_evaluation_pool'))
split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
tr,te=next(split.split(X[pool],y[pool],groups[pool]))
test=pool[te];testgroups=set(groups[test])
augment=np.flatnonzero(eligible & df.evaluation_role.eq('augmentation_train_only'))
train=np.r_[pool[tr],[i for i in augment if groups[i] not in testgroups]].astype(int)
purged=[int(i) for i in augment if groups[i] in testgroups]
assert set(groups[train]).isdisjoint(testgroups)
assert len(np.unique(y[train]))==len(np.unique(y[test]))==2
roles=np.full(len(df),'excluded_tokenizer',dtype=object)
roles[pool]='unassigned'; roles[train]='train';roles[test]='holdout';roles[purged]='purged_augmentation_group'
splitdf=pd.DataFrame({'row_index':np.arange(len(df)),'formulation_id':df.formulation_id,'label':y,'group':groups,'role':roles,'original_role':df.evaluation_role})
splitdf.to_csv(OUT/'split_assignments.csv',index=False,encoding='utf-8-sig')
save('split_receipt.json',{'protocol_sha256':sha(OUT/'protocol.json'),'train_rows':len(train),'holdout_rows':len(test),'train_groups':len(set(groups[train])),'holdout_groups':len(testgroups),'train_classes':np.bincount(y[train],minlength=2).tolist(),'holdout_classes':np.bincount(y[test],minlength=2).tolist(),'purged_augmentation':purged,'excluded_tokenizer_rows':int((~eligible).sum())})
original=joblib.load(BASE/'model/eye_A_LR_wmean.joblib')
head=make_pipeline(SimpleImputer(strategy='constant',fill_value=0,keep_empty_features=True),StandardScaler(),LogisticRegression(C=1.0,max_iter=5000,random_state=42))
baseline=clone(original['pipeline'])
weights=compute_sample_weight('balanced',y[train])
with threadpool_limits(limits=4):
    head.fit(X[train],y[train],logisticregression__sample_weight=weights)
    baseline.fit(W[train],y[train],**{baseline.steps[-1][0]+'__sample_weight':weights})
    pc=head.predict_proba(X[test])[:,1];pl=baseline.predict_proba(W[test])[:,1]
def metric(y,p):
    pred=(p>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'threshold':.5,'MCC':float(matthews_corrcoef(y,pred)),'ROC_AUC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}
metrics={'ChemBERTa_Eye':metric(y[test],pc),'LR_same_training_split':metric(y[test],pl)}
joblib.dump({'format':'eye-chemberta-frozen-epa-v1','pipeline':head,'threshold':.5,'encoder_sha256':json.loads((OUT/'encoding_summary.json').read_text())['encoder_sha256'],'target':'EPA I/II=1; III/IV=0','representation':'weighted_mean_384','protocol_sha256':sha(OUT/'protocol.json'),'training_rows':train.tolist(),'evaluation':'internal_group_holdout_only'},OUT/'eye_chemberta_head.joblib')
joblib.dump({'pipeline':baseline,'threshold':.5,'training_rows':train.tolist(),'target':'EPA I/II=1; III/IV=0'},OUT/'lr_refit_comparator.joblib')
pd.DataFrame({'row_index':test,'formulation_id':df.iloc[test].formulation_id.to_numpy(),'group':groups[test],'observed_eye_epa_binary':y[test],'chemberta_score':pc,'chemberta_prediction':(pc>=.5).astype(int),'lr_score':pl,'lr_prediction':(pl>=.5).astype(int)}).to_csv(OUT/'holdout_predictions.csv',index=False,encoding='utf-8-sig')
rng=np.random.default_rng(20261109);ug=np.unique(groups[test]);deltas=[]
for _ in range(2000):
    sample=rng.choice(ug,len(ug),replace=True)
    ix=np.concatenate([np.flatnonzero(groups[test]==g) for g in sample])
    if len(np.unique(y[test][ix]))<2: continue
    deltas.append(float(matthews_corrcoef(y[test][ix],pc[ix]>=.5)-matthews_corrcoef(y[test][ix],pl[ix]>=.5)))
ci=np.quantile(deltas,[.025,.975]).tolist()
train_sm=set(link[link.row_index.isin(train)].smiles);test_sm=set(link[link.row_index.isin(test)].smiles)
summary={'status':'trained_and_internally_evaluated','endpoint':'Eye EPA I/II versus III/IV; no Zein labels','metrics':metrics,'delta_MCC':metrics['ChemBERTa_Eye']['MCC']-metrics['LR_same_training_split']['MCC'],'paired_group_bootstrap_delta_MCC_95CI':ci,'bootstrap_valid':len(deltas),'split':json.loads((OUT/'split_receipt.json').read_text()),'shared_individual_component_smiles':len(train_sm & test_sm),'test_unique_component_smiles':len(test_sm),'independent_external_EPA_evaluation':False,'encoder_frozen':True,'original_LR_preserved':True,'limitations':['Internal heldout formulations, not independent external experiments','Individual ingredient overlap allowed; not unseen-component generalization','Prior data and LR hyperparameter selection exposure','Upstream MLM corpus overlap unknown; no supervised encoder fine-tuning','No cosmetic safety claim; zero means EPA III/IV, not no eye irritation']}
save('summary.json',summary)
save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__})
assert all(sha(Path(p))==v for p,v in manifest.items())
print(json.dumps(summary,ensure_ascii=False,indent=2))
