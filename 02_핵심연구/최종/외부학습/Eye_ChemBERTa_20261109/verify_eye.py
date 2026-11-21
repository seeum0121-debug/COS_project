from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score,matthews_corrcoef,confusion_matrix
from predict_eye import predict,validate

OUT=Path(__file__).resolve().parent
BASE=OUT.parent.parent/'외부 검증/Eye_cosmetic_validation_20261109'
checks={}
def check(n,v):
    checks[n]=bool(v)
    if not v: raise AssertionError(n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((OUT/'summary.json').read_text())
a=pd.read_csv(OUT/'split_assignments.csv');p=pd.read_csv(OUT/'holdout_predictions.csv')
train=a[a.role=='train'].row_index.to_numpy();test=a[a.role=='holdout'].row_index.to_numpy()
X=np.load(OUT/'chemberta_wmean.npy').astype(np.float64);W=np.load(BASE/'training_reference/wmean.npy')
check('complete_attrition_accounting',len(a)==673 and a.role.eq('train').sum()==533 and a.role.eq('holdout').sum()==118 and a.role.eq('excluded_tokenizer').sum()==20 and a.role.eq('purged_augmentation_group').sum()==2)
check('no_label_group_leakage',set(a.loc[train,'group']).isdisjoint(set(a.loc[test,'group'])))
check('augmentation_never_tested',a.loc[test,'original_role'].eq('existing_evaluation_pool').all())
check('prediction_row_alignment',set(p.row_index)==set(test))
for name,file,matrix,col in [('ChemBERTa_Eye','eye_chemberta_head.joblib',X,'chemberta_score'),('LR_same_training_split','lr_refit_comparator.joblib',W,'lr_score')]:
    b=joblib.load(OUT/file);pipe=b['pipeline']
    check(name+'_fit_row_manifest',b['training_rows']==train.tolist())
    actual=pipe.predict_proba(matrix[p.row_index])[:,1]
    check(name+'_reloaded_predictions',np.allclose(actual,p[col],atol=1e-12,rtol=0))
    m=s['metrics'][name];y=p.observed_eye_epa_binary.to_numpy();pred=p[col].to_numpy()>=.5
    check(name+'_independent_metrics',abs(roc_auc_score(y,p[col])-m['ROC_AUC'])<1e-12 and abs(matthews_corrcoef(y,pred)-m['MCC'])<1e-12 and confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()==[m['TN'],m['FP'],m['FN'],m['TP']])
    imputed=pipe.steps[0][1].transform(matrix[train])
    check(name+'_train_only_scaler',np.allclose(pipe.steps[1][1].mean_,imputed.mean(0),rtol=1e-6,atol=1e-6))
    check(name+'_optimizer_converged',bool((pipe.steps[-1][1].n_iter_<5000).all()))
for matrix,name in [(X,'encoder'),(W,'LR')]:
    hashes=lambda ix:{hashlib.sha256(np.nan_to_num(matrix[i],nan=0).tobytes()).hexdigest() for i in ix}
    check(name+'_no_exact_vector_leakage',hashes(train).isdisjoint(hashes(test)))
link=pd.read_csv(BASE/'training_reference/component_linkage.csv')
# Deliberately choose a multi-component heldout formulation for true end-to-end inference.
counts=link.groupby('row_index').size();chosen=next(int(i) for i in test if counts.loc[i]>1)
g=link[link.row_index==chosen]
payload={'formulation_id':g.formulation_id.iloc[0],'components':[{'smiles':r.smiles,'weight':float(r.weight)} for _,r in g.iterrows()]}
(OUT/'smoke_input.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
result=predict(payload)
(OUT/'smoke_output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
expected=float(p.loc[p.row_index==chosen,'chemberta_score'].iloc[0])
check('raw_smiles_to_Eye_end_to_end',abs(result['score']-expected)<1e-5 and result['endpoint']=='eye_epa')
for label,payload in [('empty',{'components':[]}),('missing_weight',{'components':[{'smiles':'CCO'}]}),('negative_weight',{'components':[{'smiles':'CCO','weight':-1}]}),('zero_sum',{'components':[{'smiles':'CCO','weight':0}]}),('nonfinite',{'components':[{'smiles':'CCO','weight':float('nan')}]}),('invalid_SMILES',{'components':[{'smiles':'not_a_molecule','weight':1}]})]:
    caught=False
    try:validate(payload)
    except ValueError:caught=True
    check('reject_'+label,caught)
manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
check('original_encoder_LR_data_protocol_unchanged',all(sha(Path(k))==v for k,v in manifest.items()))
check('external_evaluation_not_claimed',s['independent_external_EPA_evaluation'] is False)
check('separate_eye_endpoint','Zein' not in result['endpoint'])
result={'status':'passed','checks':checks,'smoke_row':chosen,'smoke_score':result['score'],'expected_score':expected,'scope':'Split/feature leakage, train-only scaling, reloaded model metrics, raw-SMILES frozen-encoder inference, invalid inputs and original artifact hashes. No independent external EPA data verified.'}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',len(checks),'checks')
