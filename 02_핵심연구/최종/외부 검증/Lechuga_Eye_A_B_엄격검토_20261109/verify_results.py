from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
out=Path(__file__).resolve().parent
base=out.parent/'Eye_cosmetic_validation_20261109'
sys.path.insert(0,str(base/'model'))
import predict as P
s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
r=pd.read_csv(out/'A_all_rows.csv');a=pd.read_csv(out/'A_aggregated_predictions.csv');reg=pd.read_csv(out/'structure_audit.csv')
checks={}
def check(n,v):
    checks[n]=bool(v)
    assert v,n
check('all_rows_accounted',len(r)==205 and len(r[r.coverage=='unresolved'])==133 and len(r[r.coverage=='partial_components'])==51 and len(r[r.coverage=='all_named_components'])==21)
check('unresolved_not_scored',r.loc[r.coverage=='unresolved','eye_score'].isna().all())
check('no_invented_UVCB_smiles',reg.loc[reg.structure_status!='named_identity_resolved','smiles'].isna().all())
f=a[a.coverage=='all_named_components']
check('all_positive_baseline',len(f)==18 and f.eye_prediction.eq(1).all() and f.zein_label.sum()==14)
check('independent_auc',abs(roc_auc_score(f.zein_label,f.eye_score)-s['all_named_components']['cross_endpoint_roc_auc'])<1e-12)
check('independent_rank',abs(spearmanr(f.eye_score,f.zein_number).statistic-s['all_named_components']['spearman'])<1e-12)
check('cutoff_from_averaged_ZN',a.zein_label.eq((a.zein_number>=200).astype(int)).all())
check('scaffold_gate',all(not MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(sm)) for sm in reg.smiles.dropna()) and s['B']['scaffold_groups']==1)
check('B_not_trained',s['B']['trained'] is False and s['B']['status']=='blocked_before_training')
bundle=P.load_bundle(base/'model/eye_A_LR_wmean.joblib')
sm=reg.set_index('ingredient').smiles.to_dict()
row=f[f.product2.notna()].iloc[0]
vec=P.vectorize_components(bundle['feature_names'],[sm[row.product1],sm[row.product2]],[1,1])
pipe=bundle['pipeline']; transformed=pipe[:-1].transform(vec.to_numpy());lr=pipe.steps[-1][1]
score=float((1/(1+np.exp(-(transformed@lr.coef_.T+lr.intercept_))))[0,0])
check('manual_logistic_mixture_score',abs(score-row.eye_score)<1e-12)
manifest=json.loads((out/'input_manifest.json').read_text(encoding='utf-8'))
check('source_model_protocol_unchanged',all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in manifest.items()))
check('report_warns_baseline','기준선과 동일' in (out/'결과요약.txt').read_text(encoding='utf-8-sig'))
(out/'verification.json').write_text(json.dumps({'status':'passed','checks':checks,'scope':'Independent metric recomputation, one manual logistic mixture inference, scaffold and preservation checks; not original assay validation'},ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',len(checks),'checks')
