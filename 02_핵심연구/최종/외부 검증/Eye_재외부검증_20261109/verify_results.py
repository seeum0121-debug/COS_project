"""Independent reconstruction of source-defined vectors and reported statistics."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('locked',ROOT/'model/predict.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
b=p.load_bundle(ROOT/'model/eye_A_LR_wmean.joblib')
reg=pd.read_csv(ROOT/'inputs/structure_registry.csv').set_index('component_id')
result=pd.read_csv(ROOT/'outputs/quat_predictions.csv')
features=np.load(ROOT/'outputs/quat_features.npy')
fg=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)

def independent_component(key):
    m=Chem.MolFromSmiles(reg.loc[key,'smiles'])
    frag=sorted(Chem.GetMolFrags(m,asMols=True),key=lambda x:(x.GetNumHeavyAtoms(),Descriptors.MolWt(x)))[-1]
    desc=Descriptors.CalcMolDescriptors(frag)
    return np.r_[fg.GetFingerprintAsNumPy(m),[desc[name[5:]] for name in b['feature_names'][2048:]]]

# Direct reconstruction from source Table 1; do not read the expanded component table.
source_recipe={'DDAC':1.5*.8*.6,'BAC12':1.5*.8*.4*.4,'BAC14':1.5*.8*.4*.5,'BAC16':1.5*.8*.4*.1}
errors=[]
for fid,extra in [('Q_T1_C2',{}),('Q_T1_C1',{'EDTA':.9,'NaOH':.33})]:
    recipe={**source_recipe,**extra};total=sum(recipe.values())
    arr=np.stack([independent_component(k) for k in recipe]);weight=np.array(list(recipe.values()))/total
    finite=np.isfinite(arr)
    independently_built=np.sum(np.where(finite,arr,0)*weight[:,None],axis=0)/np.sum(finite*weight[:,None],axis=0)
    row=int(np.flatnonzero(result.formulation_id==fid)[0])
    error=float(np.nanmax(abs(independently_built-features[row])))
    errors.append(error)
    assert np.allclose(independently_built,features[row],rtol=1e-12,atol=1e-10,equal_nan=True)
    score=float(p.predict_features(b,independently_built[None,:]).eye_A_score.iloc[0])
    assert abs(score-result.iloc[row].model_score)<1e-12

# Published day-7 values in column order, independent of the long-form observation CSV.
source_day7=[7.7,30,17.7,1.7,6.3,2,3,.3,12.3,5,21,1,3.7,15.3,8.3]
assert np.allclose(result.day7,source_day7)
ranks_x=result.model_score.rank(method='average').to_numpy()
ranks_y=pd.Series(source_day7).rank(method='average').to_numpy()
correlation=float(np.corrcoef(ranks_x,ranks_y)[0,1])
stats=pd.read_csv(ROOT/'outputs/rank_association.csv')
assert abs(correlation-stats.iloc[0].spearman_day7)<1e-12
source_pairs=[(1,0),(2,3),(4,5),(6,7)]
correct=sum(np.sign(source_day7[t]-source_day7[c])==np.sign(result.iloc[t].model_score-result.iloc[c].model_score) for c,t in source_pairs)
summary=json.loads((ROOT/'outputs/summary.json').read_text(encoding='utf-8'))
assert correct==summary['paired_direction_agreement']
current_model=ROOT.parent.parent/'최종모델/eye_A_LR_wmean.joblib'
assert hashlib.sha256(current_model.read_bytes()).hexdigest()==hashlib.sha256((ROOT/'model/eye_A_LR_wmean.joblib').read_bytes()).hexdigest()
manifest=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
verification=dict(status='passed',source_table1_rebuilt_vectors=2,max_vector_error=max(errors),
    published_day7_values_checked=15,rank_correlation_independently_recomputed=correlation,
    prespecified_pairs_independently_recomputed=len(source_pairs),pair_direction_agreement=int(correct),
    current_final_model_unchanged=True,input_manifest_unchanged=True)
(ROOT/'outputs/independent_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
checks={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*') if f.is_file() and f.name!='artifact_checksums.json' and '__pycache__' not in f.parts and 'plot_cache' not in f.parts}
(ROOT/'artifact_checksums.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False))
