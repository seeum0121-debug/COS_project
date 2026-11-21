"""Independent arithmetic checks on source-to-feature-to-score and grouped metrics."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Read-only access to original package, including its established regression fixtures.
    sys.path.insert(0,str(ROOT.parents[1]/'최종모델'))
    import predict
    bundle=joblib.load(ROOT/'model/eye_A_LR_wmean.joblib')
    names=bundle['feature_names']; threshold=bundle['threshold']
    raw=pd.read_csv(ROOT/'sources/cosmetic_validation/flea_tick_shampoo.csv')
    results=pd.read_csv(ROOT/'outputs/shampoo_predictions.csv')
    vectors=np.load(ROOT/'outputs/shampoo_features.npy')
    ids=json.loads((ROOT/'outputs/shampoo_feature_row_ids.json').read_text(encoding='utf-8'))
    gen=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    descriptors=[name.removeprefix('desc_') for name in names[2048:]]
    for _,r in results[results.score.notna()].iterrows():
        g=raw[(raw['product']==r['product']) & raw.resolved].copy()
        weights=((g.pct_low+g.pct_high)/2).to_numpy(float)
        individual=[]
        for smi in g.canonical_smiles:
            mol=Chem.MolFromSmiles(smi)
            largest=max(Chem.GetMolFrags(mol,asMols=True),key=lambda x:(x.GetNumHeavyAtoms(),Descriptors.MolWt(x)))
            desc=Descriptors.CalcMolDescriptors(largest,missingVal=np.nan)
            individual.append(np.r_[gen.GetFingerprintAsNumPy(mol),[desc[n] for n in descriptors]])
        individual=np.array(individual)
        # All component values are finite here; independent simple weighted average is valid.
        assert np.isfinite(individual).all()
        x=np.average(individual,weights=weights,axis=0)
        np.testing.assert_allclose(x,vectors[ids.index(r['product'])],rtol=1e-12,atol=1e-8)
        expected=bundle['pipeline'].predict_proba(x[None,:])[0,1]
        np.testing.assert_allclose(r.score,expected,rtol=0,atol=1e-12)
        assert r.prediction==int(expected>=threshold)
        assert np.isclose(r.represented_pct,weights.sum())
        assert r.score_low-1e-12 <= expected <= r.score_high+1e-12
    # AUC computed by pairwise ordering, independent of sklearn.metrics.
    scored=results.dropna(subset=['score'])
    yy=scored.label.to_numpy(int); pp=scored.prediction.to_numpy(int)
    tp=int(((yy==1)&(pp==1)).sum());tn=int(((yy==0)&(pp==0)).sum())
    fp=int(((yy==0)&(pp==1)).sum());fn=int(((yy==1)&(pp==0)).sum())
    auc=np.mean([float(a>b)+.5*float(a==b) for a in scored.loc[scored.label==1,'score'] for b in scored.loc[scored.label==0,'score']])
    summary=json.loads((ROOT/'outputs/summary.json').read_text(encoding='utf-8'))
    m=summary['metrics'][0]
    assert [tp,tn,fp,fn]==[m['TP'],m['TN'],m['FP'],m['FN']]
    assert auc==m['ROC_AUC']
    assert (tp,tn,fp,fn)==(0,0,2,2)
    groups=pd.read_csv(ROOT/'outputs/shampoo_registration_groups.csv')
    assert len(groups)==3 and sorted(groups.n_products)==[1,1,2]
    for _,g in groups.iterrows():
        source=scored[scored.registration_group==g.registration_group]
        np.testing.assert_allclose(g.score,source.score.mean(),rtol=0,atol=1e-14)
    assert (scored.score_low<threshold).all()
    assert (scored.score_high>=threshold).sum()==2
    assert results.loc[results['product'].str.startswith('Davis'),'score'].isna().all()
    fixture=json.loads((ROOT/'model/smoke_test_cases.json').read_text(encoding='utf-8'))
    for c in fixture['cases']:
        x=predict.vectorize_components(names,c['smiles'],c['amounts'])
        p=predict.predict_features(bundle,x).iloc[0]
        np.testing.assert_allclose(p.eye_A_score,c['expected_score'],rtol=0,atol=1e-10)
    original=ROOT.parents[1]/'최종모델'
    checks=json.loads((original/'checksums.json').read_text(encoding='utf-8'))
    for name,h in checks['files'].items(): assert sha(original/name)==h,name
    cir=pd.read_csv(ROOT/'outputs/CIR_eye_audit_predictions.csv')
    assert len(cir)==31 and cir.score.notna().sum()==22
    assert cir.loc[cir.CAS=='151-21-3','score'].isna().all()
    assert cir.loc[cir.CAS=='61789-40-0','score'].isna().all()
    for _,g in cir.dropna(subset=['score']).groupby('CAS'):
        np.testing.assert_allclose(g.score,g.score.iloc[0],rtol=0,atol=1e-12)
    out={'status':'passed','independently_rebuilt_shampoo_vectors':4,
         'manual_confusion_counts':{'TP':tp,'TN':tn,'FP':fp,'FN':fn},
         'pairwise_AUC':auc,'registration_groups':3,'range_flip_products':2,
         'original_model_package_checksums_unchanged':True,
         'saved_model_regression_cases':len(fixture['cases']),
         'CIR_no_invented_dilution_vehicle':True}
    (ROOT/'outputs/independent_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    (ROOT/'artifact_checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_checksums.json'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
