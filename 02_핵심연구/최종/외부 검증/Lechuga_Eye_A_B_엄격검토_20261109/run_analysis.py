from pathlib import Path
import sys, json, hashlib, html
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, roc_auc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
BASE=ROOT/'Eye_cosmetic_validation_20261109'
sys.path.insert(0,str(BASE/'model'))
import predict as P

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
source=BASE/'sources/cosmetic_validation/lechuga2025_surfactant_mixture_zein.csv'
model=BASE/'model/eye_A_LR_wmean.joblib'
before={str(p):digest(p) for p in [source,model,OUT/'protocol.json']}
bundle=P.load_bundle(model)
data=pd.read_csv(source).fillna({'product2':'','proportion':''})
identities={
'Lauramine Oxide':(15433,'CCCCCCCCCCCC[N+](C)(C)[O-]','lauramine%20oxide'),
'Myristamine Oxide':(18739,'CCCCCCCCCCCCCC[N+](C)(C)[O-]','myristamine%20oxide'),
'Sodium lauryl sulfate':(3423265,'CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]','sodium%20lauryl%20sulfate'),
'Disodium Lauroyl Glutamate':(11337766,'CCCCCCCCCCCC(=O)NC(CCC(=O)[O-])C(=O)[O-].[Na+].[Na+]','disodium%20lauroyl%20glutamate'),
'Disodium Capryloyl Glutamate: 30% - 37%':(71587566,'CCCCCCCC(=O)NC(CCC(=O)[O-])C(=O)[O-].[Na+].[Na+]','disodium%20capryloyl%20glutamate')}
registry=[]
for name in sorted(set(data.product1)|set(data.product2)-{''}):
    item=identities.get(name)
    registry.append({'ingredient':name,'structure_status':'named_identity_resolved' if item else 'unresolved_no_representative_structure',
        'smiles':item[1] if item else '', 'pubchem_cid':item[0] if item else None,
        'source_url':'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/'+item[2]+'/property/CanonicalSMILES/JSON' if item else '',
        'experimental_lot_identity_confirmed':False,
        'note':'PubChem API returned this named identity; commercial lot composition/purity not established' if item else 'No exact single structure established; coco/EO/alkyl/glucoside averages are not substituted'})
pd.DataFrame(registry).to_csv(OUT/'structure_audit.csv',index=False,encoding='utf-8-sig')
training=np.load(BASE/'training_reference/wmean.npy')
rows=[]
for i,r in data.iterrows():
    names=[r.product1]+([r.product2] if r.product2 else [])
    known=[j for j,n in enumerate(names) if n in identities]
    out=r.to_dict(); out['source_row']=i+2
    out['coverage']='all_named_components' if len(known)==len(names) else 'partial_components' if known else 'unresolved'
    out['unresolved_components']='; '.join(n for n in names if n not in identities)
    out['amount_basis']='nominal relative ratio; absolute active amount unverified'
    out['contains_SLS']='Sodium lauryl sulfate' in names
    out['eye_score']=None;out['eye_prediction']=None;out['training_feature_matches']=None
    if known:
        weights=[1.] if len(names)==1 else [float(x) for x in str(r.proportion).split('-')]
        if len(weights)!=len(names): raise ValueError('Invalid mixture ratio')
        v=P.vectorize_components(bundle['feature_names'],[identities[names[j]][1] for j in known],[weights[j] for j in known])
        pred=P.predict_features(bundle,v).iloc[0]
        out['eye_score']=float(pred.eye_A_score);out['eye_prediction']=int(pred.eye_A_prediction)
        out['feature_hash']=hashlib.sha256(np.nan_to_num(v.to_numpy(),nan=0.).tobytes()).hexdigest()
        out['training_feature_matches']=int(np.all(np.isclose(training,v.to_numpy(),atol=1e-8,rtol=1e-10,equal_nan=True),axis=1).sum())
    rows.append(out)
raw=pd.DataFrame(rows)
raw.to_csv(OUT/'A_all_rows.csv',index=False,encoding='utf-8-sig')
keys=['product1','product2','proportion','concentration_pct','coverage']
agg=raw[raw.eye_score.notna()].groupby(keys,dropna=False,as_index=False).agg(zein_number=('zein_number','mean'),n_replicates=('zein_number','size'),eye_score=('eye_score','first'),eye_prediction=('eye_prediction','first'),feature_hash=('feature_hash','first'),training_feature_matches=('training_feature_matches','max'),contains_SLS=('contains_SLS','max'))
agg['zein_label']=(agg.zein_number>=200).astype(int)
agg.to_csv(OUT/'A_aggregated_predictions.csv',index=False,encoding='utf-8-sig')
def metrics(df):
    if not len(df): return {'n':0}
    y=df.zein_label.astype(int);yp=df.eye_prediction.astype(int)
    rho=float(spearmanr(df.eye_score,df.zein_number).statistic) if df.eye_score.nunique()>1 and df.zein_number.nunique()>1 else None
    return {'n':len(df),'unique_inputs':int(df.feature_hash.nunique()),'label_counts':{str(k):int(v) for k,v in y.value_counts().items()},'agreement_count':int((y==yp).sum()),'descriptive_agreement':float((y==yp).mean()),'confusion_matrix_labels_0_1':confusion_matrix(y,yp,labels=[0,1]).tolist(),'cross_endpoint_roc_auc':float(roc_auc_score(y,df.eye_score)) if y.nunique()==2 else None,'spearman':rho,'not_EPA_accuracy':True}
full=agg[agg.coverage=='all_named_components']
partial=agg[agg.coverage=='partial_components']
scaffolds={}
for n,(_,s,_) in identities.items():
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) or 'ACYCLIC'
    scaffolds[n]=scaffold
# Component-connected groups: no ingredient may cross train/test.
parent={n:n for n in identities}
def find(n):
    while parent[n]!=n: n=parent[n]
    return n
for _,r in full.iterrows():
    if r.product2: parent[find(r.product2)]=find(r.product1)
groups=sorted({find(n) for n in identities})
b={'status':'blocked_before_training','trained':False,'endpoint':'zein_binary_not_eye_EPA','named_chemical_identities':len(identities),'fully_named_analysis_units':len(full),'unique_model_inputs':int(full.feature_hash.nunique()),'component_connected_groups':len(groups),'scaffold_groups':len(set(scaffolds.values())),'scaffolds':scaffolds,'reasons':['All retained named identities are acyclic: one scaffold group; scaffold-disjoint train/test cannot be formed under the existing protocol.','Absolute active amount and commercial lot composition are not confirmed; nominal ratio is not substituted for amount_pct.','Final Eye A LR has no ChemBERTa encoder to attach a new head. A separate archived ChemBERTa encoder exists, but using it is a different model lineage.'], 'available_separate_encoder':'C:/Users/DS/COS/최종/검증/skin_team_audit_20261108/source/pretraining/best/model.safetensors','existing_eye_model_modified':False,'next_required':'Obtain verified structures/amounts and data supporting independent evaluation, or explicitly redesign scaffold validation for acyclic surfactants before training. No random-row fallback.'}
save('B_readiness.json',b)
summary={'A_status':'completed_frozen_cross_endpoint_comparison','model':'Eye A LR wmean','threshold':float(bundle['threshold']),'raw_rows':len(raw),'unique_source_ingredients':len(registry),'resolved_named_identities':len(identities),'coverage_raw_counts':{str(k):int(v) for k,v in raw.coverage.value_counts().items()},'all_named_components':metrics(full),'all_named_without_exact_training_match':metrics(full[full.training_feature_matches==0]),'all_named_without_SLS':metrics(full[~full.contains_SLS]),'partial_components_diagnostic_only':metrics(partial),'B':b,'caveats':['Named PubChem identities do not prove purity or complete commercial sample identity.','Nominal mixture ratios not verified as absolute active mass; 30–37% active material remains a partial commercial input.','ZN>=200 is an operational source threshold, not verified Eye EPA truth or confirmed regulatory cutoff.','Existing dataset has prior outcome exposure; results are exploratory.']}
save('summary.json',summary)
assert all(digest(Path(k))==v for k,v in before.items())
save('input_manifest.json',before)
text=f'''Eye 모델 Lechuga A→B 실행 결과

A: 기존 Eye A LR 모델·임계값 {bundle['threshold']:.6f} 고정 계산 완료.
원자료 {len(raw)}행, 성분명 {len(registry)}종 중 PubChem에서 명칭별 단일구조를 확인한 5종만 입력.
UVCB 대표구조 가정은 사용하지 않음. 단일 명칭 구조 확인이 실험 상업원료의 순도 확인은 아님.
전체 성분명이 해결된 행: {len(raw[raw.coverage=='all_named_components'])}; 반복 집계 후 {len(full)}단위, 고유 모델 입력 {full.feature_hash.nunique()}개.
해결된 일부 성분만 사용한 결과와 전 성분명 해결 결과는 분리.
전 성분명 해결 비교: {json.dumps(metrics(full),ensure_ascii=False)}
SLS 제외: {json.dumps(metrics(full[~full.contains_SLS]),ensure_ascii=False)}
위 일치율/AUC는 Zein 대리 라벨과의 기술 통계이며 EPA 눈자극 성능이 아님.
18단위 모두 양성으로 예측했으므로 일치 14/18(77.8%)는 전부 양성이라고 답하는 기준선과 동일함. 음성 4건은 모두 오분류하여 분별력을 입증하지 못함.
같은 명목 배합은 농도가 달라도 동일 모델 입력이 됨. 총 6개 입력을 18개 독립 화학 사례로 해석하지 않음.
혼합비는 명목 상대비로만 사용했고 절대 활성성분 함량으로 환산하지 않음. Zein 200 경계의 공식 규정 근거는 독립 확인하지 못함.

B: 학습 적격성 검토까지 진행했으며 학습은 보류.
5개 화학종은 모두 비고리형으로 scaffold 그룹이 {len(set(scaffolds.values()))}개뿐임. PDF와 기존 protocol의 scaffold-disjoint 검증 불가.
성분 연결 그룹은 {len(groups)}개이며, 반복 농도를 무작위 분할해 표본 수를 부풀리지 않음.
최종 Eye A LR는 ChemBERTa가 아니며 새 head를 붙일 backbone이 없음. 별도 보관 ChemBERTa는 존재하나 다른 모델 계보임.
절대 함량과 상업원료 조성도 미확인. 이 조건을 숨기고 별도 모델을 학습하거나 성능을 주장하지 않음.
기존 Eye 모델·입력·임계값은 변경하지 않음.

해석: A에서 점수를 계산했지만 엄격한 EPA 외부검증을 완료한 것이 아님. B를 완료했다고 보고하지 않음.
재현: python run_analysis.py
출처: structure_audit.csv의 PubChem API URL 및 기존 원본 CSV. Zenodo 원문 접근 2회 HTTP 403.
'''
(OUT/'결과요약.txt').write_text(text,encoding='utf-8-sig')
(OUT/'A_B_결과보고서.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Eye A→B 엄격 검토</title><style>body{font-family:Malgun Gothic,sans-serif;max-width:1100px;margin:40px auto;line-height:1.7}pre{white-space:pre-wrap}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Eye 모델 Lechuga A→B 결과</h1><pre>'+html.escape(text)+'</pre><h2>전 성분명 해결 분석단위</h2>'+full.to_html(index=False,escape=True)+'</html>',encoding='utf-8')
print(json.dumps({'coverage':summary['coverage_raw_counts'],'A':summary['all_named_components'],'B_status':b['status'],'scaffold_groups':b['scaffold_groups']},ensure_ascii=False,indent=2))
