"""Locked-model diagnostic comparison against newly reviewed external cases."""
from pathlib import Path
import base64, hashlib, html, importlib.util, json, math, os
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'Eye_cosmetic_validation_20261109'
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save(name,df): df.to_csv(OUT/name,index=False,encoding='utf-8-sig')
manifest=json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
assert all(sha(ROOT/k)==v for k,v in manifest.items())
protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('locked_predict',ROOT/'model/predict.py')
predict=importlib.util.module_from_spec(spec); spec.loader.exec_module(predict)
bundle=predict.load_bundle(ROOT/'model/eye_A_LR_wmean.joblib')
assert bundle['threshold']==protocol['threshold']
reg=pd.read_csv(ROOT/'inputs/structure_registry.csv').set_index('component_id')
forms=pd.read_csv(ROOT/'inputs/quat_formulations.csv')
comps=pd.read_csv(ROOT/'inputs/quat_components.csv')
obs=pd.read_csv(ROOT/'inputs/quat_observations.csv')
paired=pd.read_csv(ROOT/'inputs/prespecified_pairs.csv')
excluded=pd.read_csv(ROOT/'inputs/excluded_cases.csv')
train=pd.read_csv(OLD/'training_reference/input_snapshot.csv',low_memory=False)
link=pd.read_csv(OLD/'training_reference/component_linkage.csv')
train_x=np.load(OLD/'training_reference/wmean.npy')
assert train_x.shape==(673,2265)

def vector(selected):
    return predict.vectorize_components(bundle['feature_names'],reg.loc[selected.component_id,'smiles'].tolist(),selected.pct.to_numpy()).to_numpy()[0]
def feature_key(row):
    return hashlib.sha256(np.nan_to_num(np.round(row,10),nan=-999999).astype('<f8').tobytes()).hexdigest()
def spearman(x,y):
    a,b=rankdata(x),rankdata(y)
    return None if len(a)<3 or np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
def auc7(group):
    q=group.set_index('day').loc[[1,2,3,4,7],'draize_score']
    assert not q.isna().any()
    return float(np.trapezoid(q.to_numpy(),q.index.to_numpy())/6)

rows=[]; vectors=[]; overlap=[]; water_rows=[]; water_vectors=[]
for f in forms.to_dict('records'):
    fid=f['formulation_id']
    c=comps.loc[comps.formulation_id==fid]
    x=vector(c.loc[c.role=='primary_disclosed_nonwater'])
    vectors.append(x)
    prediction=predict.predict_features(bundle,x[None,:]).iloc[0]
    observations=obs.loc[obs.formulation_id==fid]
    days=observations.set_index('day').draize_score
    matches=np.flatnonzero(np.isclose(train_x,x[None,:],rtol=1e-9,atol=1e-10,equal_nan=True).all(axis=1))
    matched_ids=train.iloc[matches].formulation_id.tolist()
    row={**f,'model_score':float(prediction.eye_A_score),'model_prediction':int(prediction.eye_A_prediction),
        'day1':float(days.loc[1]),'day7':float(days.loc[7]),'auc_days1_7_div6':auc7(observations),
        'feature_key':feature_key(x),'training_feature_matches':len(matches),'training_matched_ids':' | '.join(matched_ids)}
    rows.append(row)
    overlap.append(dict(formulation_id=fid,exact_feature_match_count=len(matches),matched_training_ids=' | '.join(matched_ids),original_study_independence='unverified'))
    if c.role.eq('sensitivity_only_explicit_added_water').any():
        wx=vector(c)
        wp=predict.predict_features(bundle,wx[None,:]).iloc[0]
        water_vectors.append(wx)
        water_rows.append(dict(formulation_id=fid,nonwater_score=row['model_score'],water_included_score=float(wp.eye_A_score),
            nonwater_prediction=row['model_prediction'],water_included_prediction=int(wp.eye_A_prediction),
            prediction_changed=row['model_prediction']!=int(wp.eye_A_prediction),
            explicitly_added_water_pct=f['explicit_added_water_pct'],represented_with_water_pct=float(c.pct.sum()),unresolved_stock_pct=f['unresolved_stock_pct'],
            interpretation='Sensitivity only: water absent from all final training components; not a replacement validated representation'))
results=pd.DataFrame(rows); x=np.stack(vectors)
water=pd.DataFrame(water_rows)
save('quat_predictions.csv',results); np.save(OUT/'quat_features.npy',x)
save('water_sensitivity.csv',water); np.save(OUT/'water_sensitivity_features.npy',np.stack(water_vectors))
save('training_feature_overlap.csv',pd.DataFrame(overlap))
dump('quat_feature_row_order.json',results.formulation_id.tolist())

# Exact represented mixtures can recur at distinct concentrations and outcomes.
groups=results.groupby('feature_key',sort=False).agg(
    n_cases=('formulation_id','size'),ids=('formulation_id',lambda s:' | '.join(s)),
    model_score=('model_score','mean'),day7_mean=('day7','mean'),day7_min=('day7','min'),day7_max=('day7','max'),
    auc7_mean=('auc_days1_7_div6','mean'),training_matches=('training_feature_matches','max')).reset_index()
save('represented_input_groups.csv',groups)
stats=[]
for name,df,target,area in [('all_15_related_cases',results,'day7','auc_days1_7_div6'),
                           ('unique_represented_inputs',groups,'day7_mean','auc7_mean'),
                           ('unique_inputs_without_training_feature_match',groups.loc[groups.training_matches==0],'day7_mean','auc7_mean')]:
    stats.append(dict(analysis=name,n=len(df),spearman_day7=spearman(df.model_score,df[target]),
                      spearman_auc_days1_7=spearman(df.model_score,df[area]),
                      interpretation='Descriptive endpoint consistency; not EPA classifier accuracy; no independent-sample p-value'))
statistics=pd.DataFrame(stats);save('rank_association.csv',statistics)

pairrows=[]; indexed=results.set_index('formulation_id')
for p in paired.to_dict('records'):
    a,b=indexed.loc[p['control']],indexed.loc[p['treatment']]
    observed=float(b.day7-a.day7); predicted=float(b.model_score-a.model_score)
    match=(np.sign(observed)==np.sign(predicted)) if abs(predicted)>1e-12 else False
    pairrows.append({**p,'control_day7':float(a.day7),'treatment_day7':float(b.day7),
        'observed_change':observed,'control_model_score':float(a.model_score),'treatment_model_score':float(b.model_score),
        'model_score_change':predicted,'direction_agrees':bool(match),'includes_training_feature_match':bool(a.training_feature_matches or b.training_feature_matches)})
pairs=pd.DataFrame(pairrows);save('prespecified_pair_results.csv',pairs)

# One additional registration family, strictly kept as a partial-composition label-proxy check.
sds=json.loads((ROOT/'inputs/zodiac_source.json').read_text(encoding='utf-8'))
sdsrows=[];sdsvectors=[]
for scenario in ['low','midpoint','high']:
    ingredients=[]
    for c in sds['resolved']:
        value=(c['low']+c['high'])/2 if scenario=='midpoint' else c[scenario]
        ingredients.append(dict(component_id=c['component_id'],pct=value))
    cx=pd.DataFrame(ingredients);vx=vector(cx);sdsvectors.append(vx)
    px=predict.predict_features(bundle,vx[None,:]).iloc[0]
    matches=np.flatnonzero(np.isclose(train_x,vx[None,:],rtol=1e-9,atol=1e-10,equal_nan=True).all(axis=1))
    sdsrows.append(dict(formulation_id=sds['formulation_id'],scenario=scenario,model_score=float(px.eye_A_score),model_prediction=int(px.eye_A_prediction),
        label_proxy=0,label_proxy_category='III/IV',agrees_with_label_proxy=int(px.eye_A_prediction)==0,
        represented_pct=float(cx.pct.sum()),unrepresented_pct=100-float(cx.pct.sum()),training_feature_matches=len(matches),
        strict_validation_eligible=False))
sdsresults=pd.DataFrame(sdsrows);save('additional_shampoo_predictions.csv',sdsresults);np.save(OUT/'additional_shampoo_features.npy',np.stack(sdsvectors))

# Structure novelty, without treating common ingredients as original-study leakage.
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
generator=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
training_structures=link.drop_duplicates('smiles').copy()
training_structures['canonical']=training_structures.smiles.map(lambda s:Chem.MolToSmiles(Chem.MolFromSmiles(s)))
tfps=[generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in training_structures.smiles]
sim=[]
for key,r in reg.iterrows():
    fp=generator.GetFingerprint(Chem.MolFromSmiles(r.smiles));values=DataStructs.BulkTanimotoSimilarity(fp,tfps);nearest=int(np.argmax(values))
    sim.append(dict(component_id=key,cas=r.cas,smiles=r.smiles,exact_structure_in_training=bool(training_structures.canonical.eq(r.smiles).any()),
        maximum_training_tanimoto=float(values[nearest]),nearest_training_cas=training_structures.iloc[nearest]['cas']))
similarity=pd.DataFrame(sim);save('component_training_similarity.csv',similarity)
meta_columns=[c for c in ['formulation_id','formulation_name','source_url','source','record_id'] if c in train]
meta=train[meta_columns].fillna('').astype(str).agg(' | '.join,axis=1)
search_terms=['6583181','4992266','2724-792','RF2066']
metadata_matches=[]
for term in search_terms:
    hits=train.loc[meta.str.contains(term,case=False,regex=False),'formulation_id'].tolist()
    metadata_matches.append(dict(term=term,n_matches=len(hits),matched_ids=' | '.join(hits)))
save('training_metadata_overlap.csv',pd.DataFrame(metadata_matches))

# Verify scores independently from the saved pipeline's transforms and coefficients.
all_x=np.vstack([x,np.stack(water_vectors),np.stack(sdsvectors)])
pipeline=bundle['pipeline']; transformed=all_x.copy()
for _,step in pipeline.steps[:-1]: transformed=step.transform(transformed)
estimator=pipeline.steps[-1][1]
manual_scores=expit(transformed@estimator.coef_[0]+estimator.intercept_[0])
pipeline_scores=predict.predict_features(bundle,all_x).eye_A_score.to_numpy()
assert np.max(np.abs(manual_scores-pipeline_scores))<1e-12
assert all(sha(ROOT/k)==v for k,v in manifest.items())
old_hashes=json.loads((OLD/'artifact_checksums.json').read_text(encoding='utf-8'))
assert all(sha(OLD/k)==v for k,v in old_hashes.items())
assert sha(ROOT/'model/eye_A_LR_wmean.joblib')==protocol['model_sha256']
assert len(results)==15 and len(pairs)==4 and len(water)==7 and len(sdsresults)==3
assert results.epa_category.isna().all() and results.binary_reference.isna().all()
for p in pairrows:
    ca=indexed.loc[p['control']];tr=indexed.loc[p['treatment']]
    assert ca.quat_family==tr.quat_family and ca.quat_active_pct==tr.quat_active_pct
verification=dict(input_and_model_hashes_unchanged=True,original_run_hashes_checked=len(old_hashes),
    manual_logistic_score_max_error=float(np.max(np.abs(manual_scores-pipeline_scores))),
    patent_EPA_labels_not_invented=True,all_prespecified_equal_concentration_pairs_verified=True,
    frozen_threshold=protocol['threshold'],refitted=False)
dump('verification.json',verification)

summary=dict(screened=29,quat_predicted=15,held=13,additional_shampoo=1,strict_EPA_external_validation_cases=0,
    rank_associations=stats,paired_direction_agreement=int(pairs.direction_agrees.sum()),paired_total=len(pairs),
    pairs_without_training_feature_match=int((~pairs.includes_training_feature_match).sum()),
    direction_agreement_without_training_feature_match=int(pairs.loc[~pairs.includes_training_feature_match,'direction_agrees'].sum()),
    exact_training_feature_matched_cases=int((results.training_feature_matches>0).sum()),
    unique_represented_inputs=len(groups),identical_input_groups_with_different_day7=int(((groups.n_cases>1)&(groups.day7_max>groups.day7_min)).sum()),
    water_sensitivity_prediction_changes=int(water.prediction_changed.sum()),water_sensitivity_cases=len(water),
    new_shampoo_midpoint=sdsresults.loc[sdsresults.scenario=='midpoint'].iloc[0].to_dict(),
    threshold=protocol['threshold'],formal_EPA_accuracy=None,formal_EPA_AUC=None,model_refit=False,
    interpretation='Executed external-case diagnostic evaluation; current sources do not support an independent full-formulation EPA accuracy estimate.')
dump('summary.json',summary)

os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
for matched,part in results.groupby(results.training_feature_matches>0):
    axes[0].scatter(part.day7,part.model_score,s=50,alpha=.8,label='학습 입력과 일치' if matched else '새 입력',marker='x' if matched else 'o')
axes[0].set(xlabel='원문 7일 눈자극 총점',ylabel='고정 모델 점수',title='혼합물 15조건: 높고 낮음 비교')
axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
for i,r in enumerate(water.to_dict('records')):
    axes[1].plot([0,1],[r['nonwater_score'],r['water_included_score']],marker='o',alpha=.7,label=r['formulation_id'])
axes[1].set(xticks=[0,1],xticklabels=['물을 제외한 성분','명시된 물도 포함'],ylabel='고정 모델 점수',title='물 입력에 따른 민감도 (7조건)')
axes[1].axhline(bundle['threshold'],color='#b64040',ls='--',label='고정 임계값')
axes[1].grid(alpha=.2)
fig.suptitle('탐색적 외부 사례 평가 — EPA 정확도 검증과 구분',fontsize=13)
fig.tight_layout();fig.savefig(OUT/'external_diagnostics.png',dpi=180);plt.close(fig)

with pd.ExcelWriter(ROOT/'재외부검증_결과.xlsx',engine='openpyxl') as writer:
    for sheet,data in [('혼합물15조건',results),('순위비교',statistics),('배합쌍4개',pairs),('물입력민감도',water),
        ('추가샴푸1제품',sdsresults),('입력중복',groups),('학습중복',pd.DataFrame(overlap)),('성분유사도',similarity),
        ('제외13조건',excluded),('원시험점수',obs),('입력성분',comps),('구조목록',reg.reset_index())]:
        data.to_excel(writer,sheet_name=sheet,index=False)
        ws=writer.sheets[sheet];ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
        for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=min(60,max(13,len(str(col[0].value))+2))

rho=stats[0]['spearman_day7']; novel=stats[-1]['spearman_day7']
mid=summary['new_shampoo_midpoint']
txt=f'''Eye 재외부검증 결과 — 2026-11-09

새 자료 29조건을 검토하여 혼합물 15조건과 추가 샴푸 1제품을 계산했습니다. 13조건은 보류했습니다.
기존 모델과 임계값 {bundle['threshold']:.6f}을 유지했으며 재학습하지 않았습니다.

혼합물 15조건의 모델 점수와 원문 7일 눈자극 총점의 순위 상관: {rho:.3f}.
같은 입력을 묶고 학습 입력과 일치하는 항목을 제외한 {stats[-1]['n']}개 입력의 순위 상관: {novel:.3f}.
상관은 -1~1이며 양수일수록 자극이 높은 조건에 모델 점수도 높아지는 경향입니다. EPA 분류 정확도나 자극 확률을 뜻하지 않습니다.
모델의 학습 목표는 EPA 이진분류이며 자극 총점 예측이 아닙니다. 따라서 순위·변화 방향 비교는 보조 진단으로만 해석해야 합니다.
사전에 선정한 동일 살균성분 농도 배합쌍에서 자극 변화 방향 일치: {summary['paired_direction_agreement']}/{len(pairs)}.
학습 입력과 일치한 사례가 포함된 쌍을 제외하면: {summary['direction_agreement_without_training_feature_match']}/{summary['pairs_without_training_feature_match']}.
15조건 중 학습 데이터와 같은 특징 벡터를 가진 조건: {summary['exact_training_feature_matched_cases']}개.
15조건은 {len(groups)}개의 구분 가능한 모델 입력이 되었습니다. 농도가 달라도 정규화 후 같은 입력이 될 수 있습니다.
물 포함 여부에 따라 분류가 바뀐 조건: {summary['water_sensitivity_prediction_changes']}/{len(water)}. 물 포함은 별도 민감도 계산이며, 학습자료에는 물 성분 입력이 없습니다.

추가 샴푸 Zodiac 2724-792: 중간 함량 점수 {mid['model_score']:.4f}, 판정 {mid['model_prediction']}, 공식 라벨 대리값 0(III/IV).
라벨과 {'일치' if mid['agrees_with_label_proxy'] else '불일치'}했습니다. 입력에 반영한 성분 함량은 {mid['represented_pct']:.2f}%에 불과하며, 한 제품으로 성능을 추정하지 않습니다.

이번 자료로 엄격한 EPA 외부검증 정확도·AUC를 계산하지 않았습니다.
특허에는 동물별 각막·홍채·결막 원점수가 없어 EPA 등급을 확정하지 않았고, 원문의 총점·변화 방향만 비교했습니다.
7일 총점이 남아 있다는 이유만으로 EPA 양성을 부여하지 않았습니다. EPA 판정에는 조직별 양성 기준과 회복 기록이 필요합니다.
혼합물 입력은 투입 전 성분 기준이며, EDTA와 NaOH의 중화·pH·알려지지 않은 원료 용매를 모델이 충분히 반영하지 못합니다.
6개 조건(특허 표4)은 병합된 상업용 원료 열 머리글의 위치를 확정하지 못해 보류했습니다.
샴푸 특허 7배합은 주요 상업용 계면활성제·고분자의 정확한 구조 등을 확인하지 못해 보류했습니다. 점수 원자료는 저장했습니다.
표1 대조군 첫날 점수는 표의 41.7을 사용하고, 본문의 47.1과 다르다는 점을 기록했습니다. 주 분석은 7일값이므로 이 차이에 영향받지 않습니다.
관측하지 않은 날짜의 대시는 0으로 채우지 않았습니다. 원시험 독립성은 식별자 부족으로 확정할 수 없습니다.

보고서: 재외부검증_보고서.html
엑셀: 재외부검증_결과.xlsx
재현: python run_validation.py
모델·입력·이전 결과 해시와 수동 로지스틱 계산 대조 통과.
'''
(ROOT/'결과요약.txt').write_text(txt,encoding='utf-8-sig')
def table(df): return df.to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}')
png=base64.b64encode((OUT/'external_diagnostics.png').read_bytes()).decode()
body='<h1>Eye 재외부검증 결과</h1><p class="lead">새 사례 계산 완료. 자료의 한계로 EPA 분류 정확도는 확정하지 않았습니다.</p>'
body+='<pre>'+html.escape(txt)+'</pre>'
body+=f'<img alt="외부 사례 점수 및 물 입력 민감도" src="data:image/png;base64,{png}">'
body+='<h2>순위 비교</h2>'+table(statistics.drop(columns='interpretation'))
body+='<h2>같은 살균성분 농도에서 배합 변경</h2>'+table(pairs.drop(columns='selection'))
body+='<h2>혼합물별 결과</h2>'+table(results[['formulation_id','quat_family','quat_active_pct','day7','model_score','model_prediction','training_feature_matches','represented_nonwater_pct']])
body+='<h2>추가 샴푸: 라벨 대리값과 비교</h2>'+table(sdsresults)
body+='<h2>보류한 사례</h2>'+table(excluded)
body+='<h2>원문</h2><ul>'
for title,url in [('혼합물 특허','https://patents.google.com/patent/US6583181B1/en'),('샴푸 특허','https://patents.google.com/patent/US4992266A/en'),('추가 샴푸 SDS',sds['sds_url']),('추가 샴푸 EPA 라벨',sds['label_url']),('EPA 조직별 기준: NICEATM 2025 표1','https://ntp.niehs.nih.gov/sites/default/files/2025-01/Phase3_FormulationRpt_FD_Jan2025_508.pdf')]:
    body+=f'<li><a href="{html.escape(url,quote=True)}">{html.escape(title)}</a></li>'
body+='</ul><p><a href="재외부검증_결과.xlsx">엑셀 결과</a> · <a href="protocol.json">분석 전 고정한 기준</a> · <a href="outputs/verification.json">검산 결과</a></p>'
(ROOT/'재외부검증_보고서.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Eye 재외부검증</title><style>body{max-width:1280px;margin:32px auto;padding:0 24px;font-family:system-ui,sans-serif;line-height:1.65;color:#172337}.lead{background:#fff1d6;border-left:5px solid #d18b00;padding:18px}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}td,th{border:1px solid #d4dce5;padding:8px}th{background:#edf2f8}img{max-width:100%}a{color:#0759b5}</style>'+body+'</html>',encoding='utf-8')
pointer=ROOT.parent/'검증결과_안내.txt';previous=pointer.read_text(encoding='utf-8-sig')
marker='[최신 재외부검증: 2026-11-09]'
if not previous.startswith(marker):pointer.write_text(marker+'\nEye_재외부검증_20261109/재외부검증_보고서.html\n혼합물 15조건 + 추가 샴푸 1제품 계산. 엄격한 EPA 정확도 추정은 불가.\n\n'+previous,encoding='utf-8-sig')
checks={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_checksums.json' and '__pycache__' not in p.parts and 'plot_cache' not in p.parts}
(ROOT/'artifact_checksums.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
