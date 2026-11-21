"""Explain the negative score correlation without tuning the frozen model."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('locked',ROOT/'model/predict.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
b=p.load_bundle(ROOT/'model/eye_A_LR_wmean.joblib')
model_hash=hashlib.sha256((ROOT/'model/eye_A_LR_wmean.joblib').read_bytes()).hexdigest()
reg=pd.read_csv(ROOT/'inputs/structure_registry.csv').set_index('component_id')
res=pd.read_csv(ROOT/'outputs/quat_predictions.csv')
x=np.load(ROOT/'outputs/quat_features.npy')
base_idx=int(np.flatnonzero(res.formulation_id=='Q_T1_C2')[0])
treat_idx=int(np.flatnonzero(res.formulation_id=='Q_T1_C1')[0])
base=x[base_idx]
extra=[]
for key,mass in [('EDTA',.9),('NaOH',.33)]:
    v=p.vectorize_components(b['feature_names'],[reg.loc[key,'smiles']]).to_numpy()[0]
    extra.append((key,mass,v,np.isfinite(v)))
assert np.isfinite(base).all()
denominator=np.full(len(base),1.2)
for key,mass,v,finite in extra:denominator+=mass*finite
pipeline=b['pipeline']
scaler=pipeline.steps[-2][1];lr=pipeline.steps[-1][1]
beta=lr.coef_[0]/scaler.scale_
contributions=[];reconstruction=np.zeros(len(base))
for key,mass,v,finite in extra:
    delta=np.where(finite,mass*(np.nan_to_num(v)-base)/denominator,0)
    reconstruction+=delta
    contributions.append(dict(component=key,actual_wt_pct=mass,nominal_normalized_weight=mass/2.43,
        logit_change_contribution=float(delta@beta),undefined_component_descriptors=int((~finite).sum())))
assert np.allclose(reconstruction,x[treat_idx]-base,rtol=1e-10,atol=1e-10)
actual=float(pipeline.decision_function(x[[treat_idx]])[0]-pipeline.decision_function(x[[base_idx]])[0])
assert abs(sum(c['logit_change_contribution'] for c in contributions)-actual)<1e-12
rho=float(res.model_score.corr(res.day7,method='spearman'))
cutoffs=[dict(threshold=t,n_positive=int((res.model_score>=t).sum()),score_spearman=rho) for t in [.3,b['threshold'],.5,.6,.7]]
train=pd.read_csv(ROOT.parent/'Eye_cosmetic_validation_20261109/training_reference/input_snapshot.csv',low_memory=False)
label_mapping=pd.crosstab(train.epa_class,train.binary_label)
assert label_mapping.loc[[1,2],0].sum()==0 and label_mapping.loc[[3,4],1].sum()==0
assert pipeline.classes_.tolist()==[0,1]
result=dict(
    correlation_is_prethreshold=True,score_spearman=rho,threshold_scenarios=cutoffs,
    positive_score_column_verified=True,positive_target='EPA I/II',negative_target='EPA III/IV',
    control_score=float(res.iloc[base_idx].model_score),treated_score=float(res.iloc[treat_idx].model_score),
    control_day7=float(res.iloc[base_idx].day7),treated_day7=float(res.iloc[treat_idx].day7),
    actual_logit_change=actual,exact_computational_contributions=contributions,
    interpretation='Contributions explain this frozen computation, not chemical toxicity or causal biological mechanisms.',
    caveats=['EPA binary classification and day-7 composite severity are different targets.',
        'Nominal as-added ingredients do not represent neutralized species or final pH.',
        'Water omission and relative normalization remove absolute dilution information.',
        'These external cases cannot establish an optimal EPA cutoff without reliable binary labels.'],
    model_sha256=model_hash,model_or_threshold_changed=False)
(HERE/'threshold_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
pd.DataFrame(cutoffs).to_csv(HERE/'threshold_comparison.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(contributions).to_csv(HERE/'component_contributions.csv',index=False,encoding='utf-8-sig')
note='''임계값 진단

−0.57은 양성·음성 판정 전 모델 점수와 원문 7일 자극 총점의 순위 상관입니다. 임계값을 바꿔도 점수와 순서는 바뀌지 않습니다.
임계값 0.30, 0.481487, 0.50, 0.60, 0.70에서 양성 수는 달라지지만 점수의 순위 상관은 같습니다.
학습자료의 EPA I/II=1, III/IV=0 대응과 predict_proba 양성 열은 일치했습니다.

예시 배합에서 EDTA 0.9%와 NaOH 0.33%가 추가됩니다. 모델은 물 등을 제외한 2.43%의 성분끼리 다시 비율을 계산합니다.
그 결과 EDTA는 평균 분자 특성에서 약 37.0%, NaOH는 약 13.6% 비중을 차지합니다. 두 성분이 점수를 올리는 수학적 기여를 계산했습니다.
이 계산은 해당 모델이 이 입력에 어떻게 반응하는지에 대한 설명이며, EDTA 또는 NaOH의 실제 자극 위험을 추정한 값이 아닙니다.
모델에는 최종 pH·중화 후 화학종·절대 희석 농도를 별도로 표현하는 입력이 없습니다. 이 한계와 학습에서 익숙하지 않은 성분 때문에 배합 효과를 놓칠 가능성이 있습니다.

원래 학습 목표는 EPA 두 범주 분류이며, 7일 자극 총점 예측이 아닙니다. 이 보조 지표만으로 EPA 정확도가 역전되었다고 판단해서는 안 됩니다.
현재 임계값이 이 분야에서도 최적이라는 뜻은 아닙니다. 적합성은 확실한 EPA 정답이 있는 별도 검증 자료로 판단해야 합니다.
이번 자료에 맞추어 임계값을 고르지 않았으며, 모델·임계값·기존 예측 파일을 수정하지 않았습니다.
'''
(HERE/'임계값_원인진단.txt').write_text(note,encoding='utf-8-sig')
assert model_hash==hashlib.sha256((ROOT/'model/eye_A_LR_wmean.joblib').read_bytes()).hexdigest()
print(json.dumps(result,ensure_ascii=False,indent=2))
