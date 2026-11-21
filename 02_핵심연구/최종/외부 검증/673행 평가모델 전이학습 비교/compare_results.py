"""Compare the rerun against the preserved 606-row-evaluation reconstruction."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from common import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parent


def main():
    previous = ROOT / 'previous_result'
    old = joblib.load(previous / 'eye_A_LR_wmean.joblib')
    new = joblib.load(ROOT / 'source_model/eye_A_final.joblib')
    old_data, new_data = np.load(previous / 'features.npz'), np.load(ROOT / 'features.npz')
    for name in old_data.files:
        np.testing.assert_equal(old_data[name], new_data[name])
    X = new_data['X']
    old_protocol, new_protocol = read_json(previous / 'protocol.json'), read_json(ROOT / 'protocol.json')
    fixed = ['mapping_sha256', 'target', 'loss', 'class_weight', 'lambda', 'penalize_intercept',
             'preprocessing', 'grouping', 'evaluation', 'threshold', 'optimizer', 'bootstrap']
    assert all(old_protocol[k] == new_protocol[k] for k in fixed)
    same_preprocessing = True
    for (_, before), (_, after) in zip(old['pipeline'].steps[:-1], new['pipeline'].steps[:-1]):
        for attr in ['statistics_', 'mean_', 'var_', 'scale_', 'n_samples_seen_']:
            if hasattr(before, attr):
                np.testing.assert_equal(getattr(before, attr), getattr(after, attr))
    np.testing.assert_equal(old['pipeline'][:-1].transform(X), new['pipeline'][:-1].transform(X))
    old_est, new_est = old['pipeline'].steps[-1][1], new['pipeline'].steps[-1][1]
    coef_delta = float(np.max(np.abs(old_est.coef_ - new_est.coef_)))
    intercept_delta = float(np.max(np.abs(old_est.intercept_ - new_est.intercept_)))
    old_preds, new_preds = pd.read_csv(previous / 'oof_predictions.csv'), pd.read_csv(ROOT / 'oof_predictions.csv')
    assert old_preds.formulation_id.tolist() == new_preds.formulation_id.tolist()
    methods = ['frozen_source', 'source_centered_transfer', 'zero_centered_control']
    differences = {}
    rows = new_preds[['formulation_id', 'label', 'feature_group']].copy()
    for method in methods:
        diff = new_preds[method + '_score'] - old_preds[method + '_score']
        changed = new_preds[method + '_prediction'] != old_preds[method + '_prediction']
        differences[method] = {'max_abs_score_difference': float(diff.abs().max()),
                               'changed_predictions': int(changed.sum())}
        rows[method + '_previous_score'] = old_preds[method + '_score']
        rows[method + '_all673_score'] = new_preds[method + '_score']
        rows[method + '_difference'] = diff
    rows.to_csv(ROOT / 'row_comparison.csv', index=False, encoding='utf-8-sig')
    before, after = read_json(previous / 'metrics.json'), read_json(ROOT / 'metrics.json')
    report = {
        'source_models': {'previous_sha256': sha256(previous / 'eye_A_LR_wmean.joblib'),
                          'all673_sha256': sha256(ROOT / 'source_model/eye_A_final.joblib'),
                          'previous_evaluation_rows': 606, 'new_evaluation_rows': 673,
                          'both_training_rows': 673},
        'source_thresholds': {'previous': old['threshold'], 'all673': new['threshold']},
        'identical_source_feature_names': old['feature_names'] == new['feature_names'],
        'source_preprocessing_identical': same_preprocessing,
        'source_coefficient_max_abs_difference': coef_delta,
        'source_intercept_max_abs_difference': intercept_delta,
        'target_features_labels_groups_unchanged': True,
        'transfer_protocol_unchanged': True,
        'prediction_differences': differences,
        'previous_metrics': before, 'all673_source_metrics': after,
        'scope': '16-target-row, 15-group OOF reconstruction; source-cohort overlap unresolved; not an independent evaluation of the full-target refit.'}
    write_json(ROOT / 'comparison.json', report)
    current = after['source_centered_transfer']
    note = f'''673행 전체 평가 버전을 기반으로 한 Eye 전이학습 재실행

사용한 기반 모델: source_model/eye_A_final.joblib
원본 경로: C:\\Users\\DS\\COS\\최종\\최최종모델\\eye_A_final.joblib
기반 모델의 고유 평가 대상: 673행. 기반 모델 최종 학습도 673행.
기존 기반 모델: 학습 673행, 과거 평가 대상 606행.

같게 유지한 조건: Gettings 1996의 16제형, 13성분 재구성 매핑,
2,265개 특징, 15그룹 leave-one-group-out, 기존 전처리 고정,
클래스 균형 로지스틱 손실, 기존 계수 중심 L2, lambda=1, 전이 후 임계값=0.5.

실제 재실행 결과: {current['correct']}/{current['n']} ({current['accuracy']:.1%}),
MCC {current['MCC']:.12f}, ROC-AUC {current['ROC_AUC']:.12f},
균형정확도 {current['balanced_accuracy']:.12f}, 혼동행렬 {current['confusion_matrix']}.

기반 모델의 계수 최대 차이: {coef_delta}; 절편 최대 차이: {intercept_delta}.
두 기반 모델의 저장된 전처리 파라미터와 목표 데이터 변환값은 동일하다.
전이학습 OOF 점수 최대 차이: {differences['source_centered_transfer']['max_abs_score_difference']}.
전이학습 OOF 판정이 바뀐 제형: {differences['source_centered_transfer']['changed_predictions']}개.
기반 모델 자체의 저장 임계값: {old['threshold']} -> {new['threshold']}.
전이학습은 양쪽 모두 0.5를 사용하므로 이 임계값 변경이 전이 판정에 영향을 주지 않는다.

이 결과는 목표 16행의 그룹 교차검증이다. 673행 내부 교차검증 성능과 다르며,
원본 그림의 13/16 결과를 복구한 것이 아니다. 원본 성분 매핑은 확보되지 않았다.
16행과 기반 학습 673행의 중복은 확인되지 않았으므로 독립 외부검증으로 해석하지 않는다.

eye_transfer_final.joblib은 교차검증 후 16행 전체로 다시 학습한 예측용 모델이다.
성능 근거는 oof_predictions.csv와 30개 fold_models이며, 전체 재학습 모델의
학습 행 예측 refit_predictions_NOT_validation.csv는 성능 검증에 사용하지 않는다.
계수만 출발점으로 사용하며 최종 학습 알고리즘은 common.py의 objective이다.
저장된 sklearn 분류기의 C 등 파라미터는 원본 컨테이너에서 상속된 값이다.

재실행: python train.py --output-dir 새_출력_폴더
검증: python verify.py
비교: python compare_results.py
예측: python predict.py --input examples/CTFA1996_HZT.json

기존 최종 모델 패키지와 두 이미지 파일은 수정하지 않았다.
'''
    (ROOT / 'README.txt').write_text(note, encoding='utf-8-sig')
    print(json.dumps({k: v for k, v in report.items() if 'metrics' not in k}, ensure_ascii=False, indent=2))
    print('Transfer metrics:', current)


if __name__ == '__main__':
    main()
