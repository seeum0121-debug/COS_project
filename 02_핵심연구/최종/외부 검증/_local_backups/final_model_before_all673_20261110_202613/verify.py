"""Check provenance, feature compatibility, held-out predictions and final inference."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
from threadpoolctl import threadpool_limits

from common import FORMAT, read_json, write_json, sha256, vectorize, feature_group, objective, score_bundle

ROOT = Path(__file__).resolve().parent


def main():
    checks = read_json(ROOT/'checksums.json')
    for name, expected in checks['files'].items():
        path = (ROOT/name).resolve()
        assert path.is_relative_to(ROOT) and sha256(path) == expected, name
    for name, info in read_json(ROOT/'source_manifest.json').items():
        assert sha256(ROOT/name) == info['sha256'], name
    protocol = read_json(ROOT/'protocol.json')
    assert sha256(ROOT/'mapping.json') == protocol['mapping_sha256']
    assert sha256(ROOT/'source_model/eye_A_LR_wmean.joblib') == protocol['source_model_sha256']
    source = joblib.load(ROOT/'source_model/eye_A_LR_wmean.joblib')
    model = joblib.load(ROOT/'eye_transfer_final.joblib')
    assert model['format'] == FORMAT and model['threshold'] == .5
    assert model['metadata'] == read_json(ROOT/'metadata.json')
    assert model['mapping'] == read_json(ROOT/'mapping.json')
    assert model['feature_names'] == source['feature_names'] == read_json(ROOT/'feature_names.json')
    assert model['metadata']['original_model_recovered'] is False
    assert model['metadata']['role'] == 'all_16_target_rows_refit_for_prediction'
    assert model['metadata']['mapping_sha256'] == protocol['mapping_sha256']
    assert model['metadata']['protocol_sha256'] == sha256(ROOT/'protocol.json')

    data = np.load(ROOT/'features.npz')
    X, y, groups = data['X'], data['y'], data['groups']
    ids = data['formulation_ids'].tolist()
    assert X.shape == (16, 2265) and np.bincount(y).tolist() == [5, 11]
    assert len(set(groups)) == 15
    payloads = read_json(ROOT/'data/payloads.json')
    assert ids == [p['formulation_id'] for p in payloads]
    old_spec = importlib.util.spec_from_file_location('original_eye_predict', ROOT/'source_model/predict.py')
    original = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(original)
    mapping = model['mapping']
    by_id = {entry['id']: entry for entry in mapping['entries']}
    max_feature_error = 0.
    for i, payload in enumerate(payloads):
        features, audit = vectorize(payload, model['feature_names'], mapping)
        np.testing.assert_allclose(features, X[i], rtol=0, atol=1e-12, equal_nan=True)
        selected = [row for row in audit['component_audit'] if row['mapped_id'] and row['amount_pct'] > 0]
        original_features = original.vectorize_components(
            model['feature_names'], [by_id[row['mapped_id']]['canonical_smiles'] for row in selected],
            [row['amount_pct'] for row in selected]).to_numpy()[0]
        np.testing.assert_allclose(features, original_features, rtol=0, atol=1e-12, equal_nan=True)
        max_feature_error = max(max_feature_error, float(np.nanmax(np.abs(features-original_features))))
        assert feature_group(features) == groups[i]
    np.testing.assert_array_equal(model['pipeline'][:-1].transform(X), source['pipeline'][:-1].transform(X))
    for new_step, old_step in zip(model['pipeline'].steps[:-1], source['pipeline'].steps[:-1]):
        assert new_step[0] == old_step[0]
        for attr in ['statistics_', 'mean_', 'var_', 'scale_', 'n_samples_seen_']:
            if hasattr(old_step[1], attr):
                np.testing.assert_array_equal(getattr(new_step[1], attr), getattr(old_step[1], attr))

    predictions = pd.read_csv(ROOT/'oof_predictions.csv')
    assert predictions.formulation_id.tolist() == ids
    assert predictions.feature_group.tolist() == groups.tolist()
    np.testing.assert_array_equal(predictions.label, y)
    scores = read_json(ROOT/'metrics.json')
    folds = read_json(ROOT/'fold_log.json')
    exposure = {method: np.zeros(len(y), dtype=int) for method in
                ['source_centered_transfer', 'zero_centered_control']}
    for fold in folds:
        train = np.array([ids.index(fid) for fid in fold['train_ids']])
        test = np.array([ids.index(fid) for fid in fold['test_ids']])
        assert set(train) | set(test) == set(range(len(y)))
        assert not set(train) & set(test) and not set(groups[train]) & set(groups[test])
        assert len(set(groups[test])) == 1
        bundle = joblib.load(ROOT/fold['model_file'])
        assert bundle['metadata']['train_ids'] == fold['train_ids']
        assert bundle['metadata']['test_ids'] == fold['test_ids']
        actual = bundle['pipeline'].predict_proba(X[test])[:, 1]
        expected = predictions[fold['method']+'_score'].to_numpy()[test]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        exposure[fold['method']][test] += 1
    assert len(folds) == 30
    for method, count in exposure.items():
        np.testing.assert_array_equal(count, np.ones(len(y)))
        values = predictions[method+'_score'].to_numpy()
        decision = (values >= .5).astype(int)
        np.testing.assert_array_equal(predictions[method+'_prediction'], decision)
        assert confusion_matrix(y, decision, labels=[0, 1]).tolist() == scores[method]['confusion_matrix']
        np.testing.assert_allclose(matthews_corrcoef(y, decision), scores[method]['MCC'], atol=1e-14)
        np.testing.assert_allclose(roc_auc_score(y, values), scores[method]['ROC_AUC'], atol=1e-14)
    frozen = source['pipeline'].predict_proba(X)[:, 1]
    np.testing.assert_allclose(frozen, predictions.frozen_source_score, rtol=0, atol=1e-12)
    assert model['metadata']['oof_evaluation'] == scores['source_centered_transfer']

    # Independently check the analytic derivative, including the penalized intercept.
    rng = np.random.default_rng(123)
    design = np.c_[rng.normal(size=(7, 4)), np.ones(7)]
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    theta, anchor = rng.normal(size=5), rng.normal(size=5)
    analytic = objective(theta, design, labels, anchor)[1]
    numeric = []
    for i in range(5):
        delta = np.zeros(5)
        delta[i] = 1e-5
        numeric.append((objective(theta+delta, design, labels, anchor)[0] -
                        objective(theta-delta, design, labels, anchor)[0])/(2e-5))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-7)

    final_estimator = model['pipeline'].steps[-1][1]
    source_estimator = source['pipeline'].steps[-1][1]
    theta = np.r_[final_estimator.coef_.ravel(), final_estimator.intercept_]
    anchor = np.r_[source_estimator.coef_.ravel(), source_estimator.intercept_]
    design = np.c_[source['pipeline'][:-1].transform(X), np.ones(len(y))]
    loss, gradient = objective(theta, design, y, anchor, protocol['lambda'])
    assert np.max(np.abs(gradient)) < 1e-5
    assert loss < objective(anchor, design, y, anchor, protocol['lambda'])[0]
    np.testing.assert_allclose(expit(design @ theta), model['pipeline'].predict_proba(X)[:, 1], atol=1e-13)
    refit = pd.read_csv(ROOT/'refit_predictions_NOT_validation.csv')
    np.testing.assert_allclose(refit.score, model['pipeline'].predict_proba(X)[:, 1], atol=1e-12)
    invalid = [
        {'components': []},
        {'components': [{'name': 'water'}]},
        {'components': [{'name': 'water', 'amount_pct': True}]},
        {'components': [{'name': 'water', 'amount_pct': -1}]},
        {'components': [{'name': 'water', 'amount_pct': float('nan')}]},
        {'components': [{'name': 'water', 'amount_pct': 0}]},
        {'components': [{'name': 'water', 'amount_pct': 80}, {'name': 'glycerin', 'amount_pct': 40}]},
        {'components': [{'name': 'unmapped', 'amount_pct': 100}]},
    ]
    for payload in invalid:
        try:
            score_bundle(model, payload)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid input accepted: {payload}')
    example = ROOT/'examples/CTFA1996_HZT.json'
    expected = score_bundle(model, read_json(example))
    with tempfile.TemporaryDirectory(prefix='eye_transfer_cli_') as temporary:
        result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'predict.py'), '--input', str(example)],
                                cwd=temporary, check=True, capture_output=True, encoding='utf-8', timeout=120)
        actual = json.loads(result.stdout)
        assert actual == expected
    summary = {'status': 'passed', 'checksummed_files': len(checks['files']),
               'source_feature_recipe_max_abs_error': max_feature_error,
               'source_preprocessing_unchanged': True, 'held_out_fold_models_checked': len(folds),
               'each_row_predicted_once_per_method': True, 'held_out_group_leakage': False,
               'metrics_recalculated': True, 'objective_gradient_finite_difference_checked': True,
               'final_fit_gradient_max_abs': float(np.max(np.abs(gradient))),
               'invalid_inputs_rejected': len(invalid), 'cli_from_other_directory': True,
               'original_model_recovered': False,
               'scope': 'software and reconstruction checks; no independent external validation'}
    write_json(ROOT/'verification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
