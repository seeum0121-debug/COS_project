"""Separate observed evaluation-cohort changes from changed-training-protocol contrasts."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / '모델 실험' / 'eye' / '모델 최종' / 'aggregation_compare_20261109'
NEW = ROOT.parent / '최최종모델'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame, rule):
    y = frame.y_true.to_numpy()
    predicted = frame.prediction.to_numpy() if rule == 'tuned' else (frame.score.to_numpy() >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'rows': len(frame), 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp),
            'MCC': matthews_corrcoef(y, predicted), 'ROC-AUC': roc_auc_score(y, frame.score),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


def main():
    checksums = json.loads((NEW / 'checksums.json').read_text(encoding='utf-8'))
    for name in ('oof_predictions.csv', 'input_snapshot.csv', 'comparison_verification.json'):
        assert digest(NEW / name) == checksums[name]
    data = pd.read_csv(NEW / 'input_snapshot.csv', low_memory=False)
    old = pd.read_csv(OLD / 'oof_predictions.csv')
    old = old[old.label == 'A'].copy()
    new = pd.read_csv(NEW / 'oof_predictions.csv')
    original = np.flatnonzero(data.evaluation_role.eq('existing_evaluation_pool'))
    added = np.flatnonzero(data.evaluation_role.eq('augmentation_train_only'))
    old_identity = json.loads((OLD / 'config.json').read_text(encoding='utf-8'))['identity']
    import epa_harness as u
    for (model, rep, seed, fold), rows in old.groupby(['model', 'representation', 'seed', 'fold']):
        path = OLD / 'results' / 'A' / rep / model / str(seed) / f'fold-{fold}' / 'outer.json'
        result = u.load_record(path, old_identity)
        frame = rows.set_index('row_index').loc[result['test']]
        np.testing.assert_allclose(frame.score, result['scores'], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(frame.prediction, result['predictions'])
        np.testing.assert_array_equal(frame.y_true, result['y_true'])
    conditions = {'old_augmented_eval606': (old, original),
                  'new_protocol_eval606': (new[new.row_index.isin(original)], original),
                  'new_protocol_eval673': (new, np.arange(673)),
                  'new_protocol_added67': (new[new.row_index.isin(added)], added)}
    records = []
    for condition, (frame, expected) in conditions.items():
        for (model, rep, seed), rows in frame.groupby(['model', 'representation', 'seed']):
            assert sorted(rows.row_index) == sorted(expected)
            np.testing.assert_array_equal(rows.y_true, (data.final_label.iloc[rows.row_index].to_numpy() <= 2).astype(int))
            for rule in ('tuned', 'fixed05'):
                records.append({'condition': condition, 'model': model, 'representation': rep,
                                'seed': seed, 'decision_rule': rule, **metrics(rows, rule)})
    results = pd.DataFrame(records)
    results.to_csv(ROOT / 'existing_protocol_metrics.csv', index=False, encoding='utf-8-sig')
    measures = ['sensitivity', 'specificity', 'MCC', 'ROC-AUC']
    summary = results.groupby(['condition', 'model', 'representation', 'decision_rule'])[measures].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(ROOT / 'existing_protocol_summary.csv', index=False, encoding='utf-8-sig')
    deltas = []
    for model, rep, seed, rule in results[['model', 'representation', 'seed', 'decision_rule']].drop_duplicates().itertuples(index=False, name=None):
        subset = results[(results.model == model) & (results.representation == rep) & (results.seed == seed) & (results.decision_rule == rule)].set_index('condition')
        for contrast, a, b in [('changed_protocol_on_same606', 'new_protocol_eval606', 'old_augmented_eval606'),
                               ('evaluation_population_same_predictions', 'new_protocol_eval673', 'new_protocol_eval606')]:
            deltas.append({'contrast': contrast, 'model': model, 'representation': rep, 'seed': seed,
                           'decision_rule': rule, **{m: float(subset.loc[a, m] - subset.loc[b, m]) for m in measures}})
    pd.DataFrame(deltas).to_csv(ROOT / 'existing_protocol_deltas.csv', index=False, encoding='utf-8-sig')
    receipt = {'status': 'passed', 'old_oof_sha256': digest(OLD / 'oof_predictions.csv'),
               'new_oof_sha256': digest(NEW / 'oof_predictions.csv'), 'metric_rows': len(results),
               'interpretation': {
                   'changed_protocol_on_same606': 'Same evaluation identities, but changed splits/training sizes/tuning. Descriptive combined effect, not an isolated causal factor.',
                   'evaluation_population_same_predictions': 'Same new fitted outer-model predictions. Changes only reported population from original606 to all673; nonlinear metrics are not additive per-record averages.',
                   'fixed05': 'Prespecified diagnostic cutoff, not tuned against these held-out outcomes and not a deployment recommendation.',
                   'limitations': 'Repeated seeds are correlated; no independent external performance or confidence interval.'}}
    (ROOT / 'existing_analysis_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Existing-protocol decomposition verified:', len(results), 'metric rows', flush=True)


if __name__ == '__main__':
    main()
