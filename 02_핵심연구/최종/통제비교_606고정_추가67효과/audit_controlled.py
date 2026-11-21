"""Independent reconstruction of fixed-606 evidence; only this audit marks complete."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import importlib.metadata
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import epa_harness as u
import run_controlled as r

ROOT = Path(__file__).resolve().parent


def check(ok, message):
    if not ok:
        raise ValueError(message)


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)


def compare_table(actual, expected, keys):
    check(set(actual.columns) == set(expected.columns), 'Column mismatch')
    check(not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), 'Duplicate table keys')
    a, b = actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index, exact=False)
    for col in b:
        if pd.api.types.is_numeric_dtype(b[col]):
            equal(a[col], b[col])
        else:
            np.testing.assert_array_equal(a[col], b[col])


def verify_config():
    path = ROOT / 'controlled_config.json'
    identity = r.read_json(path)['identity']
    cfg = u.load_record(path, identity)
    check(u.hash_json(cfg) == identity, 'Configuration identity mismatch')
    check(set(cfg['input_hashes']) == set(r.INPUTS) and set(cfg['code_hashes']) == set(r.CODES), 'Incomplete hash manifest')
    for field in ('input_hashes', 'code_hashes'):
        for name, digest in cfg[field].items():
            check(u.file_hash(ROOT / name) == digest, f'Changed artifact: {name}')
    check(cfg['source_path'] == str(r.BASELINE), 'Wrong baseline path')
    check(u.file_hash(r.BASELINE / 'config.json') == cfg['source_config_sha256'], 'Changed source config')
    source = u.load_record(r.BASELINE / 'config.json', cfg['source_identity'])
    check(u.hash_json(source) == cfg['source_identity'], 'Source identity mismatch')
    for package in ('numpy', 'pandas', 'scikit-learn', 'xgboost'):
        check(importlib.metadata.version(package) == source['environment'][package],
              f'Historical/live environment mismatch: {package}')
    check(source['artifact_hashes'] == cfg['source_artifact_hashes'], 'Source artifact manifest mismatch')
    check(source.get('source_hashes', {}) == cfg['source_hashes'], 'Source provenance mismatch')
    for name, digest in source['artifact_hashes'].items():
        check(u.file_hash(r.BASELINE / name) == digest, f'Changed source artifact: {name}')
    for name in (*r.INPUTS, 'epa_harness.py'):
        check(u.file_hash(ROOT / name) == source['artifact_hashes'][name], 'Copied input mismatch')
    check(cfg['baseline_splits'] == source['splits'] and cfg['groups'] == source['groups'], 'Historical boundaries changed')
    check(cfg['grids'] == {m: source['grids'][m] for m in r.MODELS}, 'Historical grids changed')
    expected_paths = set()
    for job in r.jobs():
        path = r.job_path(job, True)
        expected_paths.add(str((path / 'outer.json').relative_to(r.BASELINE)))
        expected_paths.update(str((path / 'candidates' / (c['candidate_id'] + '.json')).relative_to(r.BASELINE)) for c in cfg['grids'][job['model']])
    check(set(cfg['baseline_hashes']) == expected_paths and len(expected_paths) == 1890, 'Incomplete baseline manifest')
    for name, digest in cfg['baseline_hashes'].items():
        check(u.file_hash(r.BASELINE / name) == digest, f'Changed baseline result: {name}')
    check(cfg['environment'] == dict(python=platform.python_version(), platform=platform.platform(),
                                    packages={n: importlib.metadata.version(n) for n in cfg['environment']['packages']}), 'Environment changed')
    return cfg, identity


def audit_splits(cfg, df, y):
    evaluation = set(np.flatnonzero(df.evaluation_role.eq('existing_evaluation_pool')).tolist())
    added = set(np.flatnonzero(df.evaluation_role.eq('augmentation_train_only')).tolist())
    check(len(evaluation) == 606 and len(added) == 67 and len(df) == 673, 'Invalid cohorts')
    check(sorted(evaluation) == cfg['evaluation'], 'Evaluation pool changed')
    check(df.formulation_id.notna().all() and df.formulation_id.is_unique, 'Invalid formulation IDs')
    groups = np.asarray(cfg['groups'])
    check(groups.shape == (673,) and not pd.isna(groups).any(), 'Invalid groups')
    for col in ('comp_group_id', 'split_group_id'):
        check(df[col].notna().all() and df[col].astype(str).str.strip().ne('').all(), 'Missing group identifiers')
        check(pd.DataFrame({'key': df[col], 'group': groups}).groupby('key')['group'].nunique().eq(1).all(), 'Identifier crosses groups')
    for rep in r.REPS:
        X = np.load(ROOT / (rep + '.npy'))
        check(X.shape == (673, 2265) and not np.isinf(X).any(), 'Invalid features')
        values = np.round(np.nan_to_num(X, nan=0.), 6)
        values[values == 0] = 0.
        seen = {}
        for row, group in zip(values, groups):
            key = row.tobytes()
            check(key not in seen or seen[key] == group, 'Feature collision crosses groups')
            seen[key] = group
    check(set(cfg['splits']) == {str(s) for s in r.SEEDS}, 'Unexpected seeds')
    for seed, folds in cfg['splits'].items():
        check(len(folds) == 5 and sorted(i for f in folds for i in f['test']) == sorted(evaluation), 'Outer coverage')
        for new, old in zip(folds, cfg['baseline_splits'][seed]):
            check(new['test'] == old['test'] and new['calibration'] == old['calibration'] == new['train'], 'Not fixed historical partitions')
            check(len(new['inner']) == len(old['inner']) == 3, 'Inner fold count')
            for p, q in zip(new['inner'], old['inner']):
                check(p['validation'] == q['validation'] and p['train'] == [i for i in q['train'] if i in evaluation], 'Inner boundary changed')
            for split, augmentation in ((new, False), (old, True)):
                tr, te, cal = (set(split[k]) for k in ('train', 'test', 'calibration'))
                check(all(len(split[k]) == len(set(split[k])) for k in ('train', 'test', 'calibration')), 'Duplicate split indices')
                check(cal == evaluation - te and te <= evaluation and not tr & te, 'Invalid outer partition')
                allowed = {i for i in added if groups[i] not in set(groups[list(te)])} if augmentation else set()
                check(tr == cal | allowed and not set(groups[list(tr)]) & set(groups[list(te)]), 'Outer training leakage/policy')
                check(sorted(i for p in split['inner'] for i in p['validation']) == sorted(cal), 'Calibration coverage')
                for p in split['inner']:
                    a, b = set(p['train']), set(p['validation'])
                    blocked = set(groups[list(b | te)])
                    allowed = {i for i in added if groups[i] not in blocked} if augmentation else set()
                    check(len(a) == len(p['train']) and len(b) == len(p['validation']), 'Duplicate inner index')
                    check(b <= cal and a == (cal - b) | allowed and not a & b, 'Inner train policy')
                    check(not set(groups[list(a)]) & blocked, 'Inner group leakage')
                    for indices in (a, b):
                        check(set(y[list(indices)]) == {0, 1}, 'Missing inner class')
                for indices in (tr, te, cal):
                    check(set(y[list(indices)]) == {0, 1}, 'Missing outer class')


def threshold(scores, y):
    # Independently enumerate the historical interior-threshold objective.
    scores, y = np.asarray(scores), np.asarray(y)
    values = np.unique(scores)
    check(len(values) > 1 and set(y) == {0, 1}, 'Invalid threshold inputs')
    positive, negative = int(sum(y == 1)), int(sum(y == 0))
    options = []
    for lo, hi in zip(values[:-1], values[1:]):
        midpoint = lo / 2. + hi / 2.
        cutoff = float(midpoint if lo < midpoint < hi else hi)
        pred = scores >= cutoff
        tp, tn = int(sum(pred & (y == 1))), int(sum(~pred & (y == 0)))
        fp, fn = negative - tn, positive - tp
        mcc = (tp * tn - fp * fn) / np.sqrt(float((tp + fp) * positive * negative * (tn + fn)))
        options.append((abs(tp * negative - tn * positive), -mcc, cutoff))
    return min(options)[2]


def audit_records(cfg, identity, df, y):
    rows, settings, frozen = [], [], {}
    matrices = {rep: np.load(ROOT / (rep + '.npy'), mmap_mode='r') for rep in r.REPS}
    for condition in ('augmented', 'noaugmentation'):
        baseline = condition == 'augmented'
        ident = cfg['source_identity'] if baseline else identity
        for job in r.jobs():
            path = r.job_path(job, baseline)
            split = cfg['baseline_splits' if baseline else 'splits'][str(job['seed'])][job['fold']]
            candidates = []
            for c in cfg['grids'][job['model']]:
                cp = path / 'candidates' / (c['candidate_id'] + '.json')
                frozen[str(cp)] = u.file_hash(cp)
                value = u.load_record(cp, ident)
                check(value['candidate'] == c and value['calibration_rows'] == split['calibration'], 'Candidate identity/coverage')
                scores = np.asarray(value['calibration_scores'])
                check(scores.shape == (len(split['calibration']),) and np.isfinite(scores).all()
                      and ((scores >= 0) & (scores <= 1)).all(), 'Invalid inner scores')
                lookup = dict(zip(split['calibration'], scores))
                aucs = [roc_auc_score(y[p['validation']], [lookup[i] for i in p['validation']]) for p in split['inner']]
                equal(value['fold_aucs'], aucs)
                equal(value['mean_auc'], np.mean(aucs))
                candidates.append(value)
            check(len(candidates) == 20, 'Candidate count')
            best = candidates[int(np.argmax([c['mean_auc'] for c in candidates]))]
            cutoff = threshold(best['calibration_scores'], y[split['calibration']])
            op = path / 'outer.json'
            frozen[str(op)] = u.file_hash(op)
            result = u.load_record(op, ident)
            check(all(result[k] == v for k, v in job.items()), 'Outer job mismatch')
            check(not baseline or result['label'] == 'A', 'Baseline EPA arm mismatch')
            check(result['test'] == split['test'] and result['train_rows'] == split['train']
                  and result['calibration_rows'] == split['calibration'], 'Outer coverage')
            check(result['selected_candidate'] == best['candidate'] and result['candidates_evaluated'] == 20, 'Wrong winner')
            check(result['threshold'] == cutoff, 'Wrong inner-only threshold')
            scores = np.asarray(result['scores'])
            check(scores.shape == (len(split['test']),) and np.isfinite(scores).all()
                  and ((scores >= 0) & (scores <= 1)).all(), 'Invalid heldout scores')
            np.testing.assert_array_equal(result['y_true'], y[split['test']])
            np.testing.assert_array_equal(result['predictions'], scores >= cutoff)
            if not baseline:
                ep = path / 'estimator.joblib'
                frozen[str(ep)] = u.file_hash(ep)
                check(frozen[str(ep)] == result['estimator_sha256'], 'Estimator checksum mismatch')
                est = joblib.load(ep)
                X = matrices[job['representation']]
                tr, te = split['train'], split['test']
                check(list(est.classes_) == [0, 1], 'Class orientation')
                params = est.named_steps['model'].get_params()
                check(all(params[k] == v for k, v in best['candidate']['params'].items()), 'Estimator settings differ')
                check(params['random_state'] == 42, 'Estimator random seed changed')
                imputer = est.named_steps['imputer']
                check(imputer.strategy == 'constant' and imputer.fill_value == 0 and imputer.keep_empty_features, 'Imputation policy changed')
                equal(imputer.statistics_, np.zeros(X.shape[1]))
                if job['model'] == 'lr':
                    scaler = est.named_steps['scaler']
                    train = np.nan_to_num(X[tr], nan=0.)
                    np.testing.assert_allclose(scaler.mean_, train.mean(axis=0), rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(scaler.var_, train.var(axis=0), rtol=1e-12, atol=1e-10)
                    check(int(scaler.n_samples_seen_) == len(tr), 'Scaler sample count')
                else:
                    check('scaler' not in est.named_steps, 'Unexpected tree scaler')
                with threadpool_limits(limits=1):
                    equal(est.predict_proba(X[te])[:, 1], scores)
            settings.append(dict(condition=condition, **job, threshold=cutoff,
                                 candidate_id=best['candidate']['candidate_id'], params=json.dumps(best['candidate']['params'], sort_keys=True)))
            for i, score in zip(split['test'], scores):
                rows.append(dict(condition=condition, **job, row_index=i, source_truth=int(df.final_label.iloc[i]),
                                 y_true=int(y[i]), score=float(score), tuned_threshold=cutoff,
                                 tuned_prediction=int(score >= cutoff), fixed05_prediction=int(score >= .5)))
    return pd.DataFrame(rows), pd.DataFrame(settings), frozen


def summary(frame, keys, columns):
    grouped = frame.groupby(keys)[columns]
    means, stds = grouped.mean(), grouped.std(ddof=1)
    return pd.concat([means.add_suffix('_mean'), stds.add_suffix('_std')], axis=1).reset_index()


def audit_tables(oof, settings, cfg):
    check(len(oof) == 21816, 'OOF count')
    for _, block in oof.groupby(r.KEYS):
        check(sorted(block.row_index.tolist()) == cfg['evaluation'], 'OOF coverage')
    expected = {'oof_predictions.csv': (oof, r.KEYS + ['fold', 'row_index']),
                'selected_settings.csv': (settings, r.KEYS + ['fold'])}
    for suffix, keys in [('seed', r.KEYS), ('fold', r.KEYS + ['fold'])]:
        records = []
        for key, block in oof.groupby(keys):
            for rule, column in [('tuned', 'tuned_prediction'), ('fixed0.5', 'fixed05_prediction')]:
                records.append(dict(zip(keys, key), rule=rule, rows=len(block), **r.metrics(block.y_true, block.score, block[column])))
        expected['metrics_by_' + suffix + '.csv'] = (pd.DataFrame(records), keys + ['rule'])
    seeds = expected['metrics_by_seed.csv'][0]
    groupkeys = r.KEYS[:-1] + ['rule']
    expected['metrics_summary.csv'] = (summary(seeds, groupkeys, r.METRICS), groupkeys)
    pairkeys = ['representation', 'model', 'seed', 'rule']
    augmented = seeds[seeds.condition.eq('augmented')].set_index(pairkeys)
    noadd = seeds[seeds.condition.eq('noaugmentation')].set_index(pairkeys)
    check(set(augmented.index) == set(noadd.index), 'Unpaired seeds')
    deltas = augmented[r.METRICS].subtract(noadd[r.METRICS]).reset_index()
    expected['paired_deltas.csv'] = (deltas, pairkeys)
    pairgroups = ['representation', 'model', 'rule']
    expected['paired_delta_summary.csv'] = (summary(deltas, pairgroups, r.METRICS), pairgroups)
    expected['threshold_summary.csv'] = (summary(settings, r.KEYS[:-1], ['threshold']), r.KEYS[:-1])
    distribution_keys = r.KEYS[:-1] + ['candidate_id', 'params']
    counts = settings.groupby(distribution_keys).size().rename('fold_count').reset_index()
    expected['selected_settings_distribution.csv'] = (counts, distribution_keys)
    for name, (frame, keys) in expected.items():
        compare_table(pd.read_csv(ROOT / name), frame, keys)
    # Text is generated from independently reconstructed tables, not trusted CSV metrics.
    check((ROOT / 'RESULTS.txt').read_text(encoding='utf-8') == r.results_text({k: v[0] for k, v in expected.items()}), 'Summary text mismatch')
    return list(expected) + ['RESULTS.txt']


def main():
    with r.exclusive_run():
        identity = None
        try:
            cfg, identity = verify_config()
            outputs = ['controlled_config.json', 'oof_predictions.csv', 'selected_settings.csv',
                       'selected_settings_distribution.csv', 'threshold_summary.csv', 'metrics_by_seed.csv',
                       'metrics_by_fold.csv', 'metrics_summary.csv', 'paired_deltas.csv', 'paired_delta_summary.csv', 'RESULTS.txt']
            initial = {name: u.file_hash(ROOT / name) for name in outputs}
            df = pd.read_csv(ROOT / 'input_snapshot.csv', low_memory=False)
            labels = pd.to_numeric(df.final_label, errors='raise').to_numpy()
            check(np.isfinite(labels).all() and set(labels) <= {1, 2, 3, 4}, 'Invalid source labels')
            y = (labels <= 2).astype(int)
            audit_splits(cfg, df, y)
            oof, settings, frozen = audit_records(cfg, identity, df, y)
            audit_tables(oof, settings, cfg)
            check(verify_config()[1] == identity, 'Config changed during audit')
            check(initial == {name: u.file_hash(ROOT / name) for name in outputs}, 'Reports changed during audit')
            check(all(u.file_hash(path) == digest for path, digest in frozen.items()), 'Evidence changed during audit')
            u.atomic_json(ROOT / 'audit.json', dict(status='passed', identity=identity, output_hashes=initial,
                          record_hashes=frozen, baseline_outer=90, baseline_candidates=1800,
                          noadd_outer=90, noadd_candidates=1800, oof_rows=21816,
                          historical_model_reconstruction=False, noadd_model_reconstruction=True,
                          scope=r.SCOPE))
            r.status(identity, 'complete', 90, audit_sha256=u.file_hash(ROOT / 'audit.json'))
            print('CONTROLLED AUDIT PASSED: 90 paired outer folds; 21816 OOF predictions', flush=True)
        except BaseException as exc:
            if identity is not None:
                r.status(identity, 'audit_failed', 90, error=repr(exc))
            raise


if __name__ == '__main__':
    main()
