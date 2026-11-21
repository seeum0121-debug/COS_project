"""Fixed historical 606-row folds: historical augmentation versus fresh no-add fits."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score
import epa_harness as u

ROOT = Path(__file__).resolve().parent
BASELINE = Path('C:/Users/DS/COS/최종/모델 실험/eye/모델 최종/aggregation_compare_20261109')
SEEDS = (42, 202, 777)
MODELS = ('lr', 'rf', 'xgb')
REPS = ('wmean', 'max')
INPUTS = ('input_snapshot.csv', 'wmean.npy', 'max.npy')
CODES = ('run_controlled.py', 'audit_controlled.py', 'test_controlled.py', 'epa_harness.py')
METRICS = ['sensitivity', 'specificity', 'MCC', 'ROC-AUC', 'gap', 'TN', 'FP', 'FN', 'TP']
KEYS = ['condition', 'representation', 'model', 'seed']
SCOPE = ('기존 606행과 과거 outer5/inner3 분할을 고정한 추가 데이터 정책 비교입니다. '
         '튜닝 및 학습 가중치의 반응을 포함하며, 새 673행 분할 효과를 분리하지 않습니다. '
         '고정 0.5는 진단용이며 새로 최적화한 임계값이 아닙니다. 외부 검증이 아니며 '
         '상관된 3개 seed의 표준편차는 기술통계이지 독립 표본 신뢰구간이 아닙니다. '
         '과거 augmentation 결과는 모델 바이너리 없이 재사용한 역사적 산출물입니다. 최종 모델 교체/재학습은 없습니다.')
CACHE = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_historical_environment(environment):
    for package in ('numpy', 'pandas', 'scikit-learn', 'xgboost'):
        historical = environment[package]
        live = importlib.metadata.version(package)
        require(live == historical,
                f'Historical/live environment mismatch: {package}: {historical} != {live}; augmentation-only comparison refused')


def jobs():
    return [dict(representation=r, model=m, seed=s, fold=f)
            for s in SEEDS for f in range(5) for m in ('rf', 'lr', 'xgb') for r in REPS]


def job_path(job, baseline=False):
    base = BASELINE / 'results' / 'A' if baseline else ROOT / 'results' / 'noaugmentation'
    return base / job['representation'] / job['model'] / str(job['seed']) / f"fold-{job['fold']}"


def validate_splits(splits, evaluation, groups, y, augmented):
    evaluation = set(evaluation)
    universe = set(range(len(y)))
    added = universe - evaluation
    require(set(splits) == {str(s) for s in SEEDS}, 'Unexpected seeds')
    for folds in splits.values():
        require(len(folds) == 5, 'Expected five outer folds')
        require(sorted(i for f in folds for i in f['test']) == sorted(evaluation), 'Outer coverage')
        for f in folds:
            for name in ('train', 'test', 'calibration'):
                require(len(f[name]) == len(set(f[name])) and set(f[name]) <= universe, 'Invalid row indices')
            tr, te, cal = (set(f[k]) for k in ('train', 'test', 'calibration'))
            require(cal == evaluation - te and not tr & te, 'Outer calibration partition')
            expected = cal | {i for i in added if groups[i] not in set(groups[list(te)])} if augmented else cal
            require(tr == expected, 'Outer training policy mismatch')
            require(not set(groups[list(tr)]) & set(groups[list(te)]), 'Outer group leakage')
            require(len(f['inner']) == 3, 'Expected three inner folds')
            require(sorted(i for p in f['inner'] for i in p['validation']) == sorted(cal), 'Inner coverage')
            for p in f['inner']:
                a, b = set(p['train']), set(p['validation'])
                require(len(a) == len(p['train']) and len(b) == len(p['validation']), 'Duplicate inner rows')
                blocked = set(groups[list(b | te)])
                expected = (cal - b) | {i for i in added if groups[i] not in blocked} if augmented else cal - b
                require(a == expected and b <= cal and not a & b, 'Inner training policy mismatch')
                require(not set(groups[list(a)]) & blocked, 'Inner group leakage')
                for rows in (a, b):
                    require(set(y[list(rows)]) == {0, 1}, 'Inner requires both classes')
            for rows in (tr, te, cal):
                require(set(y[list(rows)]) == {0, 1}, 'Outer requires both classes')


def noadd_splits(source, evaluation):
    keep = set(evaluation)
    return {seed: [dict(train=f['calibration'].copy(), test=f['test'].copy(),
                        calibration=f['calibration'].copy(),
                        inner=[dict(train=[i for i in p['train'] if i in keep],
                                    validation=p['validation'].copy()) for p in f['inner']])
                   for f in folds] for seed, folds in source.items()}


def candidate_record(path, identity, candidate, split, y):
    value = u.load_record(path, identity)
    require(value['candidate'] == candidate, f'Candidate mismatch: {path}')
    require(value['calibration_rows'] == split['calibration'], 'Calibration row/order mismatch')
    scores = np.asarray(value['calibration_scores'], dtype=float)
    require(scores.shape == (len(split['calibration']),) and np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all(), 'Invalid candidate scores')
    lookup = dict(zip(split['calibration'], scores))
    aucs = [float(roc_auc_score(y[p['validation']], [lookup[i] for i in p['validation']])) for p in split['inner']]
    np.testing.assert_allclose(value['fold_aucs'], aucs, rtol=0, atol=1e-12)
    np.testing.assert_allclose(value['mean_auc'], np.mean(aucs), rtol=0, atol=1e-12)
    return value


def winner(candidates):
    # Python max keeps the first exact tie, preserving the source grid order.
    return max(candidates, key=lambda c: c['mean_auc'])


def validate_outer(job, path, identity, split, grid, y, baseline=False):
    candidates = [candidate_record(path / 'candidates' / (c['candidate_id'] + '.json'), identity, c, split, y) for c in grid]
    best = winner(candidates)
    result = u.load_record(path / 'outer.json', identity)
    for key, value in job.items():
        require(result[key] == value, 'Job mismatch')
    if baseline:
        require(result['label'] == 'A', 'Baseline label mismatch')
    for key, expected in [('test', split['test']), ('train_rows', split['train']), ('calibration_rows', split['calibration']),
                          ('selected_candidate', best['candidate']), ('candidates_evaluated', 20)]:
        require(result[key] == expected, f'Outer mismatch: {key}')
    cutoff = u.threshold(best['calibration_scores'], y[split['calibration']])
    require(result['threshold'] == cutoff, 'Threshold mismatch')
    scores = np.asarray(result['scores'])
    require(scores.shape == (len(split['test']),) and np.isfinite(scores).all()
            and ((scores >= 0) & (scores <= 1)).all(), 'Invalid outer scores')
    np.testing.assert_array_equal(result['y_true'], y[split['test']])
    np.testing.assert_array_equal(result['predictions'], scores >= cutoff)
    if not baseline:
        require(u.file_hash(path / 'estimator.joblib') == result['estimator_sha256'], 'Estimator changed')
    return result


def prepare():
    source_path = BASELINE / 'config.json'
    sid = read_json(source_path)['identity']
    source = u.load_record(source_path, sid)
    require(u.hash_json(source) == sid, 'Source identity mismatch')
    for name, digest in source['artifact_hashes'].items():
        require(u.file_hash(BASELINE / name) == digest, f'Source artifact changed: {name}')
    for name in (*INPUTS, 'epa_harness.py'):
        require(u.file_hash(ROOT / name) == source['artifact_hashes'][name], f'Copied artifact changed: {name}')
    df = pd.read_csv(ROOT / 'input_snapshot.csv', low_memory=False)
    require(len(df) == 673 and df.formulation_id.notna().all() and df.formulation_id.is_unique, 'Expected 673 unique rows')
    require(df.evaluation_role.value_counts().to_dict() == {'existing_evaluation_pool': 606, 'augmentation_train_only': 67}, 'Cohort mismatch')
    evaluation = np.flatnonzero(df.evaluation_role.eq('existing_evaluation_pool')).tolist()
    y = u.epa_labels(df.final_label)['A']
    groups = np.asarray(source['groups'])
    require(groups.shape == (673,) and not pd.isna(groups).any(), 'Invalid groups')
    for rep in REPS:
        X = np.load(ROOT / (rep + '.npy'), mmap_mode='r')
        require(X.shape == (673, 2265) and not np.isinf(X).any(), 'Invalid feature matrix')
    validate_splits(source['splits'], evaluation, groups, y, True)
    splits = noadd_splits(source['splits'], evaluation)
    validate_splits(splits, evaluation, groups, y, False)
    grids = {m: source['grids'][m] for m in MODELS}
    verify_historical_environment(source['environment'])
    for grid in grids.values():
        require(len(grid) == 20 and len({c['candidate_id'] for c in grid}) == 20, 'Invalid grid')
        require(all(Path(c['candidate_id']).name == c['candidate_id'] for c in grid), 'Unsafe candidate identifier')
    hashes = {}
    for job in jobs():
        path = job_path(job, True)
        paths = [path / 'outer.json', *[path / 'candidates' / (c['candidate_id'] + '.json') for c in grids[job['model']]]]
        before = {str(p.relative_to(BASELINE)): u.file_hash(p) for p in paths}
        validate_outer(job, path, sid, source['splits'][str(job['seed'])][job['fold']], grids[job['model']], y, True)
        require(before == {str(p.relative_to(BASELINE)): u.file_hash(p) for p in paths}, 'Baseline changed during validation')
        hashes.update(before)
    cfg = dict(protocol='fixed606-augmentation-policy-v1', source_identity=sid,
               source_config_sha256=u.file_hash(source_path), source_path=str(BASELINE),
               source_hashes=source.get('source_hashes', {}), source_artifact_hashes=source['artifact_hashes'],
               baseline_hashes=hashes, baseline_splits=source['splits'], splits=splits,
               groups=source['groups'], evaluation=evaluation, grids=grids,
               input_hashes={n: u.file_hash(ROOT / n) for n in INPUTS},
               code_hashes={n: u.file_hash(ROOT / n) for n in CODES},
               environment=dict(python=platform.python_version(), platform=platform.platform(),
                                packages={n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib')}),
               baseline_provenance='Reused audited historical artifacts; estimator binaries were not saved.',
               scope=SCOPE, outer_jobs=90, candidate_count=20, estimator_seed=42)
    identity = u.hash_json(cfg)
    u.save_record(ROOT / 'controlled_config.json', identity, cfg)
    return cfg, identity


def cached_data(identity):
    global CACHE
    if CACHE is None or CACHE[0] != identity:
        cfg = u.load_record(ROOT / 'controlled_config.json', identity)
        require(u.hash_json(cfg) == identity, 'Config identity mismatch')
        for collection in ('input_hashes', 'code_hashes'):
            for name, digest in cfg[collection].items():
                require(u.file_hash(ROOT / name) == digest, f'Stale {name}')
        df = pd.read_csv(ROOT / 'input_snapshot.csv', low_memory=False)
        CACHE = (identity, cfg, df, u.epa_labels(df.final_label)['A'],
                 {r: np.load(ROOT / (r + '.npy'), mmap_mode='r') for r in REPS})
    return CACHE[1:]


def run_fold(job, identity):
    cfg, df, y, matrices = cached_data(identity)
    path = job_path(job)
    split = cfg['splits'][str(job['seed'])][job['fold']]
    grid = cfg['grids'][job['model']]
    if (path / 'outer.json').exists():
        return validate_outer(job, path, identity, split, grid, y)
    X, model = matrices[job['representation']], job['model']
    candidates = []
    for candidate in grid:
        cp = path / 'candidates' / (candidate['candidate_id'] + '.json')
        if not cp.exists():
            scores = np.full(len(df), np.nan)
            aucs = []
            for p in split['inner']:
                est = u.fit_weighted(u.build_estimator(model, candidate['params']), X[p['train']], y[p['train']])
                score = u.predict_scores(est, X[p['validation']], model)
                scores[p['validation']] = score
                aucs.append(float(roc_auc_score(y[p['validation']], score)))
            u.save_record(cp, identity, dict(candidate=candidate, calibration_rows=split['calibration'],
                          calibration_scores=scores[split['calibration']].tolist(), fold_aucs=aucs, mean_auc=float(np.mean(aucs))))
        candidates.append(candidate_record(cp, identity, candidate, split, y))
    best = winner(candidates)
    cutoff = u.threshold(best['calibration_scores'], y[split['calibration']])
    est = u.fit_weighted(u.build_estimator(model, best['candidate']['params']), X[split['train']], y[split['train']])
    scores = u.predict_scores(est, X[split['test']], model)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / 'estimator.joblib.tmp'
    joblib.dump(est, temporary)
    os.replace(temporary, path / 'estimator.joblib')
    result = dict(**job, test=split['test'], train_rows=split['train'], calibration_rows=split['calibration'],
                  y_true=y[split['test']].tolist(), scores=scores.tolist(), predictions=(scores >= cutoff).astype(int).tolist(),
                  threshold=float(cutoff), selected_candidate=best['candidate'], candidates_evaluated=20,
                  estimator_sha256=u.file_hash(path / 'estimator.joblib'))
    u.save_record(path / 'outer.json', identity, result)
    return validate_outer(job, path, identity, split, grid, y)


def metrics(y, scores, predictions):
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return dict(sensitivity=float(sens), specificity=float(spec), gap=float(abs(sens - spec)),
                MCC=float(matthews_corrcoef(y, predictions)), **{'ROC-AUC': float(roc_auc_score(y, scores))},
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def paired_deltas(seeds):
    keys = ['representation', 'model', 'seed', 'rule']
    a = seeds[seeds.condition.eq('augmented')].set_index(keys)[METRICS].sort_index()
    b = seeds[seeds.condition.eq('noaugmentation')].set_index(keys)[METRICS].sort_index()
    require(a.index.equals(b.index), 'Pairing mismatch')
    return (a - b).reset_index()


def aggregate(frame, keys, columns):
    out = frame.groupby(keys)[columns].agg(['mean', 'std'])
    out.columns = ['_'.join(c) for c in out.columns]
    return out.reset_index()


def report_tables(oof, settings):
    tables = {}
    for suffix, keys in [('seed', KEYS), ('fold', KEYS + ['fold'])]:
        rows = []
        for key, block in oof.groupby(keys, sort=True):
            for rule, col in [('tuned', 'tuned_prediction'), ('fixed0.5', 'fixed05_prediction')]:
                rows.append(dict(zip(keys, key), rule=rule, rows=len(block),
                                 **metrics(block.y_true, block.score, block[col])))
        tables['metrics_by_' + suffix + '.csv'] = pd.DataFrame(rows)
    seeds = tables['metrics_by_seed.csv']
    tables['metrics_summary.csv'] = aggregate(seeds, KEYS[:-1] + ['rule'], METRICS)
    deltas = paired_deltas(seeds)
    tables['paired_deltas.csv'] = deltas
    tables['paired_delta_summary.csv'] = aggregate(deltas, ['representation', 'model', 'rule'], METRICS)
    tables['threshold_summary.csv'] = aggregate(settings, KEYS[:-1], ['threshold'])
    tables['selected_settings.csv'] = settings
    tables['selected_settings_distribution.csv'] = settings.groupby(KEYS[:-1] + ['candidate_id', 'params']).size().reset_index(name='fold_count')
    tables['oof_predictions.csv'] = oof
    return tables


def collect(cfg, identity):
    _, df, y, _ = cached_data(identity)
    rows, settings = [], []
    for condition in ('augmented', 'noaugmentation'):
        baseline = condition == 'augmented'
        ident = cfg['source_identity'] if baseline else identity
        for job in jobs():
            split = cfg['baseline_splits' if baseline else 'splits'][str(job['seed'])][job['fold']]
            result = validate_outer(job, job_path(job, baseline), ident, split, cfg['grids'][job['model']], y, baseline)
            settings.append(dict(condition=condition, **job, threshold=result['threshold'],
                                 candidate_id=result['selected_candidate']['candidate_id'],
                                 params=json.dumps(result['selected_candidate']['params'], sort_keys=True)))
            for i, score in zip(result['test'], result['scores']):
                rows.append(dict(condition=condition, **job, row_index=i, source_truth=int(df.final_label.iloc[i]),
                                 y_true=int(y[i]), score=score, tuned_threshold=result['threshold'],
                                 tuned_prediction=int(score >= result['threshold']), fixed05_prediction=int(score >= .5)))
    oof = pd.DataFrame(rows).sort_values(KEYS + ['row_index']).reset_index(drop=True)
    for _, block in oof.groupby(KEYS):
        require(block.row_index.tolist() == sorted(cfg['evaluation']), 'OOF exact coverage mismatch')
    require(len(oof) == 21816, 'Expected paired 606*3*6*2 predictions')
    return report_tables(oof, pd.DataFrame(settings))


def results_text(tables):
    lines = ['606 고정 분할: 추가 67행 학습 정책 통제 비교', SCOPE,
             '차이 방향: augmented - noaugmentation. AUC는 두 결정 규칙에서 동일합니다.',
             '평균 ± 표준편차는 seed별 paired 차이의 기술통계입니다.']
    for row in tables['paired_delta_summary.csv'].itertuples(index=False):
        lines.append(f'{row.model}/{row.representation} {row.rule}: ΔMCC {row.MCC_mean:+.6f} ± {row.MCC_std:.6f}, '
                     f'Δ민감도 {row.sensitivity_mean:+.6f}, Δ특이도 {row.specificity_mean:+.6f}')
    return '\n'.join(lines) + '\n'


def summarize(cfg, identity):
    tables = collect(cfg, identity)
    for name, frame in tables.items():
        temporary = ROOT / (name + '.tmp')
        frame.to_csv(temporary, index=False, encoding='utf-8-sig')
        os.replace(temporary, ROOT / name)
    temporary = ROOT / 'RESULTS.txt.tmp'
    temporary.write_text(results_text(tables), encoding='utf-8')
    os.replace(temporary, ROOT / 'RESULTS.txt')


@contextmanager
def exclusive_run():
    with (ROOT / '.controlled.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def status(identity, state, completed, **extra):
    u.atomic_json(ROOT / 'status.json', dict(identity=identity, status=state, completed_outer=completed,
                  total_outer=90, updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **extra))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=6)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    with exclusive_run():
        cfg, identity = prepare()
        _, _, y, _ = cached_data(identity)
        completed, pending = 0, []
        try:
            for job in jobs():
                if (job_path(job) / 'outer.json').exists():
                    validate_outer(job, job_path(job), identity, cfg['splits'][str(job['seed'])][job['fold']], cfg['grids'][job['model']], y)
                    completed += 1
                else:
                    pending.append(job)
            status(identity, 'prepared' if args.prepare_only else 'running', completed)
            print(f'completed_outer={completed}/90', flush=True)
            if args.prepare_only:
                return
            if not args.summarize_only:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    for future in as_completed([pool.submit(run_fold, j, identity) for j in pending]):
                        future.result()
                        completed += 1
                        status(identity, 'running', completed)
                        print(f'completed_outer={completed}/90', flush=True)
            require(completed == 90, 'Incomplete run; cannot summarize')
            # Recheck immutable dependencies after potentially long training.
            require(prepare()[1] == identity, 'Dependencies changed during run')
            summarize(cfg, identity)
            status(identity, 'completed_pending_audit', completed)
        except BaseException as exc:
            status(identity, 'failed', completed, error=repr(exc))
            raise


if __name__ == '__main__':
    main()
