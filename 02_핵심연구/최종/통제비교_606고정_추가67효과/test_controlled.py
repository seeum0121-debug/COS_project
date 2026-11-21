"""Small synthetic contract tests; no production fitting or artifact modification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import epa_harness as u
import run_controlled as r
import audit_controlled as a


def synthetic_splits():
    evaluation = list(range(30))
    y = np.tile([0, 1], 16)
    groups = np.repeat(np.arange(16), 2)
    folds = []
    for fold in range(5):
        test = list(range(fold * 6, fold * 6 + 6))
        calibration = [i for i in evaluation if i not in test]
        inner = []
        for k in range(3):
            valid = calibration[k * 8:(k + 1) * 8]
            inner.append(dict(train=[i for i in calibration if i not in valid] + [30, 31], validation=valid))
        folds.append(dict(train=calibration + [30, 31], test=test, calibration=calibration, inner=inner))
    return {str(seed): copy.deepcopy(folds) for seed in r.SEEDS}, evaluation, groups, y


class ControlledTests(unittest.TestCase):
    def test_environment_mismatch_fails_closed(self):
        historical = dict.fromkeys(('numpy', 'pandas', 'scikit-learn', 'xgboost'), '1.2.3')
        for package in historical:
            with self.subTest(package=package):
                def live(name):
                    return '9.9.9' if name == package else historical[name]
                with patch.object(r.importlib.metadata, 'version', side_effect=live):
                    with self.assertRaisesRegex(ValueError, package):
                        r.verify_historical_environment(historical)
        with patch.object(r.importlib.metadata, 'version', side_effect=historical.__getitem__):
            r.verify_historical_environment(historical)

    def test_noadd_excludes_added_and_preserves_order(self):
        old, evaluation, groups, y = synthetic_splits()
        r.validate_splits(old, evaluation, groups, y, True)
        new = r.noadd_splits(old, evaluation)
        r.validate_splits(new, evaluation, groups, y, False)
        self.assertIn(30, old['42'][0]['train'])
        for seed, folds in new.items():
            for f, previous in zip(folds, old[seed]):
                self.assertEqual(f['train'], previous['calibration'])
                self.assertEqual(f['test'], previous['test'])
                for p, q in zip(f['inner'], previous['inner']):
                    self.assertEqual(p['validation'], q['validation'])
                    self.assertEqual(set(p['train']) | set(p['validation']), set(f['train']))
                    self.assertFalse(set(p['train']) & {30, 31})
                    self.assertFalse(set(groups[p['train']]) & set(groups[p['validation'] + f['test']]))

    def test_rejects_added_row_and_group_leakage(self):
        old, evaluation, groups, y = synthetic_splits()
        new = r.noadd_splits(old, evaluation)
        bad = copy.deepcopy(new)
        bad['42'][0]['inner'][0]['train'].append(30)
        with self.assertRaises(ValueError):
            r.validate_splits(bad, evaluation, groups, y, False)
        badgroups = groups.copy()
        badgroups[6:8] = badgroups[0]
        with self.assertRaises(ValueError):
            r.validate_splits(new, evaluation, badgroups, y, False)

    def test_rejects_duplicate_validation(self):
        old, evaluation, groups, y = synthetic_splits()
        new = r.noadd_splits(old, evaluation)
        new['42'][0]['inner'][0]['validation'].append(6)
        with self.assertRaises(ValueError):
            r.validate_splits(new, evaluation, groups, y, False)

    def test_baseline_checksum_and_metric_corruption_rejected(self):
        candidate = dict(candidate_id='test', params={})
        split = dict(calibration=list(range(6)), inner=[dict(validation=[0, 1]), dict(validation=[2, 3]), dict(validation=[4, 5])])
        y = np.tile([0, 1], 3)
        payload = dict(candidate=candidate, calibration_rows=list(range(6)),
                       calibration_scores=[.1, .9, .2, .8, .3, .7], fold_aucs=[1., 1., 1.], mean_auc=1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'baseline.json'
            u.save_record(path, 'historical-identity', payload)
            r.candidate_record(path, 'historical-identity', candidate, split, y)
            with self.assertRaises(ValueError):
                u.load_record(path, 'wrong-identity')
            envelope = json.loads(path.read_text(encoding='utf-8'))
            envelope['payload']['calibration_scores'][0] = .95
            path.write_text(json.dumps(envelope), encoding='utf-8')
            with self.assertRaises(ValueError):
                r.candidate_record(path, 'historical-identity', candidate, split, y)
            # Even rechecksumming a corrupted metric cannot bypass reconstruction.
            envelope['payload_sha256'] = u.hash_json(envelope['payload'])
            path.write_text(json.dumps(envelope), encoding='utf-8')
            with self.assertRaises(AssertionError):
                r.candidate_record(path, 'historical-identity', candidate, split, y)

    def test_exact_tie_preserves_grid_order(self):
        candidates = [dict(candidate='first', mean_auc=.7), dict(candidate='second', mean_auc=.7)]
        self.assertEqual(r.winner(candidates)['candidate'], 'first')
        self.assertEqual(r.winner(list(reversed(candidates)))['candidate'], 'second')
        candidates[1]['mean_auc'] = np.nextafter(.7, 1.)
        self.assertEqual(r.winner(candidates)['candidate'], 'second')

    def test_inner_threshold_objective(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        for scores in ([.2, .3, .4, .5, .6, .7], [.1, .8, .1, .8, .4, .5]):
            self.assertEqual(a.threshold(scores, y), u.threshold(scores, y))
        with self.assertRaises(ValueError):
            a.threshold([.5] * 6, y)

    def test_paired_delta_sign_and_decision_rule_diagnostic(self):
        rows, settings = [], []
        for condition, cutoff in [('augmented', .4), ('noaugmentation', .5)]:
            settings.append(dict(condition=condition, representation='wmean', model='lr', seed=42,
                                 fold=0, threshold=cutoff, candidate_id='x', params='{}'))
            for i, (truth, score) in enumerate(zip([0, 0, 1, 1], [.1, .3, .45, .9])):
                rows.append(dict(condition=condition, representation='wmean', model='lr', seed=42,
                                 fold=0, row_index=i, source_truth=1 if truth else 4, y_true=truth,
                                 score=score, tuned_threshold=cutoff, tuned_prediction=int(score >= cutoff),
                                 fixed05_prediction=int(score >= .5)))
        tables = r.report_tables(pd.DataFrame(rows), pd.DataFrame(settings))
        delta = tables['paired_deltas.csv'].set_index('rule')
        self.assertGreater(delta.loc['tuned', 'MCC'], 0)
        self.assertEqual(delta.loc['tuned', 'sensitivity'], .5)
        self.assertEqual(delta.loc['fixed0.5', 'MCC'], 0)
        self.assertTrue(delta['ROC-AUC'].eq(0).all())
        byseed = tables['metrics_by_seed.csv']
        self.assertTrue(byseed.groupby('condition')['ROC-AUC'].nunique().eq(1).all())
        # Independent recomputation agrees, and a sign reversal is detected.
        expected = delta.reset_index()
        damaged = expected.copy()
        damaged['MCC'] = -damaged['MCC']
        with self.assertRaises(AssertionError):
            a.compare_table(damaged, expected, ['representation', 'model', 'seed', 'rule'])


if __name__ == '__main__':
    unittest.main()
