"""Check the user's reference result without altering models or predictions."""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT/'oof_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    pairs = [(int(row['label']), float(row['source_centered_transfer_score'])) for row in rows]
    tn = sum(y == 0 and score < .5 for y, score in pairs)
    fp = sum(y == 0 and score >= .5 for y, score in pairs)
    fn = sum(y == 1 and score < .5 for y, score in pairs)
    tp = sum(y == 1 and score >= .5 for y, score in pairs)
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn-fp*fn)/denominator if denominator else 0.
    positive = [score for y, score in pairs if y == 1]
    negative = [score for y, score in pairs if y == 0]
    auc = sum(float(p > n)+.5*float(p == n) for p in positive for n in negative)/(len(positive)*len(negative))
    observed = {'rows': len(rows), 'feature_groups': len({r['feature_group'] for r in rows}),
                'correct': tn+tp, 'confusion_matrix': [[tn, fp], [fn, tp]],
                'MCC': mcc, 'ROC_AUC': auc}
    expected_mcc = (10*3-2*1)/math.sqrt(12*11*5*4)
    expected_auc = 42/55
    expected = {'rows': 16, 'feature_groups': 15, 'correct': 13,
                'confusion_matrix': [[3, 2], [1, 10]], 'MCC': expected_mcc, 'ROC_AUC': expected_auc}
    checks = {key: (math.isclose(observed[key], value, rel_tol=0, abs_tol=1e-12)
                    if isinstance(value, float) else observed[key] == value)
              for key, value in expected.items()}
    status = {'status': 'reference_metrics_match' if all(checks.values()) else 'reference_metrics_mismatch',
              'user_requirement_met': False,
              'original_artifacts_recovered': False,
              'expected': expected, 'observed': observed, 'checks': checks,
              'note': 'Aggregate agreement alone cannot establish original-model reproduction. Original mapping, code and held-out predictions are still required.',
              'required_original_artifacts': [
                  'cosmetic_validation/test_evidence_collection/eye_partial_transfer/',
                  '세팅/생성스크립트/run_gettings_eye_partial_transfer.py'],
              'existing_model_disposition': 'Retained as an unaccepted reconstruction; no score, threshold or label changed.'}
    (ROOT/'reference_reproduction_status.json').write_text(
        json.dumps(status, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
