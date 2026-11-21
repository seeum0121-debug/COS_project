"""Freeze source/model inputs and protocol before external prediction."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
FINAL = ROOT.parents[1]
MODEL = FINAL / '최종모델'
TRAIN = FINAL / '모델 실험/eye/모델 최종/aggregation_compare_20261109'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = {}
    checks = json.loads((MODEL / 'checksums.json').read_text(encoding='utf-8'))
    for name, expected in checks['files'].items():
        assert sha(MODEL / name) == expected, name
    for dest, source in [('model', MODEL), ('training_reference', TRAIN)]:
        (ROOT / dest).mkdir(exist_ok=True)
        names = (['eye_A_LR_wmean.joblib', 'predict.py', 'metadata.json', 'feature_names.json',
                  'requirements-lock.txt', 'smoke_test_cases.json', 'checksums.json'] if dest == 'model'
                 else ['input_snapshot.csv', 'component_linkage.csv', 'wmean.npy'])
        for name in names:
            target = ROOT / dest / name
            if target.exists():
                assert sha(target) == sha(source / name)
            else:
                shutil.copyfile(source / name, target)
            manifest[str(target.relative_to(ROOT))] = {'source': str(source / name), 'sha256': sha(target)}
    metadata = json.loads((MODEL / 'metadata.json').read_text(encoding='utf-8'))
    for name in ['input_snapshot.csv', 'component_linkage.csv', 'wmean.npy']:
        assert sha(ROOT / 'training_reference' / name) == metadata['source_artifact_hashes'][name]
    (ROOT / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    protocol = {
        'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model_sha256': sha(MODEL / 'eye_A_LR_wmean.joblib'),
        'model': 'Eye A LogisticRegression weighted mean, all-data refit 673 rows',
        'positive': 'EPA I/II', 'negative': 'EPA III/IV',
        'threshold': metadata['threshold'], 'operator': '>=',
        'refit_or_external_threshold_selection': False,
        'primary_task': 'Five flea/tick animal shampoo products, frozen-model external case evaluation',
        'scope': 'Animal pesticide shampoos; do not generalize to human cosmetics.',
        'inputs': 'Only source resolved SMILES with disclosed quantities; midpoint for ranges. No invented water, UVCB or balance structures.',
        'sensitivity': 'Enumerate all low/high corners of represented ingredient ranges plus midpoint. Unknown ingredient amounts are not modeled.',
        'grouping': 'Report product rows and first-two-segment EPA-registration groups separately; average scores within registration group; no claim of independent original studies.',
        'deduplication_selection': 'No product is chosen based on outcomes; all members retained in product table; family aggregation fixed as mean.',
        'leakage': ['CAS component overlap (not automatically leakage)', 'canonical structure overlap', 'represented mixture signature', 'exact feature vector matches', 'product-name/registration/source-url matching'],
        'strict_full_formulation_gate': 'Full represented composition, compatible independently audited endpoint, no unresolved original-study independence. Otherwise exploratory partial-composition case comparison.',
        'CIR': 'Secondary ingredient-only diagnostic; use supplied labels without retuning. Separate neat vs diluted; no unknown vehicle inference; exclude known CAS/SMILES mismatch SLS. No claim of confirmed external validation.',
        'CIR_grouping': 'Report repeated-CAS contradictions and equal-CAS weighted metrics; repeated concentrations are not independent chemicals.',
        'zein': 'Excluded from eye classification metrics: protein solubilization endpoint is not an in-vivo EPA eye outcome.',
        'uncertainty': 'No inferential CI or significance claim for 3 registration groups; use raw counts and explicit limitations.',
    }
    p = ROOT / 'protocol.json'
    if not p.exists():
        p.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Inputs verified and protocol frozen:', ROOT)


if __name__ == '__main__':
    main()
