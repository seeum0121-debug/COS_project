"""Prepare an isolated rerun with the model evaluated on all 673 source rows."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / '최종 모델'
SOURCE = ROOT.parent.parent / '최최종모델'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (ROOT / 'protocol.json').exists():
        raise FileExistsError('Comparison has already been prepared')
    source_meta = read(SOURCE / 'metadata.json')
    assert source_meta['training_rows'] == source_meta['evaluation_unique_rows'] == 673
    assert source_meta['model'] == 'lr' and source_meta['representation'] == 'wmean'
    expected = '5d5c0532cd12fd00927c9564ee28dc9c95a8bb396a049c705c77dcafaf5b28e5'
    assert sha(SOURCE / 'eye_A_final.joblib') == expected
    # Preserve all prior source documents and the fixed reconstruction inputs.
    for name in ['data', 'examples', 'reference']:
        shutil.copytree(PREVIOUS / name, ROOT / name)
    for name in ['common.py', 'predict.py', 'mapping.json', 'requirements-lock.txt',
                 'requirements-rebuild.txt', 'seal_package.py']:
        shutil.copy2(PREVIOUS / name, ROOT / name)
    (ROOT / 'source_model').mkdir()
    for name in ['eye_A_final.joblib', 'metadata.json', 'predict.py',
                 'feature_names.json', 'requirements-lock.txt']:
        shutil.copy2(SOURCE / name, ROOT / 'source_model' / name)
    (ROOT / 'previous_result').mkdir()
    for name in ['metrics.json', 'oof_predictions.csv', 'metadata.json', 'protocol.json',
                 'bootstrap_ci.json']:
        shutil.copy2(PREVIOUS / name, ROOT / 'previous_result' / name)
    old_model = PREVIOUS / 'source_model/eye_A_LR_wmean.joblib'
    shutil.copy2(old_model, ROOT / 'previous_result/eye_A_LR_wmean.joblib')
    shutil.copy2(PREVIOUS / 'features.npz', ROOT / 'previous_result/features.npz')

    protocol = read(PREVIOUS / 'protocol.json')
    protocol['source_model_sha256'] = expected
    protocol['source_model_file'] = 'source_model/eye_A_final.joblib'
    protocol['source_model_format'] = 'eye-A-grouped-v1'
    protocol['source_model_training_rows'] = 673
    protocol['source_model_evaluation_unique_rows'] = 673
    protocol['comparison_design'] = 'Change only the source bundle to the all-673-evaluated final refit; retain the 16 target rows, 13-entry mapping, feature recipe, 15 groups, optimizer, lambda, 0.5 transfer threshold, and bootstrap seed.'
    write(ROOT / 'protocol.json', protocol)

    train = (PREVIOUS / 'train.py').read_text(encoding='utf-8')
    train = train.replace('source_model/eye_A_LR_wmean.joblib', 'source_model/eye_A_final.joblib')
    train = train.replace("assert source['format'] == 'eye-A-lr-wmean-v1'", "assert source['format'] == 'eye-A-grouped-v1'\n    assert source['representation'] == 'wmean'\n    assert source['metadata']['evaluation_unique_rows'] == 673")
    train = train.replace("'source_model_sha256': protocol['source_model_sha256'],", "'source_model_sha256': protocol['source_model_sha256'],\n                'source_model_file': protocol['source_model_file'],\n                'source_model_training_rows': 673, 'source_model_evaluation_unique_rows': 673,")
    train = train.replace('New reconstruction | n=16', 'Source evaluated on all 673 rows | target n=16')
    (ROOT / 'train.py').write_text(train, encoding='utf-8')

    verify = (PREVIOUS / 'verify.py').read_text(encoding='utf-8')
    verify = verify.replace('source_model/eye_A_LR_wmean.joblib', 'source_model/eye_A_final.joblib')
    verify = verify.replace("model['feature_names'], [by_id[row['mapped_id']]['canonical_smiles']", "source, [by_id[row['mapped_id']]['canonical_smiles']")
    (ROOT / 'verify.py').write_text(verify, encoding='utf-8')

    manifest = {name: info for name, info in read(PREVIOUS / 'source_manifest.json').items()
                if not name.startswith('source_model/')}
    for path in sorted((ROOT / 'source_model').iterdir()):
        manifest[path.relative_to(ROOT).as_posix()] = {'original_path': str(SOURCE / path.name), 'sha256': sha(path)}
    for path in sorted((ROOT / 'previous_result').iterdir()):
        original = old_model if path.suffix == '.joblib' else PREVIOUS / path.name
        manifest[path.relative_to(ROOT).as_posix()] = {'original_path': str(original), 'sha256': sha(path)}
    write(ROOT / 'source_manifest.json', manifest)
    print('Prepared', ROOT)
    print('Source SHA256:', expected)


if __name__ == '__main__':
    main()
