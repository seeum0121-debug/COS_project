"""Freeze an explicit reconstruction whitelist; cache primary-source structures."""
from pathlib import Path
import json
import time

import requests

from common import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parent
ENTRIES = [
    ('water', '7732-18-5', ['water']),
    ('glycerin', '56-81-5', ['glycerin', 'glycerol']),
    ('propylene_glycol', '57-55-6', ['propylene glycol']),
    ('hexylene_glycol', '107-41-5', ['hexylene glycol']),
    ('citric_acid', '77-92-9', ['citric acid']),
    ('lactic_acid', '50-21-5', ['lactic acid']),
    ('benzoic_acid', '65-85-0', ['benzoic acid']),
    ('methylparaben', '99-76-3', ['methylparaben', 'methyl paraben']),
    ('propylparaben', '94-13-3', ['propylparaben', 'propyl paraben']),
    ('bht', '128-37-0', ['bht']),
    ('sodium_chloride', '7647-14-5', ['sodium chloride', 'sodium choride']),
    ('edta', '60-00-4', ['edta']),
    ('tetrasodium_edta', '64-02-8', ['tetrasodium edta']),
]


def main():
    if (ROOT/'eye_transfer_final.joblib').exists():
        raise RuntimeError('Mapping is frozen; do not modify a fitted package')
    entries = []
    folder = ROOT/'data/sources/pubchem'
    folder.mkdir(exist_ok=True, parents=True)
    for key, cas, aliases in ENTRIES:
        path = folder/(key+'.json')
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/ConnectivitySMILES,InChIKey,MolecularFormula/JSON'
        if path.exists():
            raw = read_json(path)['response']
        else:
            for attempt in range(4):
                try:
                    response = requests.get(url, timeout=25)
                    response.raise_for_status()
                    raw = response.json()
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(2)
            write_json(path, {'request_url': url, 'response': raw})
        properties = raw['PropertyTable']['Properties']
        assert len(properties) == 1, (key, properties)
        item = properties[0]
        entries.append({'id': key, 'aliases': aliases, 'cas': cas,
                        'pubchem_cid': item['CID'], 'canonical_smiles': item['ConnectivitySMILES'],
                        'inchikey': item['InChIKey'], 'formula': item['MolecularFormula'],
                        'source_url': url})
        print(key, item['CID'], flush=True)
        time.sleep(.3)
    write_json(ROOT/'mapping.json', {
        'status': 'new reconstruction whitelist; original mapping.json unavailable',
        'selection': '13 explicit solvent/humectant, acid, preservative, salt/chelating identities. Fixed before evaluation; not an exhaustive list of resolvable compounds.',
        'normalization': 'case-fold and collapse spaces; listed aliases only; stock percentages and composite ingredient names are NOT stripped or guessed',
        'unknown_policy': 'exclude from structural features but retain in coverage audit; no carrier or active-fraction imputation',
        'identity_scope': 'Candidate molecular structures; supplier grade, hydration, stereochemical composition and batch identity are not established.',
        'entries': entries})
    write_json(ROOT/'protocol.json', {
        'schema': 'eye-transfer-reconstruction-v1', 'method': 'source_centered_transfer',
        'original_artifacts_available': False, 'original_13_ingredient_mapping_available': False,
        'source_model_sha256': 'd71394e691ac5072e4ce900a53b47ddbb65c8685961fe2c9536411acfbbc58a2',
        'candidate_study': 'Gettings 1996', 'quantity_tolerance_percent_points': .11,
        'target': 'in vivo undiluted EPA I/II=1, III/IV=0; never HET-CAM 10% classification',
        'mapping_sha256': sha256(ROOT/'mapping.json'),
        'loss': 'mean(class_weight * (logaddexp(0, logit)-y*logit)) + 0.5*lambda*||theta-anchor||^2',
        'class_weight': 'n_training/(2*n_class), recomputed within each fold',
        'lambda': 1., 'penalize_intercept': True,
        'preprocessing': 'frozen source imputer/scaler; never fitted on target data',
        'grouping': 'identical pooled feature vectors rounded to 12 decimals, with a fixed NaN sentinel',
        'evaluation': 'leave-one-feature-group-out; only held-out rows contribute OOF scores',
        'final_fit': 'one new coefficient vector fitted on all 16 target rows after OOF evaluation',
        'threshold': .5,
        'optimizer': {'method': 'L-BFGS-B', 'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50},
        'bootstrap': {'group_resamples': 5000, 'seed': 20261110, 'refit': False},
        'source_overlap': 'unresolved; no independent external-validation claim',
        'reference_values_not_training_targets': {'n': 16, 'groups': 15, 'negative': 5, 'positive': 11,
            'correct': 13, 'MCC_rounded': .545, 'ROC_AUC_rounded': .764,
            'confusion_matrix': [[3, 2], [1, 10]]},
        'reconstruction_assumptions': 'Original mapping, source script, fold coefficients, exact loss normalization and optimizer settings unavailable. Whitelist and weighted-mean loss are new explicit choices, not recovered facts. No metric-driven selection.'})
    print('Mapping and protocol frozen before fitting.', flush=True)


if __name__ == '__main__':
    main()
