"""Frozen-model cosmetics linkage. All writes are restricted to this directory."""
from pathlib import Path
import os, sys, json, re, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
OUT = ROOT / 'outputs'

def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def cas_valid(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{2,7}-\d{2}-\d', value.strip()):
        return False
    digits = value.strip().replace('-', '')
    return sum((i + 1) * int(d) for i, d in enumerate(digits[-2::-1])) % 10 == int(digits[-1])

def textval(value):
    return '' if pd.isna(value) else str(value).strip()

INDUSTRIAL = {
    1184789: 'pigment tint product; industrial/craft title',
    1287724: 'black printer toner; cosmetic toner category mismatch',
    1127047: 'Hastelloy alloy; hair spray category mismatch',
    1124899: 'nitriding stop-off coating',
    1128704: 'way oil lubricant',
    1313892: 'SX printer toner; cosmetic toner category mismatch',
    278757: 'metallic ink pad ink; nail polish category mismatch',
    1269083: 'Expand-O-Flash roofing adhesive; eye makeup category mismatch',
}
AMBIGUOUS = {
    1224850: 'craft glitter spray versus cosmetic use unresolved',
    1229183: 'covers-over aerosol: cosmetic identity unconfirmed',
    1182581: 'generic isopropanol; cosmetic product use unconfirmed',
    1170205: 'occupational hand scrub identity unconfirmed',
    1210188: 'occupational protective cream; cosmetic scope unconfirmed',
    1308593: 'skin conditioner identity/use unconfirmed',
    1213101: 'cream cleanser; cosmetic versus surface cleaning unresolved',
    1389026: 'capote product identity unconfirmed',
    279288: 'All Clear product identity unconfirmed',
}
OTHER_SCOPE = {
    1210158: 'lens/surface cleaning product',
    1136354: 'lens cleaning fluid',
    1360349: 'lens cleaning anti-static fluid',
    1360350: 'lens cleaning anti-static fluid',
    280489: 'personal lubricant outside selected cosmetics scope',
    280490: 'personal lubricant outside selected cosmetics scope',
    279394: 'callus removal treatment outside selected scope',
    1687802: 'back patch treatment',
}
COSMETIC_TYPES = {
    'bath oil', 'body spray', 'nail polish', 'bar soap', 'hand/body lotion',
    'body wash', 'hair styling', 'shampoo', 'bubble bath', 'lip balm', 'lip gloss',
    'body scrub', 'hair spray', 'hand soap', 'body powder', 'body mist',
    'hair color - temporary', 'face powder', 'bath paints/crayons',
    'hair color - permanent', 'hair conditioner', 'hair conditioner - leave-in',
    'eye shadow', 'eye makeup',
}
MEDICAL_TYPES = {'antiseptic', 'hand sanitizer', 'denture cleaner', 'teeth whitener',
                 'toothpaste', 'contact care', 'eye drops'}
PLAUSIBLE = {274869, 276874, 277610, 277611, 277420, 272237, 272257,
             281277, 277810, 277278, 279820}

def product_status(did, g):
    title = ' '.join(g['Product Name'].dropna().astype(str).unique()).lower()
    cats = set(g['PUC Product Type'].dropna().astype(str).str.strip())
    if did in INDUSTRIAL: return 'excluded_industrial', INDUSTRIAL[did]
    if did in AMBIGUOUS: return 'unresolved_product', AMBIGUOUS[did]
    if did in OTHER_SCOPE: return 'outside_study_scope', OTHER_SCOPE[did]
    if cats & MEDICAL_TYPES or any(x in title for x in ['sunscreen', 'sun screen', 'sunblock', 'spf 30', 'spf 50', 'spf50']):
        return 'outside_study_scope', 'medical/oral-care/disinfectant/sun-protection scope excluded by study rule'
    if did in PLAUSIBLE:
        return 'supplementary_title_only', 'cosmetic use plausible from title; product category missing'
    if cats & COSMETIC_TYPES:
        return 'primary_cosmetic', 'cosmetic category consistent with product title (metadata screen)'
    return 'unresolved_product', 'insufficient product identity/category'

def bounds(row):
    lo, hi, center = [pd.to_numeric(row.get(k), errors='coerce') for k in
                     ['Lower Weight Fraction', 'Upper Weight Fraction', 'Central Weight Fraction']]
    if np.isfinite(lo) and np.isfinite(hi) and 0 <= lo <= hi <= 1:
        return float(lo), float(hi), 'reported_range'
    if np.isfinite(center) and 0 <= center <= 1:
        return float(center), float(center), 'reported_central'
    return np.nan, np.nan, 'unusable_or_missing'

def main():
    assert cas_valid('64-17-5') and cas_valid('7732-18-5')
    assert not cas_valid('5281-04-09') and not cas_valid('64-17-6')
    frames = []
    for name in ['chemexpo_personal_care.csv', 'chemexpo_eye.csv']:
        df = pd.read_csv(ROOT / 'inputs' / name)
        df['source_file'] = name
        df['source_csv_row'] = np.arange(len(df)) + 2
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)
    allrows['document_id'] = allrows['Data Document ID'].astype(int)
    products = []
    for did, g in allrows.groupby('document_id', sort=True):
        status, reason = product_status(did, g)
        products.append(dict(document_id=did, product_name=' | '.join(g['Product Name'].dropna().unique()),
            product_type=' | '.join(g['PUC Product Type'].dropna().unique()),
            source_files=' | '.join(g.source_file.unique()), source_record_count=len(g),
            status=status, reason=reason,
            source_url=f'https://comptox.epa.gov/chemexpo/datadocument/{did}/'))
    products = pd.DataFrame(products)
    save(products, 'product_audit.csv')
    allrows = allrows.merge(products[['document_id', 'status']], on='document_id', validate='many_to_one')
    allrows['cas'] = allrows['Curated CAS'].map(textval)
    allrows['raw_cas_clean'] = allrows['Raw CAS'].map(textval)
    allrows['curated_cas_valid'] = allrows.cas.map(cas_valid)
    allrows['raw_cas_valid'] = allrows.raw_cas_clean.map(cas_valid)
    # Curated identities can legitimately differ. Conservative main analysis removes mismatches
    # until their identity is independently reviewed, rather than repairing them automatically.
    allrows['raw_curated_disagreement'] = (allrows.raw_cas_valid & allrows.curated_cas_valid &
                                           (allrows.raw_cas_clean != allrows.cas))
    allrows['eligible_identity'] = allrows.curated_cas_valid & ~allrows.raw_curated_disagreement
    br = pd.DataFrame([bounds(r) for _, r in allrows.iterrows()], columns=['lower', 'upper', 'amount_basis'])
    allrows = pd.concat([allrows, br], axis=1)
    save(allrows, 'ingredient_audit.csv')

    primary = allrows[(allrows.status == 'primary_cosmetic') & allrows.eligible_identity]
    extended = allrows[allrows.status.isin(['primary_cosmetic', 'supplementary_title_only']) & allrows.eligible_identity]
    eye = primary[primary['PUC Product Type'].isin(['eye shadow', 'eye makeup'])]
    universes = {'primary': set(primary.cas) - {'7732-18-5'},
                 'extended': set(extended.cas) - {'7732-18-5'},
                 'eye_specific': set(eye.cas) - {'7732-18-5'}}
    component = pd.read_csv(ROOT / 'inputs/production_components.csv')
    component['cas'] = component.cas.map(textval)
    component['cas_valid'] = component.cas.map(cas_valid)
    snapshot = pd.read_csv(ROOT / 'inputs/production_rows.csv', low_memory=False)
    config = json.loads((ROOT / 'inputs/comparison_config.json').read_text(encoding='utf-8'))['payload']
    assert len(config['groups']) == len(snapshot) == 673
    component['group_id'] = component.row_index.map(dict(enumerate(config['groups'])))
    counts = []
    for cas, g in primary.groupby('cas'):
        c = component[(component.cas == cas) & component.cas_valid]
        counts.append(dict(cas=cas, ingredient_names=' | '.join(g['Curated Chemical Name'].dropna().unique()),
            cosmetic_documents=g.document_id.nunique(), eye_cosmetic_documents=eye[eye.cas == cas].document_id.nunique(),
            original_evaluation_rows=c[c.evaluation_role == 'existing_evaluation_pool'].row_index.nunique(),
            augmentation_rows=c[c.evaluation_role != 'existing_evaluation_pool'].row_index.nunique(),
            water_excluded_from_subgroups=cas == '7732-18-5'))
    cas_table = pd.DataFrame(counts).sort_values(['original_evaluation_rows', 'cosmetic_documents'], ascending=False)
    save(cas_table, 'cosmetic_cas_overlap.csv')
    membership = pd.DataFrame(dict(row_index=np.arange(len(snapshot)),
        formulation_id=snapshot.formulation_id, evaluation_role=snapshot.evaluation_role,
        group_id=config['groups']))
    assert np.allclose(component.groupby('row_index').weight.sum().to_numpy(), 1)
    for name, universe in universes.items():
        matched = component[component.cas_valid & component.cas.isin(universe)]
        membership[name + '_weight_share'] = membership.row_index.map(matched.groupby('row_index').weight.sum()).fillna(0)
        membership[name + '_cas'] = membership.row_index.map(matched.groupby('row_index').cas.agg(lambda s: '|'.join(sorted(set(s))))).fillna('')
    membership['any_primary_nonwater'] = membership.primary_weight_share > 0
    membership['majority_primary_nonwater'] = membership.primary_weight_share >= .5
    membership['no_primary_nonwater'] = ~membership.any_primary_nonwater
    save(membership, 'row_membership.csv')
    oof = pd.read_csv(ROOT / 'inputs/comparison_oof.csv')
    oof = oof[(oof.label == 'A') & (oof.representation == 'wmean') & (oof.model == 'lr')].copy()
    oof = oof.merge(membership, on=['row_index', 'formulation_id'], validate='many_to_one')
    assert len(oof) == 1818 and oof.row_index.nunique() == 606
    assert set(oof.evaluation_role) == {'existing_evaluation_pool'}
    assert oof.groupby(['seed', 'row_index']).size().eq(1).all()
    save(oof, 'subgroup_oof_predictions.csv')
    group_masks = {'original_606': np.ones(len(oof), bool),
        'any_cosmetic_nonwater': oof.any_primary_nonwater.to_numpy(),
        'majority_cosmetic_nonwater': oof.majority_primary_nonwater.to_numpy(),
        'no_cosmetic_nonwater': oof.no_primary_nonwater.to_numpy(),
        'supplementary_extended_any': (oof.extended_weight_share > 0).to_numpy(),
        'eye_specific_any': (oof.eye_specific_weight_share > 0).to_numpy()}
    result = []
    for name, mask in group_masks.items():
        for seed in sorted(oof.seed.unique()):
            g = oof[mask & (oof.seed == seed).to_numpy()]
            if len(g):
                tn, fp, fn, tp = confusion_matrix(g.y_true, g.prediction, labels=[0, 1]).ravel()
                npos, nneg = int(tp + fn), int(tn + fp)
                sensitivity = tp / npos if npos else np.nan
                specificity = tn / nneg if nneg else np.nan
                auc = roc_auc_score(g.y_true, g.score) if npos and nneg else np.nan
                mcc = matthews_corrcoef(g.y_true, g.prediction) if npos and nneg else np.nan
            else:
                tn = fp = fn = tp = npos = nneg = 0
                sensitivity = specificity = auc = mcc = np.nan
            result.append(dict(subgroup=name, seed=seed, n=len(g), groups=g.group_id.nunique(),
                positives=npos, negatives=nneg, tp=tp, fn=fn, tn=tn, fp=fp,
                sensitivity=sensitivity, specificity=specificity, mcc=mcc, roc_auc=auc,
                sparse=(len(g) < 20 or g.group_id.nunique() < 10 or min(npos, nneg) < 5)))
    metrics = pd.DataFrame(result)
    save(metrics, 'subgroup_metrics_by_seed.csv')
    summaries = []
    for name, g in metrics.groupby('subgroup', sort=False):
        rec = dict(subgroup=name, **{c: g.iloc[0][c] for c in ['n', 'groups', 'positives', 'negatives', 'sparse']})
        for metric in ['sensitivity', 'specificity', 'mcc', 'roc_auc']:
            rec[metric + '_mean'] = g[metric].mean()
            rec[metric + '_seed_sd'] = g[metric].std(ddof=1)
        summaries.append(rec)
    summary = pd.DataFrame(summaries)
    save(summary, 'subgroup_metrics_summary.csv')
    audit = dict(source_rows=len(allrows), unique_documents=len(products),
        source_files={x: {'rows': len(g), 'documents': g.document_id.nunique()} for x, g in allrows.groupby('source_file')},
        product_status_counts={k: int(v) for k, v in products.status.value_counts().items()},
        primary_cosmetic_records=len(allrows[allrows.status == 'primary_cosmetic']),
        invalid_nonmissing_curated_cas=int((~allrows.curated_cas_valid & allrows.cas.ne('')).sum()),
        raw_curated_identity_disagreements=int(allrows.raw_curated_disagreement.sum()),
        nonwater_cosmetic_cas=len(universes['primary']),
        nonwater_cosmetic_cas_in_original_evaluation=int(((cas_table.original_evaluation_rows > 0) & ~cas_table.water_excluded_from_subgroups).sum()),
        original_evaluation_rows=606, augmentation_training_only_rows=67,
        label_use='Only original EPA evaluation labels in saved held-out predictions. No CIR-derived labels.',
        inference='Shared ingredient CAS indicates ingredient-use overlap, not cosmetic product domain validation.')
    (OUT / 'linkage_summary.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    print(summary.to_string(index=False))
    print(cas_table[cas_table.original_evaluation_rows > 0].to_string(index=False))

if __name__ == '__main__': main()
