"""Exact frozen-LR bounds and sampled input scenarios; not product risk estimates."""
from pathlib import Path
import os, sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import expit
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'inference'))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
from predict import load_bundle, vectorize_components, predict_features
OUT = ROOT / 'outputs'

def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig')

def project_reference(lower, upper):
    midpoint = (lower + upper) / 2
    if midpoint.sum() <= 1:
        return midpoint, 'midpoint_assumption'
    left, right = 0., 1.
    for _ in range(80):
        shift = (left + right) / 2
        if np.clip(midpoint - shift, lower, upper).sum() > 1:
            left = shift
        else:
            right = shift
    return np.clip(midpoint - right, lower, upper), 'midpoint_projected_to_mass_constraint_assumption'

def optimize_ratio(effect, lower, upper, capacity, maximize=False):
    # Charnes-Cooper transform: w_i = a_i / sum(a), t = 1 / sum(a).
    n = len(lower)
    objective = np.r_[(-effect if maximize else effect), 0.]
    rows, rhs = [], []
    for i in range(n):
        upper_row = np.zeros(n + 1); upper_row[i] = 1; upper_row[-1] = -upper[i]
        lower_row = np.zeros(n + 1); lower_row[i] = -1; lower_row[-1] = lower[i]
        rows.extend([upper_row, lower_row]); rhs.extend([0., 0.])
    result = linprog(objective, A_ub=rows, b_ub=rhs,
        A_eq=[np.r_[np.ones(n), 0.]], b_eq=[1.],
        bounds=[(0, None)] * n + [(1 / capacity, 1 / lower.sum() if lower.sum() > 0 else None)],
        method='highs')
    if not result.success:
        raise ValueError(result.message)
    weights, t = result.x[:-1], result.x[-1]
    amounts = weights / t
    assert np.all(amounts >= lower - 1e-9) and np.all(amounts <= upper + 1e-9)
    assert amounts.sum() <= capacity + 1e-9 and np.isclose(weights.sum(), 1)
    return amounts, float(weights @ effect)

def main():
    bundle = load_bundle(ROOT / 'inference/eye_A_LR_wmean.joblib')
    threshold = float(bundle['threshold'])
    pipeline = bundle['pipeline']
    imputer, scaler, classifier = [step for _, step in pipeline.steps]
    assert np.all(imputer.statistics_ == 0)
    beta = classifier.coef_[0] / scaler.scale_
    intercept = float(classifier.intercept_[0] - beta @ scaler.mean_)
    registry = pd.read_csv(OUT / 'structure_registry.csv').set_index('cas')
    audit = pd.read_csv(OUT / 'ingredient_audit.csv')
    plan = json.loads((ROOT / 'case_plan.json').read_text(encoding='utf-8'))
    components = pd.read_csv(ROOT / 'inputs/production_components.csv')
    generator = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    training = components.drop_duplicates('smiles').copy()
    training['canonical_smiles'] = training.smiles.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
    training_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in training.smiles]
    novelty = []
    vectors = {}
    for cas, r in registry.iterrows():
        vec = vectorize_components(bundle['feature_names'], [r.smiles]).to_numpy()[0]
        assert np.isfinite(vec).all(), f'Nonfinite descriptors: cannot use exact linear pooling for {cas}'
        vectors[cas] = vec
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(r.smiles))
        fp = generator.GetFingerprint(Chem.MolFromSmiles(r.smiles))
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, training_fps))
        nearest = training.iloc[int(similarities.argmax())]
        novelty.append(dict(cas=cas, smiles=r.smiles,
            exact_cas_in_training=bool(components.cas.eq(cas).any()),
            exact_structure_in_training=bool(training.canonical_smiles.eq(canonical).any()),
            nearest_training_tanimoto=float(similarities.max()),
            nearest_training_cas=nearest.cas, nearest_training_smiles=nearest.smiles,
            interpretation='Descriptive fingerprint similarity only; no validated applicability-domain cutoff'))
    save(pd.DataFrame(novelty), 'case_ingredient_training_similarity.csv')
    np.savez_compressed(ROOT / 'cache/component_features.npz', **vectors)
    records, ingredients, extreme_rows, scenario_frames, response_rows = [], [], [], [], []
    maximum_error = 0.
    n_scenarios = 8192
    for case in plan['cases']:
        did = case['document_id']
        g = audit[audit.document_id == did].copy().reset_index(drop=True)
        assert len(g) and g.source_file.nunique() == 1
        assert g[['lower', 'upper']].notna().all().all()
        lower, upper = g.lower.to_numpy(float), g.upper.to_numpy(float)
        assert 0 < lower.sum() <= 1 and np.all((lower >= 0) & (upper >= lower) & (upper <= 1))
        represented = g.cas.isin(case['represented_cas']).to_numpy()
        assert g.loc[represented, 'eligible_identity'].all()
        assert set(g.loc[represented, 'cas']) == set(case['represented_cas'])
        smiles = [registry.loc[cas, 'smiles'] for cas in g.loc[represented, 'cas']]
        matrix = np.stack([vectors[cas] for cas in g.loc[represented, 'cas']])
        effect = matrix @ beta
        capacity = 1 - lower[~represented].sum()
        ref_all, reference_basis = project_reference(lower, upper)
        ref = ref_all[represented]
        low_amount, low_effect = optimize_ratio(effect, lower[represented], upper[represented], capacity)
        high_amount, high_effect = optimize_ratio(effect, lower[represented], upper[represented], capacity, True)
        low_score, high_score = float(expit(intercept + low_effect)), float(expit(intercept + high_effect))
        reference_features = vectorize_components(bundle['feature_names'], smiles, ref)
        ref_score = float(predict_features(bundle, reference_features).eye_A_score.iloc[0])
        for i, row in g.iterrows():
            ingredients.append(dict(document_id=did, title=case['title'], raw_name=row['Raw Chemical Name'],
                cas=row.cas, lower_fraction=lower[i], upper_fraction=upper[i],
                represented=bool(represented[i]), smiles=registry.loc[row.cas, 'smiles'] if represented[i] else '',
                omission_reason='' if represented[i] else case['omitted_reason'],
                reference_fraction=float(ref_all[i]), reference_basis=reference_basis))
        for label, amounts, expected in [('min', low_amount, low_score), ('max', high_amount, high_score)]:
            actual = float(predict_features(bundle,
                vectorize_components(bundle['feature_names'], smiles, amounts)).eye_A_score.iloc[0])
            maximum_error = max(maximum_error, abs(expected - actual))
            assert abs(expected - actual) < 1e-10
            for cas, amount in zip(g.loc[represented, 'cas'], amounts):
                extreme_rows.append(dict(document_id=did, bound=label, cas=cas, amount=float(amount),
                    normalized_weight=float(amount / amounts.sum()), score=actual,
                    omitted_total_lower=float(lower[~represented].sum()),
                    omitted_note='Endpoint feasibility uses omitted lower bounds; unknown remainder is not featurized'))
        rng = np.random.default_rng(20261109 + did)
        samples, accepted, total = [], 0, 0
        while accepted < n_scenarios:
            batch = rng.uniform(lower, upper, size=(32768, len(g)))
            batch = batch[batch.sum(axis=1) <= 1 + 1e-12]
            samples.append(batch); accepted += len(batch); total += 32768
            if total > 2000000: raise ValueError('Insufficient feasible scenarios')
        samples = np.concatenate(samples)[:n_scenarios]
        amounts = samples[:, represented]
        weights = amounts / amounts.sum(axis=1, keepdims=True)
        scores = expit(intercept + weights @ effect)
        # Pipeline equivalence on 32 sampled scenarios, both extrema and reference.
        actual = predict_features(bundle, weights[:32] @ matrix).eye_A_score.to_numpy()
        maximum_error = max(maximum_error, float(np.max(np.abs(scores[:32] - actual))))
        assert np.max(np.abs(scores[:32] - actual)) < 1e-10
        assert scores.min() >= low_score - 1e-10 and scores.max() <= high_score + 1e-10
        assert np.all(samples >= lower - 1e-10) and np.all(samples <= upper + 1e-10)
        assert np.all(samples.sum(axis=1) <= 1 + 1e-10)
        sf = pd.DataFrame(samples, columns=['amount_' + str(c) for c in g.cas])
        sf.insert(0, 'scenario_id', np.arange(n_scenarios))
        sf.insert(0, 'document_id', did)
        sf['represented_fraction_sum'] = amounts.sum(axis=1)
        sf['listed_fraction_sum'] = samples.sum(axis=1)
        sf['score_resolved_subset_only'] = scores
        sf['subset_above_frozen_threshold'] = scores >= threshold
        scenario_frames.append(sf)
        threshold_crosses = low_score < threshold <= high_score
        records.append(dict(document_id=did, title=case['title'], tier=case['tier'],
            interpretation='Conditional normalized resolved-subset sensitivity; NOT a complete cosmetic product prediction',
            listed_components=len(g), represented_components=int(represented.sum()),
            omitted_components=int((~represented).sum()), all_listed_components_resolved=False,
            represented_mass_lower=float(lower[represented].sum()),
            represented_mass_upper_feasible=float(min(upper[represented].sum(), capacity)),
            reference_represented_mass=float(ref.sum()), reference_listed_mass=float(ref_all.sum()),
            reference_unknown_remainder=max(0., float(1 - ref_all.sum())), reference_basis=reference_basis,
            exact_score_min=low_score, reference_score=ref_score, exact_score_max=high_score,
            exact_score_width=high_score - low_score, frozen_threshold=threshold,
            threshold_crossing_possible=bool(threshold_crosses),
            scenario_q025=float(np.quantile(scores, .025)), scenario_median=float(np.median(scores)),
            scenario_q975=float(np.quantile(scores, .975)),
            scenario_threshold_fraction=float(np.mean(scores >= threshold)),
            scenario_n=n_scenarios, scenario_acceptance_fraction=accepted / total,
            omission_reason=case['omitted_reason'],
            warning='Scenario quantiles/fractions are sensitivity summaries, not confidence intervals or biological probabilities.'))
        # One-dimensional composition curve: ethanol amount varies in source range; water
        # and omitted ingredient amounts stay at the feasible reference. Drop infeasible points.
        if did == 277977:
            idx = list(g.cas).index('64-17-5')
            for value in np.linspace(lower[idx], upper[idx], 201):
                composition = ref_all.copy(); composition[idx] = value
                if composition.sum() > 1 + 1e-12: continue
                a = composition[represented]
                s = float(expit(intercept + (a / a.sum()) @ effect))
                response_rows.append(dict(document_id=did, ethanol_fraction=value,
                    water_fraction=float(composition[list(g.cas).index('7732-18-5')]),
                    omitted_fraction=float(composition[~represented].sum()), score_resolved_subset_only=s,
                    unknown_remainder=max(0., float(1 - composition.sum()))))
            # Independent dense-grid check of the two-component ratio extrema.
            aa, bb = np.meshgrid(np.linspace(lower[represented][0], upper[represented][0], 501),
                                  np.linspace(lower[represented][1], upper[represented][1], 501))
            feasible = aa + bb <= capacity + 1e-12
            grid_scores = expit(intercept + (aa * effect[0] + bb * effect[1]) / (aa + bb))
            assert grid_scores[feasible].min() >= low_score - 1e-10
            assert grid_scores[feasible].max() <= high_score + 1e-10
    result = pd.DataFrame(records)
    save(result, 'uncertainty_cases.csv')
    save(pd.DataFrame(ingredients), 'uncertainty_case_ingredients.csv')
    save(pd.DataFrame(extreme_rows), 'uncertainty_extreme_compositions.csv')
    pd.concat(scenario_frames, ignore_index=True).to_csv(OUT / 'uncertainty_scenarios.csv.gz', index=False, compression='gzip')
    save(pd.DataFrame(response_rows), 'ethanol_composition_response.csv')
    verification = dict(model_sha256=hashlib.sha256((ROOT / 'inference/eye_A_LR_wmean.joblib').read_bytes()).hexdigest(),
        threshold=threshold, case_count=len(result), total_scenarios=n_scenarios * len(result),
        exact_formula_pipeline_max_abs_error=maximum_error,
        all_component_features_finite=True, all_scenarios_within_reported_bounds=True,
        all_scenarios_mass_feasible=True, all_scenarios_within_exact_score_bounds=True,
        two_component_dense_grid_consistent=True, fully_structure_resolved_cases=0,
        conditional_partial_representation_cases=len(result),
        model_refit=False, threshold_refit=False, external_cosmetic_validation=False)
    (OUT / 'uncertainty_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(result[['document_id', 'title', 'represented_components', 'exact_score_min', 'reference_score', 'exact_score_max',
                  'threshold_crossing_possible', 'reference_represented_mass']].to_string(index=False))
    print(json.dumps(verification, indent=2))

if __name__ == '__main__': main()
