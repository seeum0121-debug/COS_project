"""Frozen Eye A model external case evaluation; no fitting or threshold optimization."""
from pathlib import Path
import base64
import hashlib
import html
import importlib.metadata
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'model'))
from predict import load_bundle, predict_features, vectorize_components

OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv(name, frame):
    frame.to_csv(OUT / name, index=False, encoding='utf-8-sig')


def canonical(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f'Invalid structure: {smi}')
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def signature(smiles, weights):
    groups = {}
    for s, w in zip(smiles, weights):
        key = canonical(s)
        groups[key] = groups.get(key, 0.) + float(w)
    total = sum(groups.values())
    return '|'.join(f'{s}:{w/total:.8f}' for s, w in sorted(groups.items()) if w > 0)


def metrics(name, frame, weight=None):
    d = frame.dropna(subset=['score', 'label']).copy()
    y, p = d.label.astype(int).to_numpy(), d.prediction.astype(int).to_numpy()
    w = np.ones(len(d)) if weight is None else d[weight].to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1], sample_weight=w).ravel()
    return dict(analysis=name, n_rows=len(d), effective_weight=float(w.sum()),
                negative_rows=int((y == 0).sum()), positive_rows=int((y == 1).sum()),
                TN=float(tn), FP=float(fp), FN=float(fn), TP=float(tp),
                accuracy=float((tn+tp)/w.sum()) if len(d) else None,
                sensitivity=float(tp/(tp+fn)) if tp+fn else None,
                specificity=float(tn/(tn+fp)) if tn+fp else None,
                balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2) if tp+fn and tn+fp else None,
                MCC=float(matthews_corrcoef(y, p, sample_weight=w)) if len(d) else None,
                ROC_AUC=float(roc_auc_score(y, d.score, sample_weight=w)) if len(set(y)) == 2 else None,
                interpretation='Exploratory source-label agreement; not validated human-cosmetic safety performance')


def main():
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    provenance = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    adjudication = json.loads((ROOT / 'source_adjudication.json').read_text(encoding='utf-8'))
    for f in provenance['files']:
        assert f['available'] and sha(ROOT / 'sources' / f['path']) == f['sha256']
    model_path = ROOT / 'model/eye_A_LR_wmean.joblib'
    assert sha(model_path) == protocol['model_sha256']
    for line in (ROOT / 'model/requirements-lock.txt').read_text().splitlines():
        name, version = line.split('==')
        assert importlib.metadata.version(name) == version, line
    bundle = load_bundle(model_path)
    assert bundle['threshold'] == protocol['threshold']
    names, threshold = bundle['feature_names'], bundle['threshold']
    meta = bundle['metadata']
    for name in ['input_snapshot.csv', 'component_linkage.csv', 'wmean.npy']:
        assert sha(ROOT / 'training_reference' / name) == meta['source_artifact_hashes'][name]
    train = pd.read_csv(ROOT / 'training_reference/input_snapshot.csv', low_memory=False)
    components = pd.read_csv(ROOT / 'training_reference/component_linkage.csv')
    xt = np.load(ROOT / 'training_reference/wmean.npy')
    assert len(train) == 673 and xt.shape == (673, len(names))
    training_keys = {i: signature(g.smiles, g.weight) for i, g in components.groupby('row_index')}
    train_smiles = sorted(set(canonical(s) for s in components.smiles))
    generator = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    train_fp = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles]
    components['canonical'] = components.smiles.map(canonical)
    table_cols = ['formulation_id', 'formulation_name', 'source_reference', 'source_url', 'component_casrns', 'epa_class', 'evaluation_role']
    leakage_rows = []
    component_overlap = []

    def leakage(kind, case, smiles, amounts, features, product='', registration='', source_url=''):
        key = signature(smiles, amounts)
        sig_ids = [i for i, k in training_keys.items() if k == key]
        exact = np.flatnonzero(np.all(np.isclose(xt, features, rtol=1e-9, atol=1e-9, equal_nan=True), axis=1))
        terms = [re.escape(t) for t in [product, registration, source_url] if t]
        mask = (train[table_cols].fillna('').astype(str).apply(lambda s: s.str.contains('|'.join(terms), case=False, regex=True)).any(axis=1)
                if terms else pd.Series(False, index=train.index))
        record = dict(kind=kind, case=case, represented_signature=key,
                      same_represented_mixture_training_count=len(sig_ids),
                      same_feature_training_count=len(exact),
                      name_registration_url_match_count=int(mask.sum()),
                      matching_training_ids=' | '.join(train.iloc[sorted(set(sig_ids+list(exact)))].formulation_id),
                      metadata_matching_ids=' | '.join(train.loc[mask, 'formulation_id']),
                      original_study_independence='unverified: training/source study crosswalk incomplete')
        leakage_rows.append(record)
        return record

    def audit_components(frame, kind, case, smi_col, cas_col, name_col):
        for _, row in frame.iterrows():
            smi = canonical(row[smi_col]); cas = str(row[cas_col])
            fp = generator.GetFingerprint(Chem.MolFromSmiles(smi))
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, train_fp))
            hits = components[(components.cas.astype(str) == cas) | (components.canonical == smi)]
            component_overlap.append(dict(kind=kind, case=case, ingredient=row[name_col], CAS=cas,
                canonical_smiles=smi, training_CAS_match=bool((components.cas.astype(str) == cas).any()),
                training_structure_match=bool((components.canonical == smi).any()),
                max_training_Morgan_Tanimoto=float(sims.max()),
                nearest_training_smiles=train_smiles[int(sims.argmax())],
                matched_training_formulations=' | '.join(sorted(hits.formulation_id.unique()))))

    raw = pd.read_csv(ROOT / 'sources/cosmetic_validation/flea_tick_shampoo.csv')
    raw.insert(0, 'source_csv_row', np.arange(2, len(raw)+2))
    raw['pct_mid'] = (raw.pct_low + raw.pct_high)/2
    product_rows, scenarios, vectors = [], [], []
    vector_ids = []
    for product, g in raw.groupby('product', sort=False):
        assert g.eye_binary.nunique() == 1 and g.epa_reg.nunique() == 1
        reg = str(g.epa_reg.iloc[0]); family = '-'.join(reg.split('-')[:2])
        label = int(g.eye_binary.iloc[0])
        assert label == int(g.eye_epa_category.iloc[0] in ['I', 'II'])
        usable = g[g.resolved & g.canonical_smiles.notna() & g.pct_mid.gt(0)].copy()
        mid = float(usable.pct_mid.sum())
        row = dict(product=product, epa_reg=reg, registration_group=family,
                   label=label, source_epa_category=g.eye_epa_category.iloc[0],
                   represented_pct=mid, represented_pct_low=float(usable.pct_low.sum()),
                   represented_pct_high=float(usable.pct_high.sum()),
                   unrepresented_pct=100-mid, known_declared_pct=float(g.pct_mid.sum()),
                   resolved_ingredients=len(usable), declared_records=len(g),
                   source_url=g.source_url.iloc[0], label_evidence=adjudication['shampoo'][reg]['label_basis'],
                   source_issue=adjudication['shampoo'][reg]['issue'],
                   strict_full_formulation_eligible=False,
                   status='partial_composition_diagnostic' if len(usable) else 'abstain_no_resolved_structure',
                   score=np.nan, prediction=np.nan, score_low=np.nan, score_high=np.nan,
                   range_prediction_stable=False)
        if len(usable):
            smiles, amounts = usable.canonical_smiles.tolist(), usable.pct_mid.tolist()
            x = vectorize_components(names, smiles, amounts)
            p = predict_features(bundle, x).iloc[0]
            row.update(score=float(p.eye_A_score), prediction=int(p.eye_A_prediction))
            vectors.append(x.to_numpy()[0]); vector_ids.append(product)
            leakage('shampoo', product, smiles, amounts, x.to_numpy()[0], product, family, g.source_url.iloc[0])
            audit_components(usable, 'shampoo', product, 'canonical_smiles', 'cas', 'chemical_name')
            options = [[float(a)] if a == b else [float(a), float(b)] for a, b in zip(usable.pct_low, usable.pct_high)]
            scores = []
            for k, corner in enumerate(itertools.product(*options)):
                pred = predict_features(bundle, vectorize_components(names, smiles, corner)).iloc[0]
                scores.append(float(pred.eye_A_score))
                scenarios.append(dict(product=product, scenario=f'corner_{k}', amounts=json.dumps(corner),
                    represented_pct=sum(corner), score=float(pred.eye_A_score), prediction=int(pred.eye_A_prediction)))
            row.update(score_low=min(scores), score_high=max(scores),
                       range_prediction_stable=bool((min(scores) >= threshold) == (max(scores) >= threshold)))
            assert min(scores)-1e-12 <= row['score'] <= max(scores)+1e-12
        product_rows.append(row)
    products = pd.DataFrame(product_rows)
    products['correct'] = np.where(products.score.notna(), products.prediction == products.label, np.nan)
    available = products[products.score.notna()].copy()
    grouped = available.groupby('registration_group', as_index=False).agg(
        product=('product', lambda x: ' | '.join(x)), n_products=('product', 'size'),
        label=('label', 'first'), score=('score', 'mean'),
        score_low=('score_low', 'min'), score_high=('score_high', 'max'))
    assert available.groupby('registration_group').label.nunique().max() == 1
    grouped['prediction'] = (grouped.score >= threshold).astype(int)
    grouped['aggregation'] = 'unweighted mean of member midpoint scores; descriptive only'
    grouped['correct'] = grouped.prediction == grouped.label
    result_metrics = [metrics('shampoo_product_rows_partial_composition', available),
                      metrics('shampoo_registration_groups_partial_composition', grouped)]

    cir_all = pd.read_csv(ROOT / 'sources/cosmetic_validation/cir_ingredients.csv')
    cir_all.insert(0, 'source_csv_row', np.arange(2, len(cir_all)+2))
    cir = cir_all[cir_all['endpoint(skin/eye)'] == 'eye'].copy()
    cir.rename(columns={'성분명':'ingredient', '농도(%)':'concentration_pct', '이진라벨':'label'}, inplace=True)
    cir['score'], cir['prediction'] = np.nan, np.nan
    cir['case_id'] = cir.source_csv_row.map(lambda n: f'CIR_row_{n}')
    cir['status'] = 'unresolved_structure'
    cir['composition_status'] = np.where(cir.concentration_pct.eq(100), 'source_neat_single_ingredient', 'diluted_or_formulated_vehicle_unavailable')
    cir['label_audit'] = 'repository-derived category; underlying recovery data/study identity not fully audited'
    for idx, row in cir.iterrows():
        if row.CAS == '151-21-3':
            cir.loc[idx, 'status'] = 'excluded_CAS_SMILES_mismatch'
            continue
        if not row.resolved or pd.isna(row.canonical_smiles):
            continue
        x = vectorize_components(names, [row.canonical_smiles], [row.concentration_pct])
        p = predict_features(bundle, x).iloc[0]
        cir.loc[idx, ['score', 'prediction', 'status']] = [p.eye_A_score, int(p.eye_A_prediction), 'ingredient_only_diagnostic']
        if row.CAS == '122-99-6':
            cir.loc[idx, 'label_audit'] = 'SCCS checked: opacity persists at day15; binary I/II supported, exact II not established'
        elif row.CAS == '56-81-5':
            cir.loc[idx, 'label_audit'] = 'CIR checked: nonirritating rabbit tests up to 100%; source row aggregates studies'
        leakage('CIR', row.case_id, [row.canonical_smiles], [row.concentration_pct], x.to_numpy()[0])
        audit_components(cir.loc[[idx]], 'CIR', row.case_id, 'canonical_smiles', 'CAS', 'ingredient')
    cir['strict_external_eligible'] = False
    cir['correct'] = np.where(cir.score.notna(), cir.prediction == cir.label, np.nan)
    valid = cir[cir.score.notna()].copy()
    valid['equal_CAS_weight'] = 1/valid.groupby('CAS').CAS.transform('size')
    result_metrics.extend([metrics('CIR_ingredient_only_source_label_rows', valid),
                           metrics('CIR_neat_only_source_label_rows', valid[valid.concentration_pct.eq(100)]),
                           metrics('CIR_equal_CAS_weighted_source_label_comparison', valid, 'equal_CAS_weight')])
    conflict = valid.groupby(['CAS', 'ingredient'], as_index=False).agg(
        rows=('label','size'), distinct_labels=('label','nunique'),
        concentrations=('concentration_pct', lambda x: ' | '.join(map(str,x))),
        score_min=('score','min'), score_max=('score','max'))
    conflict['label_conflict_for_same_ingredient_representation'] = conflict.distinct_labels > 1
    leaks = pd.DataFrame(leakage_rows)
    overlap = pd.DataFrame(component_overlap)
    metric_table = pd.DataFrame(result_metrics)
    zein = pd.read_csv(ROOT / 'sources/cosmetic_validation/lechuga2025_surfactant_mixture_zein.csv')
    summary = dict(model='Eye A LR weighted mean', threshold=threshold, refitted=False,
        source_commit=provenance['commit'], latest_main_verified=provenance['latest_main_verified'],
        source_files_recovered=len(provenance['files']), shampoo_products=len(products),
        shampoo_scored=len(available), shampoo_abstained=int(products.score.isna().sum()),
        shampoo_range_unstable_products=int((~available.range_prediction_stable).sum()),
        shampoo_registration_groups_scored=len(grouped), strict_full_formulation_cases=0,
        CIR_total_rows=len(cir_all), CIR_eye_rows=len(cir), CIR_scored=len(valid),
        CIR_neat_scored=int(valid.concentration_pct.eq(100).sum()), CIR_unique_scored_CAS=int(valid.CAS.nunique()),
        zein_rows_excluded=len(zein), CIR_contradictory_CAS=int((conflict.distinct_labels > 1).sum()),
        same_feature_training_matches=int(leaks.same_feature_training_count.sum()),
        same_represented_mixture_training_matches=int(leaks.same_represented_mixture_training_count.sum()),
        shampoo_metadata_training_matches=int(leaks[leaks.kind=='shampoo'].name_registration_url_match_count.sum()),
        original_study_independence='unverified; no absence-of-overlap guarantee from missing identifiers',
        metrics=result_metrics)
    for name, frame in [('shampoo_predictions.csv',products),('shampoo_registration_groups.csv',grouped),
                        ('shampoo_range_scenarios.csv',pd.DataFrame(scenarios)),('shampoo_components.csv',raw),
                        ('CIR_eye_audit_predictions.csv',cir),('CIR_same_chemical_conflicts.csv',conflict),
                        ('metrics.csv',metric_table),('training_overlap_audit.csv',leaks),('component_similarity.csv',overlap)]:
        csv(name, frame)
    np.save(OUT / 'shampoo_features.npy', np.stack(vectors))
    dump('shampoo_feature_row_ids.json', vector_ids)
    dump('summary.json', summary)
    with pd.ExcelWriter(ROOT / '외부검증_결과.xlsx', engine='openpyxl') as writer:
        pd.DataFrame([{'항목':k,'값':str(v)} for k,v in summary.items() if k!='metrics']).to_excel(writer, sheet_name='요약',index=False)
        for name, data in [('샴푸_제품별',products),('샴푸_등록번호별',grouped),('함량범위',pd.DataFrame(scenarios)),
                           ('샴푸_원료',raw),('지표_참고용',metric_table),('CIR_eye',cir),('CIR_성분반복',conflict),
                           ('학습중복감사',leaks),('성분유사도',overlap)]:
            data.to_excel(writer,sheet_name=name,index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes='A2'; sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width=min(55,max(12,len(str(col[0].value))+3))
    make_report(summary, products, grouped, metric_table, cir, conflict, leaks)
    fixture=json.loads((ROOT/'model/smoke_test_cases.json').read_text(encoding='utf-8'))
    for c in fixture['cases']:
        score=predict_features(bundle,vectorize_components(names,c['smiles'],c['amounts'])).iloc[0]
        np.testing.assert_allclose(score.eye_A_score,c['expected_score'],rtol=0,atol=1e-10)
        assert int(score.eye_A_prediction)==c['expected_prediction']
    assert sha(model_path)==protocol['model_sha256']
    verification=dict(status='passed', source_blob_hashes_verified=True, training_snapshot_hashes_verified=True,
        frozen_threshold_verified=True, model_unchanged=True, locked_environment_verified=True,
        saved_model_regression_fixtures=len(fixture['cases']),
        source_rows_accounted_for=len(raw)==28 and len(cir_all)==78,
        no_fit_called=True, grouped_metrics_account_for_shared_registration=True)
    dump('verification.json',verification)
    (ROOT/'artifact_checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_checksums.json'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_report(summary, products, grouped, metric_table, cir, conflict, leaks):
    os.environ['MPLCONFIGDIR']=str(ROOT/'plot_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    short=['Davis','No-Bite','Adams Cleansing','Zodiac','Adams Plus']
    axes[0].barh(short,products.represented_pct,color='#267d86')
    axes[0].set(xlabel='Represented nominal mass (%)',xlim=(0,100),ylim=(-.5,4.5),title='Most composition is not represented')
    for i,x in enumerate(products.represented_pct): axes[0].text(x+1,i,f'{x:.3f}%',va='center')
    for i,row in products.iterrows():
        if pd.notna(row.score):
            axes[1].errorbar(row.score,i,xerr=[[max(0,row.score-row.score_low)],[max(0,row.score_high-row.score)]],
                fmt='o',color='#c46036' if row.label else '#226496',capsize=5)
        else: axes[1].text(.05,i,'Abstain: no resolved structure',va='center')
    axes[1].set(yticks=range(5),yticklabels=short,xlim=(0,1),ylim=(-.5,4.5),xlabel='Frozen Eye A score (not calibrated probability)',title='Partial-composition diagnostic')
    axes[1].axvline(summary['threshold'],color='black',ls='--',label=f"Fixed cutoff {summary['threshold']:.4f}")
    axes[1].legend(loc='lower right',fontsize=8)
    fig.savefig(OUT/'shampoo_results.png',dpi=180);plt.close(fig)
    img=base64.b64encode((OUT/'shampoo_results.png').read_bytes()).decode()
    m=summary['metrics'][0]; g=summary['metrics'][1]
    table=products[['product','source_epa_category','represented_pct','score','prediction','correct','score_low','score_high','status']].copy()
    table.columns=['제품','참고 EPA 등급','구조 표현 함량(%)','모델 점수','예측(1=I/II)','라벨 일치','범위 최저','범위 최고','상태']
    notes='''<p>점수는 화장품 자극 확률이 아닙니다. 음성은 EPA III/IV를 뜻하며 무자극을 뜻하지 않습니다.
이번 샴푸는 반려동물용 살충제 제형입니다. 일반 사람용 화장품의 성능으로 확대 해석할 수 없습니다.</p>
<p>미확인 성분을 물이나 대표 계면활성제로 임의 대체하지 않았습니다. 범위의 중간 함량을 사용하고, 구조가 확인된 원료의 최저·최고 함량 조합도 계산했습니다.
이 범위는 알려진 원료만 바꾼 조건부 분석이며, 공개되지 않은 제형 전체의 불확실성을 포함하지 않습니다.</p>
<p>Zodiac 2724-485와 Adams Plus 2724-485-270은 같은 기본 EPA 등록번호입니다. 제품별 결과는 유지하되 등록번호별 평균 점수로 별도 집계했습니다.
같은 계열을 독립 음성 표본 두 개로 해석하면 안 됩니다. 원 시험 식별자가 없어 모든 연구의 독립성까지 확정한 것은 아닙니다.</p>
<p>원본 노트북은 ChemBERTa 기반 예전 모델과 0.5 기준을 사용했습니다. 이번 분석은 현재 최종 Eye A LR 모델과 고정 임계값을 사용했으며, 외부 라벨로 재학습하거나 기준을 바꾸지 않았습니다.
서로 다른 모델의 결과 차이가 모델 우열을 입증하지는 않습니다.</p>'''
    rawlinks=''.join(f'<li><a href="{html.escape(r.source_url)}">{html.escape(r["product"])} 원본 SDS</a></li>' for _,r in products.iterrows())
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Eye 외부 검증 결과</title>
<style>body{{font:16px/1.65 'Malgun Gothic',sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#20323a}}h1,h2{{color:#173d4a}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{border:1px solid #d5dee2;padding:9px;vertical-align:top}}th{{background:#eaf2f4}}.box{{background:#eff6f7;border-left:5px solid #267d86;padding:16px}}.scroll{{overflow-x:auto}}small{{color:#52616b}}img{{max-width:100%}}</style>
<h1>Eye 최종 모델 외부 검증 결과</h1><p>2026-11-09 · Eye A / LogisticRegression / weighted mean · 임계값 {summary['threshold']:.12f}</p>
<div class="box"><b>샴푸 5종 중 4종 계산 완료: 제공 라벨과 {int(m['TP']+m['TN'])}/4 일치.</b>
<p>제품별 민감도 {m['sensitivity']:.1%}, 특이도 {m['specificity']:.1%}, MCC {m['MCC']:.3f}, ROC-AUC {m['ROC_AUC']:.3f}.
같은 등록번호를 묶으면 {len(grouped)}계열, 일치 {int(g['TP']+g['TN'])}/{len(grouped)}, MCC {g['MCC']:.3f}입니다.</p>
<p><b>이는 공개된 부분 조성에 대한 탐색적 외부 사례 비교입니다.</b> 제형 전체와 원 시험 독립성이 모두 확인된 정식 외부 검증 표본은 0건입니다.</p></div>
<p><b>함량 범위에 따른 판정 불안정:</b> Zodiac과 Adams Plus는 공개된 에탄올 함량 범위 안에서 점수가 임계값을 가로지릅니다.
중간 함량의 0/4 일치를 모든 가능한 조성에서의 결과로 해석하면 안 됩니다. No-Bite와 Adams Cleansing의 음성 예측은 계산한 함량 범위에서 유지됩니다.</p>
<h2>제품별 실제 계산</h2><div class="scroll">{table.to_html(index=False,float_format=lambda x:f'{x:.6f}',na_rep='계산 불가')}</div>
<img src="data:image/png;base64,{img}" alt="샴푸별 구조 표현 비중과 고정 모델 점수">
{notes}<h2>등록번호를 묶은 결과</h2><div class="scroll">{grouped.to_html(index=False,float_format=lambda x:f'{x:.6f}')}</div>
<h2>학습자료 중복 확인</h2><p>최종 학습 673행과 성분 CAS·구조, 표현된 조성, 특성 벡터, 샴푸 제품명·등록번호·출처 URL을 대조했습니다.
샴푸의 이름·등록번호·출처 URL 일치는 {summary['shampoo_metadata_training_matches']}건입니다.
공통 성분은 학습 누출과 같지 않습니다. 정확한 입력 일치 여부는 <code>outputs/training_overlap_audit.csv</code>, 원료별 최근접 유사도는 <code>outputs/component_similarity.csv</code>에 있습니다.</p>
<div class="scroll">{leaks[leaks.kind=='shampoo'].drop(columns=['represented_signature']).to_html(index=False)}</div>
<h2>CIR 원료 자료: 별도 보조 분석</h2><p>전체 {summary['CIR_total_rows']}행 중 eye {summary['CIR_eye_rows']}행입니다. 구조 미해결 행과 SLS의 CAS/SMILES 불일치 1행을 제외하여 {summary['CIR_scored']}행({summary['CIR_unique_scored_CAS']}종)을 계산했습니다.
100% 원료로 기재된 {summary['CIR_neat_scored']}행을 따로 집계했습니다. 희석 시험은 용매·기제가 없으므로 원료 구조만 입력한 참고값이며 시험액 재현 예측이 아닙니다.
같은 성분에서 농도별 라벨이 달라지는 {summary['CIR_contradictory_CAS']}종도 확인했습니다. 단일 성분의 정규화 가중평균은 농도를 바꾸어도 같은 벡터가 됩니다.</p>
<p>SLS 행은 나트륨염 CAS에 중성 산 구조가 연결되어 제외했습니다. Phenoxyethanol은 원문에서 15일에도 한 마리의 각막 혼탁이 남아 있어 정확한 EPA II는 확정할 수 없지만 I/II 이진 양성과는 부합합니다.
그 외 CIR 환산 라벨 전체를 원 시험까지 재확인한 것은 아닙니다. 표의 CIR 수치는 제공된 라벨과의 보조 비교입니다.</p>
<div class="scroll">{metric_table.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<h2>제외한 데이터와 원본 추적</h2><p>Zein {summary['zein_rows_excluded']}행은 단백질 용해 지표여서 EPA 눈자극 성능 계산에서 제외했습니다.
원본 폴더의 6개 파일은 로컬 Git 객체에서 복구하여 Git blob 및 SHA-256 해시를 검증했습니다. 사용 커밋은 <code>{summary['source_commit']}</code>입니다.
네트워크 제한으로 최신 main과의 동일성은 확인하지 못했습니다. 원본, 모델 복사본, 학습 기준 자료, 실행 코드, 프로토콜과 해시를 함께 보관했습니다.</p>
<p><a href="외부검증_결과.xlsx">전체 결과 Excel</a> · <a href="outputs/summary.json">수치 요약</a> · <a href="protocol.json">사전 고정 평가 규칙</a> · <a href="source_adjudication.json">원문 판독 기록</a></p>
<h2>근거 자료</h2><ul>{rawlinks}<li><a href="{adjudication_url('epa_category_reference')}">EPA 눈자극 분류 기준</a></li>
<li><a href="{adjudication_url('registration_group_reference')}">EPA 기본 등록과 유통사 등록 관계</a></li></ul>
<p><small>원본 파일 및 최종 모델 변경 없음 · 외부 자료로 모델 학습/임계값 선택 없음 · 임의 구조 또는 물 보충 없음.</small></p></html>'''
    (ROOT/'결과보고서.html').write_text(text,encoding='utf-8')
    (ROOT/'결과요약.txt').write_text(
        f"Eye A LR 고정 모델 외부 사례 비교\n임계값: {summary['threshold']}\n샴푸 5종 중 4종 계산, 1종 구조 미해결로 제외.\n"
        f"제품별 일치 {int(m['TP']+m['TN'])}/4, 민감도 {m['sensitivity']:.4f}, 특이도 {m['specificity']:.4f}, MCC {m['MCC']:.4f}, AUC {m['ROC_AUC']:.4f}.\n"
        f"기본 EPA 등록번호별 {len(grouped)}계열로 중복 보정한 수치도 별도 제공.\n"
        "부분 조성에 대한 탐색적 비교이며, 화장품 완제품의 독립 외부 검증 성공을 뜻하지 않음.\n"
        "전체 조성과 원 시험 독립성을 모두 확인한 정식 검증 표본은 0건.\n"
        "원본 6개 파일은 지정 Git 저장소의 로컬 객체에서 복구·해시 검증. 최신 main 확인은 네트워크 제한으로 미완료.\n"
        "상세 결과: 결과보고서.html / 외부검증_결과.xlsx\n재실행: python run_validation.py\n",encoding='utf-8')


def adjudication_url(key):
    return json.loads((ROOT/'source_adjudication.json').read_text(encoding='utf-8'))[key]


if __name__=='__main__':
    main()
