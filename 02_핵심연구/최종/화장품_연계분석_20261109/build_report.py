"""Create Korean review artifacts and check the read-only source boundary."""
from pathlib import Path
import os, sys, json, hashlib, html, base64, platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
sys.path.insert(0, str(ROOT / 'inference'))
from predict import load_bundle, predict_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = Path('C:/Windows/Fonts/malgun.ttf')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 140
plt.rcParams['savefig.dpi'] = 180
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def savefig(fig, name):
    fig.savefig(FIG / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def frame_html(df, digits=3):
    return df.to_html(index=False, border=0, float_format=lambda v: f'{v:.{digits}f}', na_rep='계산 불가', escape=True)

def image_html(name, caption):
    encoded = base64.b64encode((FIG / name).read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

def main():
    linkage = json.loads((OUT / 'linkage_summary.json').read_text(encoding='utf-8'))
    uv = json.loads((OUT / 'uncertainty_verification.json').read_text(encoding='utf-8'))
    products = pd.read_csv(OUT / 'product_audit.csv')
    subgroups = pd.read_csv(OUT / 'subgroup_metrics_summary.csv')
    cases = pd.read_csv(OUT / 'uncertainty_cases.csv')
    similarity = pd.read_csv(OUT / 'case_ingredient_training_similarity.csv')
    cas_overlap = pd.read_csv(OUT / 'cosmetic_cas_overlap.csv')
    registry = pd.read_csv(OUT / 'structure_registry.csv')
    bundle = load_bundle(ROOT / 'inference/eye_A_LR_wmean.joblib')
    features = np.load(ROOT / 'cache/component_features.npz')
    names = {'123-86-4': '아세트산부틸', '141-78-6': '아세트산에틸', '144-55-8': '탄산수소나트륨',
             '64-17-5': '에탄올', '67-63-0': '이소프로판올', '77-92-9': '구연산',
             '7732-18-5': '물', '7757-82-6': '황산나트륨'}
    diagnostics = []
    for cas in features.files:
        prediction = predict_features(bundle, features[cas][None, :]).iloc[0]
        diagnostics.append(dict(cas=cas, name=names[cas], score=float(prediction.eye_A_score),
            above_frozen_threshold=bool(prediction.eye_A_prediction), threshold=float(bundle['threshold']),
            purpose='Input plausibility diagnostic only; not a measured irritation label or validated pure-chemical test.'))
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(OUT / 'single_ingredient_diagnostics.csv', index=False, encoding='utf-8-sig')
    water = float(diagnostics.loc[diagnostics.cas == '7732-18-5', 'score'].iloc[0])
    assert diagnostics.above_frozen_threshold.all()

    colors = ['#206b84', '#80abb9', '#b1b9c2', '#d6aa6d', '#ad6464']
    order = ['primary_cosmetic', 'supplementary_title_only', 'outside_study_scope', 'unresolved_product', 'excluded_industrial']
    labels = ['주 분석 화장품 문서', '제품명 기반 보조 문서', '연구 범위 밖', '제품 용도 미확정', '산업용 등 제외']
    counts = [linkage['product_status_counts'][s] for s in order]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(labels[::-1], counts[::-1], color=colors[::-1])
    for i, count in enumerate(counts[::-1]): ax.text(count + 1, i, str(count), va='center')
    ax.set_xlim(0, 88); ax.set_xlabel('문서 수 (고유 제품 수와 다를 수 있음)')
    ax.set_title('원자료 140개 문서의 제품명·분류 점검', loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.01, -.02, '원본 CSV 두 파일은 그대로 보존. 분류 오류는 이번 분석의 선별표에만 반영.', fontsize=9)
    savefig(fig, '01_product_screening.png')

    sg = subgroups[subgroups.subgroup.isin(['original_606', 'any_cosmetic_nonwater', 'no_cosmetic_nonwater'])].copy()
    group_labels = ['전체 606개\n양성 115 / 음성 491', '공통성분 13개\n양성 10 / 음성 3', '공통성분 없는 593개\n양성 105 / 음성 488']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharex=True)
    for ax, metric, title in zip(axes, ['sensitivity', 'specificity'], ['민감도', '특이도']):
        ax.bar(range(3), sg[metric + '_mean'], yerr=sg[metric + '_seed_sd'], color=['#206b84', '#c07754', '#8796a5'], capsize=4)
        ax.set_xticks(range(3), group_labels); ax.set_ylim(0, 1.12); ax.set_title(title)
        for i, value in enumerate(sg[metric + '_mean']): ax.text(i, value + .035, f'{value:.3f}', ha='center')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('기존 평가 예측값의 성분 중복 하위집단 분석', fontweight='bold')
    fig.text(.02, -.025, '3개 seed 평균 ± 표준편차. 신뢰구간이 아님. 13개 하위집단은 작고 불균형하여 성능 일반화 불가.', fontsize=9)
    fig.tight_layout(); savefig(fig, '02_subgroup_metrics.png')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    titles = ['바디스프레이 · 부분 조성', '네일폴리시 · 부분 조성', '배스피즈 · 부분 조성 (보조)']
    for i, row in enumerate(cases.itertuples()):
        ax.plot([row.exact_score_min, row.exact_score_max], [i, i], color='#206b84', lw=7, solid_capstyle='round')
        ax.plot(row.reference_score, i, 'o', color='#e9973e', ms=7, zorder=3)
        ax.text(row.exact_score_max + .003, i, f'{row.exact_score_min:.4f}–{row.exact_score_max:.4f}', va='center', fontsize=10)
    ax.axvline(bundle['threshold'], color='#b84b4b', linestyle='--', label=f"기존 임계값 {bundle['threshold']:.3f}")
    ax.set_yticks(range(3), titles); ax.invert_yaxis(); ax.set_xlim(.465, .66); ax.set_ylim(2.65, -.65)
    ax.set_xlabel('구조가 확인된 성분만 정규화한 입력의 모델 점수')
    ax.set_title('보고된 함량 범위에서 가능한 점수 범위', loc='left', fontweight='bold')
    ax.legend(loc='lower left', frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.text(.01, -.025, '완제품 예측 아님 · 주황 점은 가정한 기준 조성 · 범위는 입력의 수학적 극값이며 통계적 신뢰구간이 아님', fontsize=9)
    savefig(fig, '03_conditional_composition_bounds.png')

    fig, ax = plt.subplots(figsize=(10, 5))
    diag = diagnostics.sort_values('score')
    ax.barh(diag.name, diag.score, color=['#c05a53' if c == '7732-18-5' else '#809aaa' for c in diag.cas])
    ax.axvline(bundle['threshold'], color='#333', linestyle='--', label=f"기존 임계값 {bundle['threshold']:.3f}")
    for i, row in enumerate(diag.itertuples()): ax.text(row.score + .006, i, f'{row.score:.3f}', va='center')
    ax.set_xlim(0, .72); ax.set_ylim(-1.4, len(diag) - .4)
    ax.set_xlabel('단일 구조 입력의 모델 점수 (실측 자극성 자료 아님)')
    ax.set_title('적용 전 점검: 물 입력도 양성 기준을 초과', loc='left', fontweight='bold')
    ax.legend(loc='lower left', frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.text(.01, -.025, '8개 성분 모두 임계값 초과 → 이 성분들의 가중평균 조성도 양성 유지. 이를 제품 예측 신뢰성으로 해석할 수 없음.', fontsize=9)
    savefig(fig, '04_input_plausibility_diagnostic.png')

    curve = pd.read_csv(OUT / 'ethanol_composition_response.csv')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(curve.ethanol_fraction * 100, curve.score_resolved_subset_only, color='#206b84', lw=2.4)
    ax.set_xlabel('원 조성 중 에탄올 함량 (%)'); ax.set_ylabel('부분 조성 모델 점수')
    ax.set_title('조성 반응 예시: 물 함량을 고정하고 에탄올 범위만 변경', loc='left', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.01, -.02, f"물 {curve.water_fraction.iloc[0] * 100:.1f}% 고정, 제외 혼합성분 {curve.omitted_fraction.iloc[0] * 100:.1f}% 가정. 총량 100% 초과 조건 제외. 인과적 효과/처방 추천 아님.", fontsize=8.5)
    savefig(fig, '05_composition_response.png')

    # Exact analytical explanation for threshold stability.
    method = '''METHODS AND LIMITS
Frozen model: Eye A LR weighted-mean, 2265 features; no fit or threshold change.
Endpoint A: EPA categories I/II versus III/IV. Negative is not equivalent to zero irritation.
Subgroup source: saved outer held-out predictions, 606 original rows x 3 seeds.
Augmentation: 67 rows were used in prior model training, not in subgroup evaluation.
Three seed repetitions are not 1818 independent substances. SD across seeds is not a confidence interval.
Product screening: document titles and product categories; metadata screen only, not independently verified complete formulations.
Primary CAS overlap excludes water and valid raw/curated identity disagreements. No CAS auto-correction.
Composition weights refer to represented source components, not assured full product composition.
Conditional cases are purposively selected before computing scores, based on simple discrete constituents and amount ranges.
No inference of water, vehicle, fragrance structures, polymer grades or undisclosed percentages.
For all-finite component features f_i, the frozen LR logit is:
  intercept - beta dot scaler_mean + sum(a_i * (beta dot f_i)) / sum(a_i)
where beta = fitted_logistic_coefficient / fitted_scaler_scale.
Consequently the mixture logit is a convex combination of component logits.
All eight represented components score above the fixed threshold; all their weighted-mean mixtures must also do so.
Threshold stability in these cases is therefore a model property, not evidence of reliable product classification.
Exact extrema use Charnes-Cooper LP: w_i=a_i/sum(a), t=1/sum(a), sum(w)=1;
  lower_i*t <= w_i <= upper_i*t; t >= 1/(1 - omitted_lower_sum).
LP extrema allow omitted quantities at their lower bounds. Unknown remainder stays outside the feature representation.
Scenarios: independent uniform draws for ALL listed component bounds, retain total <=1; 8192 scenarios per case.
The distribution within source ranges is an assumption. Quantiles and threshold fractions are not confidence intervals/risk probabilities.
Reference composition: source midpoints; if sum exceeds one, Euclidean box-constrained projection to sum=1.
This reference is assumed, never a measured composition. Original source fractions already use 0..1 units.
Whole-molecule Morgan bits: radius=2, 2048 bits, no chirality. Descriptors: largest fragment, same saved recipe.
Ionic SMILES are kept intact; no acid-base reactions, pH, solution chemistry, aerosol exposure or mineral/polymer grades modeled.
Bath Fizz acid/bicarbonate reactions are not represented by the ingredient weighted mean.
Training fingerprint similarity is descriptive; no validated applicability-domain threshold was fitted.
The water check is an input plausibility diagnostic, not a newly collected experiment or external performance metric.
No CIR-to-EPA converted labels were used. No external cosmetic accuracy, sensitivity or specificity can be estimated here.
'''
    (ROOT / 'METHODS.txt').write_text(method, encoding='utf-8-sig')
    sources = [
        dict(title='US EPA Chemical and Products Database', url='https://www.epa.gov/chemical-research/chemical-and-products-database-cpdat', use='Chemical/product-use evidence; not assay labels'),
        dict(title='ChemExpo download terminology', url='https://comptox.epa.gov/chemexpo/static/user_guide/download_terminology.html', use='Document/chemical identity and composition field interpretation'),
        dict(title='Johns Manville Expand-O-Flash installation instructions', url='https://www.jm.com/content/dam/jm/global/en/commercial-roofing/Installation-Application_Guides/EOF%20EJ%20Installation%20Instructions.pdf', use='Roofing/building-joint use supports rejecting adhesive from eye cosmetics'),
        dict(title='PubChem PUG REST', url='https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest', use='CAS-to-structure lookup; raw property/synonym responses saved in evidence'),
    ]
    (ROOT / 'evidence/source_register.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')

    # Verify all snapshotted source contents. Also compare every original path's size/mtime.
    manifest = json.loads((ROOT / 'input_manifest.json').read_text(encoding='utf-8'))
    input_checks = []
    for relative, info in manifest.items():
        copied = ROOT / relative
        original = Path(info.get('source', ''))
        original_hash = sha(original) if original.is_file() else None
        input_checks.append(dict(file=relative, snapshot_sha256=sha(copied), expected_sha256=info['sha256'],
            original_sha256=original_hash, snapshot_matches=(sha(copied) == info['sha256']),
            original_matches=(original_hash == info['sha256']) if original_hash is not None else None))
    assert all(c['snapshot_matches'] for c in input_checks)
    assert all(c['original_matches'] is not False for c in input_checks)
    baseline = json.loads((ROOT / 'original_file_inventory.json').read_text(encoding='utf-8'))
    differences = []
    for relative, old in baseline.items():
        path = ROOT.parent / relative
        if not path.exists(): differences.append(dict(path=relative, change='missing')); continue
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns] != old:
            differences.append(dict(path=relative, change='size_or_mtime', before=old, after=[stat.st_size, stat.st_mtime_ns]))
    check = dict(verified_at_utc=datetime.now(timezone.utc).isoformat(), existing_files_checked=len(baseline),
        changed_or_missing_original_files=differences, input_hash_checks=input_checks,
        original_files_unchanged=not differences,
        scope='Original files under the final directory; baseline excludes the newly created analysis folder.',
        model_unchanged=uv['model_sha256'] == 'd71394e691ac5072e4ce900a53b47ddbb65c8685961fe2c9536411acfbbc58a2',
        scientific_checks=uv)
    (OUT / 'verification.json').write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')
    assert not differences, f'Existing filesystem changed: {differences[:5]}'
    text = f'''화장품 연계 분석 결과 — 2026-11-09

결론
화장품과 연결되는 성분 근거 및 조성 불확실성 분석은 완료했다.
그러나 현재 Eye 모델을 화장품 완제품 자극성 판정에 바로 사용하기는 어렵다.
특히 물 단독 입력 점수가 {water:.6f}로, 기존 임계값 {bundle['threshold']:.6f}를 넘어 양성으로 분류된다.
물의 실제 위해성을 뜻하는 결과가 아니다. 현재 입력 표현/학습 범위/판정 기준의 화장품 전이 한계를 드러내는 진단이다.
기존 모델 선정과 파일은 유지했다. 이번 결과로 모델·임계값을 임의 수정하지 않았다.

1. 무엇을 분석했나
- 두 ChemExpo 원자료: 1,472개 성분 기록, 중복 제외 140개 문서.
- 제품명과 분류가 일치하는 주 분석 화장품 문서 77개, 제품명 기반 보조 문서 11개.
- 연구 범위 밖 35개, 용도 미확정 9개, 산업용 등 제외 8개.
- 원 eye 자료 21개 문서 중 eye makeup으로 분류된 접착제는 제외했다. 남은 eye shadow 문서는 3개이다.
- 산업용 합금, 프린터 토너, 잉크도 화장품 분류에 섞여 있어 분석 대상에서 제외했다.
- 원 CSV는 변경하지 않고 제외 이유를 outputs/product_audit.csv에 기록했다.
- 문서 수는 독립된 제품 수와 같지 않으며, 자료의 'full composition' 파일명만으로 전성분 완전성을 보증하지 않는다.

2. 화장품 공통성분 하위집단
- 주 분석 문서에서 물을 제외한 유효 CAS 190종. 기존 평가자료와 겹치는 성분은 7종.
- 원 평가 606개 중 공통성분이 있는 표본 13개(9개 그룹, 양성 10 / 음성 3).
- 기존 outer-fold 예측값을 재사용: 민감도 1.000, 특이도 0.000, MCC 0.000. 3개 seed 모두 전부 양성 판정.
- AUC 평균 0.811은 작은 표본의 순위 지표일 뿐이며 화장품 성능이 좋다는 근거로 쓰지 않는다.
- 공통성분 가중치가 50% 이상인 표본은 3개, 모두 양성. 음성이 없어 AUC/특이도 산출 불가.
- eye shadow 성분과 직접 겹치는 원 평가 표본은 0개.
- 전체 606개 기존 성능: 민감도 0.704, 특이도 0.678, MCC 0.306, AUC 0.719 (3개 seed 평균).
- 67개 추가학습 표본은 평가 수치에 합치지 않았다. 이번 화장품 자료에는 정답 라벨을 만들어 붙이지 않았다.
- 606×3회 예측은 1,818개 독립 표본이 아니다. 하위집단 역시 기존 농약 중심 자료이지 화장품 시험 자료가 아니다.

3. 함량 불확실성 및 조성 반응 사례
- 바디스프레이: 구조 확인 2/3성분, 조건부 점수 {cases.iloc[0].exact_score_min:.4f}–{cases.iloc[0].exact_score_max:.4f}.
- 네일폴리시: 구조 확인 3/4성분, 조건부 점수 {cases.iloc[1].exact_score_min:.4f}–{cases.iloc[1].exact_score_max:.4f}.
- 배스피즈(제품명 기반 보조): 구조 확인 3/4성분, 조건부 점수 {cases.iloc[2].exact_score_min:.4f}–{cases.iloc[2].exact_score_max:.4f}.
- 실제 공개 함량 범위와 총량 제약을 사용했다. 8개 단일 구조의 PubChem CAS/동의어 응답을 보관했다.
- 3사례 × 8,192 = 24,576개 시나리오와 선형계획법의 수학적 점수 극값을 계산했다.
- 점수는 구조가 확인된 일부 성분의 가중치를 재정규화한 결과이다. 완제품 점수/안전성 판정이 아니다.
- 제외 성분: 에톡실화 경화 피마자유, 니트로셀룰로오스, 노녹시놀-10. 임의 대체구조를 넣지 않았다.
- 함량 중앙값은 측정값이 아니며, 합이 100%를 넘으면 제약 안으로 투영한 가정값을 사용했다.
- 사례 선택은 점수를 보기 전 조성·구조·함량 가용성으로 했다. 전수 대표성이나 최적 처방을 주장하지 않는다.
- 구조 확인 성분 중 5/8종은 학습 구조와 정확히 일치하지 않았다. 유사도는 참고값이고 적용영역 합격 기준이 아니다.
- 배스피즈의 산-중탄산염 반응, pH, 물리적 자극, 에어로졸 노출, 고분자/광물 등급은 반영하지 못한다.

4. 판정이 유지되어도 신뢰성이 검증된 것은 아니다
- 세 사례의 함량 범위에서는 임계값을 넘나드는 변화가 없었다.
- 그러나 사용한 8개 성분의 단독 입력이 모두 이미 양성이다. 물 입력도 {water:.6f}이다.
- 현재 LR+가중평균의 혼합 logit은 성분 logit의 가중평균이다.
- 따라서 이 성분들끼리 비율을 바꿔도 계속 양성이 되는 것은 수학적으로 예상되는 동작이다.
- 이 안정성을 '함량 불확실성에도 정확한 화장품 모델'이라는 근거로 제시하면 안 된다.
- EPA I/II 대 III/IV 분류이고, 음성 역시 무자극/안전을 뜻하지 않는다. 점수는 보정된 실제 위험확률이 아니다.

5. 보고서·발표에 쓰는 권장 문장
“화장품 성분 사용 자료와 기존 평가자료의 성분 중복을 분석하고, 공개 함량 범위에 따른
고정 모델의 출력 민감도를 평가하였다. 공통성분 하위집단은 13개로 제한적이었으며,
부분 조성 사례와 물 단독 입력 점검에서 화장품 영역으로의 직접 적용에 한계가 확인되었다.
따라서 본 분석은 화장품 관련 적용 가능성과 한계의 탐색이며, 화장품 완제품의 외부 성능 검증을 의미하지 않는다.”

6. 권장하는 다음 판단
- 기존 원자료 범위의 모델 비교·최종 선정 결과와 화장품으로의 적용 주장을 구분한다.
- 이번 추가 분석은 논문/발표의 '화장품 연계 및 적용 한계'로 활용한다.
- 현재 모델로 화장품의 안전/위험을 단정하거나, 물 결과 하나에 맞춰 임계값을 바꾸지 않는다.
- 화장품용 예측 성능을 주장하려면 타깃에 맞는 라벨 자료와 독립 평가 설계가 필요하다.
- 현 단계에서는 근거 없는 재튜닝보다 이번 한계를 명시하는 편을 권장한다.

7. 검증과 원본 보존
- 모델 SHA256 일치, 모델 재학습/임계값 재선정 없음.
- 분석식과 실제 저장 파이프라인 예측의 최대 오차 {uv['exact_formula_pipeline_max_abs_error']:.3g}.
- 시나리오 함량·질량 제약, 수학적 극값, 원 평가 606개 및 추가학습 제외 확인.
- 기존 파일 {len(baseline):,}개의 크기/수정시각 비교: 변경·삭제 0개. 사용 원본의 SHA256도 일치.
- 모든 새 코드, 사본, 결과는 현재 새 폴더 안에만 저장했다. GitHub 반영 없음.

먼저 볼 파일
- 결과보고서.html: 그래프가 포함된 단독 열람 보고서.
- outputs/subgroup_metrics_summary.csv: 기존 평가 하위집단 요약.
- outputs/uncertainty_cases.csv: 부분 조성 함량 범위 결과.
- outputs/single_ingredient_diagnostics.csv: 물 포함 단일 입력 점검.
- outputs/product_audit.csv: 문서별 포함/제외 근거.
- outputs/verification.json: 수치 및 원본 보존 검증.
- METHODS.txt, protocol.json, case_plan.json: 방법과 사전 분석·사례 선정 기준.

재현 순서 (이 폴더에서 실행)
python -B analyze_linkage.py
python -B prepare_cases.py
python -B analyze_uncertainty.py
python -B build_report.py
setup_analysis.py는 최초 사본 생성용으로 재실행하지 않는다.
Python/라이브러리 버전은 environment.json 참조. prepare_cases는 캐시가 있으면 네트워크 재조회하지 않는다.

출처
''' + '\n'.join(s['title'] + ': ' + s['url'] for s in sources) + '\n'
    (ROOT / '결과요약.txt').write_text(text, encoding='utf-8-sig')

    group_display = subgroups[['subgroup', 'n', 'groups', 'positives', 'negatives', 'sensitivity_mean', 'specificity_mean', 'mcc_mean', 'roc_auc_mean']].copy()
    group_display.columns = ['하위집단', '표본 수', '그룹 수', '양성', '음성', '민감도', '특이도', 'MCC', 'AUC']
    group_display['하위집단'] = ['전체', '화장품 공통성분 포함', '공통성분 비중 50% 이상', '공통성분 없음', '제품명 보조 문서까지 포함', '아이섀도 성분 포함']
    case_display = cases[['title', 'listed_components', 'represented_components', 'exact_score_min', 'reference_score', 'exact_score_max']].copy()
    case_display.columns = ['사례 (부분 조성)', '공개 성분 수', '표현 성분 수', '가능한 최소 점수', '가정한 기준 점수', '가능한 최대 점수']
    cas_display = cas_overlap[cas_overlap.original_evaluation_rows > 0][['cas', 'ingredient_names', 'cosmetic_documents', 'original_evaluation_rows']].copy()
    cas_display.columns = ['CAS', '성분명', '화장품 문서 수', '기존 평가 표본 수']
    contents = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>화장품 연계 분석 결과</title><style>
body{{font-family:'Malgun Gothic',sans-serif;background:#eef2f4;color:#1b2d36;margin:0;line-height:1.75}}main{{max-width:1050px;margin:auto;padding:38px;background:white}}
h1{{font-size:30px;margin-bottom:8px}}h2{{margin-top:38px;border-top:1px solid #ddd;padding-top:24px}}.subtitle{{color:#56717e}}.finding{{padding:20px;border-left:5px solid #b54e48;background:#fff0ee}}
.note{{padding:16px;background:#f1f6f8}}table{{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}}td,th{{border-bottom:1px solid #ddd;padding:10px;text-align:left}}th{{background:#eef4f6}}.scroll{{overflow-x:auto}}
figure{{margin:28px 0}}img{{max-width:100%;height:auto}}figcaption{{font-size:13px;color:#526975}}a{{color:#176c8a}}li{{margin:8px 0}}@media print{{body{{background:white}}main{{padding:0}}h2,figure{{break-inside:avoid}}}}
</style><main><h1>화장품 연계 분석 결과</h1><p class="subtitle">2026-11-09 · Eye A / LR / 가중평균 · 기존 모델 및 원본 유지</p>
<div class="finding"><strong>현재 모델을 화장품 완제품 판정에 바로 적용하기는 어렵습니다.</strong><br>
화장품과의 성분 연결과 조성 민감도 분석은 수행했습니다. 그러나 공통성분 평가 표본은 13개에 불과하고,
물 단독 입력도 {water:.3f}로 양성 기준 {bundle['threshold']:.3f}을 넘었습니다. 이 결과는 물의 실제 위해성이 아니라 모델 적용 한계를 보여줍니다.</div>
<h2>1. 원자료부터 다시 확인했습니다</h2><p>총 1,472개 성분 기록, 중복을 제외한 140개 문서를 점검했습니다.
제품명과 화장품 분류가 일치하는 77개 문서를 주 분석에, 제품명만으로 용도가 추정되는 11개 문서를 보조 분석에 사용했습니다.
눈 화장품으로 분류된 건축용 접착제와 프린터 토너·산업용 합금·잉크 등을 제외했습니다.
이는 자료의 메타데이터에 대한 선별이며, 문서 수를 독립 제품 수나 완전한 처방 수로 해석하지 않습니다.</p>
{image_html('01_product_screening.png', '문서별 선별 기준은 outputs/product_audit.csv에 모두 기록했습니다.')}
<h2>2. 공통성분은 있지만 평가 근거는 제한적입니다</h2><p>물 이외의 화장품 성분 CAS 190종 중 원 평가자료와 겹치는 것은 7종입니다.
이 성분을 포함하는 원 평가 표본은 13개(양성 10, 음성 3)입니다. 기존 임계값에서 13개 모두 양성으로 예측됐습니다.
이는 화장품 제품의 시험 자료가 아니라 기존 농약 중심 평가자료의 하위집단입니다.</p>
<div class="scroll">{frame_html(group_display)}</div>
<p class="note">모두 기존 outer-fold 예측값의 3개 seed 평균입니다. 67개 추가학습 표본은 평가에 넣지 않았습니다.
표본 13개의 AUC 0.811만 강조하면 안 됩니다. 특이도는 0이며, 화장품 성능 일반화에 필요한 규모와 구성이 부족합니다.
공통성분 비중 50% 이상은 양성 3개뿐이고, 아이섀도 성분과 직접 겹치는 원 평가 표본은 없습니다.</p>
{image_html('02_subgroup_metrics.png', '오차 막대는 seed 간 표준편차이며 통계적 신뢰구간이 아닙니다.')}
<div class="scroll">{frame_html(cas_display)}</div><p>성분별 표본 수는 서로 중복될 수 있습니다. 보존제·용매의 사용 중복이 화장품 처방 전체의 유사성을 뜻하지 않습니다.</p>
<h2>3. 공개 함량 범위에 따른 출력 변화를 계산했습니다</h2><p>점수를 보기 전에 세 사례를 선택하고, 총량 100% 이하 제약 안에서 24,576개 시나리오와
수학적 점수 극값을 계산했습니다. 함량 범위는 원자료 그대로 사용했습니다.
중앙값은 가정값이며, 바디스프레이처럼 중앙값 합이 100%를 넘는 경우 제약을 만족하도록 조정했습니다.</p>
<div class="scroll">{frame_html(case_display, 4)}</div>
<p class="note"><strong>모두 일부 성분만 표현한 조건부 분석입니다.</strong> 에톡실화 경화 피마자유·니트로셀룰로오스·노녹시놀-10은
단일 구조로 대체하지 않았습니다. 확인된 성분만 정규화했으므로 완제품의 예측값이 아닙니다.
배스피즈는 제품명 기반 보조 사례이며, 산과 중탄산염의 반응이나 pH 변화도 모델에 반영되지 않습니다.</p>
{image_html('03_conditional_composition_bounds.png', '구조 확인 성분의 조건부 점수 범위. 생물학적 위험확률이나 통계적 신뢰구간이 아닙니다.')}
{image_html('05_composition_response.png', '보고된 범위 내 비율 변화 예시이며, 성분의 인과적 독성 효과나 안전한 처방을 추천하는 그래프가 아닙니다.')}
<h2>4. 판정 유지보다 물 입력 점검이 더 중요합니다</h2><p>세 사례에서 판정은 뒤집히지 않았습니다. 그러나 사용한 8개 성분의 단독 입력이 모두 양성입니다.
LR과 가중평균 표현에서는 혼합 logit이 각 성분 logit의 가중평균이므로, 이 성분들로 만든 조성이 계속 양성으로 남는 것은 예상되는 동작입니다.
따라서 이를 정확성이나 신뢰성이 확인된 결과로 제시하면 안 됩니다.</p>
{image_html('04_input_plausibility_diagnostic.png', '물 입력 점수 0.631은 모델 적용 전 점검 결과입니다. 물의 실제 위해성 평가 결과가 아닙니다.')}
<p>사용 구조 8종 중 5종은 학습 구조와 정확히 일치하지 않습니다. 물의 가장 가까운 학습 성분과의 Morgan 유사도는 약 0.071입니다.
이 유사도는 참고 지표이며 검증된 적용영역 합격 기준은 아닙니다.
점수는 보정된 실제 자극 확률이 아니고, EPA III/IV를 묶은 음성 역시 무자극 또는 안전과 같지 않습니다.</p>
<h2>5. 연구에서의 활용과 권장 결론</h2><p><strong>기존 자료 범위의 최종 모델 선정 결과는 유지하되, 이번 자료는 화장품 연계 및 적용 한계 분석으로 활용하는 것을 권장합니다.</strong>
물 결과 하나에 맞춘 임계값 조정이나 추가 튜닝은 하지 않았습니다. 화장품용 성능을 주장하려면 적합한 라벨 자료와 독립 평가가 필요합니다.</p>
<blockquote class="note">화장품 성분 사용 자료와 기존 평가자료의 성분 중복을 분석하고, 공개 함량 범위에 따른 고정 모델의 출력 민감도를 평가하였다.
공통성분 하위집단은 13개로 제한적이었으며, 부분 조성 사례와 물 단독 입력 점검에서 화장품 영역으로의 직접 적용에 한계가 확인되었다.
따라서 본 분석은 화장품 관련 적용 가능성과 한계의 탐색이며, 화장품 완제품의 외부 성능 검증을 의미하지 않는다.</blockquote>
<h2>6. 재현성과 원본 보존</h2><ul><li>기존 모델 SHA256 일치. 모델·임계값 재학습 없음. GitHub 변경 없음.</li>
<li>분석식과 저장 파이프라인 점수의 최대 오차: {uv['exact_formula_pipeline_max_abs_error']:.3g}.</li>
<li>기존 파일 {len(baseline):,}개 크기·수정시각 비교: 변경·삭제 0개. 사용 원본의 해시 일치.</li>
<li>모든 코드·입력 사본·결과·출처 응답은 이 새 폴더 안에 저장.</li></ul>
<p>상세 수치: outputs/의 CSV · 방법: METHODS.txt · 재현 명령: 결과요약.txt · 검증: outputs/verification.json</p>
<h2>출처</h2><ul>''' + ''.join(f'<li><a href="{s["url"]}">{html.escape(s["title"])}</a> — {html.escape(s["use"])}</li>' for s in sources) + '</ul></main></html>'
    (ROOT / '결과보고서.html').write_text(contents, encoding='utf-8')
    import scipy, sklearn, joblib, rdkit
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        scipy=scipy.__version__, scikit_learn=sklearn.__version__, joblib=joblib.__version__,
        rdkit=rdkit.__version__, matplotlib=matplotlib.__version__)
    (ROOT / 'environment.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    # An Excel workbook provides a convenient review copy if the existing environment supports it.
    try:
        import openpyxl
        workbook_tables = {'제품선별': products, '공통성분': cas_overlap, '하위집단성능': subgroups,
            '함량범위결과': cases, '성분단독점검': diagnostics, '구조유사도': similarity,
            '사례성분': pd.read_csv(OUT / 'uncertainty_case_ingredients.csv')}
        with pd.ExcelWriter(ROOT / '분석결과.xlsx', engine='openpyxl') as writer:
            for sheet, table in workbook_tables.items():
                table.to_excel(writer, sheet_name=sheet, index=False)
                ws = writer.sheets[sheet]; ws.freeze_panes = 'A2'; ws.auto_filter.ref = ws.dimensions
                for col in ws.columns:
                    length = max(len(str(cell.value or '')) for cell in list(col)[:100])
                    ws.column_dimensions[col[0].column_letter].width = min(max(length + 2, 12), 60)
    except ImportError:
        pass
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
             if p.is_file() and p.name != 'artifact_checksums.json'}
    (ROOT / 'artifact_checksums.json').write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(report=str(ROOT / '결과보고서.html'), original_files_checked=len(baseline),
        changed_files=len(differences), water_score=water, artifacts=len(files)), ensure_ascii=True, indent=2))

if __name__ == '__main__': main()
