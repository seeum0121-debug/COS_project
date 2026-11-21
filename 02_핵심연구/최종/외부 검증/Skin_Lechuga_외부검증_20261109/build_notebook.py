"""skin 외부검증 보고서 노트북(보고서.ipynb) 생성."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(t):
    return nbf.v4.new_markdown_cell(t.strip())


def code(t):
    return nbf.v4.new_code_cell(t.strip())


cells = [
md("""
# SKIN 모델 — Lechuga 2025 계면활성제 외부검증 보고서

**생성일** 2026-11-09 · **대상** skin RandomForest 체크포인트 · **자료** Lechuga et al. 2025 (ACS Omega)

---

> ### 이 보고서는 eye 모델과 무관합니다
> 같은 Lechuga 원자료를 쓰지만 **모델이 다릅니다.**
> - **eye**: `eye_A_LR_wmean.joblib` — LogisticRegression, 2265 feature, 임계값 0.4815
> - **skin**: `skin_rf_checkpoint.joblib` — RandomForest 500 trees, 2267 feature → nz20 후 524, 임계값 0.33
>
> 두 모델의 수치를 합치거나 비교해 하나의 성능으로 제시하면 안 됩니다.
> eye 쪽 결과는 별도 보고서(`외부 검증/종합보고서/종합외부검증보고서.html`)에 있습니다.

### 요약

| 항목 | 값 |
|---|---|
| 사용 행 | 41 / 원자료 205 (**20.0%**) |
| 구조 해결 성분 | 10 / 22 (PubChem) |
| 고유 배합(군집) | **11** |
| 이성분 혼합물 | 1종 (3행) |
| 분류 정확도 | 0.780 (32/41) |
| 순위상관 (pred_proba vs zein number) | +0.620 |

**주의: 41행은 독립 표본이 아닙니다.** 고유 배합 11종의 농도·반복 측정이며,
유효 표본은 11에 가깝습니다. 아래에서 군집 보정 후 재계산합니다.
"""),

code("""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
HERE = Path.cwd()
RNG = np.random.default_rng(20261109)

res = pd.read_csv(HERE/'github_data'/'lechuga_external_validation_results.csv', encoding='utf-8-sig')
cache = pd.read_csv(HERE/'github_data'/'lechuga_smiles_cache.csv', encoding='utf-8-sig')

print('외부검증 결과 :', res.shape)
print('SMILES 캐시   :', cache.shape)
print('출처          :', res.source.iloc[0])
res.head(3)
"""),

md("""
## 1. 자료 범위 — 80%가 사용되지 못했습니다

원자료는 205행이지만 검증에 쓰인 것은 41행입니다. 원인은 구조 해결 실패입니다.
PubChem 조회에서 22종 중 12종이 `http_error_404`로 실패했고, 한 성분이라도
해결되지 않은 배합은 전부 제외됐습니다(`all_resolved=True`만 사용).
"""),

code("""
ok = cache[cache.status=='found']
ng = cache[cache.status!='found']
print(f'해결 {len(ok)}종 / 미해결 {len(ng)}종\\n')
print('[미해결 — 전부 다분산 에톡실레이트·글리세레스 계열]')
for n in ng.inci_name: print('  -', n)
print('\\n[해결]')
for n in ok.inci_name: print('  -', n)
"""),

md("""
### 결손이 무작위가 아닙니다 — 이 보고서의 가장 중요한 한계

미해결 12종은 전부 **EO 수가 분포를 갖는 다분산 상업 원료**입니다
(Capryleth-6/Laureth-4/Laureth-11 carboxylic acid, Glycereth-2/6/7/17 cocoate,
Polyoxyethylene alkyl ethers, Zinc Coceth Sulfate). PubChem에 단일 화합물
레코드가 없는 것이 당연합니다.

반대로 해결된 10종은 **구조가 명확하고 흔한 물질**입니다(SLS, SLES,
알킬글루코사이드, 아민옥사이드, 아실글루타메이트, 징크코코설페이트).

따라서 남은 41행은 원자료의 무작위 부분집합이 아니라 **구조가 단순한 쪽으로
치우친 표본**입니다. 여기서 얻은 정확도 0.780을 Lechuga 자료 전체나 화장품
계면활성제 일반의 성능으로 읽으면 안 됩니다.
"""),

code("""
# 표본 구성
res['proportion_filled'] = res.proportion.fillna('1').astype(str)
res['fkey'] = (res.product1.fillna('') + '|' + res.product2.fillna('')
               + '@' + res.proportion_filled)

print('고유 배합(군집):', res.fkey.nunique())
print('mixture_type   :', res.mixture_type.value_counts().to_dict())
print('농도 수준      :', sorted(res.concentration_pct.unique()))
print()
print(res.fkey.value_counts().rename('행수').to_frame())
"""),

md("""
**이성분 혼합물은 단 1종(3행)입니다.** 즉 이 검증은 사실상 단일물질 검증이며,
혼합물 집계 규칙은 검증되지 않았습니다.
"""),

md("""
## 2. 분류 성능
"""),

code("""
tp = int(((res.pred_binary==1)&(res.true_binary==1)).sum())
tn = int(((res.pred_binary==0)&(res.true_binary==0)).sum())
fp = int(((res.pred_binary==1)&(res.true_binary==0)).sum())
fn = int(((res.pred_binary==0)&(res.true_binary==1)).sum())
acc = (tp+tn)/len(res)
sens = tp/(tp+fn) if tp+fn else np.nan
spec = tn/(tn+fp) if tn+fp else np.nan
den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
mcc = (tp*tn-fp*fn)/den if den else np.nan
auc = stats.rankdata(res.pred_proba)[res.true_binary==1].mean()
auc = (auc - (res.true_binary==1).sum()/2 - 0.5)/ (res.true_binary==0).sum()

print(f'혼동행렬  TN={tn}  FP={fp}  FN={fn}  TP={tp}')
print(f'정확도    {acc:.4f}   ({tp+tn}/{len(res)})')
print(f'민감도    {sens:.4f}')
print(f'특이도    {spec:.4f}')
print(f'MCC       {mcc:+.4f}')
print(f'ROC-AUC   {auc:.4f}')
print(f'\\npred_proba 범위 {res.pred_proba.min():.4f} ~ {res.pred_proba.max():.4f}')
"""),

md("""
### 이 수치를 그대로 쓰면 안 되는 이유

행 단위로 계산한 값입니다. 41행은 11개 배합의 농도·반복 측정이므로 관측이
독립이 아닙니다. 같은 배합이 최대 5행까지 들어가 있어, 그 배합이 맞으면
정확도가 5행만큼 올라갑니다. 아래에서 배합 단위로 다시 봅니다.
"""),

code("""
# 배합 단위 집계 — 같은 배합은 1건으로
byf = res.groupby('fkey').agg(
    n_rows=('match','size'),
    pred_proba=('pred_proba','first'),
    pred_binary=('pred_binary','first'),
    true_mode=('true_binary', lambda s: int(s.mode().iloc[0])),
    true_mixed=('true_binary', lambda s: s.nunique()>1),
    zein_mean=('zein_number','mean'))
byf['correct'] = byf.pred_binary==byf.true_mode

print(f'배합 단위 정확도: {byf.correct.mean():.4f}  ({int(byf.correct.sum())}/{len(byf)})')
print(f'행 단위 정확도  : {acc:.4f}  ({tp+tn}/{len(res)})')
print(f'\\n같은 배합 안에서 참조라벨이 갈린 경우: {int(byf.true_mixed.sum())}건')
byf
"""),

md("""
## 3. 순위상관 — 군집 보정

참조값 zein number는 연속값이므로 임계값에 의존하지 않는 순위 비교가
분류 정확도보다 정보량이 많습니다. 다만 배합 단위 비독립성을 보정해야 합니다.
"""),

code("""
def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2: return np.nan
    return stats.spearmanr(x, y).statistic

x_all, y_all = res.pred_proba.to_numpy(float), res.zein_number.to_numpy(float)
obs = rho(x_all, y_all)
naive_p = stats.spearmanr(x_all, y_all).pvalue

groups = res.groupby('fkey').indices
gk = list(groups)
fscore = res.groupby('fkey').pred_proba.first()
vals = fscore.to_numpy(float)

N = 20000
perm = np.array([rho(res.fkey.map(dict(zip(fscore.index, RNG.permutation(vals)))).to_numpy(float), y_all)
                 for _ in range(N)])
perm = perm[np.isfinite(perm)]
p_cluster = float((np.abs(perm) >= abs(obs)).mean())

boot = []
for _ in range(N):
    idx = np.concatenate([groups[gk[j]] for j in RNG.integers(0, len(gk), len(gk))])
    boot.append(rho(x_all[idx], y_all[idx]))
boot = np.array([b for b in boot if np.isfinite(b)])
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f'관측 rho              : {obs:+.4f}')
print(f'행단위 p (과장됨)     : {naive_p:.3g}')
print(f'군집 치환검정 p       : {p_cluster:.4f}   <- 보고값')
print(f'군집 부트스트랩 95%CI : [{lo:+.4f}, {hi:+.4f}]')
import math
print(f'군집 수               : {len(gk)}  (서로 다른 치환 배열 최대 {math.factorial(len(gk)):,}개 -> p값 해상도 한계 {1/math.factorial(len(gk)):.2g})')
"""),

code("""
# 배합 단위(각 배합 1점)로도 확인 — 가장 보수적
r2 = rho(byf.pred_proba, byf.zein_mean)
p2 = stats.spearmanr(byf.pred_proba, byf.zein_mean).pvalue
print(f'배합 단위 (n={len(byf)}): rho = {r2:+.4f}, p = {p2:.4f}')
print('\\n행 단위와 배합 단위가 크게 다르면 반복측정이 상관을 부풀린 것입니다.')
"""),

md("""
## 4. skin 팀 개발검증 감사가 지적한 결함

`검증/skin_team_audit_20261108/skin팀_개발검증_보고서.txt`에 이미 P1 등급 문제
세 가지가 기록돼 있습니다. **이 보고서의 수치를 해석할 때 반드시 함께 읽어야 합니다.**

### P1-a. 라벨 정의 충돌
production 라벨 B는 **EPA I/II/III = 1, IV = 0**인데, 예측 출력 문구는 여전히
"양성 = EPA I/II, 음성 = EPA III/IV"라고 표시합니다. Category III의 의미가
반대로 전달됩니다.

다만 **본 외부검증의 참조라벨은 EPA가 아니라 zein number ≥ 200**이므로
이 충돌의 직접 영향은 받지 않습니다. 영향을 받는 것은 `pred_binary`를 EPA
등급으로 해석하려는 시도입니다. 그렇게 해석하면 안 됩니다.

### P1-b. 외부검증 노트북의 feature 오류
`cosmetic_validation/skin_irritation_validation.ipynb`가 성분별 descriptor에서
`resolved_pct`를 찾는데, 이 값은 제형 전체 합계여서 성분 dict에 없습니다.
감사에서 `KeyError: 'resolved_pct'` 재현을 확인했습니다. 농도 feature가 외부검증
경로에 전파되지 않았다는 뜻입니다.

→ **본 CSV의 `pred_proba`가 농도 feature를 반영한 체크포인트로 산출됐는지
확인이 필요합니다.** 아래에서 농도별 점수 변화로 간접 점검합니다.

### P1-c. 임계값 선택과 성능 평가의 미분리
공식 20회 그리드에서 평가 대상 OOF의 정답으로 임계값을 고른 뒤 같은 OOF에서
성능을 보고했습니다. 보고된 MCC·민감도·특이도는 임계값 튜닝까지 분리된
일반화 성능이 아닙니다.
"""),

code("""
# P1-b 간접 점검: 같은 배합에서 농도가 다르면 점수가 달라지는가?
chk = (res.groupby(['fkey','concentration_pct']).pred_proba.first()
          .reset_index().pivot(index='fkey', columns='concentration_pct', values='pred_proba'))
print('배합 × 농도별 pred_proba')
print(chk.to_string())
varies = res.groupby('fkey').pred_proba.nunique()
print(f'\\n농도가 달라도 점수가 동일한 배합: {int((varies==1).sum())}/{len(varies)}')
print('→ 전부 동일하면 농도 feature가 외부검증 경로에 반영되지 않은 것입니다 (P1-b와 일치).')
"""),

md("""
## 5. 결론

1. **사용된 자료는 원자료의 20%이고, 결손이 무작위가 아닙니다.** 구조가 단순한
   10종만 남고 다분산 에톡실레이트 12종이 전부 빠졌습니다. 정확도 0.780을
   화장품 계면활성제 일반의 성능으로 읽을 수 없습니다.

2. **유효 표본은 41이 아니라 11입니다.** 고유 배합 11종의 농도·반복 측정입니다.
   행 단위 통계는 전부 과장돼 있습니다.

3. **혼합물은 1종(3행)뿐입니다.** 혼합물 집계 규칙은 검증되지 않았습니다.

4. **참조라벨은 zein number이지 EPA 등급이 아닙니다.** in vitro 단백질 변성
   지표이며, 이 결과를 EPA 피부자극 분류 성능으로 제시하면 안 됩니다.

5. **감사가 지적한 P1 결함 3건이 미해결 상태입니다.** 특히 외부검증 경로의
   농도 feature 누락(P1-b)은 본 CSV의 산출 조건 자체에 영향을 줄 수 있습니다.

### 정확한 진술

> 구조가 명확한 계면활성제 10종(배합 11종)에 한해, skin 모델 점수가 zein number의
> 순위와 양의 방향으로 연관됐다. 표본이 작고 결손이 체계적이며 혼합물이 1종뿐이어서,
> 이를 화장품 계면활성제 전반에 대한 외부검증 성공으로 일반화할 수 없다.

### 권고

1. 미해결 12종에 **대표구조 규칙**을 적용해 205행 전체를 복원할 것.
   eye 쪽에서 사용한 규칙(`Lechuga2025_화장품_외부검증_20261109/resolve_structures.py`,
   R1–R9)을 그대로 쓸 수 있습니다. 다만 그 규칙은 근사이므로 한계를 명시해야 합니다.
2. **PubChem이 반환한 Capryl Glucoside 구조를 확인할 것.** 캐시의 SMILES는
   decyl(C10) glucoside이고 IUPAC도 `2-decoxy...`인데, INCI 'Capryl'은 통상 C8입니다.
   원료 사양과 대조가 필요합니다.
3. 농도 feature 전파(P1-b)를 수정한 뒤 **재산출**할 것. 수정 전후 점수를 비교해
   본 보고서 수치의 유효성을 판정해야 합니다.
4. 라벨 B 정의(P1-a)를 출력·문서·외부검증 매핑에 통일할 것.
"""),

code("""
# 재현 정보
import sklearn, rdkit, sys, hashlib
print('Python     ', sys.version.split()[0])
for m in (np, pd, sklearn):
    print(f'{m.__name__:<11}', m.__version__)
for f in sorted((HERE/'github_data').glob('*.csv')):
    print(f'\\n{f.name}')
    print('  sha256', hashlib.sha256(f.read_bytes()).hexdigest()[:32], '...')
    print('  행수  ', sum(1 for _ in f.open(encoding='utf-8-sig')) - 1)
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = HERE / "보고서.ipynb"
nbf.write(nb, out)
print(f"생성: {out}  (셀 {len(cells)}개)")
