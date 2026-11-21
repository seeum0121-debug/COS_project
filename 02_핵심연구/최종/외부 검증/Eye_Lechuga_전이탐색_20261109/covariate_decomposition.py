"""구조 표현과 농도 covariate의 기여 분해.

질문
  skin 팀 PDF에서 'RDKit + Zein 분류기'와 'ChemBERTa + Zein head'가 두 농도
  해석 모두에서 MCC 0.377로 완전히 동일했다. 서로 다른 분자 표현이 소수점
  셋째 자리까지 같은 값을 내는 것은 표현이 결과에 기여하지 않을 때 나타나는
  징후다. 그 기여를 직접 분해한다.

설계
  동일한 52행·동일 그룹·동일 GroupKFold 3-fold·동일 임계값 0.5에서
  입력 블록만 바꾸어 비교한다.
    구조만            : eye 모델의 wmean 피처 (2265차원)
    구조 + 농도       : 위에 농도·성분수·누락량 6개 covariate 결합
    농도만            : covariate 6개
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "Eye_cosmetic_validation_20261109"
sys.path.insert(0, str(BASE / "model"))
import predict as P  # noqa: E402

SMILES = {"Lauramine Oxide": "CCCCCCCCCCCC[N+](C)(C)[O-]",
          "Myristamine Oxide": "CCCCCCCCCCCCCC[N+](C)(C)[O-]"}
COV = ["concentration_pct", "per_component_pct", "represented_pct",
       "unrepresented_pct", "n_declared", "n_represented"]
SEED = 20261109


def main() -> None:
    bundle = P.load_bundle(BASE / "model" / "eye_A_LR_wmean.joblib")
    names = bundle["feature_names"]
    out = {"generated": "2026-11-09", "seed": SEED, "scenarios": {}}

    for scenario in ("total", "per_comp"):
        r = pd.read_csv(HERE / f"predictions_{scenario}.csv", encoding="utf-8-sig")
        r["rep"] = r["represented"].map(ast.literal_eval)
        fm = {k: P.vectorize_components(names, [SMILES[c] for c in k], [1.0] * len(k)).to_numpy()[0]
              for k in set(map(tuple, r["rep"]))}
        X = np.vstack([fm[tuple(v)] for v in r["rep"]])
        C = r[COV].to_numpy(float)
        y = r["label"].to_numpy(int)
        g = r["group"].to_numpy()
        folds = list(GroupKFold(n_splits=3).split(X, y, g))

        def oof(M):
            s = np.full(len(y), np.nan)
            for tr, te in folds:
                if len(set(y[tr])) < 2:
                    continue
                pipe = Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("sc", StandardScaler()),
                    ("lr", LogisticRegression(C=1.0, solver="liblinear",
                                              class_weight="balanced", random_state=SEED))])
                pipe.fit(M[tr], y[tr])
                s[te] = pipe.predict_proba(M[te])[:, 1]
            return s

        blocks = {"structure_only": X,
                  "structure_plus_concentration": np.hstack([X, C]),
                  "concentration_only": C}
        res = {}
        for label, M in blocks.items():
            s = oof(M)
            pred = (s >= 0.5).astype(int)
            res[label] = {
                "n_features": int(M.shape[1]),
                "MCC": round(float(matthews_corrcoef(y, pred)), 4),
                "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
                "ROC_AUC": round(float(roc_auc_score(y, s)), 4),
                "TP": int(((pred == 1) & (y == 1)).sum()),
                "FP": int(((pred == 1) & (y == 0)).sum()),
                "TN": int(((pred == 0) & (y == 0)).sum()),
                "FN": int(((pred == 0) & (y == 1)).sum()),
            }
        out["scenarios"][scenario] = res
        out.setdefault("design", {})
        out["design"] = {
            "n_rows": int(len(r)),
            "n_distinct_structure_vectors": int(len(np.unique(X, axis=0))),
            "n_groups": int(r["group"].nunique()),
            "fold_test_sizes": [int(len(te)) for _, te in folds],
            "note": ("구조 피처의 고유 행이 그룹 수와 같으므로, group-disjoint CV에서는 "
                     "시험군의 구조 벡터가 학습에 전혀 등장하지 않는다. 구조만 사용하는 "
                     "모델이 무작위 수준을 내는 것은 설계상의 귀결이며 분자 표현의 "
                     "성능에 대한 증거가 아니다."),
        }

    (HERE / "covariate_decomposition.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"구조 피처 고유 행 {out['design']['n_distinct_structure_vectors']} / "
          f"전체 {out['design']['n_rows']}  · 그룹 {out['design']['n_groups']}개")
    for sc, res in out["scenarios"].items():
        print(f"\n=== {sc} 해석 ===")
        print(f"{'입력 블록':<32}{'차원':>7}{'MCC':>9}{'균형정확도':>11}{'ROC-AUC':>9}")
        for k, v in res.items():
            print(f"{k:<32}{v['n_features']:>7}{v['MCC']:>+9.3f}"
                  f"{v['balanced_accuracy']:>11.3f}{v['ROC_AUC']:>9.3f}")
    print(f"\n저장: {HERE/'covariate_decomposition.json'}")


if __name__ == "__main__":
    main()
