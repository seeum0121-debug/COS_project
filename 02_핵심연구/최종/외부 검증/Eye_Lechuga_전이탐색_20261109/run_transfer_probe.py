"""Eye 모델 — Lechuga 외부 적용 평가 및 전이 탐색.

skin 팀의 `Lechuga_외부적용평가_전이학습_탐색결과_20261109.pdf` 설계를 eye
모델에 그대로 적용한 대응 실험이다. 설계를 바꾸지 않았고, 결과를 보고
임계값이나 구조 정책을 조정하지 않았다.

동일하게 맞춘 것
  - 후보 구조 2종만 사용: Lauramine Oxide(A), Myristamine Oxide(B).
    원문 사슬길이와 부합하는 단일 명확 구조이며 UVCB 가정을 쓰지 않는다.
  - A 또는 B를 하나 이상 포함하는 조건만 포함. 나머지 성분의 구조는 입력에서
    제외하되 성분명과 누락량을 기록한다.
  - 1:2 배합 및 동일 보고 조성·농도에서 이진 라벨이 충돌하는 조건은 제외.
  - 라벨은 ZN >= 200 작업 기준. EPA 등급과 동등하다고 주장하지 않는다.
  - 표현된 성분 조합(A만 / B만 / A+B)으로 GroupKFold 3-fold.
  - 농도 해석 2종을 모두 계산하고, 결과가 좋은 쪽을 채택하지 않는다.
      total    : 2성분이면 표시 농도를 절반씩 배분
      per_comp : 각 성분에 표시 농도를 그대로 적용
  - 새 head의 임계값은 0.5 고정, 결과 기반 튜닝 없음.
    기존 eye 모델은 저장된 임계값 0.4814867474694138을 유지한다.

eye에서 달라지는 것 (의도적)
  - ChemBERTa 비교군이 없다. 최종 Eye A LR는 ChemBERTa가 아니며 새 head를
    붙일 frozen backbone이 존재하지 않는다
    (Lechuga_Eye_A_B_엄격검토_20261109/protocol.json의 B_gate와 동일한 사유).
    따라서 'RDKit + Zein head'까지만 비교하고, 전이학습 비교는 수행하지 않는다.
  - 기존 eye 모델은 농도를 피처로 쓰지 않는다(metadata concentration_features=false).
    두 농도 해석에서 frozen eye 예측은 동일해야 하며, 그 점을 검증한다.
"""
from __future__ import annotations

import hashlib
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
SRC = BASE / "sources" / "cosmetic_validation" / "lechuga2025_surfactant_mixture_zein.csv"
MODEL = BASE / "model" / "eye_A_LR_wmean.joblib"
sys.path.insert(0, str(BASE / "model"))
import predict as P  # noqa: E402

A = "Lauramine Oxide"
B = "Myristamine Oxide"
# PubChem 확인 단일 명확 구조 (엄격검토 protocol의 allowed_names와 동일)
SMILES = {A: "CCCCCCCCCCCC[N+](C)(C)[O-]",
          B: "CCCCCCCCCCCCCC[N+](C)(C)[O-]"}
SEED = 20261109


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(y, pred, score):
    y = np.asarray(y, int)
    out = {"n": int(len(y)),
           "TP": int(((pred == 1) & (y == 1)).sum()),
           "FP": int(((pred == 1) & (y == 0)).sum()),
           "TN": int(((pred == 0) & (y == 0)).sum()),
           "FN": int(((pred == 0) & (y == 1)).sum())}
    out["MCC"] = round(float(matthews_corrcoef(y, pred)), 4) if len(set(pred)) > 1 or len(set(y)) > 1 else 0.0
    out["balanced_accuracy"] = round(float(balanced_accuracy_score(y, pred)), 4)
    out["ROC_AUC"] = (round(float(roc_auc_score(y, score)), 4)
                      if score is not None and len(set(y)) > 1 else None)
    return out


def build_rows() -> pd.DataFrame:
    d = pd.read_csv(SRC, encoding="utf-8-sig")
    d = d[d["zein_number"].notna()].copy()
    d["p2"] = d["product2"].fillna("")
    d = d[d["product1"].isin([A, B]) | d["p2"].isin([A, B])].copy()
    d = d[d["proportion"].astype(str) != "1-2"].copy()

    key = ["product1", "p2", "proportion", "concentration_pct"]
    nun = d.groupby(key)["binary_label(ZN>=200)"].transform("nunique")
    excluded = d[nun > 1].copy()
    d = d[nun == 1].copy()

    def represented(r):
        return [c for c in (r["product1"], r["p2"]) if c in (A, B)]

    d["represented"] = d.apply(represented, axis=1)
    d["group"] = d["represented"].map(lambda s: "A+B" if len(s) == 2 else ("A" if s == [A] else "B"))
    d["n_declared"] = d.apply(
        lambda r: 1 if not r["p2"] else 2, axis=1)
    d["n_represented"] = d["represented"].map(len)
    d["partner_unresolved"] = d.apply(
        lambda r: bool(r["p2"]) and r["p2"] not in (A, B), axis=1)
    d["label"] = d["binary_label(ZN>=200)"].astype(int)
    return d.reset_index(drop=True), excluded


def amounts(row, scenario):
    """농도 해석에 따른 성분별 표시 농도."""
    c = float(row["concentration_pct"])
    n = int(row["n_declared"])
    per = c / n if scenario == "total" else c
    return per


def main() -> None:
    bundle = P.load_bundle(MODEL)
    names = bundle["feature_names"]
    thr = bundle["threshold"]
    rows, excluded = build_rows()
    assert len(rows) == 52, f"행 수 {len(rows)} (기대 52)"

    results = {}
    frozen_scores = {}
    for scenario in ("total", "per_comp"):
        r = rows.copy()
        r["per_component_pct"] = r.apply(lambda x: amounts(x, scenario), axis=1)
        r["represented_pct"] = r["per_component_pct"] * r["n_represented"]
        r["unrepresented_pct"] = r["per_component_pct"] * (r["n_declared"] - r["n_represented"])

        # --- 피처: 표현된 성분만, 동일 비율 (1:1 또는 단일) ---
        feats, keys = [], []
        for key, grp in r.groupby(r["represented"].map(tuple), sort=True):
            comps = list(key)
            w = [1.0] * len(comps)
            feats.append(P.vectorize_components(names, [SMILES[c] for c in comps], w))
            keys.append(key)
        fmap = {k: f.to_numpy()[0] for k, f in zip(keys, feats)}
        X_struct = np.vstack([fmap[tuple(v)] for v in r["represented"]])

        # --- 1. frozen eye A LR ---
        sc = P.predict_features(bundle, X_struct)
        r["eye_score"] = sc["eye_A_score"].to_numpy()
        r["eye_pred"] = (r["eye_score"] >= thr).astype(int)
        frozen_scores[scenario] = r["eye_score"].to_numpy().copy()
        m_frozen = metrics(r["label"], r["eye_pred"].to_numpy(), r["eye_score"].to_numpy())

        # --- 교차검증 설정 ---
        gkf = GroupKFold(n_splits=3)
        groups = r["group"].to_numpy()
        y = r["label"].to_numpy()
        folds = list(gkf.split(np.zeros(len(r)), y, groups))

        def oof(X):
            s = np.full(len(r), np.nan)
            for tr, te in folds:
                if len(set(y[tr])) < 2:
                    continue
                pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
                                 ("sc", StandardScaler()),
                                 ("lr", LogisticRegression(C=1.0, solver="liblinear",
                                                           class_weight="balanced",
                                                           random_state=SEED))])
                pipe.fit(X[tr], y[tr])
                s[te] = pipe.predict_proba(X[te])[:, 1]
            return s

        # --- 2. 학습 양성비 기준모델 ---
        base = np.full(len(r), np.nan)
        for tr, te in folds:
            base[te] = y[tr].mean()
        m_base = metrics(y, (base >= 0.5).astype(int), base)

        # --- 3. 구조 없는 비교모델 (농도·성분수·누락량) ---
        X_cov = r[["concentration_pct", "per_component_pct", "represented_pct",
                   "unrepresented_pct", "n_declared", "n_represented"]].to_numpy(float)
        s_cov = oof(X_cov)
        m_cov = metrics(y, (s_cov >= 0.5).astype(int), s_cov)

        # --- 4. RDKit + Zein head (eye 피처에 새 분류기) ---
        s_rd = oof(X_struct)
        m_rd = metrics(y, (s_rd >= 0.5).astype(int), s_rd)

        results[scenario] = {
            "frozen_eye_A_LR": m_frozen,
            "training_positive_rate_baseline": m_base,
            "no_structure_covariates": m_cov,
            "RDKit_plus_zein_head": m_rd,
            "ChemBERTa_plus_zein_head": None,
        }
        r.assign(oof_cov=s_cov, oof_rdkit=s_rd).to_csv(
            HERE / f"predictions_{scenario}.csv", index=False, encoding="utf-8-sig")

    identical = bool(np.allclose(frozen_scores["total"], frozen_scores["per_comp"]))

    fold_sizes = [int(len(te)) for _, te in
                  GroupKFold(n_splits=3).split(np.zeros(len(rows)),
                                               rows["label"], rows["group"])]

    out = {
        "generated": "2026-11-09",
        "design_source": "Lechuga_외부적용평가_전이학습_탐색결과_20261109.pdf (skin팀) 설계를 eye에 적용",
        "model": "Eye A LR wmean frozen",
        "threshold_frozen": thr,
        "threshold_new_head": 0.5,
        "label": "ZN >= 200 작업 기준; EPA 등급 아님",
        "n_rows": int(len(rows)),
        "label_counts": {"positive": int(rows["label"].sum()),
                         "negative": int((rows["label"] == 0).sum())},
        "groups": rows["group"].value_counts().to_dict(),
        "fold_test_sizes": fold_sizes,
        "rows_with_unresolved_partner": int(rows["partner_unresolved"].sum()),
        "excluded_label_conflict_rows": int(len(excluded)),
        "structures_used": SMILES,
        "frozen_scores_identical_across_scenarios": identical,
        "chemberta_comparison": "수행하지 않음 — eye 계보에 frozen backbone 부재",
        "results": results,
        "source_sha256": sha256(SRC),
        "model_sha256": sha256(MODEL),
        "model_refit": False,
        "seed": SEED,
    }
    (HERE / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    flat = []
    for sc, mods in results.items():
        for mod, m in mods.items():
            if m is None:
                flat.append({"scenario": sc, "model": mod, "MCC": None,
                             "balanced_accuracy": None, "ROC_AUC": None,
                             "TP": None, "FP": None, "TN": None, "FN": None})
            else:
                flat.append({"scenario": sc, "model": mod, **{k: m[k] for k in
                            ("MCC", "balanced_accuracy", "ROC_AUC", "TP", "FP", "TN", "FN")}})
    pd.DataFrame(flat).to_csv(HERE / "metrics.csv", index=False, encoding="utf-8-sig")

    print(f"행 {len(rows)}  (양성 {int(rows['label'].sum())} / 음성 {int((rows['label']==0).sum())})")
    print(f"표현 성분 조합 그룹: {rows['group'].value_counts().to_dict()}")
    print(f"fold 시험 행 수: {fold_sizes}")
    print(f"미해결 파트너 동반 행: {int(rows['partner_unresolved'].sum())}")
    print(f"라벨 충돌 제외: {len(excluded)}행")
    print(f"frozen eye 점수가 두 농도 해석에서 동일: {identical}")
    for sc, mods in results.items():
        print(f"\n=== {sc} 해석 ===")
        print(f"{'모델':<34}{'MCC':>8}{'균형정확도':>11}{'ROC-AUC':>9}   혼동행렬")
        for mod, m in mods.items():
            if m is None:
                print(f"{mod:<34}{'—':>8}{'—':>11}{'—':>9}   수행 안 함")
                continue
            auc = f"{m['ROC_AUC']:.3f}" if m["ROC_AUC"] is not None else "—"
            print(f"{mod:<34}{m['MCC']:>+8.3f}{m['balanced_accuracy']:>11.3f}{auc:>9}"
                  f"   TP{m['TP']:>3} FP{m['FP']:>3} TN{m['TN']:>3} FN{m['FN']:>3}")
    print(f"\n저장: {HERE}")


if __name__ == "__main__":
    main()
