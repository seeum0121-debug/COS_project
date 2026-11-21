"""Eye A 고정 모델의 화장품 계면활성제 외부 비교 (Lechuga et al. 2025).

사전 지정 설계
--------------
목적    : 모델 점수가 독립 실험값(zein number)의 순위를 따라가는지 확인.
          EPA 등급 일치도가 아니다. zein number는 in vitro 단백질 변성 지표이며
          EPA 눈자극 등급과 동일한 엔드포인트가 아니다. 따라서 임계값 기반
          분류 성능(민감도/특이도)은 1차 결과로 보고하지 않는다.

주 지표  : Spearman 순위상관 (모델 점수 vs zein number)
층화    : 모델에 농도 피처가 없으므로(metadata: concentration_features=false)
          동일 배합은 농도가 달라도 피처가 같다. 농도를 교란으로 처리하여
          0.5 / 1.0 / 2.0% 각 층 안에서 상관을 계산하고, 통합값은 참고로만 본다.
반복측정 : 동일 (배합, 농도)의 반복 측정은 zein number 평균으로 집계한다.
Tier 1  : 단일물질 — 분자 표현 자체의 신호를 본다.
Tier 2  : 이성분 혼합물 — 가중평균 집계 규칙을 본다.
          Tier1과 Tier2의 상관 차이가 집계 규칙의 기여도 추정치다.

모델·임계값·학습자료는 수정하지 않는다. 라벨을 예측에 맞추지 않는다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "Eye_cosmetic_validation_20261109"
SRC = BASE / "sources" / "cosmetic_validation" / "lechuga2025_surfactant_mixture_zein.csv"
MODEL = BASE / "model" / "eye_A_LR_wmean.joblib"
LINKAGE = BASE / "training_reference" / "component_linkage.csv"

sys.path.insert(0, str(BASE / "model"))
import predict as P  # noqa: E402

RNG = np.random.default_rng(20261109)
N_BOOT = 10000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry() -> dict[str, str]:
    reg = {}
    with (HERE / "structure_registry.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            reg[row["surfactant"]] = row["input_smiles"]
    return reg


def parse_weights(proportion: str, n: int) -> list[float]:
    """'1-1' -> [1,1], '1-2' -> [1,2], '1'/결측 -> [1]."""
    if n == 1:
        return [1.0]
    text = (proportion or "").strip()
    parts = [p for p in text.split("-") if p]
    if len(parts) != n:
        raise ValueError(f"배합비 {proportion!r}가 성분 수 {n}와 맞지 않음")
    return [float(p) for p in parts]


def bootstrap_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 4:
        return float("nan"), float("nan")
    stats_ = np.empty(N_BOOT)
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    for i in range(N_BOOT):
        xi, yi = x[idx[i]], y[idx[i]]
        stats_[i] = stats.spearmanr(xi, yi).statistic if len(set(xi)) > 1 and len(set(yi)) > 1 else np.nan
    stats_ = stats_[np.isfinite(stats_)]
    if not len(stats_):
        return float("nan"), float("nan")
    return float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))


def main() -> None:
    bundle = P.load_bundle(MODEL)
    names = bundle["feature_names"]
    registry = load_registry()

    raw = pd.read_csv(SRC, encoding="utf-8-sig")
    raw = raw[raw["zein_number"].notna()].copy()

    # --- 피처 생성 (배합 단위) ---
    raw["components"] = [
        tuple(c for c in (r.product1, r.product2) if isinstance(c, str) and c.strip())
        for r in raw.itertuples()
    ]
    unknown = {c for comps in raw["components"] for c in comps if c not in registry}
    if unknown:
        raise SystemExit(f"구조 미해결 성분: {sorted(unknown)}")

    raw["formulation_key"] = [
        "|".join(comps) + "@" + str(r.proportion)
        for comps, r in zip(raw["components"], raw.itertuples())
    ]

    feature_rows, keys = [], []
    for key, grp in raw.groupby("formulation_key", sort=True):
        first = grp.iloc[0]
        comps = first["components"]
        weights = parse_weights(first["proportion"], len(comps))
        smiles = [registry[c] for c in comps]
        feature_rows.append(P.vectorize_components(names, smiles, weights))
        keys.append(key)
    features = pd.concat(feature_rows, ignore_index=True)
    preds = P.predict_features(bundle, features)
    preds.insert(0, "formulation_key", keys)
    score_map = dict(zip(preds["formulation_key"], preds["eye_A_score"]))

    # --- 반복측정 집계 ---
    raw["eye_A_score"] = raw["formulation_key"].map(score_map)
    agg = (raw.groupby(["formulation_key", "concentration_pct"], as_index=False)
              .agg(zein_number=("zein_number", "mean"),
                   n_replicates=("zein_number", "size"),
                   eye_A_score=("eye_A_score", "first"),
                   mixture_type=("mixture_type", "first"),
                   ionic_character=("ionic_character", "first"),
                   binary_label=("binary_label(ZN>=200)", "max"),
                   product1=("product1", "first"),
                   product2=("product2", "first"),
                   proportion=("proportion", "first")))
    agg["eye_A_prediction"] = (agg["eye_A_score"] >= bundle["threshold"]).astype(int)

    # --- Spearman: 농도층 × 티어 ---
    results = []
    tiers = {"all": None, "Tier1_single": "single", "Tier2_mixture": "binary_mixture"}
    for conc in sorted(agg["concentration_pct"].unique()) + ["pooled"]:
        sub0 = agg if conc == "pooled" else agg[agg["concentration_pct"] == conc]
        for tier, mt in tiers.items():
            sub = sub0 if mt is None else sub0[sub0["mixture_type"] == mt]
            n = len(sub)
            if n < 4 or sub["eye_A_score"].nunique() < 2:
                results.append({"concentration_pct": conc, "tier": tier, "n": n,
                                "spearman": np.nan, "p_value": np.nan,
                                "ci_low": np.nan, "ci_high": np.nan,
                                "n_unique_scores": int(sub["eye_A_score"].nunique())})
                continue
            x = sub["eye_A_score"].to_numpy(float)
            y = sub["zein_number"].to_numpy(float)
            r = stats.spearmanr(x, y)
            lo, hi = bootstrap_spearman(x, y)
            results.append({"concentration_pct": conc, "tier": tier, "n": n,
                            "spearman": round(float(r.statistic), 4),
                            "p_value": float(f"{r.pvalue:.3g}"),
                            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                            "n_unique_scores": int(sub["eye_A_score"].nunique())})
    res = pd.DataFrame(results)

    # --- 학습자료 구조 중복 감사 ---
    link = pd.read_csv(LINKAGE, low_memory=False, usecols=["smiles", "formulation_id"])
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    def canon(s):
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None

    train_canon = {}
    for s, fid in zip(link["smiles"], link["formulation_id"]):
        c = canon(s)
        if c:
            train_canon.setdefault(c, set()).add(fid)
    overlap = []
    for name, smi in registry.items():
        c = canon(smi)
        hits = train_canon.get(c, set())
        overlap.append({"surfactant": name, "canonical_smiles": c,
                        "in_training": bool(hits),
                        "n_training_formulations": len(hits),
                        "example_formulation_ids": ";".join(sorted(map(str, hits))[:5])})
    ov = pd.DataFrame(overlap).sort_values(["in_training", "surfactant"], ascending=[False, True])

    # --- 저장 ---
    agg.to_csv(HERE / "predictions.csv", index=False, encoding="utf-8-sig")
    res.to_csv(HERE / "rank_association.csv", index=False, encoding="utf-8-sig")
    ov.to_csv(HERE / "training_overlap_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "generated": "2026-11-09",
        "model_identity": bundle.get("identity"),
        "threshold": bundle["threshold"],
        "primary_endpoint": "Spearman rank correlation (model score vs zein number)",
        "endpoint_caveat": ("zein number는 in vitro 단백질 변성 지표로 EPA 눈자극 "
                            "등급과 동일한 엔드포인트가 아니다. 분류 성능은 1차 결과가 아니다."),
        "n_raw_rows": int(len(raw)),
        "n_unique_formulations": int(agg["formulation_key"].nunique()),
        "n_analysis_units": int(len(agg)),
        "n_surfactants": len(registry),
        "concentration_strata": sorted(float(c) for c in agg["concentration_pct"].unique()),
        "results": results,
        "training_overlap": {
            "n_overlapping_surfactants": int(ov["in_training"].sum()),
            "overlapping": ov.loc[ov["in_training"], "surfactant"].tolist(),
        },
        "source_sha256": sha256(SRC),
        "model_sha256": sha256(MODEL),
        "bootstrap_iterations": N_BOOT,
        "seed": 20261109,
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- 콘솔 ---
    print(f"분석 단위 {len(agg)}행 (고유 배합 {agg['formulation_key'].nunique()}종, 원자료 {len(raw)}행)")
    print(f"모델 점수 범위 {agg['eye_A_score'].min():.4f} ~ {agg['eye_A_score'].max():.4f}"
          f"  (임계값 {bundle['threshold']:.4f})")
    print(f"임계값 초과 예측 양성: {int(agg['eye_A_prediction'].sum())}/{len(agg)}\n")
    print("=== Spearman 순위상관 (모델 점수 vs zein number) ===")
    print(f"{'농도':>8} {'티어':<16} {'n':>4} {'rho':>8} {'95% CI':>18} {'p':>10}")
    for r in results:
        ci = (f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
              if np.isfinite(r.get("ci_low", np.nan)) else "—")
        rho = f"{r['spearman']:.3f}" if np.isfinite(r["spearman"]) else "—"
        p = f"{r['p_value']:.3g}" if np.isfinite(r["p_value"]) else "—"
        print(f"{str(r['concentration_pct']):>8} {r['tier']:<16} {r['n']:>4} {rho:>8} {ci:>18} {p:>10}")
    print("\n=== 학습자료 구조 중복 ===")
    n_ov = int(ov["in_training"].sum())
    if n_ov:
        for _, r in ov[ov["in_training"]].iterrows():
            print(f"  [중복] {r['surfactant']} — 학습 배합 {r['n_training_formulations']}건")
    else:
        print("  없음")
    print(f"\n저장 위치: {HERE}")


if __name__ == "__main__":
    main()
