"""적용범위(AD) 내부 부분집합 재분석.

질문: Tier2의 역상관(-0.344)은 외삽 때문인가, 모델 자체의 방향 오류인가.
      학습 화학공간 안쪽으로 좁혔을 때 상관이 0으로 수렴하면 외삽 설명이
      지지된다. 안쪽에서도 음으로 남으면 모델 방향 자체의 문제다.

정직성 장치
  - 본 분석은 사후(post-hoc) 부분집합 분석이다. AD 임계값은 결과를 본 뒤
    고른 값이 아니라, 관례적 분위수(p50/p75/p90/p95)를 전부 보고한다.
  - 단일 임계값만 보고하지 않는다. 임계값에 따라 결론이 바뀌면 그 사실을
    그대로 적는다.
  - 표본이 줄어 검정력이 낮아지므로, 유의성보다 CI 폭과 추정치 안정성을 본다.
  - 학습자료와 구조가 겹치는 Sodium lauryl sulfate 제외 민감도도 같이 낸다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "Eye_cosmetic_validation_20261109"
sys.path.insert(0, str(BASE / "model"))
import predict as P  # noqa: E402

RNG = np.random.default_rng(20261109)
N_PERM = 20000
N_BOOT = 20000
OVERLAP_SURFACTANT = "Sodium lauryl sulfate"


def rho(x, y):
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return np.nan
    return stats.spearmanr(x, y).statistic


def cluster_tests(sub: pd.DataFrame):
    """배합 단위 군집 치환검정 + 군집 부트스트랩."""
    x = sub["eye_A_score"].to_numpy(float)
    y = sub["zein_number"].to_numpy(float)
    obs = rho(x, y)
    if not np.isfinite(obs):
        return dict(rho=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan, n_clusters=0)

    groups = sub.groupby("formulation_key").indices
    gkeys = list(groups)
    fscore = sub.groupby("formulation_key")["eye_A_score"].first()
    vals = fscore.to_numpy(float)

    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        shuffled = dict(zip(fscore.index, RNG.permutation(vals)))
        perm[i] = rho(sub["formulation_key"].map(shuffled).to_numpy(float), y)
    perm = perm[np.isfinite(perm)]
    p = float((np.abs(perm) >= abs(obs)).mean()) if len(perm) else np.nan

    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.integers(0, len(gkeys), len(gkeys))
        idx = np.concatenate([groups[gkeys[j]] for j in pick])
        boot[i] = rho(x[idx], y[idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
    return dict(rho=float(obs), p=p, ci_low=float(lo), ci_high=float(hi),
                n_clusters=len(gkeys))


def main() -> None:
    bundle = P.load_bundle(BASE / "model" / "eye_A_LR_wmean.joblib")
    names = bundle["feature_names"]
    reg = pd.read_csv(HERE / "structure_registry.csv", encoding="utf-8-sig")
    regmap = dict(zip(reg["surfactant"], reg["input_smiles"]))
    pred = pd.read_csv(HERE / "predictions.csv", encoding="utf-8-sig")

    # --- 학습 피처 분포 ---
    W = np.load(BASE / "training_reference" / "wmean.npy")
    mu, sd = np.nanmean(W, axis=0), np.nanstd(W, axis=0)
    sd[sd == 0] = 1.0

    def zdist(mat):
        z = (np.nan_to_num(mat, nan=0.0) - mu) / sd
        return np.sqrt((z ** 2).mean(axis=1))

    d_train = zdist(W)
    cuts = {f"p{q}": float(np.percentile(d_train, q)) for q in (50, 75, 90, 95)}

    # --- 배합별 거리 ---
    uniq = pred.drop_duplicates("formulation_key").copy()
    feats = []
    for _, r in uniq.iterrows():
        comps = [c for c in (r["product1"], r["product2"])
                 if isinstance(c, str) and c.strip()]
        w = [1.0] * len(comps) if len(comps) == 1 else \
            [float(p) for p in str(r["proportion"]).split("-")]
        feats.append(P.vectorize_components(names, [regmap[c] for c in comps], w).to_numpy()[0])
    uniq["feature_distance"] = zdist(np.vstack(feats))
    dmap = dict(zip(uniq["formulation_key"], uniq["feature_distance"]))
    pred["feature_distance"] = pred["formulation_key"].map(dmap)
    pred["has_overlap_component"] = (
        (pred["product1"] == OVERLAP_SURFACTANT) |
        (pred["product2"].fillna("") == OVERLAP_SURFACTANT))
    pred.to_csv(HERE / "predictions.csv", index=False, encoding="utf-8-sig")

    # --- AD 임계값별 ---
    rows = []
    for cut_name, cut in [("전체(제한없음)", np.inf), *cuts.items()]:
        base = pred[pred["feature_distance"] <= cut]
        for tier, mt in [("Tier1_single", "single"),
                         ("Tier2_mixture", "binary_mixture"),
                         ("all", None)]:
            sub = base if mt is None else base[base["mixture_type"] == mt]
            res = cluster_tests(sub) if len(sub) >= 6 else dict(
                rho=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan, n_clusters=0)
            rows.append({"ad_cut": cut_name,
                         "cut_value": None if np.isinf(cut) else round(cut, 4),
                         "tier": tier, "n": int(len(sub)), **res})
    ad = pd.DataFrame(rows)
    ad.to_csv(HERE / "in_domain_rank_association.csv", index=False, encoding="utf-8-sig")

    # --- 거리 구간별 (연속 경향) ---
    grad = []
    mix = pred[pred["mixture_type"] == "binary_mixture"].copy()
    qs = np.percentile(mix["feature_distance"], [0, 33.3, 66.7, 100])
    for i in range(3):
        lo_d, hi_d = qs[i], qs[i + 1]
        sel = mix[(mix["feature_distance"] >= lo_d) &
                  (mix["feature_distance"] <= hi_d if i == 2
                   else mix["feature_distance"] < hi_d)]
        r = cluster_tests(sel) if len(sel) >= 6 else dict(
            rho=np.nan, p=np.nan, ci_low=np.nan, ci_high=np.nan, n_clusters=0)
        grad.append({"distance_tertile": ["가까움", "중간", "멂"][i],
                     "distance_range": f"{lo_d:.3f}–{hi_d:.3f}",
                     "n": int(len(sel)), **r})
    gd = pd.DataFrame(grad)
    gd.to_csv(HERE / "distance_gradient.csv", index=False, encoding="utf-8-sig")

    # --- SLS 제외 민감도 ---
    sens = []
    for label, frame in [("전체", pred),
                         ("SLS 제외", pred[~pred["has_overlap_component"]])]:
        for tier, mt in [("Tier1_single", "single"), ("Tier2_mixture", "binary_mixture")]:
            sub = frame[frame["mixture_type"] == mt]
            r = cluster_tests(sub) if len(sub) >= 6 else {}
            sens.append({"set": label, "tier": tier, "n": int(len(sub)), **r})
    sd_df = pd.DataFrame(sens)
    sd_df.to_csv(HERE / "sls_exclusion_sensitivity.csv", index=False, encoding="utf-8-sig")

    (HERE / "in_domain_summary.json").write_text(json.dumps({
        "generated": "2026-11-09",
        "analysis_type": "post-hoc subset (AD-restricted)",
        "training_distance_cuts": cuts,
        "ad_results": rows,
        "distance_gradient": grad,
        "sls_sensitivity": sens,
        "note": ("사후 부분집합 분석. 단일 임계값을 고르지 않고 관례적 분위수를 "
                 "모두 보고한다. 표본 축소로 검정력이 낮으므로 유의성보다 "
                 "CI 폭과 추정치 안정성을 본다."),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # --- 출력 ---
    print("학습 피처거리 분위수:", {k: round(v, 3) for k, v in cuts.items()})
    print(f"외부 배합 거리: 중앙값 {uniq['feature_distance'].median():.3f}, "
          f"최소 {uniq['feature_distance'].min():.3f}\n")
    print("=== AD 임계값별 Spearman (군집 보정) ===")
    print(f"{'AD기준':<14}{'티어':<16}{'n':>4}{'군집':>5}{'rho':>9}{'95% CI':>20}{'p':>9}")
    for r in rows:
        if r["tier"] == "all":
            continue
        rr = f"{r['rho']:+.3f}" if np.isfinite(r["rho"]) else "—"
        ci = (f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
              if np.isfinite(r["ci_low"]) else "—")
        pp = f"{r['p']:.4f}" if np.isfinite(r["p"]) else "—"
        print(f"{r['ad_cut']:<14}{r['tier']:<16}{r['n']:>4}{r['n_clusters']:>5}"
              f"{rr:>9}{ci:>20}{pp:>9}")

    print("\n=== 혼합물: 학습공간 거리 3분위별 ===")
    for r in grad:
        rr = f"{r['rho']:+.3f}" if np.isfinite(r["rho"]) else "—"
        ci = (f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
              if np.isfinite(r["ci_low"]) else "—")
        print(f"  {r['distance_tertile']:<5} 거리 {r['distance_range']:<14} "
              f"n={r['n']:>3} rho={rr} CI={ci}")

    print("\n=== SLS(학습 중복) 제외 민감도 ===")
    for r in sens:
        rr = f"{r['rho']:+.3f}" if np.isfinite(r.get("rho", np.nan)) else "—"
        pp = f"{r['p']:.4f}" if np.isfinite(r.get("p", np.nan)) else "—"
        print(f"  {r['set']:<9}{r['tier']:<16} n={r['n']:>3} rho={rr} p={pp}")
    print(f"\n저장: {HERE}")


if __name__ == "__main__":
    main()
