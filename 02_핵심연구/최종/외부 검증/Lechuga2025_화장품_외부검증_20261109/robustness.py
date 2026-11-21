"""Tier2 음의 상관에 대한 강건성 검정.

문제: 103개 혼합물 분석단위는 22종 계면활성제의 조합에서 나왔다. 같은 성분을
공유하는 행들은 독립이 아니므로 일반 Spearman p값과 행단위 부트스트랩 CI는
과소추정된다. 아래 세 가지로 다시 본다.

1) 군집 치환검정  : 계면활성제 라벨을 치환(permute)하여 귀무분포를 만든다.
                    성분 공유 구조를 유지한 채 점수-결과 대응만 끊는다.
2) 군집 부트스트랩: 행이 아니라 '배합(formulation)' 단위로 재표집한다.
3) 영향도 분석    : 계면활성제 1종씩 제외하고 rho가 얼마나 흔들리는지 본다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261109)
N_PERM = 20000
N_BOOT = 20000


def rho(x, y):
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return np.nan
    return stats.spearmanr(x, y).statistic


def main() -> None:
    d = pd.read_csv(HERE / "predictions.csv", encoding="utf-8-sig")
    out = {}

    for tier, label in [("binary_mixture", "Tier2_mixture"),
                        ("single", "Tier1_single")]:
        sub = d[d["mixture_type"] == tier].copy()
        sub["p2"] = sub["product2"].fillna("")
        x = sub["eye_A_score"].to_numpy(float)
        y = sub["zein_number"].to_numpy(float)
        obs = rho(x, y)

        # 배합 키 (농도 무시) = 군집 단위
        sub["fkey"] = sub["formulation_key"]
        groups = sub.groupby("fkey").indices
        gkeys = list(groups)

        # 1) 군집 치환: 배합 단위로 점수를 섞는다
        fscore = sub.groupby("fkey")["eye_A_score"].first()
        perm = np.empty(N_PERM)
        vals = fscore.to_numpy(float)
        for i in range(N_PERM):
            shuffled = dict(zip(fscore.index, RNG.permutation(vals)))
            perm[i] = rho(sub["fkey"].map(shuffled).to_numpy(float), y)
        perm = perm[np.isfinite(perm)]
        p_perm = float((np.abs(perm) >= abs(obs)).mean())

        # 2) 군집 부트스트랩: 배합 단위 재표집
        boot = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pick = RNG.integers(0, len(gkeys), len(gkeys))
            idx = np.concatenate([groups[gkeys[j]] for j in pick])
            boot[i] = rho(x[idx], y[idx])
        boot = boot[np.isfinite(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])

        # 3) 계면활성제 1종 제외 영향도
        surfs = sorted(set(sub["product1"]) | set(s for s in sub["p2"] if s))
        loo = []
        for s in surfs:
            keep = (sub["product1"] != s) & (sub["p2"] != s)
            if keep.sum() < 8:
                continue
            loo.append({"excluded": s, "n": int(keep.sum()),
                        "rho": round(float(rho(x[keep.to_numpy()], y[keep.to_numpy()])), 4)})
        loo_df = pd.DataFrame(loo).sort_values("rho")

        out[label] = {
            "n": int(len(sub)),
            "n_clusters": int(len(gkeys)),
            "rho_observed": round(float(obs), 4),
            "cluster_permutation_p": p_perm,
            "cluster_bootstrap_ci": [round(float(lo), 4), round(float(hi), 4)],
            "ci_excludes_zero": bool(lo * hi > 0),
            "loo_rho_min": round(float(loo_df["rho"].min()), 4) if len(loo_df) else None,
            "loo_rho_max": round(float(loo_df["rho"].max()), 4) if len(loo_df) else None,
            "loo_sign_stable": bool((loo_df["rho"] < 0).all() or (loo_df["rho"] > 0).all())
            if len(loo_df) else None,
        }
        loo_df.to_csv(HERE / f"leave_one_surfactant_out_{label}.csv",
                      index=False, encoding="utf-8-sig")

        print(f"\n=== {label} ===")
        print(f"  n={len(sub)}  군집(배합)={len(gkeys)}")
        print(f"  관측 rho            : {obs:+.4f}")
        print(f"  군집 치환검정 p     : {p_perm:.4f}   (행단위 p와 비교할 것)")
        print(f"  군집 부트스트랩 CI  : [{lo:+.4f}, {hi:+.4f}]"
              f"  {'0 제외' if lo*hi>0 else '0 포함'}")
        if len(loo_df):
            print(f"  1종 제외 rho 범위   : {loo_df['rho'].min():+.4f} ~ {loo_df['rho'].max():+.4f}")
            print(f"  부호 안정성         : {'안정' if out[label]['loo_sign_stable'] else '불안정'}")
            print(f"  가장 큰 영향(제외시 rho 0에 가까워짐):")
            for _, r in loo_df.tail(3).iterrows():
                print(f"     - {r['excluded'][:45]:<45} rho={r['rho']:+.4f}")

    (HERE / "robustness.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {HERE / 'robustness.json'}")


if __name__ == "__main__":
    main()
