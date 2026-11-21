"""UVCB 구조 해결 방침에 따른 결론 민감도 분석.

배경
  동일한 Lechuga 2025 zein 자료에 대해 세 가지 분석이 서로 다른 결론을 냈다.
  차이는 모델이 아니라 '다분산 상업원료(UVCB)의 구조를 어디까지 가정하는가'
  하나뿐이다. 이 스크립트는 같은 eye 모델로 범위만 바꾸어 결론이 어떻게
  뒤집히는지 정량화한다.

범위 정의 (사전 지정)
  S5   PubChem에서 단일 명확 구조가 확인된 5종만. UVCB 가정 전면 금지.
       (Lechuga_Eye_A_B_엄격검토_20261109/protocol.json의 allowed_names)
  S10  PubChem 명칭 조회가 성공한 10종. 올리고머·EO평균·coco사슬 가정을
       PubChem이 준 대로 암묵 수용.
       (GitHub cosmetic_validation/lechuga_smiles_cache.csv의 status=found)
  S22  대표구조 규칙 R1-R9로 22종 전부 해결.
       (Lechuga2025_화장품_외부검증_20261109/structure_rules.json)

주의
  세 범위는 포함관계다: S5 ⊂ S10 ⊂ S22. 따라서 결과 차이는 전적으로
  '추가로 포함된 성분'에서 나온다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EYE = ROOT / "Lechuga2025_화장품_외부검증_20261109" / "predictions.csv"
SKIN = ROOT / "Skin_Lechuga_외부검증_20261109" / "github_data" / "lechuga_external_validation_results.csv"
STRICT = ROOT / "Lechuga_Eye_A_B_엄격검토_20261109" / "summary.json"

RNG = np.random.default_rng(20261109)
N = 20000

S5 = {"Lauramine Oxide", "Myristamine Oxide", "Sodium lauryl sulfate",
      "Disodium Lauroyl Glutamate", "Disodium Capryloyl Glutamate: 30% - 37%"}
GLUCOSIDE = {"Capryl Glucoside", "Coco Glucoside", "Lauryl Glucoside"}
S10 = S5 | GLUCOSIDE | {"Sodium laureth sulfate", "Zinc Coco-Sulfate: 27% - 30%"}
AMINE_OXIDE = {"Lauramine Oxide", "Myristamine Oxide"}


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return np.nan
    return stats.spearmanr(x, y).statistic


def cluster_stats(d: pd.DataFrame):
    """배합 단위 군집 치환검정 + 군집 부트스트랩."""
    x = d["eye_A_score"].to_numpy(float)
    y = d["zein_number"].to_numpy(float)
    obs = rho(x, y)
    if not np.isfinite(obs):
        return dict(rho=None, p=None, ci=[None, None], n_clusters=0)
    groups = d.groupby("formulation_key").indices
    gk = list(groups)
    fs = d.groupby("formulation_key")["eye_A_score"].first()
    vals = fs.to_numpy(float)
    perm = np.array([rho(d["formulation_key"].map(
        dict(zip(fs.index, RNG.permutation(vals)))).to_numpy(float), y) for _ in range(N)])
    perm = perm[np.isfinite(perm)]
    p = float((np.abs(perm) >= abs(obs)).mean()) if len(perm) else None
    boot = []
    for _ in range(N):
        idx = np.concatenate([groups[gk[j]] for j in RNG.integers(0, len(gk), len(gk))])
        boot.append(rho(x[idx], y[idx]))
    boot = np.array([b for b in boot if np.isfinite(b)])
    ci = [round(float(v), 4) for v in np.percentile(boot, [2.5, 97.5])] if len(boot) else [None, None]
    return dict(rho=round(float(obs), 4), p=p, ci=ci, n_clusters=len(gk))


def main() -> None:
    eye = pd.read_csv(EYE, encoding="utf-8-sig")
    eye["p2"] = eye["product2"].fillna("")
    all22 = set(eye["product1"]) | {x for x in eye["p2"] if x}

    def sel(S):
        return eye[eye["product1"].isin(S) & (eye["p2"].isin(S) | (eye["p2"] == ""))].copy()

    scopes = [("S5", "PubChem 단일구조 5종 · UVCB 가정 금지", S5),
              ("S10", "PubChem 조회 성공 10종 · 가정 암묵 수용", S10),
              ("S22", "대표구조 규칙 22종 · 가정 명시 사용", all22)]

    rows = []
    for key, label, S in scopes:
        d = sel(S)
        byf = d.groupby("formulation_key").agg(
            s=("eye_A_score", "first"), z=("zein_number", "mean"))
        r_row = rho(d["eye_A_score"], d["zein_number"])
        r_f = rho(byf["s"], byf["z"])
        p_f = (stats.spearmanr(byf["s"], byf["z"]).pvalue if len(byf) > 3 else np.nan)
        cs = cluster_stats(d)
        rows.append({
            "scope": key, "label": label,
            "n_surfactants": len(S), "n_rows": int(len(d)),
            "n_formulations": int(len(byf)),
            "rho_rows": round(float(r_row), 4) if np.isfinite(r_row) else None,
            "rho_formulations": round(float(r_f), 4) if np.isfinite(r_f) else None,
            "p_formulations": round(float(p_f), 4) if np.isfinite(p_f) else None,
            "cluster_rho": cs["rho"], "cluster_p": cs["p"],
            "cluster_ci_low": cs["ci"][0], "cluster_ci_high": cs["ci"][1],
            "score_min": round(float(d["eye_A_score"].min()), 4),
            "score_max": round(float(d["eye_A_score"].max()), 4),
        })
    res = pd.DataFrame(rows)

    # --- 글루코사이드 레버리지 ---
    lev = []
    for label, S in [("S10 전체", S10), ("S10 - 글루코사이드 3종", S10 - GLUCOSIDE),
                     ("S10 - 아민옥사이드 2종", S10 - AMINE_OXIDE),
                     ("S10 - SLS", S10 - {"Sodium lauryl sulfate"})]:
        d = sel(S)
        byf = d.groupby("formulation_key").agg(
            s=("eye_A_score", "first"), z=("zein_number", "mean"))
        r = rho(byf["s"], byf["z"])
        p = stats.spearmanr(byf["s"], byf["z"]).pvalue if len(byf) > 3 else np.nan
        lev.append({"subset": label, "n_formulations": int(len(byf)),
                    "rho": round(float(r), 4) if np.isfinite(r) else None,
                    "p": round(float(p), 4) if np.isfinite(p) else None})
    levdf = pd.DataFrame(lev)

    # --- 계열 대비 ---
    cls = []
    for label, S in [("글루코사이드 (UVCB 올리고머)", GLUCOSIDE),
                     ("아민옥사이드 (단일구조)", AMINE_OXIDE),
                     ("설페이트계", {"Sodium lauryl sulfate", "Sodium laureth sulfate",
                                   "Zinc Coco-Sulfate: 27% - 30%"}),
                     ("아실글루타메이트", {"Disodium Lauroyl Glutamate",
                                      "Disodium Capryloyl Glutamate: 30% - 37%"})]:
        d = sel(S)
        if not len(d):
            continue
        cls.append({"class": label, "n": int(len(d)),
                    "score_min": round(float(d["eye_A_score"].min()), 4),
                    "score_max": round(float(d["eye_A_score"].max()), 4),
                    "zein_min": round(float(d["zein_number"].min()), 2),
                    "zein_max": round(float(d["zein_number"].max()), 2),
                    "zein_median": round(float(d["zein_number"].median()), 2)})
    clsdf = pd.DataFrame(cls)

    # --- skin 보고값 재계산 ---
    sk = pd.read_csv(SKIN, encoding="utf-8-sig")
    sk["fk"] = (sk["product1"].fillna("") + "|" + sk["product2"].fillna("")
                + "@" + sk["proportion"].fillna("1").astype(str))
    skf = sk.groupby("fk").agg(s=("pred_proba", "first"), z=("zein_number", "mean"),
                              pb=("pred_binary", "first"),
                              tb=("true_binary", lambda v: int(v.mode().iloc[0])))
    skin_info = {
        "n_rows": int(len(sk)), "n_formulations": int(len(skf)),
        "accuracy_rows": round(float(sk["match"].mean()), 4),
        "accuracy_formulations": round(float((skf["pb"] == skf["tb"]).mean()), 4),
        "rho_rows": round(float(rho(sk["pred_proba"], sk["zein_number"])), 4),
        "rho_formulations": round(float(rho(skf["s"], skf["z"])), 4),
        "p_formulations": round(float(stats.spearmanr(skf["s"], skf["z"]).pvalue), 4),
    }
    strict = json.loads(STRICT.read_text(encoding="utf-8")) if STRICT.exists() else {}

    out = {"generated": "2026-11-09",
           "question": "UVCB 구조 해결 방침이 결론을 결정하는가",
           "scopes_nested": "S5 ⊂ S10 ⊂ S22",
           "eye_by_scope": rows,
           "glucoside_leverage": lev,
           "class_contrast": cls,
           "skin_recomputed": skin_info,
           "strict_review_reported": {
               k: strict.get(k) for k in list(strict)[:12]} if strict else {},
           "bootstrap_iterations": N, "seed": 20261109}
    (HERE / "sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    res.to_csv(HERE / "scope_comparison.csv", index=False, encoding="utf-8-sig")
    levdf.to_csv(HERE / "glucoside_leverage.csv", index=False, encoding="utf-8-sig")
    clsdf.to_csv(HERE / "class_contrast.csv", index=False, encoding="utf-8-sig")

    print("=== 구조 해결 범위별 eye 모델 (S5 ⊂ S10 ⊂ S22) ===")
    print(f"{'범위':<5}{'성분':>4}{'행':>5}{'배합':>5}{'행 rho':>9}{'배합 rho':>10}"
          f"{'배합 p':>8}{'군집 p':>8}{'군집 95% CI':>20}")
    for r in rows:
        ci = (f"[{r['cluster_ci_low']:+.3f},{r['cluster_ci_high']:+.3f}]"
              if r["cluster_ci_low"] is not None else "—")
        cp = f"{r['cluster_p']:.3f}" if r["cluster_p"] is not None else "—"
        print(f"{r['scope']:<5}{r['n_surfactants']:>4}{r['n_rows']:>5}{r['n_formulations']:>5}"
              f"{r['rho_rows']:>+9.3f}{r['rho_formulations']:>+10.3f}"
              f"{r['p_formulations']:>8.3f}{cp:>8}{ci:>20}")
    print("\n=== 글루코사이드 레버리지 (배합 단위) ===")
    for r in lev:
        print(f"  {r['subset']:<26} n={r['n_formulations']:>2}  "
              f"rho={r['rho']:+.3f}  p={r['p']:.3f}")
    print("\n=== 계열별 점수·zein 대비 ===")
    for r in cls:
        print(f"  {r['class']:<26} n={r['n']:>2}  "
              f"score {r['score_min']:.3f}~{r['score_max']:.3f}  "
              f"zein {r['zein_min']:.1f}~{r['zein_max']:.1f} (중앙 {r['zein_median']:.1f})")
    print("\n=== skin 재계산 ===")
    for k, v in skin_info.items():
        print(f"  {k:<24} {v}")
    print(f"\n저장: {HERE}")


if __name__ == "__main__":
    main()
