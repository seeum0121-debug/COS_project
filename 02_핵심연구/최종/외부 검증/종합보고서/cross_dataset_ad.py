"""4개 외부자료의 적용범위(AD)를 같은 척도로 재계산하고 교차 비교.

핵심 질문
  Lechuga(화장품 계면활성제)에서 혼합물 역상관(-0.344)은 적용범위 안에서
  사라졌다. 같은 음의 상관을 보인 quat 혼합물(-0.570/-0.725)도 같은가?
  같다면 '학습공간 밖 외삽'이라는 단일 설명으로 두 자료가 통합된다.
  다르다면 두 자료는 서로 다른 원인을 가진다.

척도 통일
  학습 wmean.npy(673행)로 평균·표준편차를 구해 표준화 유클리드 거리를 쓴다.
  Lechuga 분석(applicability_domain.py)과 동일한 정의다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
COSM = ROOT / "Eye_cosmetic_validation_20261109"
QUAT = ROOT / "Eye_재외부검증_20261109"
LECH = ROOT / "Lechuga2025_화장품_외부검증_20261109"


def main() -> None:
    W = np.load(COSM / "training_reference" / "wmean.npy")
    mu, sd = np.nanmean(W, axis=0), np.nanstd(W, axis=0)
    sd[sd == 0] = 1.0

    def zdist(mat):
        z = (np.nan_to_num(np.asarray(mat, dtype=float), nan=0.0) - mu) / sd
        return np.sqrt((z ** 2).mean(axis=1))

    d_train = zdist(W)
    cuts = {f"p{q}": float(np.percentile(d_train, q)) for q in (50, 75, 90, 95)}

    out = {"training_distance_percentiles": {k: round(v, 4) for k, v in cuts.items()},
           "datasets": {}}

    # ---------- quat 혼합물 ----------
    qp = pd.read_csv(QUAT / "outputs" / "quat_predictions.csv")
    qf = np.load(QUAT / "outputs" / "quat_features.npy")
    order = json.loads((QUAT / "outputs" / "quat_feature_row_order.json").read_text(encoding="utf-8"))
    if len(order) != len(qf) or len(qp) != len(qf):
        raise SystemExit("quat 행 수 불일치")
    qp = qp.set_index("formulation_id").loc[order].reset_index()
    qp["feature_distance"] = zdist(qf)

    q = qp.dropna(subset=["day7", "model_score"]).copy()
    rows = []
    for name, cut in [("전체", np.inf), *cuts.items()]:
        sub = q[q["feature_distance"] <= cut]
        if len(sub) >= 4 and sub["model_score"].nunique() > 1:
            r = stats.spearmanr(sub["model_score"], sub["day7"])
            rows.append({"ad_cut": name, "n": int(len(sub)),
                         "spearman_day7": round(float(r.statistic), 4),
                         "p_value": float(f"{r.pvalue:.3g}")})
        else:
            rows.append({"ad_cut": name, "n": int(len(sub)),
                         "spearman_day7": None, "p_value": None})
    out["datasets"]["quat_mixture"] = {
        "source": "US6583181B1 Tables 1-4",
        "n": int(len(q)),
        "distance_min": round(float(q["feature_distance"].min()), 4),
        "distance_median": round(float(q["feature_distance"].median()), 4),
        "distance_max": round(float(q["feature_distance"].max()), 4),
        "pct_beyond_training_p95": round(
            float((q["feature_distance"] > cuts["p95"]).mean() * 100), 1),
        "ad_stratified": rows,
    }

    # ---------- Lechuga ----------
    lp = pd.read_csv(LECH / "predictions.csv", encoding="utf-8-sig")
    lm = lp[lp["mixture_type"] == "binary_mixture"]
    ls = lp[lp["mixture_type"] == "single"]
    out["datasets"]["lechuga_surfactant"] = {
        "source": "Lechuga et al. 2025 ACS Omega",
        "n_mixture": int(len(lm)), "n_single": int(len(ls)),
        "distance_min": round(float(lp["feature_distance"].min()), 4),
        "distance_median": round(float(lp["feature_distance"].median()), 4),
        "distance_max": round(float(lp["feature_distance"].max()), 4),
        "pct_beyond_training_p95": round(
            float((lp["feature_distance"] > cuts["p95"]).mean() * 100), 1),
    }

    # ---------- 샴푸 (부분조성) ----------
    try:
        sf = np.load(COSM / "outputs" / "shampoo_features.npy")
        sp = pd.read_csv(COSM / "outputs" / "shampoo_predictions.csv", encoding="utf-8-sig")
        ds = zdist(sf)
        out["datasets"]["flea_tick_shampoo"] = {
            "source": "EPA 등록 벼룩·진드기 샴푸 라벨",
            "n_feature_rows": int(len(sf)),
            "n_prediction_rows": int(len(sp)),
            "distance_min": round(float(ds.min()), 4),
            "distance_median": round(float(np.median(ds)), 4),
            "distance_max": round(float(ds.max()), 4),
            "pct_beyond_training_p95": round(float((ds > cuts["p95"]).mean() * 100), 1),
            "caveat": "입력에 반영된 조성이 0.779~3.93%뿐이라 거리 자체의 해석이 제한적",
        }
    except Exception as exc:  # noqa: BLE001
        out["datasets"]["flea_tick_shampoo"] = {"error": str(exc)}

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "cross_dataset_ad.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    qp.to_csv(HERE / "quat_with_distance.csv", index=False, encoding="utf-8-sig")

    print("학습 거리 분위수:", {k: round(v, 3) for k, v in cuts.items()})
    print("\n=== quat 혼합물: AD 임계값별 Spearman (model_score vs day7) ===")
    for r in rows:
        s = f"{r['spearman_day7']:+.4f}" if r["spearman_day7"] is not None else "—"
        p = f"{r['p_value']:.3g}" if r["p_value"] is not None else "—"
        print(f"  {r['ad_cut']:<8} n={r['n']:>3}  rho={s:>9}  p={p:>9}")
    qd = out["datasets"]["quat_mixture"]
    print(f"\n  거리: 최소 {qd['distance_min']:.3f} / 중앙 {qd['distance_median']:.3f} "
          f"/ 최대 {qd['distance_max']:.3f}")
    print(f"  학습 p95 밖: {qd['pct_beyond_training_p95']}%")
    print("\n=== 자료별 거리 비교 ===")
    for k, v in out["datasets"].items():
        if "error" in v:
            print(f"  {k}: {v['error']}")
            continue
        print(f"  {k:<22} 중앙 {v['distance_median']:.3f}  "
              f"p95밖 {v['pct_beyond_training_p95']}%")
    print(f"\n저장: {HERE / 'cross_dataset_ad.json'}")


if __name__ == "__main__":
    main()
