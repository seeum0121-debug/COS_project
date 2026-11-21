"""화장품 계면활성제 22종이 학습자료(EPA 농약 제형)의 화학공간 안에 있는가.

Tier1 무신호 / Tier2 역상관을 해석하려면, 이 물질들이 모델이 학습한 영역
안에 있는지부터 봐야 한다. 밖에 있다면 결과는 '모델이 틀렸다'가 아니라
'외삽했다'로 읽어야 한다.

지표
  1) 성분 단위 Tanimoto 최근접 이웃 (Morgan r2, 2048bit) — 학습 성분 283종 대비
  2) 배합 단위 피처공간 거리 — 학습 673행의 wmean 피처 중심까지 마할라노비스
     대용(표준화 유클리드) 거리의 분위수 비교
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "Eye_cosmetic_validation_20261109"
sys.path.insert(0, str(BASE / "model"))
import predict as P  # noqa: E402

GEN = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)


def fp(smi):
    m = Chem.MolFromSmiles(str(smi))
    return GEN.GetFingerprint(m) if m else None


def main() -> None:
    reg = pd.read_csv(HERE / "structure_registry.csv", encoding="utf-8-sig")
    link = pd.read_csv(BASE / "training_reference" / "component_linkage.csv",
                       low_memory=False, usecols=["smiles"])

    train_fps, train_smis = [], []
    for s in link["smiles"].dropna().astype(str).unique():
        f = fp(s)
        if f is not None:
            train_fps.append(f)
            train_smis.append(s)

    rows = []
    for _, r in reg.iterrows():
        f = fp(r["input_smiles"])
        sims = np.array(DataStructs.BulkTanimotoSimilarity(f, train_fps))
        order = np.argsort(sims)[::-1]
        rows.append({
            "surfactant": r["surfactant"],
            "max_tanimoto_to_training": round(float(sims[order[0]]), 4),
            "mean_top5_tanimoto": round(float(sims[order[:5]].mean()), 4),
            "nearest_training_smiles": train_smis[order[0]][:90],
        })
    comp = pd.DataFrame(rows).sort_values("max_tanimoto_to_training")
    comp.to_csv(HERE / "component_training_similarity.csv",
                index=False, encoding="utf-8-sig")

    # 배합 단위 피처 거리
    bundle = P.load_bundle(BASE / "model" / "eye_A_LR_wmean.joblib")
    names = bundle["feature_names"]
    train = pd.read_csv(BASE / "training_reference" / "component_linkage.csv",
                        low_memory=False)
    tcols = [c for c in names if c in train.columns]
    W = np.load(BASE / "training_reference" / "wmean.npy")
    mu, sd = np.nanmean(W, axis=0), np.nanstd(W, axis=0)
    sd[sd == 0] = 1.0

    def zdist(mat):
        z = (np.nan_to_num(mat, nan=0.0) - mu) / sd
        return np.sqrt((z ** 2).mean(axis=1))

    d_train = zdist(W)
    regmap = dict(zip(reg["surfactant"], reg["input_smiles"]))
    pred = pd.read_csv(HERE / "predictions.csv", encoding="utf-8-sig")
    uniq = pred.drop_duplicates("formulation_key")
    feats = []
    for _, r in uniq.iterrows():
        comps = [c for c in (r["product1"], r["product2"])
                 if isinstance(c, str) and c.strip()]
        w = [1.0] * len(comps) if len(comps) == 1 else \
            [float(p) for p in str(r["proportion"]).split("-")]
        feats.append(P.vectorize_components(names, [regmap[c] for c in comps], w).to_numpy()[0])
    d_ext = zdist(np.vstack(feats))

    summary = {
        "n_training_components": len(train_fps),
        "component_tanimoto": {
            "min": float(comp["max_tanimoto_to_training"].min()),
            "median": float(comp["max_tanimoto_to_training"].median()),
            "max": float(comp["max_tanimoto_to_training"].max()),
            "n_below_0.3": int((comp["max_tanimoto_to_training"] < 0.3).sum()),
            "n_below_0.4": int((comp["max_tanimoto_to_training"] < 0.4).sum()),
        },
        "feature_distance": {
            "training_median": round(float(np.median(d_train)), 4),
            "training_p95": round(float(np.percentile(d_train, 95)), 4),
            "external_median": round(float(np.median(d_ext)), 4),
            "external_min": round(float(d_ext.min()), 4),
            "pct_external_beyond_training_p95": round(
                float((d_ext > np.percentile(d_train, 95)).mean() * 100), 1),
        },
    }
    (HERE / "applicability_domain.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 성분 단위 Tanimoto 최근접 (학습 성분 %d종 대비) ===" % len(train_fps))
    print(f"  최소 {comp['max_tanimoto_to_training'].min():.3f} / "
          f"중앙값 {comp['max_tanimoto_to_training'].median():.3f} / "
          f"최대 {comp['max_tanimoto_to_training'].max():.3f}")
    print(f"  Tanimoto < 0.3 인 성분: {summary['component_tanimoto']['n_below_0.3']}/22")
    print(f"  Tanimoto < 0.4 인 성분: {summary['component_tanimoto']['n_below_0.4']}/22")
    print("\n  가장 먼 5종:")
    for _, r in comp.head(5).iterrows():
        print(f"    {r['surfactant'][:44]:<44} {r['max_tanimoto_to_training']:.3f}")
    print("\n  가장 가까운 3종:")
    for _, r in comp.tail(3).iterrows():
        print(f"    {r['surfactant'][:44]:<44} {r['max_tanimoto_to_training']:.3f}")
    fd = summary["feature_distance"]
    print(f"\n=== 배합 피처공간 거리 ===")
    print(f"  학습 중앙값 {fd['training_median']:.3f} / 학습 95분위 {fd['training_p95']:.3f}")
    print(f"  외부 중앙값 {fd['external_median']:.3f} / 외부 최소 {fd['external_min']:.3f}")
    print(f"  학습 95분위를 넘는 외부 배합: {fd['pct_external_beyond_training_p95']}%")


if __name__ == "__main__":
    main()
