"""Lechuga 2025 계면활성제 22종의 대표구조 해결.

원 물질은 다분산(polydisperse) 상업 계면활성제이므로 단일 SMILES가 존재하지
않는다. 아래 규칙으로 '대표구조'를 구성하며, 이는 근사이고 논문에 한계로
명시해야 한다. 규칙은 모두 사전 지정(pre-specified)이며 예측 결과를 보고
조정하지 않는다.

R1  coco / cocamido / cocoate  -> C12 (라우르산, 코코넛 유래의 최대 성분 ~50%)
R2  capryl / capryleth         -> C8
R3  lauryl / laureth / lauroyl -> C12
R4  myrist                     -> C14
R5  명시적 탄소수 범위         -> 산술 중앙값 (C12-14→13, C16-18→17, C8-12→10)
R6  에톡실레이트 EO 수         -> 명칭의 숫자. 숫자 미표기는 상용 전형값 + 플래그
R7  알킬 폴리글루코사이드      -> DP=1 (단일 글루코스)
R8  염                         -> 학습 데이터 관례에 따라 반대이온을 별도
                                  프래그먼트로 명시 (예: .[Na+])
R9  Glycereth-n                -> 글리세롤 1차 수산기 한 곳에 EO n개를 집중
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).resolve().parent


def eo(n: int) -> str:
    """에틸렌옥사이드 n 단위."""
    return "OCC" * n


def alkyl(n: int) -> str:
    return "C" * n


def ether_carboxylic(c: int, n_eo: int) -> str:
    """알킬-O-(CH2CH2O)n-CH2COOH : -eth-n carboxylic acid."""
    return alkyl(c) + eo(n_eo) + "OCC(=O)O"


def alcohol_ethoxylate(c: int, n_eo: int) -> str:
    """알킬-O-(CH2CH2O)n-H : polyoxyethylene alkyl ether."""
    return alkyl(c) + eo(n_eo) + "O"


def glycereth_cocoate(n_eo: int) -> str:
    """코코에이트(C12) 글리세롤 에스터, 1차 수산기에 EO n개 (R9)."""
    return "CCCCCCCCCCCC(=O)OCC(O)C" + eo(n_eo) + "O"


def amine_oxide(c: int) -> str:
    return alkyl(c) + "[N+](C)(C)[O-]"


def alkyl_glucoside(c: int) -> str:
    """DP=1 알킬 글루코사이드 (R7)."""
    return alkyl(c) + "OC1OC(CO)C(O)C(O)C1O"


def acyl_glutamate_na2(c: int) -> str:
    return alkyl(c - 1) + "C(=O)NC(CCC(=O)[O-])C(=O)[O-].[Na+].[Na+]"


def alkyl_sulfate_na(c: int, n_eo: int = 0) -> str:
    return alkyl(c) + eo(n_eo) + "OS(=O)(=O)[O-].[Na+]"


def alkyl_sulfate_zn(c: int, n_eo: int = 0) -> str:
    anion = alkyl(c) + eo(n_eo) + "OS(=O)(=O)[O-]"
    return f"{anion}.{anion}.[Zn+2]"


# name -> (smiles, 적용규칙, 가정/플래그)
REGISTRY: dict[str, tuple[str, str, str]] = {
    "Capryl Glucoside": (
        alkyl_glucoside(8), "R2,R7", ""),
    "Capryleth-6 Carboxylic Acid": (
        ether_carboxylic(8, 6), "R2,R6", ""),
    "Cocamidopropylamine Oxide": (
        "CCCCCCCCCCCC(=O)NCCC[N+](C)(C)[O-]", "R1", ""),
    "Coco Glucoside": (
        alkyl_glucoside(12), "R1,R7", "Lauryl Glucoside와 대표구조 동일"),
    "Disodium Capryloyl Glutamate: 30% - 37%": (
        acyl_glutamate_na2(8), "R2,R8", "공급 원료 활성분 30-37%; 구조는 순물질 기준"),
    "Disodium Lauroyl Glutamate": (
        acyl_glutamate_na2(12), "R3,R8", ""),
    "Glycereth-17 Cocoate": (
        glycereth_cocoate(17), "R1,R6,R9", ""),
    "Glycereth-2 Cocoate": (
        glycereth_cocoate(2), "R1,R6,R9", ""),
    "Glycereth-6 Cocoate": (
        glycereth_cocoate(6), "R1,R6,R9", ""),
    "Glycereth-7 Cocoate": (
        glycereth_cocoate(7), "R1,R6,R9", ""),
    "Lauramine Oxide": (
        amine_oxide(12), "R3", ""),
    "Laureth-11 Carboxylic Acid": (
        ether_carboxylic(12, 11), "R3,R6", ""),
    "Laureth-4 Carboxylic Acid": (
        ether_carboxylic(12, 4), "R3,R6", ""),
    "Lauryl Glucoside": (
        alkyl_glucoside(12), "R3,R7", "Coco Glucoside와 대표구조 동일"),
    "Myristamine Oxide": (
        amine_oxide(14), "R4", ""),
    "Polyoxyethylene(11) alkyl(C12-14) ethers": (
        alcohol_ethoxylate(13, 11), "R5,R6", "C12-14 중앙값 C13"),
    "Polyoxyethylene(11) alkyl(C16-18) ethers": (
        alcohol_ethoxylate(17, 11), "R5,R6", "C16-18 중앙값 C17"),
    "Polyoxyethylene(6) alkyl(C8-12) ethers": (
        alcohol_ethoxylate(10, 6), "R5,R6", "C8-12 중앙값 C10"),
    "Sodium laureth sulfate": (
        alkyl_sulfate_na(12, 2), "R3,R6,R8", "EO 수 미표기: 상용 전형값 2 가정"),
    "Sodium lauryl sulfate": (
        alkyl_sulfate_na(12), "R3,R8", "학습 데이터에 동일 구조 존재(중복 확인 필요)"),
    "Zinc Coceth Sulfate": (
        alkyl_sulfate_zn(12, 3), "R1,R6,R8", "EO 수 미표기: 3 가정"),
    "Zinc Coco-Sulfate: 27% - 30%": (
        alkyl_sulfate_zn(12), "R1,R8", "공급 원료 활성분 27-30%; 구조는 순물질 기준"),
}


def main() -> None:
    rows = []
    failures = []
    for name, (smi, rules, note) in sorted(REGISTRY.items()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not mol.GetNumAtoms():
            failures.append(name)
            continue
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        largest = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Descriptors.MolWt(m)))
        rows.append({
            "surfactant": name,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "input_smiles": smi,
            "rules": rules,
            "assumption": note,
            "n_fragments": len(frags),
            "mw_total": round(Descriptors.MolWt(mol), 2),
            "mw_largest_fragment": round(Descriptors.MolWt(largest), 2),
            "heavy_atoms_largest": largest.GetNumHeavyAtoms(),
            "logp_largest": round(Descriptors.MolLogP(largest), 3),
            "tpsa_largest": round(Descriptors.TPSA(largest), 2),
        })

    if failures:
        raise SystemExit(f"구조 해결 실패: {failures}")

    out_csv = OUT / "structure_registry.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    canon = {r["surfactant"]: r["canonical_smiles"] for r in rows}
    dupes: dict[str, list[str]] = {}
    for name, smi in canon.items():
        dupes.setdefault(smi, []).append(name)
    collisions = {k: v for k, v in dupes.items() if len(v) > 1}

    (OUT / "structure_rules.json").write_text(json.dumps({
        "generated": "2026-11-09",
        "n_surfactants": len(rows),
        "rules": __doc__.split("R1")[1].strip().splitlines(),
        "canonical_collisions": collisions,
        "prespecified": True,
        "note": "다분산 상업 계면활성제의 대표구조 근사. 예측 결과를 보고 조정하지 않음.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"구조 해결 완료: {len(rows)}/22")
    print(f"저장: {out_csv}")
    if collisions:
        print("\n[주의] 대표구조가 동일해진 쌍:")
        for smi, names in collisions.items():
            print("  -", " == ".join(names))
    print("\n%-44s %9s %6s %8s" % ("surfactant", "MW(최대)", "frag", "logP"))
    for r in rows:
        print("%-44s %9.1f %6d %8.2f" % (
            r["surfactant"][:44], r["mw_largest_fragment"],
            r["n_fragments"], r["logp_largest"]))


if __name__ == "__main__":
    main()
