# -*- coding: utf-8 -*-
"""Skin Option A 재보정 전이학습 그림 — eye 전이학습 시각화와 동일한 서식.

혼동행렬 + ROC curve 2패널. 수치는 저장된 OOF 예측에서 직접 재계산하고,
results/metrics.csv 의 기록값과 일치하는지 단언한다.

근거: 보고서/evidence/COS_project/transfer_learning/skin_option_a_recalibration/results/
      predictions.csv · metrics.csv · bootstrap_ci.csv · protocol.json

사용법: python make_skin_transfer_figure.py 출력.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
RESULTS = FINAL/"보고서"/"evidence"/"COS_project"/"transfer_learning"/"skin_option_a_recalibration"/"results"

NAVY, BLUE, GREEN = "#24547f", "#287bb8", "#2f7c3b"
INK, MUTED, GRID = "#1c252c", "#59636c", "#d5dce2"

for font in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Apple SD Gothic Neo"]:
    try:
        fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
        plt.rcParams["font.family"] = [font, "DejaVu Sans"]
        break
    except ValueError:
        pass
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})

COLUMN = "frozen_plus_conc_hlb"          # 최종 조건: Option A + 농도 + HLB
THRESHOLD = 0.5


def load():
    frame = pd.read_csv(RESULTS/"predictions.csv", encoding="utf-8-sig")
    stored = pd.read_csv(RESULTS/"metrics.csv", encoding="utf-8-sig").set_index("model")
    boot = pd.read_csv(RESULTS/"bootstrap_ci.csv", encoding="utf-8-sig").set_index("model")
    protocol = json.loads((RESULTS/"protocol.json").read_text(encoding="utf-8-sig"))

    y = frame["true_binary"].to_numpy(dtype=int)
    score = frame[COLUMN].to_numpy(dtype=float)
    pred = (score >= THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    row = stored.loc[COLUMN]

    computed = {
        "n": int(len(y)), "correct": int((pred == y).sum()),
        "MCC": float(matthews_corrcoef(y, pred)),
        "ROC_AUC": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    assert cm.tolist() == [[int(row.TN), int(row.FP)], [int(row.FN), int(row.TP)]], cm
    for key, stored_key in [("MCC", "MCC"), ("ROC_AUC", "ROC_AUC"),
                            ("balanced_accuracy", "balanced_accuracy")]:
        assert np.isclose(computed[key], float(row[stored_key]), atol=1e-12), key
    assert computed["n"] == int(row.n) == protocol["n_rows"] == 41

    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    computed.update({"cm": cm, "fpr": fpr, "tpr": tpr,
                     "groups": int(frame["structure_group"].nunique()),
                     "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
                     "raw_MCC": float(stored.loc["option_a_raw_frozen", "MCC"]),
                     "raw_AUC": float(stored.loc["option_a_raw_frozen", "ROC_AUC"]),
                     "ci_lo": float(boot.loc[COLUMN, "MCC_p2.5"]),
                     "ci_hi": float(boot.loc[COLUMN, "MCC_p97.5"])})
    return computed


def draw_cm(ax, e):
    cm = e["cm"]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    tags = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max()/2 else INK
            ax.text(j, i-.07, str(cm[i, j]), ha="center", va="center",
                    fontsize=27, color=color, weight="bold")
            ax.text(j, i+.22, tags[i][j], ha="center", va="center",
                    fontsize=11, color=color)
    ax.set_xticks([0, 1], ["예측 음성\n(점수 < 0.5)", "예측 양성\n(점수 ≥ 0.5)"])
    ax.set_yticks([0, 1], ["실제 음성\n(Zein < 200)", "실제 양성\n(Zein ≥ 200)"])
    ax.tick_params(axis="both", length=0, labelsize=11, pad=7)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"혼동행렬 · n={e['n']} · {e['groups']}그룹",
                 fontsize=13, color=NAVY, pad=11)


def draw_roc(ax, e):
    ax.plot([0, 1], [0, 1], "--", color="#b7bfc5", linewidth=1)
    ax.plot(e["fpr"], e["tpr"], color=BLUE, linewidth=2.4,
            label=f"ROC (AUC={e['ROC_AUC']:.3f})")
    ax.set(xlim=(-.03, 1.04), ylim=(-.03, 1.05),
           xlabel="FPR (1−Specificity)", ylabel="TPR (Sensitivity)")
    ax.set_title("ROC curve · 저장된 OOF 점수", fontsize=13, color=NAVY, pad=11)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(labelsize=10)
    ax.grid(color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10, framealpha=.95)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def render(path):
    e = load()
    fig = plt.figure(figsize=(16, 6.5), dpi=180)
    fig.text(.5, .955, "Lechuga 2025 — frozen Skin Option A 기반 재보정 전이학습",
             ha="center", va="top", fontsize=17, weight="bold", color=NAVY)
    fig.text(.5, .902,
             f"11그룹 교차검증(OOF) · 일치 {e['correct']}/{e['n']} "
             f"({e['correct']/e['n']*100:.1f}%) · MCC {e['MCC']:.3f} · AUC {e['ROC_AUC']:.3f}",
             ha="center", va="top", fontsize=13, color=INK)
    draw_cm(fig.add_axes([.13, .245, .31, .55]), e)
    draw_roc(fig.add_axes([.565, .245, .35, .55]), e)
    footnotes = [
        "기반: frozen skin_rf의 Option A 출력 점수 고정 · 농도와 혼합물 HLB를 공변량으로 추가 · 앵커 L2(λ=1) · 판정 임계값 0.5",
        f"raw Option A의 MCC {e['raw_MCC']:.3f} · AUC {e['raw_AUC']:.3f}에서 상승 · "
        f"그룹 bootstrap 5,000회 MCC 95% 구간 [{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]",
        "정답 기준 Zein Number ≥ 200은 production이 학습한 EPA Classification과 다른 엔드포인트 · 독립 외부검증 아님",
        "근거: transfer_learning/skin_option_a_recalibration/results/predictions.csv · metrics.csv · bootstrap_ci.csv",
    ]
    for y, text in zip([.125, .089, .053, .017], footnotes):
        fig.text(.5, y, text, ha="center", va="center", fontsize=9.3, color=MUTED)
    fig.savefig(path)
    plt.close(fig)
    return e


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python make_skin_transfer_figure.py OUTPUT.png")
    info = render(sys.argv[1])
    print("saved", sys.argv[1])
    print({k: v for k, v in info.items() if k not in ("fpr", "tpr", "cm")})
