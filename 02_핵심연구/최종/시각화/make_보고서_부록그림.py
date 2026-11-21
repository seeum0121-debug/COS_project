# -*- coding: utf-8 -*-
"""COS_보고서.docx 부록 그림 1~4를 논문 서식으로 다시 그린다.

출력 비율은 보고서에 삽입된 원본과 동일하게 맞춰 docx 배치를 바꾸지 않는다.
  그림 1  1.700  초기 신경망 단계별 MCC 분포
  그림 2  1.429  조기종료 진단 (loss / test ROC-AUC)
  그림 3  2.600  comp_group_id 수정 전후 MCC
  그림 4  1.798  wmean vs max 중첩 교차검증 MCC

근거
  그림 1·2 : 시행착오/COS_pretraining_mixture_20261104/modeling/eval_b/figdata.json
  그림 3   : 최종/검증/skin_team_audit_20261108/source/skin irritation/
             notebooks/skin_fixed_group_leak_rerun.ipynb 출력표
             (수정 후 값은 skin_aggregation_fixed_grouping_results.csv 와 일치)
  그림 4   : 보고서 수록 원본 그림의 평균값(본문 0.2457 / 0.3021)과
             오차막대 길이를 그대로 옮긴 값

사용법: python make_보고서_부록그림.py [출력폴더]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------- 공통 서식
NAVY, BLUE, ORANGE = "#24547f", "#287bb8", "#bd691d"
GREEN, GREY = "#2f7c3b", "#9aa5ad"
INK, MUTED, GRID = "#1c252c", "#59636c", "#d5dce2"

for _font in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Apple SD Gothic Neo"]:
    try:
        fm.findfont(fm.FontProperties(family=_font), fallback_to_default=False)
        plt.rcParams["font.family"] = [_font, "DejaVu Sans"]
        break
    except ValueError:
        continue
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.linewidth": 0.9,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False,
    "figure.dpi": 180,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDATA = os.path.join(ROOT, "시행착오", "COS_pretraining_mixture_20261104",
                       "modeling", "eval_b", "figdata.json")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "부록")
os.makedirs(OUT, exist_ok=True)

D = json.load(open(FIGDATA, encoding="utf-8"))


def head(fig, title, subtitle, y=0.965, ys=0.912, size=15.5, ssize=10.8):
    fig.text(0.5, y, title, ha="center", va="top", fontsize=size, weight="bold", color=NAVY)
    fig.text(0.5, ys, subtitle, ha="center", va="top", fontsize=ssize, color=MUTED)


def note(fig, text, y=0.035, size=9.2):
    fig.text(0.5, y, text, ha="center", va="center", fontsize=size, color=MUTED, linespacing=1.6)


def clean(ax, xgrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y")
    ax.grid(axis="y" if xgrid else "x", visible=False)


# ================================================================ 그림 1
def figure1(path):
    stages = D["stages"]
    fig = plt.figure(figsize=(7.9, 4.647))
    ax = fig.add_axes([0.235, 0.300, 0.735, 0.545])
    rng = np.random.default_rng(7)
    ys = np.arange(len(stages))[::-1]

    for y, st in zip(ys, stages):
        v = np.array(st["mcc"])
        m, sd = v.mean(), v.std()
        ax.scatter(v, y + rng.uniform(-0.12, 0.12, len(v)), s=28, color=BLUE,
                   alpha=0.40, linewidths=0, zorder=3)
        ax.plot([m - sd, m + sd], [y + 0.30] * 2, color=NAVY, lw=2.2,
                solid_capstyle="round", zorder=4)
        ax.scatter([m], [y + 0.30], s=78, color=NAVY, zorder=5,
                   edgecolors="white", linewidths=1.8)
        ax.scatter([st["a_reported"]], [y - 0.28], s=96, color=ORANGE, marker="D",
                   zorder=5, edgecolors="white", linewidths=1.8)
        ax.annotate(f"{m:.3f} ± {sd:.3f}", xy=(m, y + 0.30), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9.6,
                    color=NAVY, weight="bold")
        ax.annotate(f"{st['a_reported']:.3f}", xy=(st["a_reported"], y - 0.28),
                    xytext=(0, -16), textcoords="offset points", ha="center",
                    fontsize=9.6, color=ORANGE)

    ax.set_yticks(ys)
    ax.set_yticklabels([s["label"] for s in stages], fontsize=10.2, color=INK,
                       linespacing=1.5)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=9.6)
    ax.set_ylim(-0.78, len(stages) - 0.32)
    ax.set_xlabel("MCC", fontsize=10.8, labelpad=6)
    clean(ax)

    ax.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, alpha=0.45, lw=0, markersize=7,
               label="반복 교차검증 회차별 값"),
        Line2D([], [], marker="o", color=NAVY, lw=2.2, markersize=9,
               markeredgecolor="white", markeredgewidth=1.8, label="평균 ± 1 표준편차"),
        Line2D([], [], marker="D", color=ORANGE, lw=0, markersize=8.5,
               markeredgecolor="white", markeredgewidth=1.8, label="단일 분할 1회 보고값"),
    ], loc="upper center", bbox_to_anchor=(0.42, -0.200), ncol=3, fontsize=9.8)

    head(fig, "초기 신경망 설정 변경 3단계의 반복 교차검증 MCC",
         "반복 group 5-fold 10회(분할·초기화 시드 동시 변경)와 단일 분할 1회 보고값의 비교 · skin 라벨 A, 양성 30건")
    note(fig, "단계 간 평균 차이(−0.067, +0.043)는 회차 간 표준편차(0.034–0.049)를 넘지 않는다.\n"
              "설정 변경의 효과와 반복에 따른 변동이 분리되지 않는다.", y=0.062)
    fig.savefig(path)
    plt.close(fig)


# ================================================================ 그림 2
def figure2(path):
    C = D["curve"]
    ep = np.array(C["epoch"])
    tr, va, auc = (np.array(C[k]) for k in ("train_loss", "val_loss", "test_auc"))
    best = int(ep[int(np.argmin(va))])

    fig = plt.figure(figsize=(7.7, 5.386))
    a0 = fig.add_axes([0.105, 0.505, 0.775, 0.325])
    a1 = fig.add_axes([0.105, 0.185, 0.775, 0.245])

    a0.plot(ep, tr, color=BLUE, lw=2.1, solid_capstyle="round")
    a0.plot(ep, va, color=ORANGE, lw=2.1, solid_capstyle="round")
    a0.annotate("train loss", xy=(ep[-1], tr[-1]), xytext=(8, 0),
                textcoords="offset points", color=BLUE, fontsize=10,
                va="center", weight="bold")
    a0.annotate("validation loss", xy=(ep[-1], va[-1]), xytext=(8, 0),
                textcoords="offset points", color=ORANGE, fontsize=10,
                va="center", weight="bold")
    a0.set_ylabel("loss", fontsize=10.5)
    a0.set_xticklabels([])

    a1.plot(ep, auc, color=GREEN, lw=2.1, solid_capstyle="round")
    a1.annotate("test ROC-AUC", xy=(ep[-1], auc[-1]), xytext=(8, 0),
                textcoords="offset points", color=GREEN, fontsize=10,
                va="center", weight="bold")
    a1.set_ylabel("test ROC-AUC", fontsize=10.5)
    a1.set_xlabel("epoch", fontsize=10.5, labelpad=5)

    for a in (a0, a1):
        a.axvline(best, color=GREY, lw=1.0, zorder=1)
        a.set_xlim(0.5, 44)
        a.tick_params(labelsize=9.4)
        clean(a, xgrid=False)
    a0.annotate(f"조기종료 채택 지점 (epoch {best})", xy=(best, va.max() * 0.46),
                xytext=(9, 0), textcoords="offset points", fontsize=9.4,
                color=MUTED, va="center")

    head(fig, "조기종료가 채택한 지점은 학습 초기 epoch에 해당",
         "skin 단일 분할 · train loss는 5 epoch 내 0 부근으로 수렴, validation loss는 epoch 2 이후 증가")
    note(fig, "validation loss 최저점이 epoch 3에 형성되어 조기종료가 학습 초기 모델을 선택한다.\n"
              "최종 배포모델의 성능이 아니라 초기 신경망의 학습 안정성을 점검한 진단 결과이다.", y=0.050)
    fig.savefig(path)
    plt.close(fig)


# ================================================================ 그림 3
# skin_fixed_group_leak_rerun.ipynb 출력표 (label, model, method) -> (전, 전sd, 후, 후sd)
GROUPFIX = {
    ("B", "ExtraTrees"): [("wmean", 0.204, 0.022, 0.210, 0.049),
                          ("top",   0.199, 0.037, 0.255, 0.021),
                          ("max",   0.218, 0.029, 0.186, 0.034),
                          ("min",   0.214, 0.029, 0.209, 0.029)],
    ("A", "RandomForest"): [("wmean", 0.153, 0.066, 0.198, 0.014),
                            ("top",   0.222, 0.024, 0.253, 0.059),
                            ("max",   0.203, 0.069, 0.242, 0.025),
                            ("min",   0.250, 0.075, 0.260, 0.016)],
}


def figure3(path):
    fig = plt.figure(figsize=(9.8, 4.9))
    panels = [(("B", "ExtraTrees"), "피부 B 라벨 · ExtraTrees", 0.070),
              (("A", "RandomForest"), "안 자극성 A 라벨 · RandomForest", 0.560)]
    for key, title, x0 in panels:
        rows = GROUPFIX[key]
        ax = fig.add_axes([x0, 0.185, 0.370, 0.555])
        idx = np.arange(len(rows))
        w = 0.30
        before = [r[1] for r in rows]
        bsd = [r[2] for r in rows]
        after = [r[3] for r in rows]
        asd = [r[4] for r in rows]
        ax.bar(idx - w / 2, before, w, color=GREY, label="수정 전 · 463그룹",
               yerr=bsd, error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 3})
        ax.bar(idx + w / 2, after, w, color=NAVY, label="수정 후 · 637그룹",
               yerr=asd, error_kw={"ecolor": MUTED, "elinewidth": 1.2, "capsize": 3})
        for i, (v, s) in enumerate(zip(before, bsd)):
            ax.text(i - w / 2, v + s + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.6, color=MUTED)
        for i, (v, s) in enumerate(zip(after, asd)):
            ax.text(i + w / 2, v + s + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.6, color=NAVY, weight="bold")
        ax.set_xticks(idx)
        ax.set_xticklabels([r[0] for r in rows], fontsize=10.2, color=INK)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.tick_params(axis="y", labelsize=9.4)
        ax.set_ylim(0, 0.40)
        ax.set_ylabel("MCC", fontsize=10.5)
        ax.set_title(title, fontsize=11.6, color=NAVY, pad=10)
        clean(ax, xgrid=False)
        if x0 < 0.3:
            ax.legend(loc="upper left", fontsize=9.6, ncol=1)

    head(fig, "조성 기반 comp_group_id 수정 전후의 MCC",
         "수정 전 vectoronly_v3 463그룹 → 수정 후 comp_group_id 637그룹 · 659행(특성 완비 646–649행)\n"
         "막대 = 반복 평균, 오차막대 = 표준편차",
         y=0.968, ys=0.912, size=15.5, ssize=10.4)
    note(fig, "집계 방식에 따라 변화의 방향이 달라, 동일 조성의 반복행을 묶는 방식이 내부 평가값을 좌우함을 보인다.\n"
              "데이터 누수 통제를 위한 민감도 분석이며 최종 production 성능표가 아니다.", y=0.052)
    fig.savefig(path)
    plt.close(fig)


# ================================================================ 그림 4
WMAX = [("전체 (n = 649)", 0.246, 0.030, 0.302, 0.029),
        ("다성분 혼합물만 (n = 274)", 0.227, 0.036, 0.323, 0.031)]


def figure4(path):
    fig = plt.figure(figsize=(8.7, 4.839))
    for i, (title, wm, wsd, mx, msd) in enumerate(WMAX):
        ax = fig.add_axes([0.095 + i * 0.480, 0.235, 0.355, 0.535])
        ax.bar([0, 1], [wm, mx], 0.52, color=[NAVY, ORANGE],
               yerr=[wsd, msd], error_kw={"ecolor": MUTED, "elinewidth": 1.3, "capsize": 5})
        for x, (v, s) in enumerate([(wm, wsd), (mx, msd)]):
            ax.text(x, v + s + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=11.6, weight="bold", color=NAVY if x == 0 else ORANGE)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["wmean\n(함량 가중평균)", "max\n(특성별 최댓값)"],
                           fontsize=10.4, color=INK, linespacing=1.5)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.tick_params(axis="y", labelsize=9.4)
        ax.set_xlim(-0.7, 1.7)
        ax.set_ylim(0, 0.42)
        ax.set_ylabel("MCC", fontsize=10.5)
        ax.set_title(title, fontsize=11.8, color=NAVY, pad=11)
        clean(ax, xgrid=False)

    head(fig, "wmean과 max 집계의 동일 표본 중첩 교차검증 MCC",
         "outer 5 × inner 3 nested CV, 20회 반복 · 막대 = 평균, 오차막대 = 표준편차")
    note(fig, "20회 반복 전부에서 max가 높았으나, 특성별 최댓값은 성분별 함량을 반영하지 않고 미량 성분의 극값에 민감하다.\n"
              "입력 표현의 해석 가능성을 함께 고려하여 production 표현은 wmean + resolved_pct로 유지하였다.", y=0.055)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    jobs = [("부록그림1_초기신경망_단계별MCC.png", figure1),
            ("부록그림2_조기종료_진단.png", figure2),
            ("부록그림3_그룹수정_전후MCC.png", figure3),
            ("부록그림4_wmean_vs_max.png", figure4)]
    for name, fn in jobs:
        p = os.path.join(OUT, name)
        fn(p)
        print("saved", p)
