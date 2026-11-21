# -*- coding: utf-8 -*-
"""Eye · Skin 외부검증 / 전이학습 시도별 MCC 모음.

사용법: python make_전이외부_MCC비교.py <출력.png>
근거
  eye  : 외부학습/Eye_ChemBERTa_20261109/summary.json
         외부 검증/Eye_Lechuga_전이탐색_20261109/summary.json
         외부 검증/Eye_cosmetic_validation_20261109/결과요약.txt
         최최종모델/metadata.json
         ㅎ.최종정리파일/전이학습.pdf (eye_partial_transfer)
  skin : ㅎ.최종정리파일/전이학습.pdf (Option A 4단계)
         skin 모델링 과정 시각화 (production 내부 CV)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import numpy as np
import sys

for cand in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Apple SD Gothic Neo"]:
    try:
        fm.findfont(fm.FontProperties(family=cand), fallback_to_default=False)
        plt.rcParams["font.family"] = [cand, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#1f4e79"; GREEN = "#2e7d32"; GREY = "#9a9a9a"; RED = "#c00000"
INK = "#111111"; INK2 = "#333333"; INK3 = "#5a5a5a"
GRID = "#d4d4d4"

W, H = 16.0, 9.6
fig = plt.figure(figsize=(W, H), dpi=135, facecolor="white")
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis("off")


def txt(x, y, s, size=10, color=INK, weight="normal", ha="left", va="baseline", **kw):
    return AX.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=3, **kw)


# 역할: F 최종채택 / M 모델 후보 / D 진단·대조 / B 내부 기준
EYE = [
    ("구조 제외 (농도+HLB)",      "[zein] n=52",  0.725, "D"),
    ("기존계수 전이학습",          "[EPA] n=16",   0.713, "F"),
    ("앵커 없는 대조",            "[EPA] n=16",   0.713, "D"),
    ("production 내부 CV",        "[EPA] n=673",  0.294, "B"),
    ("같은 분할 LR 대조",          "[EPA] n=118",  0.294, "D"),
    ("ChemBERTa frozen",          "[EPA] n=118",  0.082, "M"),
    ("frozen Eye A → zein",       "[zein] n=52",  0.000, "M"),
    ("학습 양성률 기준선",         "[zein] n=52",  0.000, "D"),
    ("RDKit + zein head",         "[zein] n=52",  0.000, "M"),
    ("frozen Eye A → Gettings",   "[EPA] n=16",   0.000, "M"),
    ("샴푸 라벨 외부검증",         "[EPA] n=4",   -1.000, "M"),
]
SKIN = [
    ("Option A + 농도 + HLB",      "[zein] n=41",  0.656, "F"),
    ("Option A + HLB",             "[zein] n=41",  0.612, "M"),
    ("Option A raw (frozen)",      "[zein] n=41",  0.568, "M"),
    ("Option A + 농도",            "[zein] n=41",  0.558, "M"),
    ("production 내부 CV",         "[EPA] n=659",  0.264, "B"),
    ("production baseline",        "[EPA] n=659",  0.234, "D"),
]
COLOR = {"F": GREEN, "M": NAVY, "D": GREY, "B": NAVY}

# ===================== 제목 =====================
txt(0.5, 0.955, "외부검증 · 전이학습 시도별 MCC 비교",
    size=24, weight="bold", color=NAVY, ha="center", va="center")
txt(0.5, 0.916,
    "내부 교차검증 MCC는 eye 0.294 · skin 0.264. 외부 자료로 보정한 전이학습이 양쪽 모두 최고값이나 표본은 각 16건 · 41건에 그친다.",
    size=12.5, color=INK2, ha="center", va="center")


def panel(rows, x0, axw, title, base_v, base_lab):
    txt(x0, 0.852, title, size=16, weight="bold", color=INK)
    n = len(rows)
    ax = fig.add_axes([x0 + 0.190, 0.268, axw, 0.562])
    ys = np.arange(n)[::-1]
    ax.axvline(0, color="#aaaaaa", lw=1.0, zorder=1)
    ax.axvline(base_v, color=NAVY, lw=1.5, ls="--", zorder=2)
    ax.text(base_v + 0.015, n - 0.55, base_lab, fontsize=10.2, weight="bold",
            color=NAVY, ha="left", va="center")
    for y, (nm, meta, v, role) in zip(ys, rows):
        c = COLOR[role]
        ax.plot([0, v], [y, y], lw=2.4, color=c, alpha=0.5, zorder=3,
                solid_capstyle="butt")
        if role == "F":
            ax.plot(v, y, marker="*", ms=22, color=c, mec="white", mew=1.2, zorder=5)
        else:
            ax.plot(v, y, marker="o", ms=11, color=c,
                    mec="white" if role != "D" else c,
                    mfc=c if role != "D" else "white", mew=1.6, zorder=5)
        ax.text(1.13, y, f"{v:.3f}", fontsize=11.6,
                weight="bold" if role in ("F", "B") else "normal",
                color=c if role != "D" else INK3,
                ha="right", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{nm}\n{meta}" for nm, meta, *_ in rows], fontsize=10,
                       color=INK2, linespacing=1.5)
    for t, (nm, meta, v, role) in zip(ax.get_yticklabels(), rows):
        if role == "F":
            t.set_fontweight("bold"); t.set_color(GREEN)
        elif role == "B":
            t.set_fontweight("bold")
    ax.set_xlim(-1.30, 1.16)
    ax.axvline(0.93, color="#dddddd", lw=0.8, zorder=1)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5])
    ax.tick_params(axis="x", labelsize=10, colors=INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("MCC", fontsize=11.5, color=INK2, fontweight="bold", labelpad=3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#999999"); ax.spines["bottom"].set_linewidth(1.0)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    return ax


panel(EYE, 0.035, 0.262, "Eye — 시도 11건", 0.294, "내부 CV 0.294")
panel(SKIN, 0.520, 0.262, "Skin — 시도 6건", 0.264, "내부 CV 0.264")

# ===================== 범례 =====================
LY = 0.145
items = [("★", GREEN, "최종 채택"), ("●", NAVY, "모델 후보"), ("○", GREY, "진단 · 대조"),
         ("--", NAVY, "내부 교차검증 기준선")]
lx = 0.035
for mk, c, lab in items:
    if mk == "--":
        AX.plot([lx, lx + 0.020], [LY, LY], ls="--", lw=1.8, color=c, zorder=3)
        lx += 0.026
    else:
        txt(lx, LY, mk, size=15 if mk == "★" else 13, color=c, va="center")
        lx += 0.018
    txt(lx, LY, lab, size=11.5, weight="bold", color=INK2, va="center")
    lx += 0.012 + len(lab) * 0.0098

# ===================== 각주 =====================
AX.plot([0.035, 0.965], [0.102, 0.102], color="#d9d9d9", lw=1.0, zorder=2)
foot = ("[EPA]는 production 학습 라벨, [zein]은 in vitro 단백질 변성 지표. 엔드포인트가 다른 항목은 직접 비교 불가.\n"
        "외부 표본 4–52건. eye 전이학습 MCC 95% 구간 [0.000, 1.000], skin 최종 [0.307, 0.892]. "
        "탐색적 비교이며 production 성능이 아니다.")
txt(0.035, 0.080, foot, size=11, color=INK3, va="top", linespacing=1.9)

out = sys.argv[1]
fig.savefig(out, facecolor="white")
print("saved", out)
