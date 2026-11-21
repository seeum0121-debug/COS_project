# -*- coding: utf-8 -*-
"""Figure 1 — 화장품 자극성 모델링을 위한 공개 데이터의 가용성.

보고서/논문 게재용 3패널 figure (A/B/C)를 PDF(벡터)와 PNG(600 dpi)로 저장한다.
수치 출처는 같은 폴더의 Figure1_caption.md 에 정리되어 있다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm
import numpy as np
import os
import sys

# ---------------------------------------------------------------- 서체
for cand in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Apple SD Gothic Neo"]:
    try:
        fm.findfont(fm.FontProperties(family=cand), fallback_to_default=False)
        plt.rcParams["font.family"] = [cand, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams.update({
    "axes.unicode_minus": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.0,
    "ytick.major.size": 0.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.4,
    "axes.labelsize": 7.8,
    "pdf.fonttype": 42,     # TrueType — 편집 가능한 벡터 텍스트
    "ps.fonttype": 42,
})

INK = "#111111"
INK2 = "#44444a"
INK3 = "#75757c"
GRID = "#d9d9d6"
BLUE = "#2a78d6"      # 단일 화학물질 단위
ORANGE = "#eb6834"    # 제형(혼합물) 포함
DARK = "#1c5cab"
ZERO = "#b4232a"

PANEL_KW = dict(fontsize=10.5, fontweight="bold", color=INK, va="top", ha="left")

# ---------------------------------------------------------------- 자료
# (A) 요구 조건을 누적할 때 남는 공개 자료
A_ROWS = [
    ("화장품 성분 인벤토리\n(CosIng 등재 성분명)", "", 15000),
    ("+ 독성 정보 연결\n(COSMOS DB)", "", 7021),
    ("+ in vivo 안자극 정답\n(REACH 등록자료 마이닝)", "", 5629),
    ("+ in vivo 피부자극 정답\n(STopTox 균형 모델링셋)", "", 2292),
    ("+ 혼합물 포함\n(OCUTOXDB)", "", 593),
    ("+ 화장품 제형·전체 배합비", "", 0),
]

# (B) 공개 자극성 데이터셋의 규모와 단위  (이름, n, 엔드포인트, 혼합물포함, 본연구)
B_ROWS = [
    ("COSMOS DB (화장품 관련 독성)", 7021, "공통", False, False),
    ("REACH 마이닝 Draize", 5629, "안자극", False, False),
    ("STopTox 안자극 (Table S4)", 3545, "안자극", False, False),
    ("STopTox 피부자극 균형셋", 2292, "피부자극", False, False),
    ("본 연구 — 안자극 제형", 673, "안자극", True, True),
    ("본 연구 — 피부자극 제형", 662, "피부자극", True, True),
    ("Cosmetics Europe DRD", 634, "안자극", False, False),
    ("OCUTOXDB", 593, "안자극", True, False),
    ("ECETOC TR 066 (피부자극·부식)", 176, "피부자극", False, False),
    ("ChemSkin 참조 DB", 100, "피부자극", False, False),
    ("CON4EI 평가셋", 80, "안자극", False, False),
    ("ECETOC 안자극 표준셋", 38, "안자극", False, False),
]

# (C) 요건 충족 매트릭스  2=충족, 1=부분·제한, 0=미충족
C_COLS = ["공개 접근", "제형(혼합물)\n단위", "성분 농도\n정보", "화장품\n대상", "in vivo\n자극성 정답"]
C_ROWS = [
    ("CosIng 성분 glossary", [2, 0, 0, 2, 0], False),
    ("COSMOS DB", [2, 0, 0, 2, 1], False),
    ("REACH 마이닝 Draize", [2, 0, 0, 0, 2], False),
    ("STopTox (안자극·피부자극)", [2, 0, 0, 0, 2], False),
    ("Cosmetics Europe DRD", [1, 0, 0, 1, 2], False),
    ("OCUTOXDB", [2, 1, 0, 0, 2], False),
    ("ECETOC · ChemSkin · CON4EI", [2, 0, 0, 0, 2], False),
    ("화장품 자극성 문헌·포스터", [2, 2, 1, 2, 1], False),
    ("본 연구 (안 673 · 피 662 제형)", [2, 2, 2, 0, 2], True),
]

# ---------------------------------------------------------------- 캔버스
FIG_W, FIG_H = 7.09, 8.55          # 180 mm 폭 (double column)
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=600, facecolor="white")
gs = GridSpec(3, 1, figure=fig, height_ratios=[1.00, 1.32, 1.30],
              left=0.265, right=0.975, top=0.963, bottom=0.070, hspace=0.46)

# ================================================================ (A)
axA = fig.add_subplot(gs[0])
yA = np.arange(len(A_ROWS))[::-1]
XMIN, XMAX = 20, 42000
for y, (lab, src, n) in zip(yA, A_ROWS):
    if n > 0:
        axA.barh(y, n - XMIN, left=XMIN, height=0.58, color=DARK, edgecolor="none", zorder=3)
        axA.text(n * 1.22, y, f"{n:,}", va="center", ha="left", fontsize=7.2,
                 color=INK, fontweight="bold", zorder=4)
    else:
        axA.barh(y, 55 - XMIN, left=XMIN, height=0.58, facecolor="none",
                 edgecolor=ZERO, linewidth=0.9, hatch="////", zorder=3)
        axA.text(68, y, "0", va="center", ha="left", fontsize=8.0,
                 color=ZERO, fontweight="bold", zorder=4)
        axA.text(100, y, "조건을 모두 만족하는 공개 자료 없음", va="center", ha="left",
                 fontsize=6.8, color=ZERO, zorder=4)

axA.set_yticks(yA)
axA.set_yticklabels([lab for lab, _, _ in A_ROWS], fontsize=7.1, color=INK,
                    linespacing=1.35)
axA.set_xscale("log")
axA.set_xlim(XMIN, XMAX)
axA.set_ylim(-0.75, len(A_ROWS) - 0.35)
axA.set_xticks([100, 1000, 10000])
axA.set_xticklabels(["100", "1,000", "10,000"])
axA.set_xlabel("남아 있는 공개 자료의 수 (물질 또는 제형, 로그 축척)", color=INK2)
axA.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right", "left"):
    axA.spines[s].set_visible(False)
axA.tick_params(axis="y", length=0, pad=2)
fig.text(0.012, 0.975, "A", **PANEL_KW)
axA.set_title("요구 조건을 누적할 때 남는 공개 자료", fontsize=8.4, fontweight="bold",
              color=INK, loc="left", pad=6)

# ================================================================ (B)
axB = fig.add_subplot(gs[1])
yB = np.arange(len(B_ROWS))[::-1]
for y, (lab, n, ep, mix, ours) in zip(yB, B_ROWS):
    col = ORANGE if mix else BLUE
    mk = "o" if mix else "s"
    axB.hlines(y, XMIN, n, color=col, linewidth=1.1, alpha=0.55, zorder=2)
    axB.plot(n, y, mk, markersize=5.2 if ours else 4.4, color=col,
             markeredgecolor="white", markeredgewidth=0.7 if ours else 0.5, zorder=4)
    axB.text(n * 1.25, y, f"{n:,}", va="center", ha="left", fontsize=6.9,
             color=INK, fontweight="bold" if ours else "normal", zorder=4)
    if ours:
        axB.add_patch(Rectangle((XMIN, y - 0.44), XMAX - XMIN, 0.88,
                                facecolor="#f1f0ec", edgecolor="none", zorder=1))

EPTAG = {"안자극": "[안]", "피부자극": "[피]", "공통": "[공]"}
axB.set_yticks(yB)
axB.set_yticklabels([f"{EPTAG[ep]} {lab}" for lab, _, ep, _, _ in B_ROWS],
                    fontsize=7.1, color=INK)
for t, (_, _, _, _, ours) in zip(axB.get_yticklabels(), B_ROWS):
    if ours:
        t.set_fontweight("bold")
axB.set_xscale("log")
axB.set_xlim(XMIN, XMAX)
axB.set_ylim(-0.75, len(B_ROWS) - 0.35)
axB.set_xticks([100, 1000, 10000])
axB.set_xticklabels(["100", "1,000", "10,000"])
axB.set_xlabel("수록 물질 또는 제형의 수 (로그 축척)", color=INK2)
axB.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.tick_params(axis="y", length=0, pad=2)
axB.legend(handles=[
    Line2D([], [], marker="s", color=BLUE, linestyle="none", markersize=4.4,
           markeredgecolor="white", markeredgewidth=0.5, label="단일 화학물질 단위"),
    Line2D([], [], marker="o", color=ORANGE, linestyle="none", markersize=4.4,
           markeredgecolor="white", markeredgewidth=0.5, label="제형(혼합물) 포함"),
], loc="lower right", fontsize=6.6, frameon=True, framealpha=1.0,
    edgecolor=GRID, borderpad=0.45, handletextpad=0.5, labelspacing=0.35)
fig.text(0.012, 0.645, "B", **PANEL_KW)
axB.set_title("공개 자극성 데이터셋의 규모와 수록 단위", fontsize=8.4, fontweight="bold",
              color=INK, loc="left", pad=6)

# ================================================================ (C)
axC = fig.add_subplot(gs[2])
nR, nC = len(C_ROWS), len(C_COLS)
axC.set_xlim(-0.5, nC - 0.5)
axC.set_ylim(-1.55, nR - 0.4)
axC.invert_yaxis()
axC.set_axis_off()

for j in range(nC):
    axC.text(j, -1.45, C_COLS[j], ha="center", va="bottom", fontsize=6.7,
             color=INK2, linespacing=1.35)
for i, (lab, vals, ours) in enumerate(C_ROWS):
    if ours:
        axC.add_patch(Rectangle((-0.5, i - 0.47), nC, 0.94,
                                facecolor="#f1f0ec", edgecolor="none", zorder=1))
    axC.text(-0.62, i, ("▶ " + lab) if ours else lab, ha="right", va="center",
             fontsize=7.1, color=INK, fontweight="bold" if ours else "normal", zorder=3)
    for j, v in enumerate(vals):
        if v == 2:
            axC.plot(j, i, "o", markersize=6.4, color=INK, zorder=3)
        elif v == 1:
            axC.plot(j, i, "o", markersize=6.4, fillstyle="left",
                     markerfacecolor=INK, markerfacecoloralt="white",
                     markeredgecolor=INK, markeredgewidth=0.9, zorder=3)
        else:
            axC.plot(j, i, "o", markersize=6.4, markerfacecolor="white",
                     markeredgecolor="#b9b9b4", markeredgewidth=0.9, zorder=3)
for i in range(nR + 1):
    axC.plot([-0.5, nC - 0.5], [i - 0.5, i - 0.5], color=GRID, linewidth=0.45, zorder=0)

axC.legend(handles=[
    Line2D([], [], marker="o", color=INK, linestyle="none", markersize=5.4, label="충족"),
    Line2D([], [], marker="o", fillstyle="left", markerfacecolor=INK,
           markerfacecoloralt="white", markeredgecolor=INK,
           markeredgewidth=0.9, color="none", linestyle="none", markersize=5.4, label="부분·제한"),
    Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#b9b9b4",
           markeredgewidth=0.9, color="none", linestyle="none", markersize=5.4, label="미충족"),
], loc="lower center", bbox_to_anchor=(0.5, -0.175), ncol=3, fontsize=6.8,
    frameon=False, handletextpad=0.45, columnspacing=1.6)
fig.text(0.012, 0.330, "C", **PANEL_KW)
axC.set_title("화장품 자극성 모델링 요건의 충족 여부", fontsize=8.4, fontweight="bold",
              color=INK, loc="left", pad=26)

# ---------------------------------------------------------------- 저장
outdir = sys.argv[1] if len(sys.argv) > 1 else "."
base = os.path.join(outdir, "Figure1_공개데이터_가용성")
fig.savefig(base + ".pdf", facecolor="white")
fig.savefig(base + ".png", facecolor="white", dpi=600)
fig.savefig(base + "_300dpi.png", facecolor="white", dpi=300)
print("saved:", base + ".pdf / .png / _300dpi.png")
