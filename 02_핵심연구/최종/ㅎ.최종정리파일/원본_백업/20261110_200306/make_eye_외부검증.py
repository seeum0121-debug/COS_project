# -*- coding: utf-8 -*-
"""Eye 전이학습 · 외부검증 — 시도한 것 전부 한눈에 보기.

사용법: python make_eye_외부검증.py <출력.png>
근거: 최종/외부 검증/* (10 트랙), 최종/외부학습/Eye_STopTox_20261108,
      최종/외부학습/Eye_ChemBERTa_20261109,
      ㅎ.최종정리파일/전이학습.pdf (eye_partial_transfer)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

RED = "#c00000"; NAVY = "#1f4e79"; ORANGE = "#d4600f"; PURPLE = "#7030a0"; GREEN = "#2e7d32"
STAGE = [RED, NAVY, ORANGE, PURPLE, GREEN]
BOXFC = "#eaf0f7"
INK = "#111111"; INK2 = "#333333"; INK3 = "#5a5a5a"
GRID = "#cfcfcf"
CM_HI = "#1f6cb0"; CM_MID = "#64a0d4"; CM_LO = "#d6e4f2"

W, H = 16.0, 11.0
fig = plt.figure(figsize=(W, H), dpi=135, facecolor="white")
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis("off")


def txt(x, y, s, size=10, color=INK, weight="normal", ha="left", va="baseline", **kw):
    return AX.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=3, **kw)


# ===================== 제목 =====================
txt(0.5, 0.960, "Eye 전이학습 · 외부검증 — 시도한 것 전부",
    size=25, weight="bold", color=NAVY, ha="center", va="center")

# ===================== 5단계 박스 =====================
stages = [
    ("01. 외부 자료 확보", "STopTox 3,545행 · Lechuga 205행\nEPA 라벨 112사례", "13개 트랙 수집"),
    ("02. 외부 보조학습", "외부 모델 점수를 피처로\nouter 90회 재실행", "일관된 개선 미확인"),
    ("03. 사전학습 전이", "ChemBERTa frozen encoder\n학습 533 · 홀드아웃 118행", "MCC 0.294 → 0.082"),
    ("04. 외부 비교 · 전이 탐색", "zein 시나리오 4종\n66배합 · 22계면활성제", "구조 밖에서만 신호"),
    ("05. 기존계수 전이학습", "Gettings 1996 제형\n16제형 · 15 구조그룹", "MCC 0.000 → 0.545"),
]
BY, BH = 0.828, 0.104
BG = 0.012
BW = (0.93 - BG * 4) / 5
for i, (ttl, body, concl) in enumerate(stages):
    x = 0.035 + i * (BW + BG)
    AX.add_patch(FancyBboxPatch((x, BY), BW, BH, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=BOXFC, ec=STAGE[i], lw=2.2, zorder=2))
    txt(x + BW / 2, BY + BH - 0.021, ttl, size=12.5, weight="bold", color=STAGE[i],
        ha="center", va="center")
    txt(x + BW / 2, BY + BH - 0.055, body, size=10.6, color=INK2, ha="center", va="center",
        linespacing=1.7)
    txt(x + BW / 2, BY + 0.015, concl, size=11.5, weight="bold", color=STAGE[i],
        ha="center", va="center")
    if i < 4:
        AX.annotate("", xy=(x + BW + BG - 0.001, BY + BH / 2), xytext=(x + BW + 0.001, BY + BH / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.6))

# ===================== 좌상: 외부 자료 규모 =====================
txt(0.035, 0.782, "외부에서 끌어온 자료 — 트랙별 규모", size=15, weight="bold", color=INK)

ax1 = fig.add_axes([0.175, 0.500, 0.290, 0.252])
srcs = [("EPA 샴푸 제품", 5, "제품"),
        ("샴푸 특허 배합", 7, "배합"),
        ("Eye 기존계수 전이학습", 16, "제형"),
        ("특허 혼합물 재검증", 29, "조건"),
        ("Lechuga zein 전이탐색", 52, "행"),
        ("검증데이터 목록", 112, "사례"),
        ("Lechuga 2025 원자료", 205, "행"),
        ("ChemBERTa 학습·홀드아웃", 651, "행"),
        ("STopTox 외부 보조학습", 3261, "행")]
ys = np.arange(len(srcs))
vals = [s[1] for s in srcs]
cols = [GREEN if s[0].startswith("Eye 기존계수") else NAVY for s in srcs]
ax1.barh(ys, vals, color=cols, height=0.62, zorder=3)
for y, (nm, v, u) in zip(ys, srcs):
    ax1.text(v * 1.18, y, f"{v:,}{u}", fontsize=10.4, weight="bold", color=INK,
             va="center", ha="left")
ax1.set_yticks(ys)
ax1.set_yticklabels([s[0] for s in srcs], fontsize=10.2, color=INK2, fontweight="bold")
ax1.set_xscale("log")
ax1.set_xlim(2, 16000)
ax1.set_xticks([10, 100, 1000, 10000])
ax1.set_xticklabels(["10", "100", "1,000", "10,000"], fontsize=9.6, color=INK2)
ax1.set_ylim(-0.7, len(srcs) - 0.3)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color("#999999"); ax1.spines[s].set_linewidth(1.0)
ax1.grid(axis="x", color=GRID, lw=0.7, zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(axis="y", length=0)

# ===================== 좌하: 트랙 목록 =====================
txt(0.035, 0.437, "13개 트랙에서 실제로 해 본 것", size=15, weight="bold", color=INK)

items = [
    ("STopTox 외부 보조학습", "3,261행으로 외부 모델을 세워 그 점수를 피처로 붙였다"),
    ("ChemBERTa 사전학습 전이", "frozen encoder 651행 — LR 0.294 대비 0.082로 내려갔다"),
    ("Lechuga zein 전이 탐색", "시나리오 4종 — 구조 피처를 빼야 신호가 났다"),
    ("Lechuga 2025 화장품 외부비교", "66배합 · 3농도, 혼합물 순위상관 −0.344"),
    ("Lechuga A·B 엄격검토", "205행 중 완전명명 18행, B는 학습 전에 막혔다"),
    ("UVCB 민감도 비교", "구조 해결 범위 S5·S10·S22로 결론이 뒤집히는지 봤다"),
    ("EPA 제품 · 특허 재외부검증", "샴푸 5제품, 특허 29조건, 배합 7건"),
    ("Eye 기존계수 전이학습", "Gettings 1996 제형 16건 · 15그룹 — 오른쪽 결과"),
]
iy = 0.396
for i, (k, v) in enumerate(items):
    y = iy - i * 0.0442
    c = GREEN if k.startswith("Eye 기존계수") else NAVY
    AX.add_patch(Rectangle((0.035, y - 0.0245), 0.0042, 0.035, fc=c, ec="none", zorder=3))
    txt(0.051, y, k, size=11.8, weight="bold", color=c, va="center")
    txt(0.051, y - 0.0205, v, size=10.8, color=INK2, va="center")

# ===================== 우: 최종 전이학습 모델 =====================
RX = 0.530
txt(RX, 0.782, "최종 전이학습 모델 — Eye 기존계수 기반", size=15, weight="bold", color=INK)
txt(RX, 0.757, "Eye A 계수를 앵커로 두고 보정 계수만 재학습 — production 모델은 그대로.",
    size=11, color=INK2)

# --- 혼동행렬 ---
cells = [[("TN", 3, CM_MID), ("FP", 2, CM_LO)],
         [("FN", 1, CM_LO), ("TP", 10, CM_HI)]]
CX0, CY0 = RX + 0.076, 0.548
CWd, CHt = 0.082, 0.064
txt(CX0 + CWd, 0.708, "혼동행렬  n = 16 · 그룹 15", size=11.5, weight="bold",
    color=NAVY, ha="center", va="center")
for r in range(2):
    for c in range(2):
        nm, v, col = cells[r][c]
        x = CX0 + c * CWd
        y = CY0 + (1 - r) * CHt
        AX.add_patch(Rectangle((x, y), CWd, CHt, fc=col, ec="white", lw=2.0, zorder=3))
        txt(x + CWd / 2, y + CHt / 2 + 0.007, str(v), size=21, weight="bold",
            color="white" if col in (CM_HI, CM_MID) else INK, ha="center", va="center")
        txt(x + CWd / 2, y + CHt / 2 - 0.019, nm, size=9.6, weight="bold",
            color="white" if col in (CM_HI, CM_MID) else INK3, ha="center", va="center")
txt(CX0 - 0.010, CY0 + CHt * 1.5, "실제 음성\n(EPA III / IV)", size=9.8, weight="bold",
    color=INK2, ha="right", va="center", linespacing=1.6)
txt(CX0 - 0.010, CY0 + CHt * 0.5, "실제 양성\n(EPA I / II)", size=9.8, weight="bold",
    color=INK2, ha="right", va="center", linespacing=1.6)
txt(CX0 + CWd * 0.5, CY0 - 0.016, "예측 음성", size=9.8, weight="bold", color=INK2,
    ha="center", va="center")
txt(CX0 + CWd * 1.5, CY0 - 0.016, "예측 양성", size=9.8, weight="bold", color=INK2,
    ha="center", va="center")

# --- ROC curve (양성 11 · 음성 5에서 재구성, AUC 0.7636) ---
txt(0.8905, 0.708, "ROC curve", size=11.5, weight="bold", color=NAVY,
    ha="center", va="center")
ax3 = fig.add_axes([0.838, 0.548, 0.105, 0.145])
fpr = [0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 1.0]
tpr = [0.0, 2 / 11, 2 / 11, 7 / 11, 7 / 11, 1.0, 1.0]
ax3.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#bbbbbb", zorder=1)
ax3.fill_between(fpr, tpr, color=CM_HI, alpha=0.10, zorder=2)
ax3.plot(fpr, tpr, lw=2.4, color=CM_HI, solid_joinstyle="miter", zorder=3)
ax3.set_xlim(-0.02, 1.02)
ax3.set_ylim(-0.02, 1.04)
ax3.set_xticks([0, 0.5, 1.0])
ax3.set_yticks([0, 0.5, 1.0])
ax3.tick_params(labelsize=8.8, colors=INK2)
ax3.set_xlabel("FPR (1−Specificity)", fontsize=9.4, color=INK2, fontweight="bold", labelpad=2)
ax3.set_ylabel("TPR (Sensitivity)", fontsize=9.4, color=INK2, fontweight="bold", labelpad=2)
ax3.text(0.97, 0.05, "AUC 0.764", fontsize=11, weight="bold", color=CM_HI,
         ha="right", va="bottom")
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax3.spines[s].set_color("#999999"); ax3.spines[s].set_linewidth(1.0)
ax3.grid(color=GRID, lw=0.6, zorder=0)
ax3.set_axisbelow(True)

# --- 지표 표 ---
TX, TY, TW = RX, 0.272, 0.435
rows_t = [("MCC", "0.000", "0.545"),
          ("ROC-AUC", "0.527", "0.764"),
          ("균형정확도", "0.500", "0.755"),
          ("적중", "11 / 16", "13 / 16")]
hh = 0.044
col = [TX, TX + 0.150, TX + 0.293]
colw = [0.150, 0.143, 0.142]
AX.add_patch(Rectangle((TX, TY + len(rows_t) * hh), TW, hh, fc=NAVY, ec="none", zorder=3))
for c, cwd, hd in zip(col, colw, ["지표", "frozen Eye A", "전이학습 후"]):
    txt(c + cwd / 2, TY + len(rows_t) * hh + hh / 2, hd, size=12, weight="bold",
        color="white", ha="center", va="center")
for r, (a, b, c_) in enumerate(rows_t):
    y = TY + (len(rows_t) - 1 - r) * hh
    AX.add_patch(Rectangle((TX, y), TW, hh, fc="#f1f5f9" if r % 2 else "white",
                           ec="#c3cedb", lw=0.8, zorder=2))
    txt(col[0] + colw[0] / 2, y + hh / 2, a, size=12, color=INK, weight="bold",
        ha="center", va="center")
    txt(col[1] + colw[1] / 2, y + hh / 2, b, size=12, color=INK3, ha="center", va="center")
    txt(col[2] + colw[2] / 2, y + hh / 2, c_, size=12.5, color=GREEN, weight="bold",
        ha="center", va="center")
AX.add_patch(Rectangle((TX, TY), TW, (len(rows_t) + 1) * hh, fc="none", ec="#8fa2b6",
                       lw=1.2, zorder=4))

# --- 한계 ---
note = ("· 재표본 5,000회 MCC 95% 신뢰구간 [−0.067, 1.000] — 0을 포함한다\n"
        "· 16제형이 기존 673행과 겹치는지 미확인 — 독립 외부검증이 아니다")
txt(RX, 0.225, note, size=11.2, color=INK2, va="top", linespacing=2.2)

out = sys.argv[1]
fig.savefig(out, facecolor="white")
print("saved", out)
