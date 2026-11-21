# -*- coding: utf-8 -*-
"""Eye Irritation 모델링 — 전체 시행착오 한눈에 보기.

사용법: python make_eye_시행착오.py <출력.png>
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm
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
BARC = [RED, NAVY, "#ed7d31", PURPLE, GREEN]
BOXFC = "#eaf0f7"
INK = "#111111"; INK2 = "#333333"; INK3 = "#5a5a5a"
GRID = "#cfcfcf"

W, H = 16.0, 11.0
fig = plt.figure(figsize=(W, H), dpi=135, facecolor="white")
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis("off")


def txt(x, y, s, size=10, color=INK, weight="normal", ha="left", va="baseline", **kw):
    return AX.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=3, **kw)


# ===================== 제목 =====================
txt(0.5, 0.960, "Eye Irritation 모델링 — 전체 시행착오 한눈에 보기",
    size=25, weight="bold", color=NAVY, ha="center", va="center")

# ===================== 5단계 박스 =====================
stages = [
    ("01. 통합 집계", "EPA · EPA+GHS를\n한 규약으로 재실행", "비교 기준 통일"),
    ("02. 진단 · 모델 보강", "앙상블 · 그룹화 · 순서형\n최악성분 · 특징집합", "5개 중 3개는 기준이 1위"),
    ("03. 2차 탐색 · 입력 수정", "표현결합 · 확률보정\n교차앙상블 · 농도 복원", "중복 331 → 70행"),
    ("04. production · 외부 보조", "outer 360회 전량 재실행\nSTopTox 3,261행 보조학습", "외부검증 적격 0행"),
    ("05. 최종 확정", "집계방식 비교 · 673행 전량\n추가 67행 통제비교", "MCC 0.306 → 0.294"),
]
BY, BH = 0.828, 0.104
BG = 0.012
BW = (0.93 - BG * 4) / 5
for i, (ttl, body, concl) in enumerate(stages):
    x = 0.035 + i * (BW + BG)
    AX.add_patch(FancyBboxPatch((x, BY), BW, BH, boxstyle="round,pad=0,rounding_size=0.012",
                                fc=BOXFC, ec=STAGE[i], lw=2.2, zorder=2))
    txt(x + BW / 2, BY + BH - 0.021, ttl, size=13, weight="bold", color=STAGE[i],
        ha="center", va="center")
    txt(x + BW / 2, BY + BH - 0.055, body, size=11, color=INK2, ha="center", va="center",
        linespacing=1.7)
    txt(x + BW / 2, BY + 0.015, concl, size=11.5, weight="bold", color=STAGE[i],
        ha="center", va="center")
    if i < 4:
        AX.annotate("", xy=(x + BW + BG - 0.001, BY + BH / 2), xytext=(x + BW + 0.001, BY + BH / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.6))

# ===================== 좌측 차트 =====================
txt(0.055, 0.782, "설정 변경 9건의 최고 MCC는 모두 기준선 오차 범위 내",
    size=15, weight="bold", color=INK)

ax1 = fig.add_axes([0.066, 0.500, 0.381, 0.252])
exps = [
    ("02", "앙상블", 0.354, True, "단일 lr"),
    ("03", "그룹화", 0.380, True, "기존"),
    ("04", "순서형", 0.342, False, "순서형"),
    ("05", "최악성분", 0.338, False, "최악"),
    ("06", "특징집합", 0.352, True, "기준특징"),
    ("07", "표현결합", 0.335, False, "결합"),
    ("08", "확률보정", 0.352, True, "무보정"),
    ("10", "교차앙상블", 0.352, True, "단일 lr"),
    ("11", "농도복원", 0.358, False, "농도피처"),
]
BASE, SD = 0.354, 0.094
ax1.axhspan(BASE - SD, BASE + SD, color="#e3ebf4", zorder=0)
ax1.axhline(BASE, color=NAVY, lw=1.8, ls="--", zorder=2)
ax1.text(8.45, BASE + SD - 0.006, "기준선 오차 ±0.094", fontsize=10.5, weight="bold",
         color="#4a5d72", ha="right", va="top")

for i, (num, nm, v, isbase, arm) in enumerate(exps):
    if v is None:
        ax1.plot(i, BASE, marker="x", ms=10, mew=2.4, color="#8a8a8a", zorder=4)
        ax1.text(i, BASE + 0.013, "중단", fontsize=9.6, weight="bold", color="#6e6e6e",
                 ha="center", va="bottom")
        continue
    col = NAVY if isbase else ORANGE
    ax1.plot(i, v, marker="o", ms=11, color=col, mec="white", mew=1.5, zorder=5)
    ax1.text(i, v + 0.011, f"{v:.3f}", fontsize=10.4, weight="bold", color=col,
             ha="center", va="bottom")
    ax1.text(i, v + 0.032, arm, fontsize=9.6, color=INK2, ha="center", va="bottom")

ax1.set_xlim(-0.6, 8.5)
ax1.set_ylim(0.248, 0.452)
ax1.set_xticks(range(len(exps)))
ax1.set_xticklabels([f"{n}\n{m}" for n, m, *_ in exps], fontsize=9.8, color=INK2,
                    fontweight="bold")
ax1.set_ylabel("최고 MCC (라벨 A)", fontsize=11.5, color=INK2, labelpad=6, fontweight="bold")
ax1.tick_params(axis="y", labelsize=10, colors=INK2)
ax1.set_yticks([0.26, 0.30, 0.34, 0.38, 0.42])
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color("#999999"); ax1.spines[s].set_linewidth(1.0)
ax1.grid(axis="y", color=GRID, lw=0.7, zorder=0)
ax1.set_axisbelow(True)

ax1.plot([], [], ls="--", lw=1.8, color=NAVY, label="기준선: 단일 lr 0.354")
ax1.plot([], [], "o", ms=9, ls="none", color=NAVY, label="기준 설정이 1위 (5/9)")
ax1.plot([], [], "o", ms=9, ls="none", color=ORANGE, label="새 설정이 1위 (4/9)")
lg = ax1.legend(loc="lower left", fontsize=10.2, frameon=False, handletextpad=0.5,
                borderaxespad=0.3, labelcolor=INK2, labelspacing=0.5)
for t in lg.get_texts():
    t.set_fontweight("bold")

# ===================== 우측 차트 =====================
txt(0.560, 0.782, "단계별 성능 기록 678행", size=15, weight="bold", color=INK)

ax2 = fig.add_axes([0.604, 0.500, 0.348, 0.252])
labels = ["01\n통합 집계", "02\n진단·모델 보강", "03\n2차 탐색·입력", "04\nproduction·외부", "05\n최종 확정"]
vals = [48, 222, 210, 108, 90]
bars = ax2.bar(range(5), vals, color=BARC, width=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 5, f"{v}행", fontsize=13.5, weight="bold",
             color=INK, ha="center", va="bottom")
ax2.set_xticks(range(5))
ax2.set_xticklabels(labels, fontsize=9.8, color=INK2, fontweight="bold")
ax2.set_ylim(0, 255)
ax2.tick_params(axis="y", labelsize=10, colors=INK2)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax2.spines[s].set_color("#999999"); ax2.spines[s].set_linewidth(1.0)
ax2.grid(axis="y", color=GRID, lw=0.7, zorder=0)
ax2.set_axisbelow(True)

# ===================== 하단 좌 =====================
txt(0.035, 0.412, "모델 보강 실험 00–11 주요 결과", size=15, weight="bold", color=INK)

items = [
    ("00  진단", "673행 중 고유 특징 벡터 444개, 중복 그룹에 331행 포함"),
    ("01  상한", "동일 입력의 라벨 충돌로 달성 상한 MCC 0.861(A) · 0.795(B)"),
    ("02·03  앙상블 · 그룹화", "앙상블 6종 및 벡터 병합 그룹화 모두 기준 미달"),
    ("04·05  순서형 · 최악성분", "4-class 순서형 0.342 · 최악 성분 pooling 0.338, 모두 기준 미달"),
    ("06·08  특징 · 보정", "특징집합 변경과 확률 보정 적용 시에도 기준 설정이 최고 성능"),
    ("11  농도 복원", "단일성분 농도 약분 수정 후 중복 행 331 → 70으로 감소"),
]
iy = 0.373
for i, (k, v) in enumerate(items):
    y = iy - i * 0.0418
    AX.add_patch(Rectangle((0.035, y - 0.0235), 0.0042, 0.033, fc=NAVY, ec="none", zorder=3))
    txt(0.050, y, k, size=11.8, weight="bold", color=NAVY, va="center")
    txt(0.050, y - 0.0195, v, size=10.8, color=INK2, va="center")

# ===================== 하단 우 =====================
txt(0.560, 0.412, "최종 확정: 606행 평가에서 673행 전량 평가로", size=15,
    weight="bold", color=INK)

TX, TY, TW = 0.560, 0.150, 0.405
rows_t = [("MCC", "0.306", "0.294 ± 0.004"),
          ("Sensitivity", "0.704", "0.691"),
          ("Specificity", "0.678", "0.678"),
          ("ROC-AUC", "0.719", "0.716")]
hh = 0.0406
col = [TX, TX + 0.150, TX + 0.282]
colw = [0.150, 0.132, 0.123]
AX.add_patch(Rectangle((TX, TY + len(rows_t) * hh), TW, hh, fc=NAVY, ec="none", zorder=3))
for c, cwd, hd in zip(col, colw, ["지표", "8부 · 606행 평가", "9부 · 673행 전량"]):
    txt(c + cwd / 2, TY + len(rows_t) * hh + hh / 2, hd, size=12, weight="bold",
        color="white", ha="center", va="center")
for r, (a, b, c_) in enumerate(rows_t):
    y = TY + (len(rows_t) - 1 - r) * hh
    AX.add_patch(Rectangle((TX, y), TW, hh, fc="#f1f5f9" if r % 2 else "white",
                           ec="#c3cedb", lw=0.8, zorder=2))
    for cx, cwd, s, wt in zip(col, colw, (a, b, c_), ("bold", "normal", "bold")):
        txt(cx + cwd / 2, y + hh / 2, s, size=12, color=INK, weight=wt, ha="center", va="center")
AX.add_patch(Rectangle((TX, TY), TW, (len(rows_t) + 1) * hh, fc="none", ec="#8fa2b6",
                       lw=1.2, zorder=4))

# ===================== 하단: 핵심 수치 =====================
kpis = [("20", "개", "실험 트랙"),
        ("20,438", "회", "하이퍼파라미터 후보 적합"),
        ("1,020", "회", "nested CV outer 모델 학습"),
        ("7", "종", "모델 계열")]
KY, KH = 0.020, 0.086
KG = 0.012
KW = (0.93 - KG * (len(kpis) - 1)) / len(kpis)
for i, (v, unit, k) in enumerate(kpis):
    x = 0.035 + i * (KW + KG)
    AX.add_patch(FancyBboxPatch((x, KY), KW, KH, boxstyle="round,pad=0,rounding_size=0.012",
                                fc="#eef3f9", ec=NAVY, lw=1.6, zorder=2))
    t = txt(x + 0.018, KY + 0.038, v, size=30, weight="bold", color=NAVY)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(AX.transData.inverted())
    txt(bb.x1 + 0.005, KY + 0.042, unit, size=13, weight="bold", color=INK2)
    txt(x + 0.018, KY + 0.016, k, size=12.5, weight="bold", color=INK2)

out = sys.argv[1]
fig.savefig(out, facecolor="white")
print("saved", out)
