# -*- coding: utf-8 -*-
"""Skin Irritation 모델링 — 전체 시행착오 한눈에 보기 (eye 판과 같은 형식).

사용법: python make_skin_시행착오.py <출력.png>
근거: skin 모델링 과정 시각화(skin팀) + modeling/eval_b/run_*.py README
      + skin irritation/*.csv 행수 직접 집계
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
BARC = [NAVY, "#ed7d31", PURPLE]
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
txt(0.5, 0.960, "Skin Irritation 모델링 — 전체 시행착오 한눈에 보기",
    size=25, weight="bold", color=NAVY, ha="center", va="center")

# ===================== 5단계 박스 =====================
stages = [
    ("01. NN 초기 트랙", "NN(MLP) 아키텍처 실험\n조기종료 · 차원축소 · threshold", "RF로 대체"),
    ("02. RF 재결정 · 집계 탐색", "재결정 3차 · 2회 번복\n설정비교 118행", "production = RF"),
    ("03. 그룹누수 · 하위집단", "comp_group_id 버그 수정\n설정비교 168행", "단일 / 다성분 분리평가"),
    ("04. wmean vs max 결정", "8차 재비교 · 1위 역전\n설정비교 128행", "wmean 유지 (해석성)"),
    ("05. 최종 production", "데이터 증강 2차\n버그수정 4건 재검증", "MCC 0.234 → 0.264"),
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

# ===================== 좌측 차트 — 집계방식 1위 8회 =====================
txt(0.055, 0.782, "8차에 걸친 집계 방식 재검증에서 1위가 고정되지 않음",
    size=15, weight="bold", color=INK)

ax1 = fig.add_axes([0.108, 0.500, 0.339, 0.252])
CATS = ["min", "top", "max", "동률", "wmean"]
rounds = [
    ("①\n구feature",   "min",   None,  "PR-AUC\n2배 이상"),
    ("②\n17차",        "top",   0.260, "ET 0.260"),
    ("③\nrange",       "max",   0.218, "ET 0.218"),
    ("④\n다성분",      "동률",  None,  "거의 동률"),
    ("⑤\n그룹수정",    "top",   0.255, "ET 0.255"),
    ("⑥\n농도추가",    "top",   0.249, "RF 0.249"),
    ("⑦\n공식 20rep",  "wmean", 0.264, "RF 0.264"),
    ("⑧\nmin·max",     "max",   0.302, "RF 0.302"),
]
xs = list(range(len(rounds)))
ys = [CATS.index(r[1]) for r in rounds]
ax1.plot(xs, ys, ls=":", lw=1.6, color="#bbbbbb", zorder=2)
for x, y, (lab, cat, mcc, note) in zip(xs, ys, rounds):
    c = NAVY if cat == "wmean" else ORANGE
    ax1.plot(x, y, "o", ms=14, color=c, mec="white", mew=1.6, zorder=5)
    ax1.text(x, y + 0.32, cat, fontsize=11, weight="bold", color=c,
             ha="center", va="bottom")
    ax1.text(x, y - 0.35, note, fontsize=8.8, color=INK3, ha="center", va="top",
             linespacing=1.5)
ax1.plot([], [], "o", ms=9, ls="none", color=NAVY, label="최종 채택 wmean이 1위 (1/8)")
ax1.plot([], [], "o", ms=9, ls="none", color=ORANGE, label="다른 방식이 1위 (7/8)")
lg = ax1.legend(loc="upper left", fontsize=9.6, frameon=False, handletextpad=0.5,
                borderaxespad=0.3, labelcolor=INK2, labelspacing=0.45)
for t in lg.get_texts():
    t.set_fontweight("bold")
ax1.set_xlim(-0.65, len(rounds) - 0.35)
ax1.set_ylim(-1.15, len(CATS) - 0.35)
ax1.set_xticks(xs)
ax1.set_xticklabels([r[0] for r in rounds], fontsize=9.2, color=INK2, fontweight="bold")
ax1.set_yticks(range(len(CATS)))
ax1.set_yticklabels(CATS, fontsize=10.6, color=INK2, fontweight="bold")
ax1.set_ylabel("1위 집계 방식", fontsize=11, color=INK2, fontweight="bold", labelpad=6)
ax1.tick_params(axis="both", length=0)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color("#999999"); ax1.spines[s].set_linewidth(1.0)
ax1.grid(axis="y", color=GRID, lw=0.7, zorder=0)
ax1.set_axisbelow(True)

# ===================== 우측 차트 — 단계별 로그 행 수 =====================
txt(0.560, 0.782, "단계별 설정비교 로그 414행", size=15, weight="bold", color=INK)

ax2 = fig.add_axes([0.620, 0.500, 0.332, 0.252])
labels = ["02\nRF재결정 · 집계탐색", "03\n그룹누수 · 하위집단", "04\nwmean / max 결정"]
vals = [118, 168, 128]
bars = ax2.bar(range(3), vals, color=BARC, width=0.56, zorder=3)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}행", fontsize=13.5, weight="bold",
             color=INK, ha="center", va="bottom")
ax2.set_xticks(range(3))
ax2.set_xticklabels(labels, fontsize=9.8, color=INK2, fontweight="bold")
ax2.set_ylim(0, 200)
ax2.tick_params(axis="y", labelsize=10, colors=INK2)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax2.spines[s].set_color("#999999"); ax2.spines[s].set_linewidth(1.0)
ax2.grid(axis="y", color=GRID, lw=0.7, zorder=0)
ax2.set_axisbelow(True)

# ===================== 하단 좌 =====================
txt(0.035, 0.420, "NN 초기 트랙 주요 결과", size=15, weight="bold", color=INK)

items = [
    ("run_baseline / run_replicate_A", "단일 분할 1회로는 우열 판정 불가 → 반복 group 5-fold 채택"),
    ("run_esdiag", "val loss가 epoch 2부터 단조 증가, 조기종료가 과도하게 작동"),
    ("run_b4", "feature 표현 12종 × 10회 반복 비교 → nz20(444차원) 채택"),
    ("run_b5", "threshold 0.05–0.95 전수 탐색, nested 적용 시 고정 0.5보다 저하"),
    ("run_b6", "아키텍처 간 MCC 차이 0.011–0.029, 반복 변동폭 ±0.03–0.05 이내"),
    ("run_final", "차원축소 후보 20회 짝지은 비교로 b4 결론 재확인"),
]
iy = 0.379
for i, (k, v) in enumerate(items):
    y = iy - i * 0.0455
    AX.add_patch(Rectangle((0.035, y - 0.0245), 0.0042, 0.035, fc=NAVY, ec="none", zorder=3))
    txt(0.051, y, k, size=11.8, weight="bold", color=NAVY, va="center")
    txt(0.051, y - 0.0205, v, size=10.8, color=INK2, va="center")

# ===================== 하단 우 =====================
txt(0.560, 0.420, "최종 production 확정: 증강 2차와 버그수정 4건", size=15,
    weight="bold", color=INK)

TX, TY, TW = 0.560, 0.205, 0.405
rows_t = [("MCC", "0.234 ± 0.020", "0.264 ± 0.020"),
          ("Sensitivity", "0.648", "0.668"),
          ("Specificity", "0.650", "0.667"),
          ("ROC-AUC", "0.702", "0.718")]
hh = 0.0380
col = [TX, TX + 0.132, TX + 0.272]
colw = [0.132, 0.140, 0.133]
AX.add_patch(Rectangle((TX, TY + len(rows_t) * hh), TW, hh, fc=NAVY, ec="none", zorder=3))
for c, cwd, hd in zip(col, colw, ["지표", "baseline", "production"]):
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

note = ("· 데이터 증강 541 → 659건, 양성 30 → 36건\n"
        "· group 5-fold 20회 반복 pooled 결과, 임계값은 검증 폴드에서만 선정")
txt(TX, 0.173, note, size=10.8, color=INK2, va="top", linespacing=1.95)

# ===================== 하단: 핵심 수치 =====================
kpis = [("12", "개", "설정비교 CSV"),
        ("414", "행", "설정비교 로그"),
        ("20", "회", "반복 group 5-fold"),
        ("637", "개", "구조 그룹 (comp_group_id)")]
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
