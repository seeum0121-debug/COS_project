# -*- coding: utf-8 -*-
"""Eye 실험 규모 한 장 요약 PNG (2026-11-10 집계).

사용법: python make_실험규모.py <출력.png>
MCC 분포 패널은 ../ 아래 저장된 결과 파일에서 714개 레코드를 직접 읽어 그립니다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm
import os, json, csv, random, statistics, sys

for cand in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "Apple SD Gothic Neo"]:
    try:
        fm.findfont(fm.FontProperties(family=cand), fallback_to_default=False)
        plt.rcParams["font.family"] = [cand, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

SURF0 = "#f4f3f0"; SURF1 = "#fcfcfb"; SURF2 = "#eceae5"
BORDER = "#dedbd4"; BORDER_S = "#c3c0b7"
INK = "#0b0b0b"; INK2 = "#52514e"; INK3 = "#7a7872"
S1 = "#2a78d6"; S2 = "#eb6834"; S3 = "#1baf7a"; S4 = "#eda100"; S7 = "#6250d6"

W, H = 16.0, 11.0
ASPECT = W / H
fig = plt.figure(figsize=(W, H), dpi=135, facecolor=SURF0)
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis("off")


def card(x, y, w, h, fc=SURF1, ec=BORDER, lw=1.0, r=0.012):
    AX.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=1))


def txt(x, y, s, size=10, color=INK, weight="normal", ha="left", va="baseline", **kw):
    return AX.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=3, **kw)


M = 0.035
RIGHT = 1 - M

# ===================== 헤더 =====================
txt(M, 0.960, "Eye 엔드포인트 — 실험 전체 규모", size=27, weight="bold")
txt(RIGHT, 0.960, "집계일 2026-11-10", size=10.5, color=INK3, ha="right")
AX.plot([M, RIGHT], [0.940, 0.940], color=BORDER_S, lw=1.8, zorder=2)

# ===================== KPI =====================
kpis = [("20", "개", "실험 트랙", True),
        ("20,438", "회", "하이퍼파라미터 후보 적합", True),
        ("1,020", "회", "nested CV outer 모델 학습", True),
        ("678", "건", "저장된 성능 레코드", False),
        ("7", "종", "모델 계열", False),
        ("3,261", "행", "외부 보조학습 자료", False)]
gap = 0.010
kw_ = (RIGHT - M - gap * (len(kpis) - 1)) / len(kpis)
ky, kh = 0.832, 0.090
for i, (v, unit, k, lead) in enumerate(kpis):
    x = M + i * (kw_ + gap)
    card(x, ky, kw_, kh, fc=SURF2 if lead else SURF1, ec=BORDER_S if lead else BORDER)
    t = txt(x + 0.013, ky + 0.042, v, size=26, weight="bold", color=S1 if lead else INK)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(AX.transData.inverted())
    txt(bb.x1 + 0.004, ky + 0.046, unit, size=12, color=INK2)
    txt(x + 0.013, ky + 0.018, k, size=10.4, color=INK2)

# ===================== 좌: 트랙별 후보 적합 =====================
CW = (RIGHT - M - 0.022) / 2
lx, ly, lh = M, 0.578, 0.235
card(lx, ly, CW, lh)
txt(lx + 0.016, ly + lh - 0.032, "실행 규모 — 트랙별 후보 적합 횟수", size=13.5, weight="bold")

fits = [("5부 농도 production", 7200),
        ("8부 집계 방식 비교", 3600),
        ("EPA+GHS 라벨조합", 2400),
        ("EPA 단독 production", 1838),
        ("6부 STopTox 외부 보조", 1800),
        ("9부 최종 모델 확정", 1800),
        ("10부 추가 67행 통제비교", 1800)]
maxc = max(c for _, c in fits)
bx0 = lx + 0.168
bw = CW - 0.168 - 0.072
rowh = 0.0215
top = ly + lh - 0.072
for i, (nm, c) in enumerate(fits):
    y = top - i * rowh
    txt(bx0 - 0.010, y, nm, size=10.4, color=INK2, ha="right", va="center")
    AX.add_patch(FancyBboxPatch((bx0, y - 0.0068), bw * c / maxc, 0.0136,
                                boxstyle="round,pad=0,rounding_size=0.0035",
                                fc=S1, ec="none", zorder=3))
    txt(lx + CW - 0.016, y, f"{c:,}", size=11, weight="bold", ha="right", va="center")

# ===================== 우: 탐색한 공간 =====================
rx = M + CW + 0.022
card(rx, ly, CW, lh)
txt(rx + 0.016, ly + lh - 0.032, "탐색한 공간 — 바꿔 본 축", size=13.5, weight="bold")

space = [("라벨 정의", 4, "A(≤2) · B(≤3) · GHS 2A · GHS 2B"),
         ("모델 계열", 7, "LR · RF · ET · SVM · XGB · LGBM · CatBoost"),
         ("분할 seed", 3, "42 · 202 · 777"),
         ("표현 · 피처안", 12, "Morgan · MACCS · 기술자 · 농도 결합"),
         ("후보 격자", 20, "모델당 하이퍼파라미터 후보")]
sw, sgap = 0.0092, 0.0028
sh = sw * ASPECT
sx0 = rx + 0.110
srowh = 0.0335
stop = ly + lh - 0.072
for i, (nm, k, desc) in enumerate(space):
    y = stop - i * srowh
    txt(rx + 0.016, y, nm, size=11.2, weight="bold", color=INK, va="center")
    for j in range(k):
        AX.add_patch(FancyBboxPatch((sx0 + j * (sw + sgap), y - sh / 2), sw, sh,
                                    boxstyle="round,pad=0,rounding_size=0.002",
                                    fc=S1, ec="none", zorder=3))
    txt(rx + CW - 0.016, y, str(k), size=15, weight="bold", color=S1, ha="right", va="center")
    txt(rx + 0.016, y - 0.0150, desc, size=9.0, color=INK3, va="center")

# ===================== 714개 레코드 MCC 분포 =====================
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _jl(rel):
    d = json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))
    return d if isinstance(d, list) else d.get("payload", [])


def _cv(rel):
    return list(csv.DictReader(open(os.path.join(ROOT, rel), encoding="utf-8-sig")))


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


R = "모델 실험/eye/claude/results/"
G = {"1부  통합 집계": [], "2부  진단 · 모델 보강": [], "3·4부  2차 탐색 · 입력 수정": [],
     "5·6부  production · 외부 보조": [], "8~10부  최종 확정": []}
for f, g in [("02_ensemble_saved_oof", 1), ("03_grouping_fix", 1), ("04_ordinal", 1),
             ("05_worst_actor", 1), ("06_representations", 1),
             ("07_combined_representation", 2), ("08_calibration", 2),
             ("10_cross_ensemble", 2), ("11_concentration", 2)]:
    key = list(G)[g]
    for r in _jl(R + f + ".json"):
        if r.get("MCC") is not None:
            G[key].append(float(r["MCC"]))
for r in _jl(R + "concentration_production/f91f3c88787bec6b/metrics.json"):
    if r.get("MCC") is not None:
        G[list(G)[3]].append(float(r["MCC"]))
for r in _cv("외부학습/Eye_STopTox_20261108/metrics_by_seed.csv"):
    v = _f(r.get("MCC"))
    if v is not None:
        G[list(G)[3]].append(v)
for p in ["label 조합 변경/EPA/runs/epa-production-20261106-no-catboost/comparison_all_metrics.csv",
          "label 조합 변경/EPA + GHS/runs/epa-ghs-eye-719c1efdda0b5a59/augmented_metrics.csv"]:
    for r in _cv(p):
        v = _f(r.get("MCC"))
        if v is not None:
            G[list(G)[0]].append(v)
for p in ["모델 실험/eye/모델 최종/aggregation_compare_20261109/metrics_summary.csv",
          "최최종모델/metrics_summary.csv", "통제비교_606고정_추가67효과/metrics_summary.csv"]:
    for r in _cv(p):
        v = _f(r.get("MCC_mean") or r.get("MCC"))
        if v is not None:
            G[list(G)[4]].append(v)

NTOT = sum(len(v) for v in G.values())
ALLV = [x for v in G.values() for x in v]

my, mh = 0.292, 0.258
card(M, my, RIGHT - M, mh)
txt(M + 0.016, my + mh - 0.032, "678개 성능 레코드가 어디에 모였나 — 측정값 전부와 데이터 상한",
    size=13.5, weight="bold")
txt(M + 0.016, my + mh - 0.054,
    f"점 하나가 저장된 측정 1건입니다. {NTOT}개 전부 MCC {max(ALLV):.3f} 이하에 모였고, "
    f"이 데이터의 이론 상한은 0.861(A) · 0.795(B)입니다.",
    size=9.4, color=INK2)

XMIN, XMAX = -0.02, 0.92
px0, px1 = M + 0.190, RIGHT - 0.022


def px(v):
    return px0 + (v - XMIN) / (XMAX - XMIN) * (px1 - px0)


ytop = my + mh - 0.098
drowh = 0.0300
rng = random.Random(7)

# 눈금
ax_y = ytop - 4 * drowh - 0.022
for t in [0.0, 0.2, 0.4, 0.6, 0.8]:
    AX.plot([px(t), px(t)], [ax_y, ytop + 0.015], color="#e4e1da", lw=0.9, zorder=1)
    txt(px(t), ax_y - 0.011, f"{t:.1f}", size=9.0, color=INK3, ha="center", va="center")
AX.plot([px(XMIN), px(XMAX)], [ax_y, ax_y], color=BORDER_S, lw=0.9, zorder=2)
txt(px(XMAX) + 0.008, ax_y - 0.011, "MCC", size=9.2, color=INK2, ha="left", va="center")

# 이론 상한
AX.plot([px(0.861), px(0.861)], [ax_y, ytop + 0.015], color=S2, lw=1.3, ls="--", zorder=4)
AX.plot([px(0.795), px(0.795)], [ax_y, ytop + 0.015], color=INK3, lw=1.1, ls="--", zorder=4)
txt(px(0.861) + 0.006, ytop + 0.019, "라벨 A 상한 0.861", size=9.2, color=S2,
    weight="bold", ha="left", va="center")
txt(px(0.795) - 0.006, ytop + 0.019, "라벨 B 상한 0.795", size=9.2, color=INK3,
    ha="right", va="center")

for i, (nm, vals) in enumerate(G.items()):
    y = ytop - i * drowh
    if i % 2 == 0:
        AX.add_patch(Rectangle((M + 0.010, y - drowh * 0.46), RIGHT - M - 0.020, drowh * 0.92,
                               fc=SURF2, ec="none", zorder=1))
    med = statistics.median(vals)
    txt(px0 - 0.014, y + 0.0062, nm, size=10.2, color=INK, ha="right", va="center")
    txt(px0 - 0.014, y - 0.0085, f"n = {len(vals)}  ·  중앙값 {med:.2f}", size=8.6,
        color=INK3, ha="right", va="center")
    xs = [px(v) for v in vals]
    ys = [y + rng.uniform(-0.0082, 0.0082) for _ in vals]
    AX.plot(xs, ys, "o", ms=3.3, color=S1, alpha=0.42, mec="none", zorder=5)
    AX.plot([px(med), px(med)], [y - 0.0105, y + 0.0105], color="#0d3f78", lw=2.2, zorder=6)

# ===================== 실험 흐름 (2행 × 5열) =====================
txt(M, 0.272, "실험이 이어진 순서", size=13.5, weight="bold")
flow = [("1부", "통합 집계", "EPA · EPA+GHS를 한 규약으로", S1),
        ("2부 · 00–06", "진단과 모델 보강", "673행 중 331행 중복 확인", S2),
        ("3부 · 07–10", "2차 탐색", "표현 · 보정 · 앙상블 모두 효과 없음", S2),
        ("4부 · 11", "입력을 고치다", "단일성분 농도 약분, 221행 → 18행", S3),
        ("5부", "production 검증", "outer 360회 전량 재실행", S4),
        ("6부", "외부 보조학습", "STopTox 3,261행으로 보조모델 학습", S7),
        ("7부", "종합", "독립 외부검증 적격 0행", S7),
        ("8부", "집계 방식 비교", "가중평균 vs 특징별 최댓값", S4),
        ("9부", "최종 모델 확정", "lr / wmean · MCC 0.294", S1),
        ("10부", "추가 67행 통제비교", "ΔMCC 부호 일관되지 않음", S3)]
NCOL = 5
fgap = 0.009
fw = (RIGHT - M - fgap * (NCOL - 1)) / NCOL
fh = 0.060
for i, (num, ttl, body, col) in enumerate(flow):
    r, c = divmod(i, NCOL)
    x = M + c * (fw + fgap)
    y = 0.193 - r * (fh + 0.008)
    card(x, y, fw, fh)
    AX.add_patch(Rectangle((x, y + 0.004), 0.0038, fh - 0.008, fc=col, ec="none", zorder=3))
    txt(x + 0.013, y + fh - 0.015, num, size=8.6, color=INK3)
    txt(x + 0.013, y + fh - 0.034, ttl, size=11.2, weight="bold")
    txt(x + 0.013, y + fh - 0.050, body, size=9.0, color=INK2)

# ===================== 푸터 =====================
AX.plot([M, RIGHT], [0.085, 0.085], color=BORDER, lw=1.0, zorder=2)
foot = ("eye 엔드포인트만 집계했습니다. skin 결과와 중단·대체된 실행은 제외했습니다.\n"
        "수행한 실험의 양과 범위를 보여주는 자료이며, 성능 우열이나 모델 채택 결론은 담고 있지 않습니다.")
txt(M, 0.067, foot, size=9.4, color=INK3, va="top", linespacing=1.6)

out = sys.argv[1]
fig.savefig(out, facecolor=SURF0)
print("saved", out, "| records:", NTOT)
