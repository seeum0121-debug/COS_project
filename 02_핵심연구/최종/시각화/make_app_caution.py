# -*- coding: utf-8 -*-
"""COS Lab 주의 규칙 엔진 — 조건·약물 × 성분 태그 매칭 구조.

사용법: python make_app_caution.py <출력.png>
근거: github.com/tjdbf061106-sketch/COS_project — app/caution/
      {conditions,medications,ingredient_tags,rules,citations}.json 직접 집계
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

NAVY = "#1f4e79"; RED = "#c00000"; ORANGE = "#d4600f"; GREEN = "#2e7d32"; PURPLE = "#7030a0"
INK = "#111111"; INK2 = "#333333"; INK3 = "#5a5a5a"
SKIN_C = "#1f6cb0"      # 피부 축
EYE_C = "#2e7d32"       # 눈 축
TIER = {"strong": "#c00000", "moderate": "#d4600f", "mixed": "#7030a0", "advisory": "#8a8a8a"}
TIER_KO = {"strong": "강함", "moderate": "보통", "mixed": "혼재", "advisory": "참고"}

W, H = 16.0, 11.0
fig = plt.figure(figsize=(W, H), dpi=135, facecolor="white")
AX = fig.add_axes([0, 0, 1, 1]); AX.set_xlim(0, 1); AX.set_ylim(0, 1); AX.axis("off")


def txt(x, y, s, size=10, color=INK, weight="normal", ha="left", va="baseline", **kw):
    return AX.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, zorder=3, **kw)


# ===================== 데이터 =====================
TAGS = [
    ("fragrance_named_allergen",     "향료 알레르겐 (EU 지정)",     "strong"),
    ("fragrance_generic_label",      "향료 (성분 비공개)",          "advisory"),
    ("isothiazolinone_preservative", "이소티아졸리논 보존제",        "strong"),
    ("anionic_surfactant_irritant",  "음이온 계면활성제 (SLS 등)",  "mixed"),
    ("benzalkonium_chloride",        "벤잘코늄염화물 (BAK)",        "strong"),
    ("alcohol_high_conc",            "에탄올 고농도 (≥15%)",        "moderate"),
    ("comedogenic_high",             "고위험 코메도제닉",            "moderate"),
    ("topical_retinoid_cosmetic",    "레티노이드 화장품 성분",       "moderate"),
    ("aha_exfoliant",                "AHA 각질제거",                "strong"),
    ("bha_salicylic_acid",           "BHA (살리실산)",              "mixed"),
    ("benzoyl_peroxide",             "벤조일퍼옥사이드",             "strong"),
    ("photosensitizing_furocoumarin","광독성 푸로쿠마린",            "strong"),
]
COLS = [
    ("acne_prone",                      "여드름성·지성",      "C", "skin"),
    ("atopic_dermatitis",               "아토피피부염",        "C", "skin"),
    ("contact_dermatitis_history",      "접촉피부염 이력",     "C", "skin"),
    ("dry_eye_disease",                 "안구건조증",          "C", "eye"),
    ("glaucoma_chronic_drops",          "녹내장 장기점안",     "C", "eye"),
    ("allergic_conjunctivitis_cl_wearer","결막염·렌즈",        "C", "eye"),
    ("isotretinoin_oral",               "경구 이소트레티노인",  "M", "skin"),
    ("topical_retinoid_rx",             "처방 레티노이드",      "M", "skin"),
    ("benzoyl_peroxide_rx",             "BPO 제제",            "M", "skin"),
    ("oral_tetracycline_antibiotic",    "테트라사이클린",       "M", "skin"),
    ("tci_topical",                     "칼시뉴린억제제",       "M", "skin"),
    ("prostaglandin_analog_drops",      "PG 점안액",           "M", "eye"),
    ("soft_contact_lens_wear",          "소프트렌즈",          "M", "eye"),
]
RULES = {
    "acne_prone": ["comedogenic_high"],
    "atopic_dermatitis": ["fragrance_named_allergen", "fragrance_generic_label",
                          "isothiazolinone_preservative", "anionic_surfactant_irritant",
                          "alcohol_high_conc"],
    "contact_dermatitis_history": ["fragrance_named_allergen", "fragrance_generic_label",
                                   "isothiazolinone_preservative", "anionic_surfactant_irritant"],
    "dry_eye_disease": ["benzalkonium_chloride"],
    "glaucoma_chronic_drops": ["benzalkonium_chloride"],
    "allergic_conjunctivitis_cl_wearer": ["benzalkonium_chloride", "fragrance_named_allergen"],
    "isotretinoin_oral": ["topical_retinoid_cosmetic", "aha_exfoliant", "bha_salicylic_acid",
                          "alcohol_high_conc", "photosensitizing_furocoumarin"],
    "topical_retinoid_rx": ["topical_retinoid_cosmetic", "benzoyl_peroxide", "aha_exfoliant",
                            "photosensitizing_furocoumarin"],
    "benzoyl_peroxide_rx": ["topical_retinoid_cosmetic"],
    "oral_tetracycline_antibiotic": ["photosensitizing_furocoumarin", "aha_exfoliant"],
    "tci_topical": ["alcohol_high_conc", "fragrance_named_allergen", "anionic_surfactant_irritant"],
    "prostaglandin_analog_drops": ["benzalkonium_chloride"],
    "soft_contact_lens_wear": ["benzalkonium_chloride"],
}
NRULE = sum(len(v) for v in RULES.values())

# ===================== 제목 =====================
txt(0.5, 0.957, "모델에서 끝나지 않았다 — COS Lab 주의 규칙 엔진",
    size=25, weight="bold", color=NAVY, ha="center", va="center")
txt(0.5, 0.925,
    "자극성 점수와 별개로, 사용자가 고른 피부·눈 조건과 병용 약물에 맞춰 문헌 근거가 있는 성분만 골라 알린다.",
    size=12.5, color=INK2, ha="center", va="center")

# ===================== 핵심 수치 =====================
kpis = [("6", "종", "적용 조건", "피부 3 · 눈 3"),
        ("7", "종", "병용 약물", "피부 5 · 눈 2"),
        ("12", "종", "성분 태그", f"별칭 160개"),
        (str(NRULE), "개", "매칭 규칙", "조건 14 · 약물 17"),
        ("12", "건", "인용 문헌", "항목 37개")]
KY, KH = 0.818, 0.082
KG = 0.011
KW = (0.93 - KG * (len(kpis) - 1)) / len(kpis)
for i, (v, unit, k, sub) in enumerate(kpis):
    x = 0.035 + i * (KW + KG)
    AX.add_patch(FancyBboxPatch((x, KY), KW, KH, boxstyle="round,pad=0,rounding_size=0.012",
                                fc="#eef3f9", ec=NAVY, lw=1.6, zorder=2))
    t = txt(x + 0.016, KY + 0.040, v, size=28, weight="bold", color=NAVY)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(AX.transData.inverted())
    txt(bb.x1 + 0.005, KY + 0.044, unit, size=12.5, weight="bold", color=INK2)
    txt(x + 0.016, KY + 0.021, k, size=12, weight="bold", color=INK2)
    txt(x + 0.016, KY + 0.007, sub, size=9.6, color=INK3)

# ===================== 매트릭스 =====================
txt(0.035, 0.770, "어떤 조건·약물에서 어떤 성분을 짚는가", size=15, weight="bold", color=INK)

MX0 = 0.272
MW = 0.965 - MX0
CW = MW / len(COLS)
RTOP = 0.598
RH = 0.0305

# 열 머리글 (세로)
for j, (cid, lab, kind, dom) in enumerate(COLS):
    cx = MX0 + CW * (j + 0.5)
    c = SKIN_C if dom == "skin" else EYE_C
    txt(cx, RTOP + 0.016, lab, size=9.8, color=c, weight="bold",
        ha="center", va="bottom", rotation=90)

# 열 그룹 띠
def band(j0, j1, label, color):
    x0 = MX0 + CW * j0 + 0.002
    x1 = MX0 + CW * j1 - 0.002
    AX.add_patch(Rectangle((x0, 0.742), x1 - x0, 0.0115, fc=color, ec="none", zorder=3))
    txt((x0 + x1) / 2, 0.761, label, size=10.6, weight="bold", color=color,
        ha="center", va="center")

band(0, 3, "피부 조건 3", SKIN_C)
band(3, 6, "눈 조건 3", EYE_C)
band(6, 11, "피부 병용약물 5", SKIN_C)
band(11, 13, "눈 2", EYE_C)

# 행 + 셀
for i, (tid, tlab, tier) in enumerate(TAGS):
    y = RTOP - i * RH
    if i % 2 == 0:
        AX.add_patch(Rectangle((0.035, y - RH * 0.46), 0.965 - 0.035, RH * 0.92,
                               fc="#f4f4f2", ec="none", zorder=1))
    AX.add_patch(Rectangle((0.038, y - 0.0052), 0.0095, 0.0104,
                           fc=TIER[tier], ec="none", zorder=4))
    txt(0.056, y, tlab, size=10.6, color=INK, va="center")
    for j, (cid, lab, kind, dom) in enumerate(COLS):
        cx = MX0 + CW * j + CW * 0.5
        hit = tid in RULES[cid]
        if hit:
            c = SKIN_C if dom == "skin" else EYE_C
            AX.add_patch(Rectangle((cx - CW * 0.33, y - RH * 0.33),
                                   CW * 0.66, RH * 0.66, fc=c, ec="white", lw=1.0, zorder=5))
        else:
            AX.plot(cx, y, marker="o", ms=2.2, color="#d0d0cc", zorder=4)
    AX.plot([MX0, 0.965], [y - RH * 0.5, y - RH * 0.5], color="#e6e6e2", lw=0.6, zorder=2)

# 세로 구분선
for j in (3, 6, 11):
    xx = MX0 + CW * j
    AX.plot([xx, xx], [RTOP - 11 * RH - RH * 0.5, RTOP + RH * 0.5],
            color="#bfbfbb", lw=1.0, zorder=6)

# ===================== 범례 =====================
LY = 0.207
txt(0.035, LY, "문헌 근거 등급", size=11, weight="bold", color=INK2, va="center")
lx = 0.125
for t in ["strong", "moderate", "mixed", "advisory"]:
    n = sum(1 for _, _, tt in TAGS if tt == t)
    AX.add_patch(Rectangle((lx, LY - 0.0052), 0.0095, 0.0104, fc=TIER[t], ec="none", zorder=4))
    txt(lx + 0.015, LY, f"{TIER_KO[t]} {n}", size=10.6, color=INK2, va="center", weight="bold")
    lx += 0.080
AX.add_patch(Rectangle((lx + 0.010, LY - 0.0062), 0.014, 0.0124, fc=SKIN_C, ec="none", zorder=4))
txt(lx + 0.030, LY, "피부 축", size=10.6, color=INK2, va="center", weight="bold")
AX.add_patch(Rectangle((lx + 0.090, LY - 0.0062), 0.014, 0.0124, fc=EYE_C, ec="none", zorder=4))
txt(lx + 0.110, LY, "눈 축", size=10.6, color=INK2, va="center", weight="bold")
txt(lx + 0.175, LY, f"채운 칸 = 규칙 있음 ({NRULE}개 / 156칸)", size=10.6,
    color=INK3, va="center")

# ===================== 하단: 설계 원칙 =====================
txt(0.035, 0.168, "설계에서 지킨 것", size=14, weight="bold", color=INK)
notes = [
    ("규칙은 성분 전체가 아니라 근거가 있는 12개 태그에만 건다",
     "156칸 중 31칸만 채워져 있다. 모든 성분에 경고를 띄우지 않는다."),
    ("자극성 점수와 분리해 보여준다",
     "모델 점수 옆이 아니라 별도 영역에 표시하고, 사전에 없는 성분은 '미확인'으로 알린다."),
    ("모든 규칙에 출처를 붙인다",
     "EU 1223/2009 Annex III, FDA AHA 지침, AAD, Uter 2020, Nagstrup 2023 등 12건 · 항목 37개."),
]
ny = 0.136
for i, (k, v) in enumerate(notes):
    y = ny - i * 0.0385
    AX.add_patch(Rectangle((0.035, y - 0.0215), 0.0042, 0.032, fc=NAVY, ec="none", zorder=3))
    txt(0.051, y, k, size=11.6, weight="bold", color=NAVY, va="center")
    txt(0.051, y - 0.0185, v, size=10.6, color=INK2, va="center")

AX.plot([0.035, 0.965], [0.028, 0.028], color="#d9d9d9", lw=1.0, zorder=2)
txt(0.035, 0.012,
    "근거: COS_project 저장소 app/caution/ 의 conditions · medications · ingredient_tags · rules · citations.json 직접 집계. "
    "매칭 실패는 안전성 판정이 아니다.",
    size=9.6, color=INK3)

out = sys.argv[1]
fig.savefig(out, facecolor="white")
print("saved", out, "| rules:", NRULE)
