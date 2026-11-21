# -*- coding: utf-8 -*-
"""보고서 부록 그림 5·6 (눈·피부 전이학습) 인쇄용 판형.

내용·수치는 ㅎ.최종정리파일의 전이학습 시각화와 같고, 보고서 본문 폭(약 5.75in)에
들어갔을 때 각주가 읽히도록 캔버스만 좁힌다. 지표는 각 모듈의 로더가 저장된
예측값에서 다시 계산하고 기록값과 대조한다.

사용법: python make_부록_전이그림.py [출력폴더]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_eye_transfer_reconstruction as eye
import make_skin_transfer_figure as skin

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "부록")
os.makedirs(OUT, exist_ok=True)

NAVY, INK, MUTED = eye.NAVY, eye.INK, eye.MUTED
W, H = 11.4, 5.95


def frame(title, subtitle):
    fig = plt.figure(figsize=(W, H), dpi=180)
    fig.text(.5, .966, title, ha="center", va="top", fontsize=15.5,
             weight="bold", color=NAVY)
    fig.text(.5, .912, subtitle, ha="center", va="top", fontsize=12, color=INK)
    return fig


def foot(fig, lines, top=.145, step=.050):
    for i, line in enumerate(lines):
        fig.text(.5, top - i*step, line, ha="center", va="center",
                 fontsize=8.8, color=MUTED)


def eye_figure(path):
    e = eye.load_evidence()
    m = e["metrics"]["source_centered_transfer"]
    fig = frame("Gettings 1996 — 673행 평가 Eye A 기반 전이학습 재구성",
                f"15그룹 교차검증(OOF) · 일치 {m['correct']}/{m['n']} "
                f"({m['accuracy']*100:.1f}%) · MCC {m['MCC']:.3f} · AUC {m['ROC_AUC']:.3f}")
    eye.draw_cm(fig.add_axes([.145, .300, .29, .485]), e)
    eye.draw_roc(fig.add_axes([.575, .300, .33, .485]), e)
    foot(fig, [
        "기반: 673행 전체 평가 Eye A(eye_A_final.joblib) · 새 13성분 매핑 · 기존 전처리 고정 · 앵커 L2(λ=1) · 판정 임계값 0.5",
        "그룹을 하나씩 제외해 얻은 예측 성적이며, 이후 16행 전체로 학습한 예측용 모델의 독립 시험 성적은 아님 · 기존 673행과의 중복 미확인",
        "근거: 외부 검증/최종 모델/oof_predictions.csv · metrics.json · protocol.json · 독립 외부검증 아님",
    ])
    fig.savefig(path)
    plt.close(fig)
    return m


def skin_figure(path):
    e = skin.load()
    fig = frame("Lechuga 2025 — frozen Skin Option A 기반 재보정 전이학습",
                f"11그룹 교차검증(OOF) · 일치 {e['correct']}/{e['n']} "
                f"({e['correct']/e['n']*100:.1f}%) · MCC {e['MCC']:.3f} · AUC {e['ROC_AUC']:.3f}")
    skin.draw_cm(fig.add_axes([.145, .300, .29, .485]), e)
    skin.draw_roc(fig.add_axes([.575, .300, .33, .485]), e)
    foot(fig, [
        "기반: frozen skin_rf의 Option A 출력 점수 고정 · 농도와 혼합물 HLB를 공변량으로 추가 · 앵커 L2(λ=1) · 판정 임계값 0.5",
        f"raw Option A의 MCC {e['raw_MCC']:.3f} · AUC {e['raw_AUC']:.3f}에서 상승 · "
        f"그룹 bootstrap 5,000회 MCC 95% 구간 [{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]",
        "정답 기준 Zein Number ≥ 200은 production이 학습한 EPA Classification과 다른 엔드포인트 · 독립 외부검증 아님",
    ])
    fig.savefig(path)
    plt.close(fig)
    return e


if __name__ == "__main__":
    for name, fn in [("부록그림5_eye_전이학습.png", eye_figure),
                     ("부록그림6_skin_전이학습.png", skin_figure)]:
        p = os.path.join(OUT, name)
        fn(p)
        print("saved", p)
