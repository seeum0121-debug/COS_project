# -*- coding: utf-8 -*-
"""COS_보고서.docx 의 부록 그림 1~6과 관련 설명문을 교체한다.

그림 1~4 : 논문 서식으로 다시 그린 PNG (make_보고서_부록그림.py)
그림 5   : ㅎ.최종정리파일/eye 전이학습 시각화.png
그림 6   : make_skin_transfer_figure.py 결과

이미지 가로폭(cx)은 그대로 두고 세로(cy)만 새 비율에 맞게 다시 계산하므로
Word 레이아웃이 밀리지 않는다. 원본은 같은 폴더에 .bak_<시각>.docx 로 남긴다.
"""
import datetime
import re
import shutil
import struct
import sys
import zipfile
from pathlib import Path

DOCX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Users\DS\Desktop\COS_보고서.docx")
OUTPUT = Path(sys.argv[2]) if len(sys.argv) > 2 else DOCX
HERE = Path(__file__).resolve().parent
APPENDIX = HERE/"부록"
REPORT = HERE.parent/"ㅎ.최종정리파일"

NEW_IMAGES = {
    "word/media/image1.png": APPENDIX/"부록그림1_초기신경망_단계별MCC.png",
    "word/media/image2.png": APPENDIX/"부록그림2_조기종료_진단.png",
    "word/media/image3.png": APPENDIX/"부록그림3_그룹수정_전후MCC.png",
    "word/media/image4.png": APPENDIX/"부록그림4_wmean_vs_max.png",
    "word/media/image5.png": APPENDIX/"부록그림5_eye_전이학습.png",
    "word/media/image6.png": APPENDIX/"부록그림6_skin_전이학습.png",
}
RID_FOR_MEDIA = {"word/media/image1.png": "rId9", "word/media/image2.png": "rId10",
                 "word/media/image3.png": "rId11", "word/media/image4.png": "rId12",
                 "word/media/image5.png": "rId13", "word/media/image6.png": "rId14"}

# 문단 번호 -> 새 본문 (번호는 document.xml 의 <w:p> 등장 순서)
NEW_TEXT = {
    216: "본문에서 인용한 그래프를 보고서 서식에 맞추어 다시 작성하였다. 수치와 비교 조건은 원 분석 "
         "결과를 그대로 사용하였고 글꼴·색·배치만 통일하였다. 초기 신경망 및 조성 그룹 민감도 그림은 "
         "최종 배포모델의 외부 검증 성능이 아니라 내부 모델 선택과 누수 점검을 위한 자료이며, "
         "wmean 대 max 비교도 동일 표본의 중첩 교차검증으로 해석한다.",
    218: "그림 1. 초기 신경망 트랙의 단계별 MCC 분포. 반복 group-aware 교차검증 10회의 회차별 값과 "
         "평균±표준편차를 단일 분할 1회 보고값과 함께 표시하였다. 단계 간 평균 차이(−0.067, +0.043)는 "
         "회차 간 표준편차(0.034–0.049)를 넘지 않아, 설정 변경의 효과와 반복에 따른 변동이 분리되지 않았다.",
    220: "그림 2. 초기 신경망 fold에서 관찰된 조기종료 진단. train loss의 급격한 감소와 validation loss의 "
         "상승이 함께 나타났으며, validation loss 최저점이 epoch 3에 형성되어 조기종료가 학습 초기 모델을 "
         "선택하였다. 과적합 및 검증 불안정성을 점검하는 근거로 사용하였다.",
    222: "그림 3. 조성 기반 comp_group_id 수정 전후의 MCC 비교. 수정 전 463그룹에서 수정 후 637그룹으로 "
         "바뀔 때 피부 B와 안 자극성 A의 집계방식별 변화를 탐색 단계에서 비교한 결과이며, 집계 방식에 따라 "
         "변화의 방향이 달랐다. 최종 생산모델의 공식 성능표가 아니다.",
    224: "그림 4. 동일 표본의 nested 비교에서 wmean과 max 집계방식의 MCC를 비교한 결과. 20회 반복 전부에서 "
         "max가 높았으나, 미량 성분 극값 민감도와 실제 제형을 대표하지 않는 가상 조성 문제를 고려해 최종 "
         "production 표현은 wmean+resolved_pct로 유지하였다.",
    227: "그림 5·6은 COS_project에 저장된 OOF 예측값에서 지표를 다시 계산하여 작성하였으며, 모델 재학습이나 "
         "평가 분할 변경은 수행하지 않았다. 각 그림은 최종 조건의 혼동행렬과 같은 예측값으로 계산한 ROC 곡선을 "
         "함께 제시하고, 비교 조건별 지표는 표 3·4에 수록하였다. 독립 외부검증이나 앱 구조 보조점수의 검증으로 "
         "해석하지 않는다.",
    229: "그림 5. 눈 표적 16제형·15개 부분특징 그룹의 교차검증 결과. 기반 계수 중심 적응의 OOF 혼동행렬은 "
         "TN/FP/FN/TP=3/2/0/11이며, 14/16 일치, MCC 0.713, ROC-AUC 0.764였다. ROC 곡선은 같은 OOF 예측 "
         "점수에서 계산하였고 판정 임계값은 0.5이다. 0 중심 대조군도 같은 분류 결과를 얻었으므로 기반 계수 "
         "활용의 추가 우월성은 확인되지 않았다. 출처: COS_project/transfer_learning/"
         "eye_source_centered_transfer/results/metrics.csv. 과거 13/16 그림을 현재 결과로 교체하였다.",
    231: "그림 6. 피부 Zein 자료 41행·11개 원료 조합 그룹의 교차검증 결과. +농도·HLB 재보정의 OOF 혼동행렬은 "
         "TN/FP/FN/TP=19/4/3/15이며, 34/41 일치, MCC 0.656, ROC-AUC 0.905였다. ROC 곡선은 같은 OOF 예측 "
         "점수에서 계산하였다. 정답의 양성 기준은 Zein Number≥200, 예측의 양성 기준은 점수≥0.5이다. 기반 "
         "피부 모델의 EPA I·II·III 라벨 및 배포 임계값 0.33과 구별한다. 다섯 조건의 지표는 표 3에 수록하였다. "
         "출처: COS_project/transfer_learning/skin_option_a_recalibration/results/metrics.csv.",
}

PARA_RE = re.compile(r"<w:p[ >].*?</w:p>|<w:p/>", re.S)
RUN_RE = re.compile(r"<w:r>.*?</w:r>|<w:r [^>]*>.*?</w:r>", re.S)
RPR_RE = re.compile(r"<w:rPr>.*?</w:rPr>", re.S)


def png_size(path):
    data = path.read_bytes()[:33]
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", data[16:24])


def escape(text):
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def rewrite_paragraph(para, text):
    """모든 run 을 지우고 첫 run 의 서식으로 된 단일 run 을 넣는다."""
    runs = RUN_RE.findall(para)
    assert runs, "run 없음"
    rprs = {m for m in (RPR_RE.search(r) for r in runs) if m}
    styles = {m.group(0) for m in rprs}
    assert len(styles) == 1, f"서식이 섞여 있음: {len(styles)}종"
    rpr = styles.pop()
    new_run = f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    first = para.index(runs[0])
    last = para.index(runs[-1]) + len(runs[-1])
    return para[:first] + new_run + para[last:]


def patch_extent(xml, rid, ratio):
    """해당 rId 를 쓰는 <w:drawing> 안의 wp:extent 와 a:ext 를 새 비율로 맞춘다."""
    anchor = xml.index(f'r:embed="{rid}"')
    start = xml.rindex("<w:drawing>", 0, anchor)
    end = xml.index("</w:drawing>", anchor) + len("</w:drawing>")
    block = xml[start:end]
    cx = int(re.search(r'<wp:extent cx="(\d+)"', block).group(1))
    cy = round(cx/ratio)
    patched, n1 = re.subn(r'<wp:extent cx="\d+" cy="\d+"/>',
                          f'<wp:extent cx="{cx}" cy="{cy}"/>', block)
    patched, n2 = re.subn(r'<a:ext cx="\d+" cy="\d+"/>',
                          f'<a:ext cx="{cx}" cy="{cy}"/>', patched)
    assert n1 == 1 and n2 == 1, (rid, n1, n2)
    assert "<a:srcRect" not in patched, f"{rid}: 잘라낸 그림"
    return xml[:start] + patched + xml[end:], cx, cy


def main():
    for path in NEW_IMAGES.values():
        if not path.exists():
            raise SystemExit(f"없는 파일: {path}")

    if OUTPUT == DOCX:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = DOCX.with_name(f"{DOCX.stem}.bak_{stamp}.docx")
        shutil.copy2(DOCX, backup)
        print("backup:", backup)

    with zipfile.ZipFile(DOCX) as zf:
        items = [(info, zf.read(info.filename)) for info in zf.infolist()]

    xml = next(data for info, data in items if info.filename == "word/document.xml").decode("utf-8")

    # 1) 본문·설명문은 건드리지 않는다. 그림과 그림 크기만 바꾼다.

    # 2) 이미지 비율에 맞춰 높이 조정
    for media, source in NEW_IMAGES.items():
        w, h = png_size(source)
        xml, cx, cy = patch_extent(xml, RID_FOR_MEDIA[media], w/h)
        print(f"size  {media}: {w}x{h} -> {cx/914400:.2f}in x {cy/914400:.2f}in")

    # 3) 다시 압축
    out = OUTPUT
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for info, data in items:
            if info.filename == "word/document.xml":
                data = xml.encode("utf-8")
            elif info.filename in NEW_IMAGES:
                data = NEW_IMAGES[info.filename].read_bytes()
            zf.writestr(info, data)
    print("saved:", out)


if __name__ == "__main__":
    sys.exit(main())
