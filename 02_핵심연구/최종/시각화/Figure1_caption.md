# 그림 1 — 캡션과 근거

그림 파일: `Figure1_공개데이터_가용성.pdf` / `.png`(600 dpi) / `_300dpi.png`
원자료: `Figure1_data.csv`
보고서 수록본: `ㅎ.최종정리파일/공개데이터가용성.png`

---

## 캡션 (보고서에 그대로 넣는 글)

**그림 1. 화장품 자극성 모델링에 쓸 수 있는 공개 데이터.**
**(A)** 모델링에 필요한 조건을 하나씩 더해 갈 때 남는 공개 자료의 수(로그 축척). 성분 이름만 보면
15,000건이 넘게 공개돼 있다. 여기에 독성 정보 연결, in vivo 자극성 정답, 혼합물 포함을 차례로
요구하면 593건까지 줄어든다. 마지막으로 "화장품 제형이면서 전체 배합비가 공개돼 있을 것"을 더하면
해당하는 자료가 없다(빗금 막대, n = 0).
**(B)** 공개 자극성 데이터셋의 수록 규모와 단위(로그 축척). 대괄호는 엔드포인트를 뜻한다
([안] 안자극, [피] 피부자극, [공] 공통). 사각형은 단일 화학물질 단위, 원형은 제형을 포함한 자료다.
규모가 큰 자료는 모두 단일 화학물질 단위이고, 제형 단위 자료는 수백 건에 그친다. 음영 행은 본 연구에서
구축한 자료다.
**(C)** 화장품 자극성 모델링에 필요한 다섯 요건의 충족 여부(● 충족, ◐ 부분, ○ 미충족). 다섯을 모두
채우는 공개 자료는 없다. 본 연구 자료는 "화장품 대상"만 빼고 네 요건을 채운다.

### 영문 캡션 (필요할 때)

**Figure 1. Public data available for modelling cosmetic irritation.**
**(A)** Records remaining as modelling requirements are added one by one (log scale). Over 15,000
entries are public at the ingredient-name level. Requiring toxicity linkage, an in vivo irritation
endpoint and mixture coverage leaves 593. Adding "a cosmetic formulation with its full declared
composition" leaves none (hatched bar, n = 0).
**(B)** Size and recording unit of public irritation datasets (log scale). Brackets denote the
endpoint ([안] eye, [피] skin, [공] both). Squares are single-substance datasets, circles are
mixture-inclusive ones. Shaded rows were compiled in this study.
**(C)** Fulfilment of five requirements for cosmetic irritation modelling (● met, ◐ partial,
○ not met).

---

## 본문에 인용할 수 있는 문장

- 혼합물을 포함한 최신 공개 안독성 데이터베이스 OCUTOXDB는 593개 물질·810개 기록이고, 그중 혼합물은
  약 20%다. 이 모델의 ECHA 외부검증 balanced accuracy는 64–71%이며, 저자들은 작은 데이터 크기를
  주요 한계로 적었다. (PMC12973229, 2025)
- 같은 논문은 혼합물이 독성 모델링 연구에서 아예 제외되는 경우가 많다고 기술한다
  ("mixtures are often excluded from toxicity modeling studies altogether").
- NC3Rs CRACK IT의 QSARs Mix 과제는 QSAR 분야가 물질과 혼합물의 피부·안자극 예측에 아직 성공하지
  못했다는 전제에서 공모됐다.
- 피부자극 공개 참조 데이터셋은 ECETOC TR 066이 176물질, ChemSkin이 100물질로 안자극보다 더 작다.

---

## 수치 출처

| 패널 | 항목 | 값 | 출처 |
|---|---|---|---|
| A, B, C | CosIng 등재 화장품 성분명 | 15,000+ | European Commission, CosIng |
| A, B, C | COSMOS DB 수록 화합물 | 7,021 | US EPA CompTox Dashboard, chemical list `COSMOSDB` |
| A, B, C | REACH 등록자료 마이닝 Draize | 5,629 (자극 1,931 / 비자극 3,698) | Luechtefeld 등, PMC5461467 |
| B, C | STopTox 안자극 관측 자료 (Table S4) | 3,545 | Borba 등, *Environ Health Perspect* 130(2):027012, 2022, doi:10.1289/EHP9341 |
| A, B, C | STopTox 피부자극 균형 모델링셋 | 2,292 (자극 1,146 / 비자극 1,146) | 동 논문 (원문 Table 1 교차확인 권장) |
| B, C | Cosmetics Europe DRD | 634물질 / 681연구 | Barroso 등, *Arch Toxicol* 91(2):521–547, 2017 |
| A, B, C | OCUTOXDB | 593물질 / 810기록, 혼합물 약 20% | PMC12973229, 2025 |
| B, C | ECETOC TR 066 피부자극·부식 참조 데이터뱅크 | 176 | ECETOC Technical Report No. 66, 1995 |
| B, C | ChemSkin 참조 DB | 100 | PMC8625019 |
| B, C | CON4EI 평가셋 | 80 (액상 38 / 고상 42) | CON4EI 프로젝트 |
| B, C | ECETOC 안자극 표준셋 | 38 | OECD TG 405 준거 |
| B, C | 본 연구 — 안자극 제형 | 673 (평가 606 + 증강 67) | `production데이터/eye_vectorized.csv` |
| B, C | 본 연구 — 피부자극 제형 | 662 | `production데이터/skin_vectorized.csv` |

---

## 주의사항

1. 공개 보고값을 인용했다. 자료마다 발표 시점이 달라 현재 수치와 차이가 있을 수 있다.
2. COSMOS DB의 7,021은 EPA CompTox가 "부분 수록, 큐레이션 진행 중"으로 표기한 값이다. 문헌에 따라
   2,300여 물질, NO(A)EL/LO(A)EL 1,357물질, TTC 서브셋 552물질 같은 다른 단위의 수치가 함께 보고된다.
3. STopTox 피부자극 2,292는 논문 본문 기술에서 가져왔다. 게재 전에 원문 Table 1로 확인하는 편이 좋다.
4. Cosmetics Europe DRD는 논문이 open access지만 데이터베이스 자체의 자유 내려받기 여부를 확인하지
   못해, (C)에서 "공개 접근"을 부분(◐)으로 표시했다.
5. (A)의 각 단계는 서로 다른 출처를 조건별로 늘어놓은 것이지, 한 모집단에서 순차적으로 걸러낸 수치가
   아니다. 조건이 깊어질수록 쓸 수 있는 공개 자료가 줄어든다는 점을 보이기 위한 배열이다.
6. 본 연구의 673·662건은 화장품이 아니라 FIFRA(농약) 등록자료 기반 제형이다. (C)에서 "화장품 대상"을
   미충족(○)으로 표시한 이유가 이것이다.
