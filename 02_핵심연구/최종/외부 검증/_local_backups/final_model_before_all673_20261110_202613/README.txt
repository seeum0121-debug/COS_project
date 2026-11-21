상태: 원본 그림 재현 미완료 — 사용자가 요구한 동일 성능을 충족하지 못함
=====================================================================
현재 eye_transfer_final.joblib는 앞서 만든 재구성본을 보존한 파일입니다.
원본 그림의 13/16, MCC 약 0.545, AUC 약 0.764를 재현한 최종본으로 사용하지 마세요.
software verification.json의 passed는 실행·수치 일관성 검사에만 해당합니다.
원본 성능과의 별도 대조: python check_reference_reproduction.py
상세 차이: reference_reproduction_status.json
완전한 재현을 위해 원래 eye_partial_transfer 폴더와
run_gettings_eye_partial_transfer.py가 필요합니다.

Eye 기존 계수 기반 전이학습 — 새로 재구성한 모델 (미채택)
====================================================

사용할 모델: eye_transfer_final.joblib
예측 코드: predict.py / common.py

1. 무엇을 만든 것인가

제공된 전이학습.pdf와 eye 전이학습 시각화.png의 방법을 기준으로,
기존 Eye A 모델의 전처리기를 고정하고 분류 계수와 절편을 미세 조정한 모델입니다.
Gettings 1996 제형 16개 전체로 최종 재학습한 모델 하나를 저장했습니다.
평가에 사용한 그룹별 모델들은 fold_models/에 별도로 보관했습니다.

원본 전이학습 모델을 복구한 파일은 아닙니다.
원본 mapping.json, 실행 스크립트, fold 계수 파일은 확보되지 않았습니다.
따라서 13개 성분 매핑표와 손실의 평균/합계 방식, 최적화 설정은 이번 재구성의
명시적 선택입니다. 원래 13개 성분 목록과 같다고 확인된 것이 아닙니다.
mapping.json과 protocol.json은 첫 학습 전에 고정했으며, 그림 성적에 맞추기 위한
성분 선택·임계값 조정·하이퍼파라미터 탐색을 하지 않았습니다.

2. 데이터와 방법

- 출처: ICCVAM HET-CAM BRD Appendix B4의 Gettings 1996 배합표,
        Appendix D5의 같은 연구·제형코드에 해당하는 원액 in vivo EPA 등급.
- 25개 후보 중 모든 함량이 숫자이며 합계가 100±0.11%인 16개 사용.
- 양성 11개(EPA I/II), 음성 5개(EPA III/IV).
- D5 원문 PDF에서 25개 라벨을 다시 추출해 기존 정리표와 대조했습니다.
- 기존 Eye A: source_model/eye_A_LR_wmean.joblib
  SHA256: d71394e691ac5072e4ce900a53b47ddbb65c8685961fe2c9536411acfbbc58a2
  기존 임계값: 0.4814867474694138. 현재 GitHub의 all673 신버전과 다른 이전 모델입니다.
- Morgan 2,048개 + RDKit descriptor 217개, 매핑된 성분 함량 가중평균.
  fingerprint는 분자 전체, descriptor는 최대 조각으로 계산합니다.
- 기존 imputer/scaler를 고정하고 계수와 절편만 최적화합니다.
- 손실: 클래스 균형 가중 logistic loss의 평균
        + 0.5 * ||새 계수·절편 - 기존 계수·절편||². λ=1.
- 클래스 가중치는 매 학습 fold의 클래스 수로 다시 계산합니다.
- 판정 임계값은 0.5, 비교는 >=입니다.
- 동일한 부분 구조 특징을 묶은 15개 그룹에 대해 leave-one-group-out 평가.
- 평가를 마친 뒤 16행 전체로 최종 예측용 모델 하나를 다시 학습했습니다.

3. 실제 재구성 결과와 원래 그림의 차이

                         원래 PDF/그림       이번 재구성 OOF
행 수 / 그룹 수          16 / 15              16 / 15
맞힌 수                  13 / 16              14 / 16
정확도                   81.25%               87.50%
MCC                      약 0.545             0.7125253031944253
ROC-AUC                  약 0.764             0.7636363636363637
TN / FP / FN / TP        3 / 2 / 1 / 10        3 / 2 / 0 / 11

AUC가 같더라도 예측 판정과 원본 매핑이 달라 동일한 모델의 복원으로 볼 수 없습니다.
원래 그림의 수치를 새 최종 모델의 검증 성적으로 붙이지 않았습니다.
기존 계수 대신 0을 기준으로 같은 방식으로 학습한 대조군도 14/16, MCC 0.713입니다.
따라서 이 결과만으로 기존 계수의 추가 이점이 입증되었다고 말할 수 없습니다.
이번 고정 OOF 점수의 그룹 bootstrap 5,000회 중 양·음성이 모두 있는 4,988회에서
MCC 95% 백분위 구간은 [0, 1], AUC 구간은 [0.3333, 1]입니다.

OOF 성적은 그룹별로 해당 제형을 학습에서 제외한 예측의 성적입니다.
eye_transfer_final.joblib는 이후 16행 전체를 학습한 예측용 파일이므로 이 파일로
학습 제형을 다시 예측한 결과는 OOF와 다릅니다. refit_predictions_NOT_validation.csv는
재학습 동작 확인용이며 검증 성적이 아닙니다.

4. 설치와 예측

확인 환경: Windows, Python 3.14.6. 현재 환경에서 검증했으며 새 가상환경의
온라인 설치까지 시험한 것은 아닙니다. 다음 명령은 이 폴더에서 실행합니다.

python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python predict.py --input examples\CTFA1996_HZT.json
.venv\Scripts\python verify.py

이미 동일한 패키지가 설치된 Python에서는 다음과 같이 실행할 수 있습니다.
python predict.py --input examples\CTFA1996_HZT.json
python predict.py --input examples\CTFA1996_HZT.json --output 새_예측결과.json

입력 예시는 data/payloads.json에도 16개가 있습니다.
JSON의 components에 원료의 name과 amount_pct를 입력합니다.
mapping.json의 명시적 이름·별칭만 인식하며 대소문자와 연속 공백을 정리합니다.
예: "Methyl paraben"과 "Methylparaben"은 같은 후보 구조로 처리합니다.
"Tetrasodium EDTA (30%)" 같은 원액 상품은 괄호를 제거하거나 활성분을 추정하지 않습니다.
목록에 없는 성분은 특징에서 제외되고 반환되는 component_audit에 남습니다.
매핑된 양수 함량 성분이 하나도 없으면 예측을 거부합니다.
모든 함량은 0~100의 유한 숫자여야 하며 합계는 0 초과 100.11 이하입니다.

주요 출력:
- eye_transfer_score: 모델의 양성 클래스 점수
- eye_transfer_prediction: 0 또는 1
- decision_threshold: 0.5
- coverage: 매핑·누락·물 이외의 매핑 함량과 각 성분 처리 내역

joblib 파일은 사전학습된 sklearn pipeline, 특징 이름, 매핑표, 메타데이터를 담은 dict입니다.
pipeline.predict()와 저장 임계값은 모두 0.5입니다.
pipeline의 LogisticRegression 객체는 추론 컨테이너로 재사용했습니다. 해당 객체의
get_params()에 남은 C 등은 원본 설정이며 이번 전이학습의 목적함수를 나타내지 않습니다.
실제 전이학습 설정은 protocol.json, 최적화 결과는 metadata.json에 기록되어 있습니다.

5. 재학습과 검증 자료

추가 준비 의존성: requirements-rebuild.txt
python prepare_data.py
python train.py --output-dir ..\최종모델_재현출력

기존 최종 모델을 덮어쓰는 학습은 거부합니다. 새 출력 폴더를 지정하세요.
원료 구조 조회는 이미 data/sources/pubchem/에 캐시되어 있어 학습과 예측에 인터넷이
필요하지 않습니다. resolve_mapping.py는 최초 패키지 구성 기록이며 최종 모델이 있는
폴더에서는 실행을 거부합니다.

주요 파일:
- eye_transfer_final.joblib: 16행 전체로 학습한 최종 예측 모델
- metadata.json / protocol.json / mapping.json: 학습 조건과 재구성 가정
- metrics.csv / metrics.json / oof_predictions.csv: 그룹별 검증 결과
- fold_models/: 전이학습 15개 + 0 기준 대조군 15개
- fold_log.json: 각 fold의 학습·시험 제형 ID와 최적화 수렴 기록
- features.npz / feature_names.json: 학습 입력과 순서
- coverage_audit.json: 성분 매핑과 입력 누락 내역
- bootstrap_ci.json: 그룹 재표본 결과
- reconstructed_oof.png: 이번 재구성의 실제 검증 그림
- data/: 원자료, 추출 라벨, 후보 제외 이유, 제형 입력
- reference/: 사용자가 제공한 원래 보고서와 그림
- source_manifest.json / checksums.json: 출처 파일 및 패키지 해시
- verification.json: 실행·무결성·특징·OOF 분리·수치 재계산 검사 결과

패키지를 의도적으로 수정한 뒤에만 seal_package.py로 해시를 갱신합니다.
정상 사용 및 무결성 확인에는 verify.py만 실행합니다.

6. 해석 범위

연구용으로 재구성한 부분 조성 모델입니다. 16행의 작은 표본이고 기존 673행
학습자료와의 중복을 배제하지 못했으므로 독립 외부검증이 아닙니다.
물 이외에 구조로 표현한 함량은 제형별 0~6.15%이며 주요 계면활성제가 많이 누락됩니다.
원료 구조는 후보 INCI 이름에 대응하는 구조이고 공급 원료의 등급·배치 동일성은 미확정입니다.
점수는 보정된 실제 자극 확률이나 개인·제품의 안전성 판정이 아닙니다.

원문 출처:
https://ntp.niehs.nih.gov/sites/default/files/iccvam/docs/ocutox_docs/ocubrd/hetcam/hetcamappb.pdf
https://ntp.niehs.nih.gov/sites/default/files/iccvam/docs/ocutox_docs/ocubrd/hetcam/hetcamappd5.pdf
성분별 PubChem 원문 조회 주소와 반환 구조는 mapping.json과 data/sources/pubchem/에 있습니다.
