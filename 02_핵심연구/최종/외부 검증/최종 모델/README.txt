Eye 기존계수 기반 전이학습 — 673행 전체 평가 모델 기반
====================================================

기반 모델: source_model/eye_A_final.joblib
673행 전체를 그룹 교차검증으로 평가한 뒤 전체 673행으로 재학습한 Eye A 모델입니다.
SHA-256: 5d5c0532cd12fd00927c9564ee28dc9c95a8bb396a049c705c77dcafaf5b28e5

현재 예측용 모델: eye_transfer_final.joblib
Gettings 1996의 16제형 전체로 최종 적합한 전이학습 모델입니다.
평가용 30개 fold 모델은 fold_models/에 별도로 보관합니다.

실제 15그룹 OOF 결과
--------------------
구분                     적중    MCC       AUC       균형정확도   TN/FP/FN/TP
기반 Eye A 고정          11/16   0.0000    0.6909    0.5000       0/5/0/11
기존계수 기반 전이학습   14/16   0.7125    0.7636    0.8000       3/2/0/11
0 기준 대조군            14/16   0.7125    0.7818    0.8000       3/2/0/11

전이학습 정확도는 87.5%입니다. 기반 모델 임계값은 0.48345327423625106,
전이학습과 대조군의 임계값은 0.5입니다. 0 기준 대조군도 같은 MCC이므로
기존 계수의 추가 이점은 이 자료에서 확인되지 않았습니다.
15그룹을 5,000회 재표본한 가운데 유효 4,988회에서 MCC 95% 백분위 구간은
[0, 1], AUC 구간은 [0.3333, 1]입니다. bootstrap은 저장 OOF 점수만 재표본합니다.

방법
----
ICCVAM Appendix B4 배합표와 D5 원액 in vivo EPA 등급을 연결했습니다.
25개 후보 중 모든 함량이 숫자이고 합계가 100±0.11%인 16제형을 선택했습니다.
양성 EPA I/II 11개, 음성 EPA III/IV 5개입니다.
고정 13성분 매핑과 함량 가중평균으로 Morgan 2,048개와 RDKit descriptor 217개를 생성합니다.
기존 imputer/scaler는 고정하고, 클래스 균형 logistic loss의 평균에 기존 계수와
절편으로부터의 L2 거리 벌점(0.5 * lambda * 거리제곱, lambda=1)을 더합니다.
클래스 가중치는 fold마다 다시 계산하고 동일 부분특징을 가진 제형은 같은 그룹으로 묶습니다.

실행
----
확인 환경: Windows / Python 3.14.6. 이 폴더에서 실행합니다.

python -m pip install -r requirements-lock.txt
python predict.py --input examples/CTFA1996_HZT.json
python verify.py

추가 재학습·원문 라벨 재추출 의존성: requirements-rebuild.txt
python -m pip install -r requirements-rebuild.txt
python train.py --output-dir ../eye_all673_transfer_reproduced

학습은 기존 모델을 덮어쓰지 않으므로 새 출력 폴더를 지정하세요.
prepare_data.py는 포함된 원문·추출 표에서 제형과 라벨을 다시 구성합니다.
학습과 예측은 저장된 자료를 사용하며 네트워크 접속이 필요하지 않습니다.
새 가상환경 온라인 설치와 다른 운영체제에서의 일치는 별도 검증이 필요합니다.

핵심 자료
---------
protocol.json / mapping.json: 고정한 학습·평가 조건과 성분 매핑
metrics.json / metrics.csv / oof_predictions.csv: 교차검증 결과
fold_log.json / fold_models/: fold별 학습·시험 제형과 모델
bootstrap_ci.json: 그룹 재표본 결과
metadata.json: 기반 모델 해시와 16행 최종 학습 정보
source_manifest.json / checksums.json / verification.json: 출처·무결성·실행 검사
refit_predictions_NOT_validation.csv: 전체 재학습 동작 확인용이며 검증 성적이 아님

joblib의 sklearn 분류기는 추론 컨테이너입니다. 상속된 C 등의 값 대신
common.py의 objective와 protocol.json이 실제 전이학습 목적함수를 정의합니다.
의도적인 패키지 변경 후에만 seal_package.py로 해시를 갱신합니다.

해석 범위
---------
원본 전이학습의 매핑표·fold 모델을 확보하지 못해 새로 재구성한 실험입니다.
현재 14/16 결과가 재현 대상이며, reference/에 보존한 과거 13/16 그림의 복구가 아닙니다.
OOF 성적과 이후 전체 16행으로 적합한 최종 모델의 학습 행 예측은 다릅니다.
16제형과 기반 학습 673행의 중복을 배제하지 못해 독립 외부검증으로 해석하지 않습니다.
물 이외에 구조로 표현한 함량은 0~6.15%로 주요 성분 누락이 크며,
점수는 보정된 실제 자극 확률이나 개인·제품의 안전성 판정이 아닙니다.

원문 출처
https://ntp.niehs.nih.gov/sites/default/files/iccvam/docs/ocutox_docs/ocubrd/hetcam/hetcamappb.pdf
https://ntp.niehs.nih.gov/sites/default/files/iccvam/docs/ocutox_docs/ocubrd/hetcam/hetcamappd5.pdf

GitHub 재현 자료
https://github.com/tjdbf061106-sketch/COS_project/tree/main/transfer_learning/eye_source_centered_transfer
