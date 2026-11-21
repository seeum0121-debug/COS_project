"""Create a separate executed notebook and report; never edit the user's notebook."""
from common import *
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
from datetime import datetime,timezone

def main():
    verify=read(HERE/'verification.json');assert verify['passed']
    audit=read(HERE/'data_audit.json');external=read(HERE/'external_validation_audit.json')
    metrics=pd.read_csv(HERE/'metrics_by_seed.csv');paired=pd.read_csv(HERE/'paired_delta_by_seed.csv')
    summary=pd.read_csv(HERE/'paired_summary.csv');complete=read(HERE/'training_complete.json')
    prep=read(HERE/'preparation_verification.json');teacher=read(HERE/'teacher_source_internal_metrics.json')
    cases=read(HERE/'descriptive_case_summary.json')
    eval_group_count=len(set(read(HERE/'run_config.json')['payload']['groups'][:606]))
    audit_rows=pd.read_csv(HERE/'stoptx_identity_audit.csv')
    # Compact, disjoint attrition accounting; all nonexclusive flags remain in CSV.
    remaining=np.ones(len(audit_rows),bool);attrition=[]
    for flag in ['excluded_overlap','excluded_invalid_standardized_parent','excluded_known_uvcb_or_polymer',
                 'excluded_identical_local_morgan','excluded_conflicting_parent','excluded_duplicate_parent']:
        n=int((remaining&audit_rows[flag].to_numpy()).sum());remaining &= ~audit_rows[flag].to_numpy()
        attrition.append({'reason_in_order':flag,'newly_excluded':n,'remaining':int(remaining.sum())})
    assert int(remaining.sum())==audit['training_rows']
    csv(HERE/'training_attrition_disjoint.csv',pd.DataFrame(attrition))
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
    colors={42:'#2a6fbb',202:'#d57923',777:'#138e7d'}
    for ax,label in zip(axes,['A','B']):
        data=paired[paired.label.eq(label)]
        for j,model in enumerate(MODELS):
            subset=data[data.model.eq(model)].sort_values('seed')
            vals=subset.delta_MCC.to_numpy()
            ax.plot([vals.min(),vals.max()],[j,j],color='#bbbbbb',lw=2)
            for seed,delta in zip(subset.seed,vals):
                ax.scatter(delta,j,s=55,color=colors[seed],label=str(seed) if j==0 else None,zorder=3)
            ax.scatter(vals.mean(),j,marker='|',s=200,color='black',zorder=4)
        ax.axvline(0,color='#555555',ls='--',lw=1)
        ax.set_yticks(range(len(MODELS)),[m.upper() for m in MODELS]);ax.set_title(f'EPA {label}')
        ax.set_xlabel('ΔMCC = STopTox 보조학습 - 대조군');ax.grid(axis='x',alpha=.2)
    axes[0].legend(title='분할 seed',loc='best',fontsize=8)
    fig.suptitle('기존 606행의 동일 grouped OOF 비교 · 점: 각 seed · 검은 선: 평균')
    fig.tight_layout();fig.savefig(HERE/'paired_delta_mcc.png',dpi=180);fig.savefig(HERE/'paired_delta_mcc.pdf');plt.close(fig)
    concise=[]
    for r in summary.itertuples():
        concise.append(f'{r.label} / {r.model.upper():5s}  대조군 {r.control_MCC_mean:.6f} ± {r.control_MCC_sd:.6f}  '
            f'보조학습 {r.aux_MCC_mean:.6f} ± {r.aux_MCC_sd:.6f}  Δ {r.delta_MCC_mean:+.6f} ± {r.delta_MCC_sd:.6f} '
            f'(범위 {r.delta_MCC_min:+.6f}~{r.delta_MCC_max:+.6f}, 양수 {r.positive_seed_count}/3)')
    robust=summary[summary.positive_seed_count.eq(3)]
    negative=summary[summary.positive_seed_count.eq(0)]
    conclusion=(f'평균 ΔMCC가 양수인 조합은 {int((summary.delta_MCC_mean>0).sum())}/6, '
        f'세 seed 모두 양수인 조합은 {len(robust)}/6이다. '
        '결과는 아래 모델·라벨별로 판단하며, 보조학습이 항상 개선된다는 결론은 내리지 않는다. '
        '3개 seed는 동일 자료의 반복 분할이므로 SD는 분할 민감도이고 독립 연구 간 신뢰구간이 아니다.')
    report=f'''Eye STopTox 외부 보조학습 및 화장품 외부검증 검토
최종 결과 폴더: {HERE}
작성 UTC: {datetime.now(timezone.utc).isoformat()}

실제 외부 관측자료로 새 보조모델을 학습했고, 이를 기존 Eye 평가 606행에서 대조군과 비교했다.
화장품 후보의 독립 EPA 정량 외부검증은 완료하지 못했다. 현재 적격 표본은 0개다.
ChemBERTa 재사용 내부 비교로 대체하지 않았고, 화장품·ChemExpo 규칙 라벨을 정답으로 평가하지 않았다.

학습 자료와 중복 처리
Table S4 원본 workbook과 추출 CSV의 3,545행/관측 Call을 일치 검증했다.
학습 허용 {audit['training_rows']}행, 양성 {audit['training_label_counts']['1']}, 음성 {audit['training_label_counts']['0']}.
최종 제외 {audit['excluded_union_all_reasons']}행. 겹치는 플래그의 수를 더하지 말 것.
배타적 제외 순서는 training_attrition_disjoint.csv, 전체 행별 사유는 stoptx_identity_audit.csv에 있다.
CAS/원 canonical/연결성/염 제거·중성화·입체 제거·tautomer parent를 로컬 전체 ICE와 production 673행의 성분 출처와 대조했다.
최초 CAS 성분목록 76중복과 달리, 식별됐지만 벡터에 없는 성분 CAS까지 넓힌 검사는 82행이다.
구조/CAS 중복 합집합 230행. 알려진 UVCB·고분자와 로컬 구조와 동일한 binary Morgan 표현도 추가 제외했다.
원 논문은 혼합물 제거를 설명하지만 공개 Table S4에서 정부가 UVCB/고분자로 분류한 CAS가 실제 발견됐다.
예비 실행은 이 입력 문제를 확인하고 중단·보존했다. 성능을 보고 제외 기준을 바꾼 것이 아니다.
최종 실행은 이 strict 폴더이며, 이전 external_aux_20261108T1705의 부분 결과는 최종 비교가 아니다.

STopTox 라벨은 GHS 1–2B 자극/부식 양성이다. EPA A(I/II), B(I/II/III)와 동일하다고 가정하지 않았다.
원 논문은 salts 정리/중성화, chemotype 정규화, 구조 기반 반복 통합 및 eye 라벨 불일치 제거를 설명한다.
이번 RDKit 절차는 보수적 재검사이며 저자의 원 소프트웨어를 완전히 재현했다는 뜻이 아니다.
Table S4는 CAS·SMILES·InChIKey·Call만 제공하므로 원시험 ID별 독립성은 미확인이다.
정부 CAS 목록은 4,353개 우선검토 물질과 별도 확인 CAS에 한정되므로, 나머지가 모두 순수 단일 물질이라고 보증하지 않는다.
자료의 잔여 구조/원시험 불확실성을 가진 조건부 전이 실험으로 해석한다.

실제 전달 방식과 동일 조건 비교
새 RandomForest 보조모델(500 trees, min_samples_leaf=2, max_features=sqrt, balanced, seed42)을
필터링된 외부 GHS 관측 라벨과 표준화 Morgan 2,048비트로 학습했다. 원래 STopTox 가중치를 재사용한 것이 아니다.
외부 모델을 고정해 현재 성분 점수를 얻고, 원래 성분 가중 평균/최대/최소/가중 표준편차 4개를 계산했다.
이 값은 제형의 보정된 자극 확률이 아니라 EPA 분류기에 전달하는 보조 특성이다.
대조군은 기존 농도 포함 2,269특성, 실험군은 같은 특성+보조특성=2,273특성이다.
원래 성분 연결은 673행 모두에서 Morgan 벡터 재구성으로 확인했다. 평가행 감소 0, 공통 평가행 606.
평가 606행은 원래 병합 규칙상 {eval_group_count}개 그룹이며, 606개의 완전히 독립된 화학물질로 해석하지 않는다.
원래 농도누락 단일성분 9행은 성분 가중치1만 유지하며 농도를 만들지 않았다. 농도 중앙값은 fold 학습행에서 계산한다.
LR/SVM/LightGBM, A/B, seed42/202/777, 원래 outer5·inner3, 모델별20후보 전체 격자,
balanced 학습 가중치, inner 평균 AUC 선택, 기존행 inner OOF의 민감도·특이도 차이 최소 threshold 규칙을 그대로 썼다.
두 arm은 같은 threshold 선택 규칙을 사용하며 각 arm의 수치 threshold는 별도로 정해진다.
67개 증강행은 train-only이고 validation/test와 연결된 그룹의 증강은 기존 split대로 차단한다.
대조군 90 outer/1,800후보는 코드·환경·행·split·threshold·체크섬을 감사해 재사용했다.
대조군 선택 모델 1개를 다시 학습해 저장 점수와 수치 일치도 확인했다.
새 실험군 90 outer/1,800후보, 총 {complete['new_target_model_fits']:,}회 target 모델 fit이 완료됐다.
기존 production의 6개 모델 360작업을 재실행/변경한 결과가 아니다. 새 비교 범위는 명시한 3개 모델이다.

결과: 각 seed의 606개 outer 예측을 합산한 MCC의 평균 ± 표본 SD
{chr(10).join(concise)}

{conclusion}
원자료는 paired_delta_by_seed.csv, 전체 성능은 metrics_by_seed.csv, 행별 예측은 outer_oof_predictions.csv.
각 fold의 실제 threshold와 성능도 metrics_by_fold.csv에 저장했다. 가장 좋은 seed만 선택하지 않는다.
보조모델의 외부 출처 corpus 내부 5-fold 진단 AUC={teacher['auc']:.6f}, MCC@0.5={teacher['mcc_at_0_5']:.6f}.
이 진단은 동일 binary Morgan을 그룹화한 source 내부 CV이며 EPA·화장품 독립 외부 성능이 아니다.
현재 평가 606행의 출처는 모두 FIFRA data이므로 이 MCC를 화장품 성능으로 옮겨 보고하지 않는다.
고유 성분의 외부학습 최근접 Morgan Tanimoto 중앙값={prep['nearest_tanimoto_median_unique_component']:.6f}.
최종 학습자료와 로컬 성분의 동일 binary Morgan은 0개다. 유사도 자체는 적용 가능성 보증이 아니다.

새 화장품 자료: 정량 검증과 사례 분석 구분
2025 baby-bath 본문 PDF와 보충자료 PDF를 추가 다운로드하고 Table 2/3 병합 셀·숫자를 시각 검증했다.
39개 시판 제품의 개별 전체 조성/시험 연결이 부족하다. Table 2는 39개 원료명의 58개 조건이며 58개 제품이 아니다.
Table 3는 대조군 포함6행, 새 혼합조건5개다. 보충자료의 102시험/306마리와 부합하며 대조군을 다시 세지 않는다.
동물 Table 3의 5% SLES와 세포시험 0.01% SLES는 다른 관측이다.
성분 %, pH범위, 원래 Draize 등급을 보존했고 w/w·w/v/유효성분 기준을 임의 확정하지 않았다.
보충 Table S1을 확인했지만 개체별 조직·시간점 점수 없이 전체 Draize 등급을 EPA A/B로 자동 변환하지 않았다.
비자극 일부 조건의 음성 호환 가능성도 시험액 조성·농도기준·개체 기록 확인 전에는 정량 검증으로 쓰지 않는다.
SLES/고분자/상업원료는 단일 확정구조로 만들지 않고, 잔량 물·pH 중간값·농도범위 중간값을 만들지 않았다.
같은 CAPB 농도/pH의 공급업체별 등급 차이는 현재 구조/농도 표현의 한계를 보여주는 사례로 남겼다.
공급업체 비교 {cases['baby_supplier_contrast_groups']}그룹 중 {cases['baby_supplier_groups_with_different_original_grades']}그룹에서 원래 등급이 달랐고,
동일 원료명/공급업체/pH기재/기재 세척조건의 농도 비교 {cases['baby_concentration_contrast_groups']}그룹 중 {cases['baby_concentration_groups_with_changed_original_grades']}그룹에서 등급이 변했다.
여기서 pH기재에는 'No adjustment'도 포함되며, 실제 측정 pH가 같다고 단정하지 않는다.
이를 독립 제품 수나 인과효과/모델 성능으로 해석하지 않는다. 상세는 baby_*_case_comparisons.csv.

IIVS: 14개 번호 제형의 기재 농도·pH·IVIS를 원 PDF 텍스트와 대조했다. 원문 dash는 그대로다.
비공개 vehicle, 농도기준, exposure/시험액, EPA 연결 자료가 부족하므로 정량 검증에는 사용하지 않았다.
제형13의 IVIS85 vs 기재평균으로 계산81.985, 제형8의 IVIS79 vs 서술분류 등 원문 불일치를 기록했다.
PDF 직접 다운로드/스크린샷은 실패해 IIVS는 텍스트 판독까지 검증했다. baby-bath는 PDF 시각 검증도 완료했다.
두 논문의 표는 사례/자료 적격성 분석만 수행했고 학습·튜닝·threshold에 투입한 행은0이다.
ChemExpo는 조성/라벨 없는 표현학습 후보로 분리하며 Eye 정답과 10% 규칙을 혼동하지 않았다.
필요한 추가자료: 시험ID별 전체 배합과 vehicle/pH조정제, %의 질량·부피·활성 기준, 원료 등급/순도/분포,
개체·조직·시간점 원점수와 회복/세척조건, 제조/시험 lot 및 반복/원시험 중복 식별자.
외부 연락은 하지 않았다.

검증 및 보존
verification.json: 신규90작업/1,800후보의 inner AUC·threshold·예측·median과 합산 성능 검증 통과.
기존 사용자 파일 {verify['protected_files_checked']:,}개 SHA256 재확인, 변경0개.
Eye 원본 SHA256: {verify['eye_input_sha256']}
기존 eye_all_experiments_v2.ipynb의 5부/6부는 변경하지 않았다. 별도 eye_external_auxiliary_validation.ipynb를 생성했다.

출처
STopTox 논문: https://pmc.ncbi.nlm.nih.gov/articles/PMC8863177/ (10.1289/EHP9341)
데이터: https://github.com/molecularmodelinglab/StopTox-External-Validation/tree/efbd5c526ba117e91d9f02c74fb4fe12efd6d79d
화장품 논문: https://pmc.ncbi.nlm.nih.gov/articles/PMC12656482/ (10.3390/toxics13110948)
보충자료: https://mdpi-res.com/d_attachment/toxics/toxics-13-00948/article_deploy/toxics-13-00948-s001.zip
IIVS: {external['iivs']['source']}
UVCB 정부 목록: https://www.canada.ca/en/environment-climate-change/services/evaluating-existing-substances/statusprioritized-substances2.html
별도 UVCB 확인: https://canadachemicals.oecd.org/ChemicalDetails.aspx?ChemicalID=E236BA68-0E99-4E50-B966-6D068A43E1D0
'''
    with (HERE/'RESULTS.txt').open('x',encoding='utf-8') as f:f.write(report)
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook(cells=[
        md('# Eye 외부 보조학습과 외부검증 검토\n\n기존 종합 노트북과 별도인 후속 결과입니다. **외부학습 효과는 기존 606행에서 비교했고, 독립 화장품 EPA 외부 성능은 미검증(적격0행)**입니다. 기존 production 결과를 덮어쓰지 않았습니다.'),
        code("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd()\nassert (ROOT/'training_complete.json').exists()\nload=lambda name: json.loads((ROOT/name).read_text(encoding='utf-8'))\ndisplay(load('verification.json'))"),
        md('## 1. 외부 관측자료와 누수 검사\n\nSTopTox GHS 라벨을 EPA 정답으로 합치지 않았습니다. CAS·표준화 구조·연결성뿐 아니라 알려진 UVCB/고분자와 로컬과 동일한 Morgan 표현을 제외했습니다. Table S4에 원시험 ID가 없어 연구 단위 독립성은 완전히 입증되지 않았습니다. [원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC8863177/)'),
        code("a=load('data_audit.json')\ndisplay({k:a[k] for k in ['stoptx_input_rows','training_rows','training_label_counts','excluded_union_all_reasons','common_evaluation_rows','evaluation_excluded_rows','uvcb_screen_scope']})\ndisplay(pd.read_csv(ROOT/'training_attrition_disjoint.csv'))"),
        md('## 2. 실제 보조학습 전달과 평가 조건\n\n필터링된 외부 관측자료로 새 RF 성분 모델을 학습하고 고정했습니다. 성분 점수의 가중평균·최대·최소·가중표준편차를 EPA 분류기에 추가합니다. RF/XGB 사이의 가중치 전이가 아닙니다.\n\n대조군2,269 vs 실험군2,273 특성, 같은606행·그룹·5×3 fold·20후보·threshold 규칙을 사용합니다. 수치 threshold는 각 arm의 inner OOF에서 정합니다. 증강67행은 train-only입니다. 대조군 저장 결과는 전체 후보와 cutoff를 감사하고 일부 재학습으로 점수 일치를 확인했습니다.'),
        code("display(load('training_complete.json'))\ndisplay(load('preparation_verification.json'))"),
        md('## 3. 외부학습 효과: 기존 평가행에서의 paired OOF\n\n평균과 SD는 세 seed의 결과입니다. seed 반복은 독립 연구 반복이 아니며, 최상의 seed만 선택하지 않습니다.'),
        code("summary=pd.read_csv(ROOT/'paired_summary.csv')\ndisplay(summary)\ndisplay(pd.read_csv(ROOT/'paired_delta_by_seed.csv'))\ndisplay(Image(filename=str(ROOT/'paired_delta_mcc.png')))"),
        md(conclusion),
        code("display(pd.read_csv(ROOT/'metrics_by_seed.csv')[['arm','label','model','seed','MCC','ROC-AUC','sensitivity','specificity','PR-AUC']])"),
        md('보조모델의 다음 수치는 **외부 출처 corpus의 내부 CV 진단**입니다. EPA 또는 화장품의 독립 외부 성능이 아닙니다.'),
        code("display(load('teacher_source_internal_metrics.json'))"),
        md('## 4. 화장품 자료의 검증 적격성\n\n[2025 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12656482/)의 39개 제품, 58개 원료 조건, 대조군 포함6개 조합 표시행을 구별했습니다. 개체별 시험 기록·농도기준·시험액/원료 조성이 부족해 EPA 정량 검증에 쓰지 않았습니다. 원문 등급을 보존하고 임의 EPA 변환·잔량 물·범위 중간값을 만들지 않았습니다.'),
        code("e=load('external_validation_audit.json')\ndisplay(e['decision'])\nt2=pd.read_csv(ROOT/'baby_table2_observation_audit.csv')\ndisplay(t2[['test_number','ingredient_reported','supplier_reported','concentration_reported','ph_reported','original_draize_label','quantitative_external_validation_allowed']])"),
        code("display(pd.read_csv(ROOT/'baby_table3_observation_audit.csv')[['sample_id','main_pct_reported','additional_ingredient','additional_pct_reported','ph_range_reported','original_draize_label','quantitative_external_validation_allowed']])\ndisplay(pd.read_csv(ROOT/'iivs_table2_observation_audit.csv')[['formulation_id','ph','ethanol_pct_raw','glycolic_acid_pct_raw','salicylic_acid_pct_raw','bcop_ivis_reported','large_arithmetic_discrepancy','quantitative_external_validation_allowed']])"),
        md('IIVS는 vehicle 미공개와 원문 수치/서술 불일치가 남습니다. 동물시험과 SIRC 세포시험 농도를 혼합하지 않았습니다. ChemExpo 함량규칙은 Eye 시험 정답이 아닙니다. 외부 후보는 학습·튜닝·threshold에 쓰지 않았고 저자에게 연락하지 않았습니다.'),
        code("display(e['baby']['missing'])\ndisplay(e['iivs']['missing'])"),
        md('공급업체·농도 비교는 같은 원료의 조건 차이를 원래 등급으로 기술한 사례 분석입니다. 독립 제품 수 또는 EPA 모델 정확도가 아닙니다.'),
        code("display(load('descriptive_case_summary.json'))\ndisplay(pd.read_csv(ROOT/'baby_supplier_case_comparisons.csv'))\ndisplay(pd.read_csv(ROOT/'baby_concentration_case_comparisons.csv'))"),
        md('## 5. 재현·한계\n\n`protocol.json`, `run_config.json`, `stoptx_identity_audit.csv`, `outer_oof_predictions.csv`, `verification.json`, `artifact_checksums.json`을 함께 보관합니다. 상세 설명은 `RESULTS.txt`에 있습니다. 결과는 LR/SVM/LightGBM과 현재 FIFRA 평가행에 한정됩니다. 기존 노트북과 입력·production 결과는 보존했습니다. 원시험 연결과 정부 목록 밖 UVCB의 잔여 불확실성이 있어 완전한 외부 독립성을 주장하지 않습니다.')
    ],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(HERE)}}).execute()
    with (HERE/'eye_external_auxiliary_validation.ipynb').open('x',encoding='utf-8') as f:nbformat.write(nb,f)
    write(HERE/'artifact_checksums.json',{str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='artifact_checksums.json'})
    print('REPORT + EXECUTED NOTEBOOK COMPLETE',flush=True)

if __name__=='__main__':main()
