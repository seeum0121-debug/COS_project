"""Evidence inventory only. No model fit, predictions, thresholds, or invented EPA labels."""
from pathlib import Path
import hashlib, json, re, shutil, html
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OLD = BASE / 'Eye_cosmetic_validation_20261109'
RERUN = BASE / 'Eye_재외부검증_20261109'
REVIEW = BASE / 'EPA_재검토_추가자료_20261109'
DATE = '2026-11-09'
COMMIT = 'a693ae5fb43197cac4dc530a5371f1682ede2c3c'
REPO = f'https://github.com/tjdbf061106-sketch/COS/blob/{COMMIT}/cosmetic_validation/'
QURL = 'https://patents.google.com/patent/US6583181B1/en'
SURL = 'https://patents.google.com/patent/US4992266A/en'
MANUAL = 'https://www.epa.gov/sites/production/files/documents/label-review-manual.pdf'
for folder in ['sources', 'input_snapshot']:
    (ROOT / folder).mkdir(exist_ok=True)

INPUTS = {
    'cir_ingredients.csv': OLD/'sources/cosmetic_validation/cir_ingredients.csv',
    'flea_tick_shampoo.csv': OLD/'sources/cosmetic_validation/flea_tick_shampoo.csv',
    'zein.csv': OLD/'sources/cosmetic_validation/lechuga2025_surfactant_mixture_zein.csv',
    'quat_formulations.csv': RERUN/'inputs/quat_formulations.csv',
    'quat_components.csv': RERUN/'inputs/quat_components.csv',
    'quat_observations.csv': RERUN/'inputs/quat_observations.csv',
    'structure_registry.csv': RERUN/'inputs/structure_registry.csv',
    'shampoo_patent_observations.csv': RERUN/'inputs/shampoo_patent_observations.csv',
    'zodiac_source.json': RERUN/'inputs/zodiac_source.json',
    'source_adjudication.json': OLD/'source_adjudication.json',
    'EPA_review.csv': REVIEW/'EPA_라벨_재검토.csv',
    'additional_sources.csv': REVIEW/'추가자료_후보와중복.csv',
    'CIR_previous_audit.csv': OLD/'outputs/CIR_eye_audit_predictions.csv',
    'rerun_summary.json': RERUN/'outputs/summary.json',
}
manifest=[]
for name, source in INPUTS.items():
    data=source.read_bytes()
    (ROOT/'input_snapshot'/name).write_bytes(data)
    manifest.append(dict(snapshot=name, original_path=str(source), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
for name in ['shampoo_recipe_web.txt','quat_tables_1_3_web.txt','quat_pdf_and_table4_web.txt','epa_criteria_web.txt']:
    shutil.copy2(RERUN/'sources'/name, ROOT/'sources'/('prior_'+name))

def read(name): return pd.read_csv(ROOT/'input_snapshot'/name).fillna('')
def number(x): return None if x == '' or pd.isna(x) else float(x)
def truth(x): return str(x).lower() == 'true'
def fmt(x): return '' if x is None or x == '' else f'{float(x):g}'
def joined(xs): return ' | '.join(str(x) for x in xs if x is not None and str(x) != '')

cases=[]; components=[]; sources=[]
def source(sid, title, url, location, status, note='', version=''):
    sources.append(dict(source_id=sid, title=title, url=url, location=location, review_status=status, limitations=note, document_version=version, checked_date=DATE))
source('EPA_GUIDE','EPA Label Review Manual',MANUAL,'PDF 76쪽, Chapter 7 Tables 6–7','공식 지침 확인','라벨 문구와 시험으로 확정한 등급은 구분. 피부 흡수 급성독성은 피부 자극과 다른 endpoint.')
source('REPO_CIR','GitHub CIR 추출표',REPO+'cir_ingredients.csv','고정 커밋의 CSV 데이터 행','로컬 보존본 확인','최신 main과의 일치 미확인; EPA등급(환산)은 저장소 작성자의 환산값.',COMMIT)
source('REPO_SDS','GitHub 샴푸 추출표',REPO+'flea_tick_shampoo.csv','고정 커밋의 CSV','로컬 보존본 확인','원래 레이블과 분자구조를 보존하며 본 목록에서 자동 교정하지 않음.',COMMIT)
source('Q_PATENT','US6583181B1',QURL,'Tables 1–4; Examples 1–5','특허 표·본문 확인','동물별 조직 점수 없음. Table 4 상업 원료 머리글 대응은 추가 확인 필요.','2003-06-24')
source('SH_PATENT','US4992266A',SURL,'Examples 1–7; Tables I–II','특허 표·본문 확인','3마리 평균, 7일 관찰. Examples 8–11은 예상 결과여서 실측 사례에서 제외.','1991-02-12')

def base(cid,dataset,name,endpoint,url,title,loc):
    return dict(case_id=cid,dataset=dataset,name=name,endpoints=endpoint,cas_or_registration='',source_title=title,source_url=url,source_location=loc,
        evidence_ids='',primary_review='미확인',original_eye_epa='',original_skin_epa='',original_eye_binary=None,original_skin_binary=None,
        original_label_confidence='',eye_epa_review='해당 endpoint 자료 없음',skin_epa_review='해당 endpoint 자료 없음',
        eye_epa_gold='',skin_epa_gold='',eye_label_proxy=None,skin_label_proxy=None,label_evidence_type='',
        eye_observation='',skin_observation='',composition_status='',composition_summary='',structure_status='',
        original_concentration_pct=None,concentration_interpretation='',test_dilution_status='미확인',verified_test_concentration_pct=None,
        concentration_basis='',vehicle_status='',ph_low=None,ph_high=None,ph_status='미확인',ph_context='',ph_source_url='',ph_source_location='',
        represented_pct_low=None,represented_pct_high=None,represented_pct_mean=None,coverage_basis='',
        study_id='',study_identity_status='고유 원시험 번호·시험물질 배치 연결 미확인',formulation_study_match='미확인',
        grouping_key='',training_independence='원시험 중복 미확인',current_use='',eye_use='',skin_use='',
        strict_epa_eligible=False,eligibility_reason='EPA 정답·시험물질 일치·원시험 독립성 미확정',
        source_exposed=True,priority='P2',next_action='',conflict_flags='',notes='',checked_date=DATE)

cir=read('cir_ingredients.csv')
prior_audit=read('CIR_previous_audit.csv').set_index('case_id')
adjud=json.loads((ROOT/'input_snapshot/source_adjudication.json').read_text(encoding='utf-8'))
report_ids={t:f'CIR_REPORT_{i:02}' for i,t in enumerate(cir['출처보고서'].unique(),1)}
for title,sid in report_ids.items():
    source(sid,title,'','표·페이지·원시험 식별자 확보 필요','서지명만 추출','이전 부분 검토 URL은 해당 사례 행에 별도 기록. 저장소의 high를 독립 검증 신뢰도로 재사용하지 않음.')
for i,row in cir.iterrows():
    endpoint=row['endpoint(skin/eye)']; ev=row['근거']; cas=str(row['CAS']); conc=number(row['농도(%)'])
    cid=f'CIR_row_{i+1}'
    c=base(cid,'CIR',row['성분명'],endpoint.upper(),'',row['출처보고서'],f'cir_ingredients.csv 데이터 행 {i+1} / 파일 줄 {i+2}; 원보고서 위치 미확인')
    c.update(cas_or_registration=cas,evidence_ids=f'REPO_CIR | {report_ids[row["출처보고서"]]}',
        primary_review='저장소 추출값 확인; 원문 미검토',original_label_confidence=row['신뢰도'],
        label_evidence_type='저장소 EPA 환산값; 공식 EPA 분류로 확인되지 않음',
        original_concentration_pct=conc,concentration_basis='저장소 농도(%) 열; 질량/부피 기준 및 실제 시험조건 원문 확인 필요',
        concentration_interpretation='CSV 기재 농도; 해당 원시험의 정확한 농도인지 미확인' if conc is not None else 'CSV 농도 결측',
        test_dilution_status='CSV 기재값만 있음; 원문 확인 필요' if conc is not None else '미확인',
        composition_status='부분 확인—성분명 및 CSV 농도',
        composition_summary=f'{row["성분명"]}: '+(fmt(conc)+'% (CSV)' if conc is not None else '농도 미상'),
        vehicle_status='CSV 100% 표기; 순도·불순물 미확인' if conc==100 else '용매·기제 전체 배합 미확인',
        structure_status='기존 단일구조 있음; 시험물질 일치 재확인 필요' if truth(row['resolved']) else '혼합물 또는 구조 미해결',
        grouping_key=f'CAS:{cas}',current_use='근거 확인 전 탐색·진단 후보',
        next_action='원보고서의 해당 시험 표·페이지와 원시험 번호 확보; 농도·기제·세척 조건 대조; endpoint별 EPA 기준 재판정',
        notes='한 행이 여러 농도·시험을 요약했을 수 있음. 성분 농도가 100%로 기재되어도 순도 확인으로 간주하지 않음.')
    c[f'original_{endpoint}_epa']=row['EPA등급(환산)']
    c[f'original_{endpoint}_binary']=int(row['이진라벨'])
    c[f'{endpoint}_observation']=ev
    c[f'{endpoint}_epa_review']='미확정—환산 근거와 시험 프로토콜 재검토 필요'
    c['eye_use']='EYE 원자료 재검토 후보' if endpoint=='eye' else '현재 행은 SKIN 시험; EYE 정답 전용 불가'
    c['skin_use']='SKIN 원자료 재검토 후보' if endpoint=='skin' else '현재 행은 EYE 시험; SKIN 정답 전용 불가'
    if cid in prior_audit.index:
        c['current_use']='이전 EYE 평가: '+str(prior_audit.loc[cid,'status'])
    flags=[]
    if '10-20%' in ev:
        c['concentration_interpretation']='CSV 15%는 근거의 10–20% 범위를 요약한 값; 실제 15% 시험으로 취급 금지'
        flags.append('농도 범위가 단일값으로 요약됨')
    elif 'up to' in ev.lower():
        c['concentration_interpretation']='근거에 up to가 있음; CSV 값이 상한/여러 시험 요약일 가능성, 정확한 시험농도 미확정'
        flags.append('농도 상한·다중 시험 요약 가능')
    for token in ['cream vehicle','hydrophilic ointment','propylene glycol','sunscreen']:
        if token in ev.lower(): c['vehicle_status']=f'기제 일부 언급: {token}; 전체 조성·원문 미확인'
    if 'rinsed' in ev.lower() or 'rinsing' in ev.lower():
        flags.append('세척 조건별 시험 분리 필요')
    if any(t in ev.lower() for t in ['cumulative','repeated-application']): flags.append('반복/누적시험—급성 endpoint 적합성 검토')
    ph=re.search(r'\bpH\s*([0-9]+(?:\.[0-9]+)?)',ev)
    if ph:
        c.update(ph_low=float(ph[1]),ph_high=float(ph[1]),ph_status='CSV 기재—원문 미확인',ph_context='시험 근거 요약의 pH; 직접 원문 검증 전',ph_source_url=REPO+'cir_ingredients.csv',ph_source_location=f'데이터 행 {i+1}')
    if cas=='151-21-3':
        c['structure_status']='오류 의심—CAS는 나트륨염, 공급 SMILES는 중성 산'
        c['priority']='P1';flags.append('CAS–SMILES 염 형태 불일치')
        c['next_action']='검증된 SLS 나트륨염 구조 확인; 원문 농도·세척별 피부/눈 결과 대조 후 EPA 재판정'
    if cas in ['56-81-5','122-99-6']:
        a=adjud['cir'][cas]
        c.update(source_url=a['url'],primary_review='이전 검토에서 원문 일부 확인; 개별 시험 전체는 미확정')
        c['source_location']+=f'; 이전 검토 PDF {a["pages"]}쪽'
        c['notes']+=' '+a['issue']
        if cas=='122-99-6' and endpoint=='eye':
            flags.append('회복 시점 서술 재확인—15일 완전 회복으로 단정 금지')
            c['eye_epa_review']='I/II 세부 등급 미확정; 이전 부분 검토에서 7일 이후 소견 확인'
    c['conflict_flags']=joined(flags)
    cases.append(c)
    components.append(dict(case_id=cid,component=row['성분명'],cas=cas,pct_low=conc,pct_high=conc,
        amount_basis='CSV 성분 농도; 시험농도·단위·기제 원문 미확인',role='repository_ingredient',smiles=row['canonical_smiles'],
        structure_status=c['structure_status'],source_url=REPO+'cir_ingredients.csv',source_location=f'데이터 행 {i+1}'))

sds=read('flea_tick_shampoo.csv'); epa=read('EPA_review.csv').set_index('epa_reg')
ph_meta={
 '69061-12-50591':(6.2,6.8,'2025-05-28',3,'SDS 제품 pH; 시험 당시 시료와 일치 미확인'),
 '1021-1723-12281':(None,None,'2015-07-06',None,'SDS 본문 접근 실패; pH 미보고라고 단정하지 않음'),
 '270-324':(7,8,'2023-09-28',4,'SDS 제품 pH; 시험 당시 시료와 일치 미확인'),
 '2724-485':(5.9,5.9,'2016-06-24',5,'SDS 제품 pH, 21.3°C; 시험 당시 시료와 일치 미확인'),
 '2724-485-270':(5.9,5.9,'2020-10-29',4,'SDS 제품 pH, 21.3°C; 시험 당시 시료와 일치 미확인'),
}
for j,(reg,rows) in enumerate(sds.groupby('epa_reg',sort=False),1):
    r=rows.iloc[0]; er=epa.loc[reg]; plo,phi,version,page,context=ph_meta[reg]
    cid='SDS_'+reg.replace('-','_'); sid=f'SDS_{j:02}'; lid=f'LABEL_{j:02}'
    source(sid,r['product']+' SDS',r['source_url'],f'Section 3 / 9 / 11; pH PDF {page}쪽' if page else '본문 접근 실패; 기존 추출표/검색 색인만 있음','원문 일부 확인' if page else '접근 실패',context,version)
    source(lid,'EPA 기본 등록 '+er['registration_family'],er['official_label_url'],f'PDF {int(er["pdf_page_1_based"])}쪽','공식 endpoint별 라벨 문구 확인','원시험 등급·현재 SDS 배합 연결은 미확정',er['label_date'])
    c=base(cid,'제품 SDS',r['product'],'EYE+SKIN',r['source_url'],r['product']+' SDS','Section 3(조성), 9(pH), 11(독성); EPA 공식 라벨 별도')
    resolved=rows[rows.resolved.map(truth)]
    low=sum(float(v) for v in resolved.pct_low if v!=''); high=sum(float(v) for v in resolved.pct_high if v!='')
    c.update(cas_or_registration=reg,evidence_ids=f'REPO_SDS | {sid} | {lid} | EPA_GUIDE',
        primary_review='SDS 일부 및 EPA 라벨 확인' if page else 'SDS 본문 접근 실패; EPA 공식 라벨 확인',
        original_eye_epa=r['eye_epa_category'],original_skin_epa=r['skin_epa_category'],
        original_eye_binary=int(r['eye_binary']),original_skin_binary=int(r['skin_binary']),
        eye_epa_review='II 대응 라벨 문구; 원시험 등급 미확정' if int(er['binary_label_proxy']) else 'III/IV 대응 라벨 문구; III 단독 확정 불가',
        eye_label_proxy=int(er['binary_label_proxy']),skin_epa_review='미확정—GHS/형용사만으로 EPA 피부등급 환산 불가',
        label_evidence_type='EPA 눈자극 문구 대리값 + SDS 독성 요약; 원시험 자료 아님',eye_observation=r['eye_evidence'],skin_observation=r['skin_evidence'],
        composition_status='부분 확인—SDS 공개 성분/범위; 전체 배합 미공개',
        composition_summary=joined([f'{v.chemical_name}: {fmt(number(v.pct_low))}–{fmt(number(v.pct_high))}%' if v.pct_low!='' else f'{v.chemical_name}: 미공개' for v in rows.itertuples()]),
        structure_status='일부 성분만 단일구조 대응; 계면활성제/UVCB/비공개 잔량 미해결',
        concentration_interpretation='성분 wt%는 제품 배합 함량. 자극 시험의 제품 희석률은 별도 미확인',
        concentration_basis='SDS Section 3 공개 함량',test_dilution_status='미확인—사용법의 희석·세척을 자극시험 조건으로 전용하지 않음',vehicle_status='전체 기제/수분 함량 미공개',
        ph_low=plo,ph_high=phi,ph_status='원문 확인—제품 pH' if plo is not None else '미확인—본문 접근 실패',ph_context=context,
        ph_source_url=r['source_url'],ph_source_location=f'PDF {page}쪽, Section 9' if page else '본문 재확보 필요',
        represented_pct_low=low,represented_pct_high=high,represented_pct_mean=(low+high)/2,
        coverage_basis='이전 모델 입력에서 단일구조로 대응한 성분의 제품 내 질량%; 정확한 전체 분자 조성의 확인률은 아님',
        grouping_key='EPA_BASIC:'+er['registration_family'],current_use='라벨 대리값과 부분 배합을 이용한 진단만 가능',
        eye_use='EYE 라벨 대리값 진단; 정식 정답 보류',skin_use='SKIN 시험 요약 있음; EPA 피부등급 재판정 필요',priority='P1',
        next_action='등록 배합 버전·원시험 번호 연결; 눈/피부 각각 원시험 점수·회복자료 확보; 전체 배합과 시험 희석률·시험 pH 확인',
        notes='같은 기본 등록번호는 묶어서 관리하되 동일 배합/시험으로 단정하지 않음. pH 범위 중앙값을 시험 pH로 넣지 않음.')
    if reg in ['69061-12-50591','1021-1723-12281']:
        c['skin_epa_review']='충돌—CSV III, 기본 등록 공식 피부 자극 문구는 II 대응; 배합·원시험 확인 전 보류'
        c['skin_label_proxy']=1
        c['conflict_flags']='SKIN 원래 III/0와 공식 라벨 II 대응 문구/1 불일치'
        c['next_action']='최우선: SKIN III 환산 근거를 공식 피부 자극 문구·원시험과 대조; 배합 버전 연결 후 판정. '+c['next_action']
    if reg=='69061-12-50591':
        c['notes']+=' 공식 2015 라벨은 pyrethrins 0.15%, PBO 0.30%, MGK264 0.50%를 기재하나 기존 CSV는 일부만 추출. SDS 버전·배합 연결 전 기존 입력에 자동 추가하지 않음.'
    if reg=='1021-1723-12281':
        c['notes']+=' 2013 공식 라벨 MGK264 0.479%는 CSV와 일치. PBO technical 0.300%=주성분 0.240%+관련성분 0.060%; 0.300% 전체를 단일구조로 둔 기존 근사 확인 필요.'
    cases.append(c)
    for _,v in rows.iterrows():
        components.append(dict(case_id=cid,component=v['chemical_name'],cas=v['cas'],pct_low=number(v['pct_low']),pct_high=number(v['pct_high']),
            amount_basis='SDS 제품 내 공개 wt%; 원시험 희석농도 아님',role=v['declared_type'],smiles=v['canonical_smiles'],
            structure_status='기존 구조 사용; technical 혼합물 가능' if truth(v['resolved']) else '미해결',source_url=v['source_url'],source_location='원래 CSV / SDS Section 3'))

z=json.loads((ROOT/'input_snapshot/zodiac_source.json').read_text(encoding='utf-8'))
registry=read('structure_registry.csv').set_index('component_id')
source('SDS_06',z['product']+' SDS',z['sds_url'],'PDF 2쪽 조성, 5쪽 pH·독성','원문 확인','파일명 2020과 달리 문서 내부 날짜는 2016-04-07','2016-04-07')
source('LABEL_06','EPA 2724-792',z['label_url'],'PDF 4쪽','공식 눈자극 문구 확인','피부 흡수 유해 문구는 피부 자극 등급 근거가 아님','2015-01-29')
c=base('Zodiac_2724_792','제품 SDS',z['product'],'EYE+SKIN',z['sds_url'],z['product']+' SDS','PDF 2쪽 조성, 5쪽 pH·독성')
c.update(cas_or_registration='2724-792',evidence_ids='SDS_06 | LABEL_06 | EPA_GUIDE',primary_review='SDS 일부 및 EPA 라벨 확인',
    eye_epa_review='III/IV 대응 라벨 문구; III 단독 확정 불가',eye_label_proxy=0,skin_epa_review='미확정—GHS 피부2를 EPA II/III로 자동 환산 불가',
    label_evidence_type='EPA 눈자극 라벨 대리값 + SDS 독성 요약',eye_observation='SDS: 토끼 눈의 경도 자극',skin_observation='SDS: 토끼 피부의 중등도 자극; GHS Skin Irritation 2',
    composition_status='부분 확인—SDS 공개 성분/범위; 전체 배합 미공개',composition_summary='Pyrethrins 0.05%; PBO 0.5%; ethanol 1.69–1.95%; coconut monoethanolamide 1.08–1.2%; 나머지 미공개',
    structure_status='PBO/ethanol만 기존 입력 대응; 혼합 계면활성제 등 미해결',
    concentration_interpretation='성분 wt%는 제품 배합 함량; 시험 제품 희석률 미확인',concentration_basis='SDS Section 3',vehicle_status='전체 기제/수분 함량 미공개',
    ph_low=5.64,ph_high=5.64,ph_status='원문 확인—제품 pH',ph_context='SDS 제품 pH; 시험 당시 시료와 일치 미확인',ph_source_url=z['sds_url'],ph_source_location='PDF 5쪽, Section 9',
    represented_pct_low=2.19,represented_pct_high=2.45,represented_pct_mean=2.32,coverage_basis='이전 모델 입력에 대응한 성분의 제품 내 질량%',
    grouping_key='EPA_BASIC:2724-792',current_use='부분 배합의 EYE 라벨 대리값 진단',eye_use='EYE 대리값 진단; 정식 정답 보류',skin_use='SKIN 시험 요약 있음; EPA 재판정 필요',
    priority='P1',next_action='원시험 번호·배합 버전 연결 및 전체 배합·시험 희석률·시험 pH 확보',notes='기존 2724-485 계열과 다름. 파일명 2020을 SDS 개정일로 사용하지 않음.')
cases.append(c)
for key,lo,hi,cas in [('PBO',.5,.5,'51-03-6'),('ethanol',1.69,1.95,'64-17-5'),('pyrethrins',.05,.05,'8003-34-7'),('coconut fatty acid monoethanolamide',1.08,1.2,'68140-00-1'),('other ingredients',None,None,'')]:
    components.append(dict(case_id=c['case_id'],component=key,cas=cas,pct_low=lo,pct_high=hi,amount_basis='SDS 제품 wt%',role='SDS 공개 성분',smiles=registry.loc[key,'smiles'] if key in registry.index else '',structure_status='기존 구조 대응' if key in registry.index else '미해결',source_url=z['sds_url'],source_location='PDF 2쪽, Section 3'))

quat=read('quat_formulations.csv'); qc=read('quat_components.csv'); qo=read('quat_observations.csv')
for _,r in quat.iterrows():
    cid=r['formulation_id']; table=int(r['table']); col=int(r['column'])
    c=base(cid,'4급암모늄 특허',f'표 {table} / 조건 {col} ({r["quat_family"]})','EYE',QURL,'US6583181B1',f'Table {table}, 왼쪽부터 조건 {col}')
    observed=qo[(qo.formulation_id==cid)&(qo.day==7)].iloc[0]['draize_score']
    parts=[f'{r["quat_family"]} 유효성분 {fmt(r["quat_active_pct"])}%']
    for k in ['EDTA','NaOH','NaCl','glycolic']:
        if float(r[k+'_pct']): parts.append(f'{k} {fmt(r[k+"_pct"])}%')
    parts.append('물 '+(fmt(number(r['explicit_added_water_pct']))+'% 추가; 원료 stock 잔량 별도' if r['explicit_added_water_pct']!='' else 'q.s.; stock 용매 조성 미확인'))
    c.update(evidence_ids='Q_PATENT',primary_review='특허 배합·pH·평균 눈자극 시계열 확인',eye_epa_review='미확정—평균 Draize 총점으로 EPA 등급 부여 불가',
        label_evidence_type='특허 평균 눈자극 시계열; 공식 EPA 분류 없음',eye_observation=f'Day 7 평균 Draize 총점 {fmt(observed)}; 전체 시계열은 별도 표',
        composition_status='부분 확인—주요 투입량/유효성분 확인; stock 용매·중화 후 화학종 미확인',composition_summary=joined(parts),
        structure_status='명시 성분/동족체 명목 구조 대응; 중화 후 EDTA 화학종 및 pH 미반영',
        original_concentration_pct=float(r['quat_active_pct']),concentration_basis='배합 중 4급암모늄 유효성분 wt%; 전체 시험물질 희석률과 다른 값',
        concentration_interpretation='특허 배합 농도 확인; 별도 시험 희석률·노출량은 분리 확인 필요',test_dilution_status='표 배합의 시험으로 서술; 추가 희석 여부/실제 투여량 별도 미확정',
        vehicle_status='물 기반 배합; 상업 원료 stock의 잔여 용매 미확인',ph_low=float(r['pH']),ph_high=float(r['pH']),
        ph_status='원문 확인—시험 배합 pH',ph_context='해당 표의 배합 pH; 시험 직전·직후 측정 시점은 미기재',ph_source_url=QURL,ph_source_location=f'Table {table}, 조건 {col}, pH 행',
        represented_pct_low=float(r['represented_nonwater_pct']),represented_pct_high=float(r['represented_nonwater_pct']),represented_pct_mean=float(r['represented_nonwater_pct']),
        coverage_basis='이전 비수계 성분 입력 합의 제품 내 wt%; 물 제외. 배합의 나머지가 전부 미확인이라는 뜻은 아님',
        study_id=f'US6583181B1/Table{table}/Column{col} (내부 사례 식별자; 원시험 번호 아님)',grouping_key='PATENT:US6583181B1',
        formulation_study_match='특허 내부 표 대응 확인; 원시험 물질·배치 외부 확인 없음',training_independence='이전 15조건의 정확 특징 일치 0건; 원시험 독립성은 미확인',
        current_use='EYE 상대 순위·배합 변화 진단',eye_use='평균 자극점수 진단 가능; EPA 정확도 산출 보류',skin_use='해당 배합의 SKIN 결과 없음; 별도 피부 시험 필요',
        priority='P1',next_action='동물별 각 조직 점수·회복/부식 여부 및 원시험 번호 확보; stock 성분과 중화 상태·투여량 확인',
        notes='Tables 5–6 살균시험의 1/128 희석을 눈자극 시험 농도로 사용하지 않음. 임계값 조정 자료로 사용하지 않음.')
    if cid=='Q_T1_C2':c['conflict_flags']='1일 평균값: 표 41.7 / 본문 47.1 불일치; 기존 분석은 표 41.7 유지'
    cases.append(c)
    for _,v in qc[qc.formulation_id==cid].iterrows():
        rr=registry.loc[v['component_id']]
        components.append(dict(case_id=cid,component=rr['name'],cas=rr['cas'],pct_low=float(v['pct']),pct_high=float(v['pct']),
            amount_basis='특허 투입/동족체 비율로 계산한 명목 wt%; 혼합 후 화학종 비율 아님',role=v['role'],smiles=rr['smiles'],structure_status='원료 명목 구조',source_url=QURL,source_location=f'Table {table}, 조건 {col}'))

for i in range(6):
    cid=f'Q_T4_C{i+1}'; ph=[7,6.65,7.1,7,7.3,7][i]; active=[1.92]*5+[1.60]
    amounts={'nominal quat active':active[i],'EDTA':[6,0,4,4,0,6][i],'NaOH':[2.4,0,1.62,1.62,0,2.4][i],'NaCl':[0,6,0,0,4,0][i]}
    c=base(cid,'4급암모늄 특허',f'표 4 / 조건 {i+1} (상업원료 대응 보류)','EYE',QURL,'US6583181B1',f'Table 4, 왼쪽부터 조건 {i+1}')
    c.update(evidence_ids='Q_PATENT',primary_review='표 수치 확인; 병합 원료 머리글 대응 미확인',eye_epa_review='미확정—평균 총점만 있음',label_evidence_type='특허 눈자극 평균 총점',
        eye_observation=f'Day 7 평균 총점 {[5,4.3,9.5,12.3,2.8,2][i]}',composition_status='부분 확인—열별 수치 있음; 상업원료명-열 대응 미확인',
        composition_summary=joined([f'{k} {fmt(v)}%' for k,v in amounts.items() if v])+f' | 물 q.s.; 실측 quat actives {[2.06,1.94,2,2,2,1.71][i]}%',
        structure_status='보류—상업원료를 열에 임의 배정하지 않음',original_concentration_pct=active[i],concentration_basis='명목 quat 유효성분 wt%; 실측 함량은 배합 요약에 별도 기록',
        concentration_interpretation='명목 함량과 실측 actives를 구분; 시험 희석률 미확정',vehicle_status='물 q.s.; 원료 stock 용매 미확인',
        ph_low=ph,ph_high=ph,ph_status='원문 확인—시험 배합 pH',ph_context='표의 열 위치 기준 배합 pH; 원료 정체 대응은 보류',ph_source_url=QURL,ph_source_location=f'Table 4, 조건 {i+1}, pH 행',
        grouping_key='PATENT:US6583181B1',current_use='기존 예측 보류; 원료 열 대응 확인 후보',eye_use='EYE 관측치 있음; 입력/정답 보완 필요',skin_use='SKIN 결과 없음; 별도 피부 시험 필요',priority='P1',
        next_action='원본 PDF Table 4 병합 머리글의 제품명·열 대응 확인; 원료 조성 및 동물별 조직 점수 확보',conflict_flags='상업원료 머리글 대응 미확인',
        notes='수치 행을 좌→우로 기록했으나 특정 상업원료/SMILES에 연결하지 않음.')
    cases.append(c)
    for k,v in amounts.items():
        if v:components.append(dict(case_id=cid,component=k,cas='',pct_low=v,pct_high=v,amount_basis='Table 4 열 위치 기준 명목 wt%',role='table4_column_value',smiles='',structure_status='상업 원료 대응 및 화학종 보류',source_url=QURL,source_location=c['source_location']))

# Commercial ingredient dosages, not pure active-molecule dosages. Zero entries are absent in Table I.
recipe={
 'ALS/ALES-1 commercial blend (Table I footnote 3)':([52.5,0,0,0,0,0,0],'combined actives 27%; internal mixture composition not resolved'),
 'Ammonium lauryl sulfate stock':([0,0,41.03,41.03,41.03,0,43.14],'29% actives'),
 'Ammonium laureth-3 sulfate stock':([0,0,7.78,7.78,7.78,0,11.19],'27% actives; EO distribution unresolved'),
 'Sodium lauryl sulfate stock':([0,40.55,0,0,0,0,0],'29% actives'),
 'Sodium laureth-3 sulfate stock':([0,10.89,0,0,0,0,0],'27% actives; EO distribution unresolved'),
 'SLES-2/CAPB aqueous blend (footnote 11)':([0,0,0,0,0,27.59,0],'49% SLES-2 + 9% CAPB actives; molecular distributions unresolved'),
 'Ethylene oxide homopolymer POLYOX WSRN60-K':([.0117,0,0,0,0,0,0],'PEG-45M; average molecular weight about 2,000,000; no monodisperse SMILES'),
 'Oleamphopropyl sulfonate stock':([0,0,2.98,0,0,0,0],'47% actives'),
 'Cocamidopropyl hydroxysultaine stock':([0,0,0,3.26,0,0,0],'43% actives'),
 'Oleamidopropyl betaine stock':([0,0,0,0,5.45,0,0],'30% actives'),
 'Lauric/myristic DEA':([0,0,0,0,0,0,2],'100% actives; 1:1 blend'),
 'Ammonium chloride 20% aqueous stock':([5,0,2.35,3,2.4,0,1],'20% active salt; stock amount not salt amount'),
 'Ammonium chloride crystals':([0,2.60,0,0,0,0,0],'as added'),
 'Sodium chloride 20% aqueous stock':([0,0,0,0,0,3,0],'20% active salt; stock amount not salt amount'),
 'Citric acid 50% aqueous stock':([0,0,.39,.14,.17,0,.18],'50% active acid'),
 'Tetrasodium EDTA':([0,0,.25,.25,.25,0,.25],'as added; purity/hydrate not verified'),
 'Preservative solution':([0,0,.01,.01,.01,0,.01],'constituents unresolved'),
 'Dye solution':([2,2,0,0,0,.30,0],'constituents unresolved'),
 'Fragrance':([.5]*7,'constituents unresolved'),
 'Deionized water added':([39.9883,43.46,44.71,44.03,42.41,68.61,41.73],'added water only; excludes water inside stocks'),
}
sp=read('shampoo_patent_observations.csv')
for i in range(7):
    cid=f'SH_Example{i+1}'; ph=[None,4.64,5.61,5.39,5.15,7.52,5.78][i]
    r=sp.loc[sp.example==i+1].iloc[0]
    c=base(cid,'샴푸 특허',f'US4992266A Example {i+1}','EYE',SURL,'US4992266A',f'Table I 배합 / Table II 결과, Example {i+1}')
    c.update(evidence_ids='SH_PATENT',primary_review='배합·평균 눈자극 결과 확인',eye_epa_review='미확정—7일 평균 총점만으로 EPA 등급 재구성 불가',label_evidence_type='특허의 토끼 3마리 평균 Draize 점수',
        eye_observation=f'24h={r["draize_mean_24h"]}; 48h={r["draize_mean_48h"]}; 72h={r["draize_mean_72h"]}; day7={r["draize_mean_day7"]}',
        composition_status='투입 원료 배합표 확인; 상업 원료의 상세 분자 조성 일부 미해결',
        composition_summary=joined([f'{k}: {fmt(values[i])}%' for k,(values,note) in recipe.items() if values[i]]),
        structure_status='보류—계면활성제 분포·고분자·향료/색소 상세 조성 미해결',
        concentration_interpretation='표의 wt%는 상업 원료 투입량. 원료 활성분%와 곱해야 활성성분 함량이며 시험 희석률과 다름',concentration_basis='Table I 투입 원료 wt%와 각주 actives',
        test_dilution_status='배합별 Draize 시험으로 서술; 추가 희석·투여량 별도 미확정',vehicle_status='추가 물과 수용액 원료 있음; 전체 분자 배합 미해결',
        ph_low=ph,ph_high=ph,ph_status='원문 확인—시험 배합 pH' if ph is not None else '미보고—표의 대시',
        ph_context='Table I 제조 후 배합 pH; 시험 직전 측정 여부 미기재' if ph is not None else 'Example 1 pH는 대시; 다른 예시의 pH로 대체하지 않음',ph_source_url=SURL,ph_source_location=f'Table I, Example {i+1}, pH 행',
        study_id=f'US4992266A/Example{i+1} (내부 사례 식별자)',grouping_key='PATENT:US4992266A',formulation_study_match='특허 내부 배합과 점수 대응 확인; 외부 원시험/배치 확인 없음',
        current_use='기존 예측 보류; 원료 상세 조성 확인 후보',eye_use='EYE 평균점수 있음; 구조/정답 보완 필요',skin_use='해당 배합 SKIN 결과 없음; 별도 피부 시험 필요',
        next_action='원료별 상세 조성·분포/분자량과 시험 투여조건 확보; 동물별 조직점수·7일 이후 회복 자료 확인',
        notes='Examples 8–11의 예상 효과를 관측 결과로 추가하지 않음. 혼합 stock 함량을 각 성분에 중복 적용하지 않음.')
    if i==0:c['notes']+=' Example 1 배합은 비교 예와 염 종류·조성도 다르므로 고분자 단독 효과의 대조쌍으로 단정하지 않음.'
    cases.append(c)
    for k,(values,note) in recipe.items():
        if values[i]:components.append(dict(case_id=cid,component=k,cas='',pct_low=values[i],pct_high=values[i],amount_basis='상업 원료 투입 wt%; '+note,role='as_supplied_ingredient',smiles='',structure_status='본 목록에서 구조 배정하지 않음',source_url=SURL,source_location=f'Table I, Example {i+1}'))

for c in cases:
    label_sources=[s for s in sources if s['source_id'].startswith('LABEL_') and s['source_id'] in c['evidence_ids'].split(' | ')]
    c['epa_source_url']=label_sources[0]['url'] if label_sources else c['source_url']
    c['epa_source_location']=label_sources[0]['location'] if label_sources else c['source_location']
    c['repository_data_url']=REPO+'cir_ingredients.csv' if c['dataset']=='CIR' else (REPO+'flea_tick_shampoo.csv' if c['case_id'].startswith('SDS_') else '')
df=pd.DataFrame(cases)
comp=pd.DataFrame(components)
src=pd.DataFrame(sources).drop_duplicates('source_id')
additional=read('additional_sources.csv')
duplicates=additional[additional.id.str.startswith('D')].copy()
zein=read('zein.csv')
zein.insert(0,'case_id',[f'ZEIN_{i+1:03}' for i in range(len(zein))])
zein['current_use']='단백질 용해 지표 보조 분석만 가능; EPA EYE/SKIN 정답 아님'
zein['strict_epa_eligible']=False

# Capture availability in the frozen training metadata without changing any model input.
training_file=OLD/'training_reference/input_snapshot.csv'
train=pd.read_csv(training_file,usecols=['epa_class','test_concentration','test_concentration_units','test_concentration_status','known_pct','pct_sum_used'])
train_header=pd.read_csv(training_file,nrows=0).columns
training_summary=pd.DataFrame([
    dict(item='최종 학습 행',value=len(train),meaning='원래 학습자료의 보존 스냅샷'),
    dict(item='test_concentration 값 존재',value=int(train.test_concentration.notna().sum()),meaning='실제 시험농도 열; 성분 공개 함량 합과 다름'),
    dict(item='test_concentration_units 값 존재',value=int(train.test_concentration_units.notna().sum()),meaning='시험 농도 단위'),
    dict(item='pH 전용 열',value=int(any(str(c).lower() in ['ph','test_ph','mixture_ph'] for c in train_header)),meaning='분자 descriptor 이름 속 ph 문자열은 pH 측정 열이 아님'),
    dict(item='known_pct 값 존재',value=int(train.known_pct.notna().sum()),meaning='공개/표현된 성분 함량 합의 정보; 시험 노출농도로 간주 금지'),
])
manifest.append(dict(snapshot='training_metadata_read_only',original_path=str(training_file),sha256=hashlib.sha256(training_file.read_bytes()).hexdigest(),bytes=training_file.stat().st_size))

evidence=[]
for c in cases:
    for domain,value,status,url,loc in [
        ('EPA-EYE',c['eye_epa_review'],c['label_evidence_type'],c['epa_source_url'],c['epa_source_location']),
        ('EPA-SKIN',c['skin_epa_review'],c['label_evidence_type'],c['epa_source_url'],c['epa_source_location']),
        ('배합',c['composition_summary'],c['composition_status'],c['source_url'],c['source_location']),
        ('농도',c['concentration_interpretation'],c['test_dilution_status'],c['source_url'],c['source_location']),
        ('pH',fmt(c['ph_low'])+('–'+fmt(c['ph_high']) if c['ph_high']!=c['ph_low'] else ''),c['ph_status'],c['ph_source_url'],c['ph_source_location'])]:
        evidence.append(dict(case_id=c['case_id'],domain=domain,value=value,status=status,context=c['ph_context'] if domain=='pH' else '',source_ids=c['evidence_ids'],source_url=url,location=loc))
evdf=pd.DataFrame(evidence)
tasks=df[['case_id','dataset','name','priority','conflict_flags','next_action','source_url','grouping_key']].sort_values(['priority','dataset','case_id'])
overview_cols=['case_id','dataset','name','endpoints','eye_epa_review','skin_epa_review','composition_status','original_concentration_pct','concentration_interpretation','test_dilution_status','ph_low','ph_high','ph_status','ph_context','structure_status','current_use','next_action','priority','conflict_flags','source_url']
labels={'case_id':'사례ID','dataset':'자료군','name':'성분·제품·조건','endpoints':'시험 부위','eye_epa_review':'EYE EPA 근거 상태','skin_epa_review':'SKIN EPA 근거 상태','composition_status':'배합 확인 수준',
 'original_concentration_pct':'원자료 기재 함량(%)','concentration_interpretation':'함량 해석·시험농도 구분','test_dilution_status':'시험 희석농도 확인 상태','ph_low':'pH 하한/값','ph_high':'pH 상한/값','ph_status':'pH 확인 상태','ph_context':'pH의 대상·조건',
 'structure_status':'구조 표현 상태','current_use':'현재 활용 범위','next_action':'다음 확인 작업','priority':'우선순위','conflict_flags':'충돌·주의 항목','source_url':'원문 링크'}
overview=df[overview_cols].rename(columns=labels)
stats=dict(candidate_case_records=len(df),by_dataset={str(k):int(v) for k,v in df.dataset.value_counts().items()},eye_records=int(df.endpoints.str.contains('EYE').sum()),skin_records=int(df.endpoints.str.contains('SKIN').sum()),
    ph_primary_document_records=int(df.ph_status.str.startswith('원문 확인').sum()),ph_formulation_records=int((df.ph_status=='원문 확인—시험 배합 pH').sum()),ph_product_records=int((df.ph_status=='원문 확인—제품 pH').sum()),ph_csv_only_records=int((df.ph_status=='CSV 기재—원문 미확인').sum()),
    strict_epa_eligible=int(df.strict_epa_eligible.sum()),skin_label_conflicts=int(df.conflict_flags.str.contains('SKIN 원래').sum()),zein_excluded_records=len(zein),duplicate_source_records=len(duplicates),component_rows=len(comp),source_rows=len(src))
notes=[
 ('목적','EPA 근거·배합·시험농도·pH의 확인 범위를 사례별로 기록. 정식 평가 데이터셋 확정이나 모델 재학습은 수행하지 않음.'),
 ('사례 수','112행 = CIR 78 + 제품 SDS 6 + 4급암모늄 특허 21 + 샴푸 특허 7. 112개의 독립 연구를 뜻하지 않음. 제품 6행은 눈/피부 요약을 함께 보유.'),
 ('EPA 상태','기존 EPA 환산값은 original_* 열에 보존. 확정 정답 eye_epa_gold / skin_epa_gold는 미확정이므로 비워 둠. 라벨 문구 대리값과 확정 시험 등급을 혼합하지 않음.'),
 ('SKIN 충돌','Davis와 No-Bite: CSV SKIN III/0, 기본 등록 공식 피부 자극 문구는 II 대응/1. 원시험·배합 버전 미확인으로 자동 교정하지 않음.'),
 ('정식 검증 가능 0건의 뜻','이 목록에서 EPA 정답·시험물질 연결·학습자료와 원시험 독립성까지 확정한 사례가 0건. 효과가 없거나 자료가 모두 무가치하다는 뜻이 아님. pH 결측만을 탈락 기준으로 두지 않음.'),
 ('pH','원문 값 32건=특허 배합 27+SDS 제품 5. CSV에만 pH 4.5가 적힌 1건은 별도. 제품 pH와 시험 시료 pH의 일치 여부를 구분.'),
 ('농도','성분 배합 wt%, 상업 원료 투입량, 원료 active%, 시험물질 희석률은 서로 다름. 100%나 pH 7을 임의로 채우지 않음.'),
 ('배합','components의 함량 기준을 반드시 함께 읽을 것. stock 투입량과 분해한 활성 성분을 한 번에 합산하지 않음. represented_pct는 기존 모델에 대응한 성분 합으로서 자료 완전성의 단일 척도가 아님.'),
 ('출처','input_snapshot은 기존 CSV/JSON을 바이트 그대로 복사. sources의 웹 자료는 도구가 반환한 발췌문이며 원본 PDF가 아님. 원문 URL·버전·위치를 출처 시트에 보존.'),
 ('확인 수준','미확인=이번에 입증하지 못함. 미보고=확인한 문서에 값이 없음. 접근 실패=문서 내용을 확인할 수 없음. 저장소 high를 원문 검증 완료로 해석하지 않음.'),
 ('누출·독립성','이미 성능/사례를 살펴본 자료군이다. 이를 보고 모델이나 임계값을 바꾼 뒤 같은 자료에서 최종 외부 성능을 주장하지 않음. 학습 원시험과의 중복을 확인하고 새 자료군을 별도로 확보할 것.'),
 ('우선 작업','① SKIN 등급 충돌 및 SLS 구조 오류 확인 ② 특허 원시험별 조직점수·회복자료/stock 조성 확보 ③ CIR 원문 표·농도·기제 대조. 우선순위는 모델 정오 여부를 사용하지 않고 정함.'),
 ('기존 학습자료','673행 스냅샷에서 test_concentration/units 값 0건, pH 전용 열 없음. known_pct를 실제 시험농도로 바꾸어 부르지 않음.'),
 ('EYE와 SKIN','CIR은 해당 endpoint 행만 사용. SDS는 눈/피부 근거를 각각 확인. 두 특허 28배합의 EYE 점수를 SKIN 정답으로 전용하지 않음. Zein 205행은 양쪽 EPA 정답에서 제외.'),
]
readme=pd.DataFrame(notes,columns=['항목','설명'])
summary=pd.DataFrame([dict(항목=k,값=json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v) for k,v in stats.items()])
dictionary=pd.DataFrame([dict(field=k,meaning=labels.get(k,{
    'eye_epa_gold':'확정한 EYE EPA 정답; 미확정은 빈칸','skin_epa_gold':'확정한 SKIN EPA 정답; 미확정은 빈칸',
    'eye_label_proxy':'기본 등록의 눈자극 문구에 따른 대리값: 1=I/II, 0=III/IV; 확정 시험 정답 아님',
    'skin_label_proxy':'기본 등록 피부 자극 문구 대리값; GHS 환산 아님; 원시험 연결 미확정',
    'verified_test_concentration_pct':'실제 투여 시험물질 희석농도로 원문 확인한 경우만 입력; 현재 모두 미확정',
    'original_concentration_pct':'CIR은 저장소 농도, quat는 배합 유효성분 농도. concentration_basis를 반드시 함께 사용',
    'grouping_key':'같은 CAS/기본 등록/특허를 관리하는 그룹; 완전히 동일한 원시험이라는 뜻은 아님',
    'evidence_ids':'출처 시트 source_id로 연결; 원문 라벨과 SDS URL이 다를 수 있음',
    'strict_epa_eligible':'본 감사에서 정식 독립 EPA 평가 요건 확인 여부; pH 단독 필수조건 아님',
    'source_exposed':'이미 살펴본 자료군; 모델 변경 후 새 잠금 검증자료로 취급하지 않음',
    'represented_pct_mean':'이전 입력에서 표현한 비율의 범위 중앙값; 새 최적화나 정답 선정 기준 아님',
    'original_label_confidence':'원저장소의 신뢰도 텍스트를 보존; 본 감사의 확인 수준과 다름',
 }.get(k,k))) for k in df.columns])

tables={'먼저읽기':readme,'집계':summary,'사례목록':overview,'사례상세':df,'성분함량':comp,'근거목록':evdf,'출처':src,'다음확인작업':tasks,
    'Zein_별도보관':zein,'중복출처_3건':duplicates,'학습정보_확인범위':training_summary,'특허_눈점수':qo,'샴푸특허_눈점수':sp,'열설명':dictionary}
for name,data in [('검증용_데이터목록',df),('성분별_함량근거',comp),('사례별_근거목록',evdf),('출처목록',src),('다음_확인작업',tasks)]:
    data.to_csv(ROOT/(name+'.csv'),index=False,encoding='utf-8-sig')
with pd.ExcelWriter(ROOT/'검증용_데이터목록.xlsx',engine='openpyxl') as writer:
    for sheet,data in tables.items():
        data.to_excel(writer,index=False,sheet_name=sheet)
        ws=writer.sheets[sheet];ws.freeze_panes='D2' if sheet in ['사례목록','사례상세'] else 'A2'
        ws.auto_filter.ref=ws.dimensions
        for cell in ws[1]:cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='24495B');cell.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[1].height=32
        for colidx,column in enumerate(data.columns,1):
            sample=[str(column)]+[str(x) for x in data[column].head(80)]
            width=min(58,max(14,max(len(s) for s in sample)*.85))
            ws.column_dimensions[get_column_letter(colidx)].width=width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment=Alignment(vertical='top',wrap_text=True)
                if isinstance(cell.value,str):
                    if cell.value.startswith('https://') and ' | ' not in cell.value:
                        cell.hyperlink=cell.value;cell.font=Font(color='176BA0',underline='single')
                    if '충돌—' in cell.value or cell.value=='P1':cell.fill=PatternFill('solid',fgColor='FFE6DD')
                    elif cell.value.startswith('원문 확인—'):cell.fill=PatternFill('solid',fgColor='DDEFE3')
                    # Keep literal source strings, including those beginning '=', as text.
                    if cell.value.startswith(('=','+','-','@')):cell.data_type='s'
            ws.row_dimensions[row[0].row].height=60 if sheet not in ['집계','성분함량','출처'] else 48
        ws.sheet_view.zoomScale=80

plain='검증용 데이터 목록 — '+DATE+'\n\n'+'\n\n'.join(k+'\n'+v for k,v in notes)
(ROOT/'읽어주세요.txt').write_text(plain,encoding='utf-8-sig')
(ROOT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

# A standalone, offline report; all data and filtering remain local.
columns=['case_id','dataset','name','endpoints','eye_epa_review','skin_epa_review','composition_status','concentration_interpretation','ph_low','ph_high','ph_status','ph_context','conflict_flags','next_action']
records=df[columns].where(pd.notna(df[columns]),None).to_dict('records')
payload=json.dumps(records,ensure_ascii=False).replace('</','<\\/')
notes_html=''.join('<p><b>'+html.escape(k)+'</b>　'+html.escape(v)+'</p>' for k,v in notes[:8])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>검증용 데이터 목록</title>
<style>body{font-family:Malgun Gothic,Arial,sans-serif;background:#f3f6f8;color:#203640;margin:28px}h1{font-size:26px}p{line-height:1.7}main{max-width:1600px;margin:auto}.cards{display:flex;gap:14px;flex-wrap:wrap}.card,section{background:white;padding:20px;border-radius:10px;margin:15px 0}.card strong{font-size:28px;display:block}input,select{padding:12px;margin:6px;border:1px solid #b7c8d0;border-radius:6px}table{border-collapse:collapse;background:white;width:100%}th,td{padding:10px;border:1px solid #dce4e8;vertical-align:top;font-size:13px;min-width:170px}th{background:#24495b;color:white;position:sticky;top:0}a{color:#176ba0}.scroll{overflow:auto;max-height:720px}.conflict{background:#fff0e7}small{color:#567}</style>
<main><h1>EPA 근거·배합·농도·pH 검증 목록</h1><p>2026-11-09 · 확인 상태를 기록한 후보 목록입니다. 독립 EPA 정답 데이터셋으로 확정한 자료는 아직 없습니다.</p>
<div class="cards"><div class="card"><strong>112</strong>후보 사례 행</div><div class="card"><strong>32</strong>원문 pH 값: 배합 27 + 제품 5</div><div class="card"><strong>2</strong>SKIN 등급 근거 충돌</div><div class="card"><strong>0</strong>정식 독립 EPA 평가 요건 확인</div></div>
<p><a href="검증용_데이터목록.xlsx">Excel 전체 목록</a> · <a href="검증용_데이터목록.csv">CSV 상세</a> · <a href="다음_확인작업.csv">다음 확인 작업</a> · <a href="출처목록.csv">원문 출처</a></p>
<section><b>먼저 확인할 사항</b>__NOTES__</section><section><input id="search" placeholder="성분·제품·사례ID 검색" size="34"><select id="group"><option value="">모든 자료군</option><option>CIR</option><option>제품 SDS</option><option>4급암모늄 특허</option><option>샴푸 특허</option></select><select id="endpoint"><option value="">EYE + SKIN 전체</option><option>EYE</option><option>SKIN</option></select><label><input type="checkbox" id="conflict">충돌·주의 항목만</label><p id="count"></p><div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div></section></main>
<script>const data=__DATA__;const labels=__LABELS__;const cols=__COLS__;const head=document.getElementById('head');const tr=document.createElement('tr');cols.forEach(k=>{let t=document.createElement('th');t.textContent=labels[k]||k;tr.appendChild(t)});head.appendChild(tr);function render(){const q=document.getElementById('search').value.toLowerCase(),g=document.getElementById('group').value,e=document.getElementById('endpoint').value,conf=document.getElementById('conflict').checked;const rows=data.filter(r=>(!q||JSON.stringify(r).toLowerCase().includes(q))&&(!g||r.dataset===g)&&(!e||r.endpoints.includes(e))&&(!conf||r.conflict_flags));document.getElementById('count').textContent=rows.length+' / 112 사례 행';const b=document.getElementById('body');b.replaceChildren();for(const r of rows){const t=document.createElement('tr');if(r.conflict_flags)t.className='conflict';for(const k of cols){const d=document.createElement('td');d.textContent=r[k]??'미확인';t.appendChild(d)}b.appendChild(t)}}['search','group','endpoint','conflict'].forEach(id=>document.getElementById(id).addEventListener('input',render));render();</script></html>'''
page=page.replace('__NOTES__',notes_html).replace('__DATA__',payload).replace('__LABELS__',json.dumps(labels,ensure_ascii=False)).replace('__COLS__',json.dumps(columns))
(ROOT/'검증목록_요약.html').write_text(page,encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
