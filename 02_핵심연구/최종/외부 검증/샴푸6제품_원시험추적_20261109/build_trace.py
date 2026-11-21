"""Trace six shampoo products to regulatory records, study citations and formula versions.
No new predictions or EPA ground-truth assignments. External requests are drafts only.
"""
from pathlib import Path
import hashlib, json, html, re
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
CAT=BASE/'검증데이터_목록_20261109'
DATE='2026-11-09'
PPLS='https://www3.epa.gov/pesticides/chem_search/ppls/'
BIB='https://www3.epa.gov/pesticides/nas/final/diazinon/appendix-2-4.pdf'
GUIDE='https://www.epa.gov/foia/guide-foia-requests-pesticide-information'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def url(file):return PPLS+file+'.pdf'
catalog_path=CAT/'검증용_데이터목록.csv'
catalog=pd.read_csv(catalog_path).fillna('')
products=catalog[catalog.dataset=='제품 SDS'].copy()
prior_sources=pd.read_csv(CAT/'출처목록.csv').fillna('')
manifest=[dict(path=str(p),sha256=sha(p)) for p in [catalog_path,CAT/'출처목록.csv']]
(ROOT/'input_snapshot').mkdir(exist_ok=True)
for p in [catalog_path,CAT/'출처목록.csv']:(ROOT/'input_snapshot'/p.name).write_bytes(p.read_bytes())

documents=[]
def doc(sid,family,date,filename,pages,kind,location,findings,access='웹 도구의 PDF 텍스트 확인'):
    documents.append(dict(source_id=sid,registration_family=family,document_date=date,source_url=url(filename) if filename else '',pdf_pages=pages,document_type=kind,location=location,findings=findings,access_status=access))
doc('D10','69061-12','2010-06-03','069061-00012-20100603',6,'명칭 변경 통지','PDF 1쪽 / 4쪽 라벨','SIVAD .15% Pyrethrin Flea & Tick Shampoo에서 SIVAD Flea & Tick Shampoo로 이름 변경. OCR의 15% 표기를 성분 농도로 사용하지 않음.')
doc('D15','69061-12','2015-09-10','069061-00012-20150910',5,'재등록 관련 승인서·라벨','PDF 1쪽 CSF / 3쪽 활성성분 / 4쪽 자극 문구','Basic와 Alternate CSF 1 모두 2012-07-19. 결정번호 414109, 411781, 410235. 공식 라벨 제조사 Davis.')
doc('N00','1021-1723','2000-02-15','001021-01723-20000215',3,'최초 조건부 등록 통지','PDF 1쪽 2(b)–(c), 2쪽','등록명 Pyrocide Flea and Tick Shampoo 51921. 1998년 10월 서신을 언급하며 눈자극 경고 수정 요구. 희석 제품의 눈자극 문구 허용 가능성을 조건부로 언급하나 시험농도 수치 없음.')
doc('N13','1021-1723','2013-12-24','001021-01723-20131224',5,'라벨 통지·신청서','PDF 3쪽 CSF 변경 없음 진술 / 4쪽 성분·눈·피부 문구','2013-11-22 신청은 보관·폐기 문구 추가이며 CSF 변경 없음이라고 진술. CSF 날짜·식별자와 1997 시험 배합 연결은 미기재.')
doc('A12','270-324','2012-03-01','000270-00324-20120301',4,'라벨 수정 승인','PDF 1쪽 / 2–3쪽 라벨','응급처치 순서 변경. D460844, D461738은 문서의 행정 참조번호로만 보존; MRID 아님.')
doc('A18','270-324','2018-01-10','000270-00324-20180110',5,'독성분류 관련 라벨 수정','PDF 1쪽 / 4쪽','독성범주를 반영하는 응급처치·주의문구 수정; Decision 533083. 눈자극 문구는 II 대응. 시험 MRID나 동물별 점수 없음.')
doc('A25','270-324','2025-06-25','000270-00324-20250625',5,'재평가 관련 라벨 수정','PDF 1쪽 / 3쪽 활성성분·브랜드','Pyriproxyfen 관련 변경, Decision 567010 / Case 473758. Adams Flea & Tick Cleansing Shampoo 브랜드명을 명시; 활성성분은 기존 수치와 일치. 2023 SDS 배치 일치를 증명하지 않음.')
doc('O12','2724-485','2012-06-19','002724-00485-20120619',None,'재등록 관련 승인서 단서','검색 색인에 노출된 승인서 첫 페이지','Basic/Alternate CSF 모두 2010-12-21. Decision 412327, 410679. acute toxicity 및 product chemistry reviews 동봉 언급.','검색 색인 발췌 확인; PDF 본문 열기는 두 차례 timeout')
doc('O18','2724-485','2018-04-05','002724-00485-20180405',6,'대체 배합 추가 승인','PDF 1쪽 / 3쪽 라벨','거품형 대체 배합 추가. Basic 2015-01-07, Alternate 1 2015-01-07, Alternate 2 2017-10-06. Decision 534916, 534918.')
doc('O19','2724-485','2019-05-23','002724-00485-20190523',None,'향 관련 라벨 통지','검색 색인에 노출된 승인서 첫 페이지','기존 대체 CSF에 따른 aloe/cucumber scent 문구 추가; Decision 547177. 배합 전체가 공개된 것은 아님.','EPA PDF 검색 색인 발췌 확인')
doc('O26','2724-485','2026-05-18','002724-00485-20260518',6,'라벨 사진 추가 통지','PDF 1쪽 / 3쪽','선택적인 개·고양이 사진 추가. Action Case 00692317; 눈의 moderate 문구 유지. 새 자극시험 결과로 해석하지 않음.')
doc('Z08','2724-792','2008-04-26','002724-00792-20080426',7,'최초 조건부 등록 통지','PDF 4쪽 동봉목록','Acute Toxicity Review 2008-04-09, Product Chemistry Review 2008-03-11를 동봉했다고 명시. 공개 PDF는 승인서 4쪽과 라벨 3쪽으로 검토보고서 본문은 포함되지 않음. 날짜는 PPLS 파일명/색인 기준.')
doc('Z15','2724-792','2015-01-29','002724-00792-20150129',5,'재등록 관련 승인서·라벨','PDF 2쪽 동봉 언급 / 3–4쪽 라벨','Decision 410827, 412069. acute toxicity 및 product chemistry reviews 동봉 언급; 공개 PDF에는 검토보고서 본문 없음.')
documents.append(dict(source_id='BIB51921',registration_family='1021-1723 연결 후보',document_date='2015-10-01 (서지 출력일)',source_url=BIB,pdf_pages=222,document_type='EPA 공개 서지',location='PDF 196쪽, MRID 44539100–08; 197쪽 09 이어짐',findings='Pyrocide Flea & Tick Shampoo 51921라는 동일 시험제품명과 MRID/실험실/연도/페이지 수 확인. 결과·농도·CSF 연결은 없음. 호스팅 경로가 diazinon이라고 해서 샴푸에 diazinon이 들어 있다고 판단하지 않음.',access_status='웹 PDF 텍스트 확인; 페이지 스크린샷은 cache miss'))
documents.append(dict(source_id='FOIA_GUIDE',registration_family='공통',document_date='2026-11-09 확인',source_url=GUIDE,pdf_pages=None,document_type='EPA 자료 접근 안내',location='Tips Related to FOIA Requests',findings='등록번호·MRID·문서종류/날짜로 필요한 자료를 특정하는 근거. 요청문안만 작성했으며 제출/발송하지 않음.',access_status='공식 웹페이지 확인'))
for _,s in prior_sources[prior_sources.source_id.str.startswith('SDS_')].iterrows():
    documents.append(dict(source_id=s.source_id,registration_family='제품별 연결표 참조',document_date=s.document_version,source_url=s.url,pdf_pages=None,document_type='기존 제품 SDS',location=s.location,findings=s.limitations,access_status='이전 목록의 확인 상태 유지: '+s.review_status))

studies=[]
for mrid,endpoint,project,year,author,lab,pages in [
 ('44539104','EYE','5145',1997,'G. Wnorowski','Product Safety Labs',26),
 ('44539105','EYE','5536',1997,'G. Wnorowski','Product Safety Labs',25),
 ('44539106','SKIN','5146',1997,'G. Wnorowski','Product Safety Labs',17),
 ('44539108','PRODUCT_CHEMISTRY','',1996,'W. Grise','McLaughlin Gormley King Co.',7)]:
    studies.append(dict(mrid=mrid,endpoint=endpoint,lab_project_number=project,study_year=year,author=author,laboratory=lab,reported_pages=pages,
        test_product_name='Pyrocide Flea & Tick Shampoo 51921',submission_root='44539100 (1998 transmittal; 원시험 아님)',
        candidate_family='1021-1723',candidate_case_id='SDS_1021_1723_12281',link_basis='EPA 서지의 시험제품명과 2000/2013 등록문서의 기본 등록명을 연결',
        link_certainty='기본 제품명 연결 후보; 현행 No-Bite 배합/배치와 일치 미확정',source_ids='BIB51921 | N00 | N13',source_url=BIB,source_location='PDF 196쪽',
        full_report_obtained=False,result_obtained=False,eye_epa_gold='',skin_epa_gold='',test_concentration='',test_ph='',
        required_next='원시험 본문 또는 EPA DER 확보; 시험물질 lot/CSF/희석·세척조건과 SDS 연결'))

review_leads=[]
for rid,fam,title,date,sid,certainty in [
 ('R_Z08_TOX','2724-792','Acute Toxicity Review','2008-04-09','Z08','명칭·날짜·기본 등록번호 확인'),
 ('R_Z08_CHEM','2724-792','Product Chemistry Review','2008-03-11','Z08','명칭·날짜·기본 등록번호 확인'),
 ('R_Z15_TOX','2724-792','Acute toxicity review enclosed with reregistration letter','','Z15','2015 승인서에서 동봉 언급; 검토보고서 작성일과 MRID 미상'),
 ('R_Z15_CHEM','2724-792','Product chemistry review enclosed with reregistration letter','','Z15','2015 승인서에서 동봉 언급; 검토보고서 작성일과 MRID 미상'),
 ('R_O12_TOX','2724-485','Acute toxicity review enclosed with reregistration letter','','O12','2012 승인서 검색 색인에 동봉 언급; 본문 미확보'),
 ('R_O12_CHEM','2724-485','Product chemistry review enclosed with reregistration letter','','O12','2012 승인서 검색 색인에 동봉 언급; 본문 미확보')]:
    review_leads.append(dict(lead_id=rid,registration_family=fam,review_title=title,review_date=date,mrid='',source_id=sid,
        confirmation_scope=certainty,full_review_obtained=False,distinct_review_verified=False,
        note='각 동봉 언급이 서로 다른 검토보고서인지 미확정. 날짜를 원시험일로 사용하지 않음.'))

formulas=[]
for fam,kind,date,sid,status in [
 ('69061-12','Basic','2012-07-19','D15','등록문서에서 존재/날짜 확인'),
 ('69061-12','Alternate 1','2012-07-19','D15','등록문서에서 존재/날짜 확인'),
 ('2724-485','Basic','2010-12-21','O12','과거 승인서 검색 색인만 확인'),
 ('2724-485','Alternate (번호 미기재)','2010-12-21','O12','과거 승인서 검색 색인만 확인'),
 ('2724-485','Basic','2015-01-07','O18','등록문서에서 존재/날짜 확인'),
 ('2724-485','Alternate 1','2015-01-07','O18','등록문서에서 존재/날짜 확인'),
 ('2724-485','Alternate 2','2017-10-06','O18','등록문서에서 존재/날짜 확인; 거품형 추가 승인과 연결')]:
    formulas.append(dict(registration_family=fam,csf_type=kind,csf_date=date,source_id=sid,confirmation_scope=status,csf_contents_obtained=False,retail_product_csf_match_verified=False,study_csf_match_verified=False))

family_info={
 '69061-12':dict(name='Sivad Flea & Tick Shampoo',documents='D10 | D15',ids='414109 | 411781 | 410235',eye='',skin='',chem='',report='개별 시험 MRID 미확인; 2015 재등록 행정기록으로 후속 요청 가능',
    csf='Basic 2012-07-19; Alternate 1 2012-07-19; 내용 미확보',active='pyrethrins 0.15%; PBO technical 0.30%; MGK264 0.50%',
    match='공식 기본 라벨에 Davis 제조사 명시. 기존 CSV는 PBO/MGK264 함량 누락; 2025 SDS와 2012 CSF의 연결 미확정',
    gap='원시험/DER·CSF 내용을 확보하지 못함. No-Bite와 PII 4.3이 같다는 사실만으로 같은 시험이라고 연결하지 않음.',
    next='2015 결정번호 3개에 연결된 눈·피부 DER/시험 MRID 및 2012 Basic/Alternate 배합 중 해당 제품 버전 확인',priority=3),
 '1021-1723':dict(name='Pyrocide Flea & Tick Shampoo 51921',documents='BIB51921 | N00 | N13',ids='2000-02-15 등록 통지; 2013-11-22 신청 / 2013-12-24 통지',eye='44539104 | 44539105',skin='44539106',chem='44539108',
    report='EYE 2건·SKIN 1건·제품화학 1건의 MRID 서지 확인; 본문 미확보',csf='2013 신청서에서 CSF 변경 없음 진술. CSF 날짜/식별자 미확인',
    active='pyrethrins 0.150%; PBO technical 0.300%; MGK264 0.479%',match='공식 기본 등록명→서지 시험명 연결 가능. 1997 시험시료→2015 No-Bite SDS의 배합·농도·배치 연결 미확정',
    gap='두 눈 시험의 차이(희석, 세척, 배합, 재시험 등) 미확인. 2000 서신의 희석 언급을 특정 MRID의 시험농도로 배정하지 않음.',
    next='MRID 44539104/05/06의 본문 또는 DER와 44539108 제품화학자료 확보; 두 눈 시험의 농도/세척 차이 및 현재 제품 연결 확인',priority=1),
 '270-324':dict(name='Farnam 2009 Insecticide',documents='A12 | A18 | A25',ids='D460844 | D461738 | Decision 533083 | Decision 567010 | Case 473758',eye='',skin='',chem='',
    report='2018 독성분류 관련 라벨 변경 이력 확보; 개별 시험 MRID 미확인',csf='확인한 승인서에서 CSF 버전 날짜 미확인',active='pyrethrins 0.075%; PBO technical 0.750%; pyriproxyfen 0.086%',
    match='2012/2025 기본 라벨 활성성분과 기존 CSV 활성성분 일치. 전체 조성·2023 SDS와 시험 배치 일치는 미확정',
    gap='활성성분 수치 일치만으로 계면활성제·용매까지 동일하다고 판단할 수 없음. 행정번호는 원시험 번호가 아님.',
    next='2018 Decision 533083의 독성검토/DER와 인용 MRID 확보; 2023 SDS에 해당하는 CSF 버전 확인',priority=3),
 '2724-485':dict(name='RF-9414 Shampoo',documents='O12 | O18 | O19 | O26',ids='412327 | 410679 | 534916 | 534918 | 547177 | Action Case 00692317',eye='',skin='',chem='',
    report='2012 독성·제품화학 검토보고서 동봉 단서; 2018 대체 배합 추가 확인; MRID 미확인',
    csf='2010-12-21 Basic/Alternate(과거 색인); 2015-01-07 Basic/Alt1; 2017-10-06 Alt2(거품형 추가)',
    active='S-methoprene 0.10%; pyrethrins 0.15%; PBO technical 1.88%',
    match='두 브랜드가 같은 기본 등록번호를 공유. 2016 Zodiac/2020 Adams SDS가 어느 CSF인지 미확정; 2017 신규 배합을 2016 SDS에 소급 연결하지 않음',
    gap='2018 거품형, 2019 대체 배합의 향 문구, 2026 사진 추가를 새 눈자극 시험으로 세지 않음. 두 브랜드를 독립 시험 2개로 세지 않음.',
    next='2012 결정기록의 독성/제품화학 검토 및 MRID 확보; 두 브랜드 각각 Basic/Alt1/Alt2와 연결; 배합 변경 시 시험자료 적용 근거 확인',priority=2),
 '2724-792':dict(name='RF2066 Shampoo',documents='Z08 | Z15',ids='2015 Decision 410827 | 412069',eye='',skin='',chem='',
    report='Acute Toxicity Review 2008-04-09 / Product Chemistry Review 2008-03-11의 존재·날짜 확인; 본문 미확보',csf='확인한 승인서에서 CSF 버전 날짜 미확인',
    active='pyrethrins 0.05%; PBO technical 0.50%',
    match='등록번호·활성성분을 통해 기본 제품 연결. 2016 SDS와 2008/2015 검토 시료의 전체 배합·배치 일치 미확정',
    gap='공개 라벨 PDF에 동봉 검토보고서 본문이 없음. 2015 동봉 언급이 2008 보고서 재사용인지 새 검토인지 미확정.',
    next='2008-04-09 급성독성 및 2008-03-11 제품화학 검토보고서 확보; 2015 결정문 인용 MRID/적용 배합 연결',priority=1),
}

rows=[]
for _,p in products.iterrows():
    family=p.grouping_key.split(':',1)[1];f=family_info[family]
    prior_sds=next(s for _,s in prior_sources.iterrows() if s.source_id.startswith('SDS_') and s.url==p.source_url)
    rows.append(dict(case_id=p.case_id,product=p['name'],reported_registration=p.cas_or_registration,basic_registration=family,registered_product_name=f['name'],
        sds_date=prior_sds.document_version,sds_url=p.source_url,registration_connection='SDS/기본 라벨의 등록번호·명칭으로 연결; 시험시료까지 연결한 것은 아님',
        document_ids=f['documents']+' | '+prior_sds.source_id,administrative_ids=f['ids'],eye_mrid_candidates=f['eye'],skin_mrid_candidates=f['skin'],product_chemistry_mrid_candidate=f['chem'],
        original_study_progress=f['report'],csf_versions=f['csf'],public_active_ingredients=f['active'],formulation_connection=f['match'],
        eye_label_proxy=p.eye_label_proxy,eye_epa_evidence=p.eye_epa_review,skin_label_proxy=p.skin_label_proxy,skin_epa_evidence=p.skin_epa_review,
        eye_epa_gold='',skin_epa_gold='',test_concentration='',test_ph='',product_ph_low=p.ph_low,product_ph_high=p.ph_high,ph_scope=p.ph_context,
        full_eye_report_obtained=False,full_skin_report_obtained=False,full_csf_obtained=False,study_to_current_product_match_verified=False,
        formal_epa_validation_eligible=False,source_study_independence='학습 메타데이터의 명칭·번호 검색 일치 없음; 원시험 독립성은 입증되지 않음',
        remaining_gap=f['gap'],next_document_to_obtain=f['next'],priority=f['priority'],status='추적 단서 확보; 확정 정답/정식 평가 보류',checked_date=DATE))
trace=pd.DataFrame(rows)
studies_df=pd.DataFrame(studies); reviews_df=pd.DataFrame(review_leads); formulas_df=pd.DataFrame(formulas); docs_df=pd.DataFrame(documents)

# A separate identifier namespace prevents administrative decisions becoming trial labels.
requests=[]
for fam,f in family_info.items():
    requests.append(dict(priority=f['priority'],registration_family=fam,product_name=f['name'],
        study_identifiers=' | '.join(x for x in [f['eye'],f['skin'],f['chem']] if x),
        administrative_identifiers=f['ids'],documents_needed=f['next'],
        fields_needed='시험물질명·lot·CSF 연결, 실제 시험농도/vehicle, 세척/노출시간, 시험 pH(있으면), 동물별 조직 점수·회복, EPA endpoint별 등급·판정근거',
        preferred_source='EPA 공개 가능 DER/급성독성 검토 및 원시험; 등록사 보유 배합-시험 연결자료',
        request_status='문안 준비만 완료; 제출/발송하지 않음',guidance_url=GUIDE))
requests_df=pd.DataFrame(requests).sort_values(['priority','registration_family'])

train_path=BASE/'Eye_cosmetic_validation_20261109/training_reference/input_snapshot.csv'
train_cols=['formulation_id','formulation_name','source','source_reference','source_url','source_formulation_id','epa_class']
train=pd.read_csv(train_path,usecols=train_cols,low_memory=False).fillna('')
terms=['44539104','44539105','44539106','44539108','51921','RF2066','RF-9414','RF9414','69061','1021-1723','270-324','2724-485','2724-792','shampoo','Sivad','Farnam']
overlap=[]
for term in terms:
    mask=train.astype(str).apply(lambda col:col.str.contains(term,case=False,regex=False)).any(axis=1)
    overlap.append(dict(query=term,metadata_columns=' | '.join(train_cols),matched_rows=int(mask.sum()),matched_ids=' | '.join(train.loc[mask,'formulation_id'].astype(str)),
        interpretation='일치 없음은 독립성의 증명이 아님. 원시험 MRID가 학습자료에 없거나 익명화되었을 수 있음.'))
overlap_df=pd.DataFrame(overlap)
manifest.append(dict(path=str(train_path),sha256=sha(train_path)))

search_log=[]
for file in sorted((ROOT/'sources').glob('shampoo_trace_*.json')):
    value=read_json(file)
    request=value.get('request',{})
    for kind,ops in request.items():
        if kind=='response_length':continue
        for op in ops:
            search_log.append(dict(source_log=file.name,operation=kind,query_or_reference=op.get('q',op.get('ref_id','')),details=json.dumps(op,ensure_ascii=False)))
search_log_df=pd.DataFrame(search_log)

summary={
 'products_traced':len(trace),'basic_registration_families':trace.basic_registration.nunique(),
 'families_with_named_study_mrid_candidates':int(trace.loc[trace.eye_mrid_candidates!='','basic_registration'].nunique()),
 'eye_study_mrid_citations':int((studies_df.endpoint=='EYE').sum()),'skin_study_mrid_citations':int((studies_df.endpoint=='SKIN').sum()),
 'product_chemistry_mrid_citations':int((studies_df.endpoint=='PRODUCT_CHEMISTRY').sum()),
 'dated_review_leads':int((reviews_df.review_date!='').sum()),'csf_version_records':len(formulas_df),
 'csf_records_verified_in_pdf_text':int((formulas_df.source_id!='O12').sum()),
 'full_original_eye_skin_reports_obtained':0,'full_review_documents_obtained':0,'formal_epa_validation_eligible':0,
 'training_metadata_name_id_matches':int(overlap_df.matched_rows.sum()),'new_predictions':False,'labels_changed':False,'external_requests_sent':False,
}
readme=pd.DataFrame([
 ('핵심 결과','6제품을 5개 기본 등록계열로 연결. No-Bite 기본 제품에서 EYE 2건, SKIN 1건, 제품화학 1건의 MRID 서지 발견. RF2066은 검토보고서 2건의 날짜까지 확인.'),
 ('확인된 것과 미확인','MRID 서지·문서 동봉 언급·CSF 날짜는 확인했지만, 원시험/검토보고서 본문과 전체 CSF는 확보하지 못함. 공개 검색 범위에서 미확보라는 뜻이며 문서가 존재하지 않거나 전부 비공개라는 뜻이 아님.'),
 ('6제품 ≠ 6독립시험','Zodiac Oatmeal과 Adams Plus는 2724-485를 공유. 한 등록번호 안에 여러 CSF가 있으며, 같은 번호를 곧 동일 시험/완전 동일 배합으로 간주하지 않음.'),
 ('정답','기존 라벨 대리값은 유지. eye_epa_gold/skin_epa_gold는 빈칸. 원시험 본문을 보지 않은 채 MRID만으로 등급·희석농도를 부여하지 않음.'),
 ('No-Bite','44539104와 44539105는 서로 다른 눈 시험의 서지. 두 시험이 희석·세척·배합·재시험 중 어떤 차이인지 미확인. 원하는 결과의 시험만 고르지 않음.'),
 ('피부 충돌','Davis/No-Bite의 CSV SKIN III와 공식 라벨의 II 대응 피부 문구 충돌은 해소되지 않음. PII 4.3 수치가 같은 이유만으로 원시험을 공유한다고 판단하지 않음.'),
 ('배합 버전','2018 RF-9414 승인서는 거품형 Alternate 2 추가를 명시. 2016 Zodiac SDS를 2017 신규 CSF에 소급 배정하지 않음. 2020 Adams SDS는 어느 CSF인지 추가 확인 필요.'),
 ('보고서 종류','MRID=EPA 제출 문서 식별자, Lab Project=실험실 번호, Decision/Case=행정번호, CSF 날짜=배합문서 버전. 검토보고서 작성일은 원시험 실시일이 아님.'),
 ('pH','제품 SDS pH 5건은 이전 목록에서 가져옴. 이번 추적으로 원시험 pH를 새로 확정한 사례는 없음. No-Bite SDS pH는 접근 실패 상태 유지.'),
 ('자료 범위','기본 등록명·활성성분·공개 라벨 이력을 우선 추적. 최신 문서를 과거 SDS에 무조건 대입하지 않음. 이 파일에 기록된 날짜는 확인한 문서 날짜이며 전체 변경 이력을 빠짐없이 확보했다고 주장하지 않음.'),
 ('중복 확인','학습 673행의 보존 메타데이터에서 이번 식별자/명칭 검색 일치 0건. 원시험 번호가 없는 학습 행이 있어 독립성 입증은 아님.'),
 ('출처 보존','sources에는 도구가 반환한 검색/PDF 텍스트 발췌와 요청 조건을 저장. 원본 PDF를 내려받았다는 뜻은 아님. O12는 PDF 접근 실패로 검색 색인 확인 수준을 명시.'),
 ('다음 실행 순서','① No-Bite의 MRID 4건 및 해당 DER ② RF2066의 날짜가 확인된 독성/제품화학 검토 ③ RF-9414의 2012 검토·브랜드별 CSF 연결 ④ Davis·Farnam 행정기록으로 시험번호 확인.'),
 ('발송 여부','자료 요청문안은 검토용으로만 작성. EPA나 제조사에 연락·제출하지 않았음.'),
],columns=['항목','설명'])

rename={'case_id':'사례ID','product':'제품','basic_registration':'기본 등록번호','registered_product_name':'공식 등록명','sds_date':'SDS 날짜',
 'eye_mrid_candidates':'EYE 원시험 MRID 후보','skin_mrid_candidates':'SKIN 원시험 MRID 후보','original_study_progress':'원시험 추적 결과','csf_versions':'확인한 배합문서 버전',
 'formulation_connection':'배합 연결 수준','remaining_gap':'아직 모르는 부분','next_document_to_obtain':'다음 확보 문서','priority':'확보 우선순위','status':'현재 상태'}
short=trace[list(rename)].rename(columns=rename)
tables={'먼저읽기':readme,'제품별연결':short,'제품별상세':trace,'원시험MRID':studies_df,'검토보고서단서':reviews_df,'배합버전':formulas_df,
 '문서이력_출처':docs_df,'자료확보목록':requests_df,'학습중복_검색':overlap_df,'검색기록':search_log_df}
for name,table in [('샴푸6제품_원시험연결표',trace),('원시험_MRID_목록',studies_df),('검토보고서_단서',reviews_df),('배합문서_버전',formulas_df),('자료확보_요청목록',requests_df),('출처_문서이력',docs_df),('학습자료_식별자검색',overlap_df),('검색기록',search_log_df)]:
    table.to_csv(ROOT/(name+'.csv'),index=False,encoding='utf-8-sig')
with pd.ExcelWriter(ROOT/'샴푸6제품_원시험연결표.xlsx',engine='openpyxl') as writer:
    for name,data in tables.items():
        data.to_excel(writer,sheet_name=name,index=False);ws=writer.sheets[name]
        ws.freeze_panes='C2' if name.startswith('제품별') else 'A2';ws.auto_filter.ref=ws.dimensions;ws.sheet_view.zoomScale=80
        for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='24495B');cell.alignment=Alignment(wrap_text=True)
        ws.row_dimensions[1].height=32
        for i,column in enumerate(data.columns,1):
            ws.column_dimensions[get_column_letter(i)].width=min(62,max(16,max(len(str(x)) for x in [column]+data[column].head(40).tolist())*.8))
        for row in ws.iter_rows(min_row=2):
            ws.row_dimensions[row[0].row].height=90 if name in ['제품별연결','제품별상세','먼저읽기','자료확보목록'] else 60
            for cell in row:
                cell.alignment=Alignment(wrap_text=True,vertical='top')
                if isinstance(cell.value,str):
                    if cell.value.startswith('https://'):
                        cell.hyperlink=cell.value;cell.font=Font(color='176BA0',underline='single')
                    if cell.value.startswith(('=','+','-','@')):cell.data_type='s'

text='샴푸 6제품 원시험·배합 버전 추적 — '+DATE+'\n\n'+'\n\n'.join(f'{r.항목}\n{r.설명}' for r in readme.itertuples())
(ROOT/'결과요약.txt').write_text(text,encoding='utf-8-sig')
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

request_text='''검토용 자료 요청문안 — 발송/제출하지 않음
실제 요청 시 요청자 정보·연락처·요청범위를 사용자가 검토하여 채워야 합니다.
EPA 안내: https://www.epa.gov/foia/guide-foia-requests-pesticide-information

Subject: Records identifying eye/skin irritation studies and formulation versions for five pesticide shampoo registrations

To the EPA Office of Pesticide Programs records staff,

I am seeking releasable records that connect the following registered shampoo products to their primary eye irritation and primary skin irritation studies and the formulation versions evaluated. Please provide available Data Evaluation Records (DERs), acute toxicity reviews, study citations, and releasable study reports. Where formulations are confidential, a non-confidential confirmation of the applicable CSF version and any bridging or waiver rationale would still be useful.

1. EPA Reg. No. 1021-1723, Pyrocide Flea & Tick Shampoo 51921:
   MRID 44539104, G. Wnorowski (1997), primary eye irritation in rabbits, Product Safety Labs project 5145, 26 pages;
   MRID 44539105, G. Wnorowski (1997), primary eye irritation in rabbits, Product Safety Labs project 5536, 25 pages;
   MRID 44539106, G. Wnorowski (1997), primary skin irritation in rabbits, Product Safety Labs project 5146, 17 pages;
   MRID 44539108, W. Grise (1996), product chemistry, McLaughlin Gormley King Co., 7 pages.
   These citations are in the EPA bibliography at the URL below, PDF page 196. Please identify differences between the two eye studies and confirm which study/formula supports No-Bite Flea & Tick Shampoo, distributor registration 1021-1723-12281, SDS dated 2015-07-06. We have not established that the test batches and this SDS are identical.
   https://www3.epa.gov/pesticides/nas/final/diazinon/appendix-2-4.pdf

2. EPA Reg. No. 2724-792, RF2066 Shampoo:
   Acute Toxicity Review dated April 9, 2008 and Product Chemistry Review dated March 11, 2008, listed as enclosures on PDF page 4 of the registration document:
   https://www3.epa.gov/pesticides/chem_search/ppls/002724-00792-20080426.pdf
   Also identify any reviews and underlying MRIDs associated with the January 29, 2015 letter, Decision Numbers 410827 and 412069, and their applicability to the Zodiac SDS dated 2016-04-07.

3. EPA Reg. No. 2724-485, RF-9414 Shampoo:
   Acute toxicity and product chemistry reviews referenced by the June 19, 2012 letter, Decision Numbers 412327 and 410679, and underlying eye/skin MRIDs. The available search-index text lists Basic and Alternate CSFs dated December 21, 2010; the full PDF could not be retrieved in our search.
   Please identify the applicable formula for Zodiac Oatmeal Conditioning Shampoo (SDS 2016-06-24) and Adams Plus Flea & Tick Shampoo with Precor (2724-485-270; SDS 2020-10-29).
   The April 5, 2018 letter lists Basic and Alternate 1 CSFs dated January 7, 2015 and Alternate 2 dated October 6, 2017 (foaming formulation addition), Decisions 534916 and 534918. Please identify any test-data bridging used across these versions.

4. EPA Reg. No. 69061-12, Sivad Flea & Tick Shampoo / Davis:
   Eye/skin irritation DERs and study MRIDs underlying the September 10, 2015 action, Decision Numbers 414109, 411781, and 410235, and applicable Basic/Alternate 1 CSFs dated July 19, 2012.
   Please clarify the relation to the Davis product SDS dated May 28, 2025. The repository reports registration 69061-12-50591; the official basic label identifies 69061-12 and separately lists establishment 50591-GA-01. Please verify the appropriate product identity rather than inferring it from similar numbers.

5. EPA Reg. No. 270-324, Farnam 2009 Insecticide / Adams Flea & Tick Cleansing Shampoo:
   Eye/skin irritation reviews, referenced MRIDs, and applicable CSF version underlying the January 10, 2018 toxicity-label amendment, Decision Number 533083, with relation to the SDS dated September 28, 2023. The June 25, 2025 label amendment, Decision 567010 / Case 473758, is an additional identity reference, not evidence of a new eye study.

For each available study/review, useful fields are the tested product name and lot, formulation identifier/version, concentration or dilution and vehicle, dose/contact duration and rinsing conditions, measured test pH if reported, animal-level tissue scores and recovery, and the EPA endpoint-specific classification and rationale.

Please provide electronic copies of releasable records and omit material that is not releasable. This draft makes no claim that all requested records or formulation details are publicly available. Please advise before incurring fees; no payment commitment is included in this draft.

Requester name: [to be completed]
Mailing address: [to be completed]
Contact: [to be completed]
'''
(ROOT/'원자료_요청문안_미발송.txt').write_text(request_text,encoding='utf-8-sig')

def link(u,label):return '<a href="'+html.escape(u,quote=True)+'">'+html.escape(label)+'</a>'
source_links='<ul>'+''.join('<li>'+link(r.source_url,r.source_id+' — '+r.registration_family+' / '+r.document_date)+' — '+html.escape(r.location)+'</li>' for r in docs_df.itertuples())+'</ul>'
report='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>샴푸 6제품 원시험 추적</title><style>body{font-family:Malgun Gothic,Arial,sans-serif;background:#f3f6f8;color:#203640;margin:28px}main{max-width:1500px;margin:auto}p,li{line-height:1.8}h1{font-size:26px}section{background:white;border-radius:10px;padding:22px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #d8e2e8;text-align:left;vertical-align:top;min-width:125px;font-size:13px}th{background:#24495b;color:white}a{color:#176ba0}.notice{background:#fff3e4;padding:18px;border-radius:8px}small{color:#526773}</style><main>
<h1>샴푸 6제품: 원시험·배합 연결 결과</h1><p>2026-11-09 · 6제품 / 기본 등록 5계열</p>
<p>원시험을 찾을 식별정보와 배합 버전 이력을 확보했습니다. 원시험 본문과 현재 SDS 배합의 일치까지 확인된 제품은 아직 없습니다.</p>
<p>__FILES__</p><section><h2>새로 확인한 근거</h2><ul>
<li>No-Bite 기본 제품: EYE MRID <b>44539104, 44539105</b>, SKIN <b>44539106</b>, 제품화학 <b>44539108</b>의 EPA 서지 확인. __BIB__</li>
<li>Zodiac 2724-792: <b>2008-04-09 급성독성 검토</b> 및 <b>2008-03-11 제품화학 검토</b>의 동봉 기록 확인. __Z08__</li>
<li>Zodiac Oatmeal / Adams Plus: 기본 등록 2724-485에 복수 CSF 존재. 2018 승인서는 거품형 대체 배합 추가를 명시. __O18__</li>
</ul><p class="notice">MRID 번호 발견 ≠ 시험 결과 확보. CSF 날짜 발견 ≠ 전체 배합 확보. 현재 공식 EPA 정답과 시험물질 연결을 확정한 제품은 0건입니다.</p></section>
<section><h2>제품별 연결표</h2><div class="scroll">__TABLE__</div></section>
<section><h2>가장 먼저 확보할 자료</h2><p>1. No-Bite의 눈/피부 MRID 3건과 제품화학 1건의 본문 또는 EPA DER를 확보합니다. 두 눈 시험의 농도·세척·배합 차이를 확인해야 합니다.</p>
<p>2. 2724-792의 날짜가 확인된 검토보고서 2건을 확보해 원시험 번호와 적용 배합을 확인합니다.</p><p>3. 2724-485는 두 브랜드가 어느 CSF 버전에 해당하는지 확인합니다. Davis와 Farnam 계열은 행정 결정번호로 원시험 번호부터 특정합니다.</p>
<p>원문 확보에 사용할 문안은 파일로 준비했으며 발송·제출하지 않았습니다. __GUIDE__</p></section>
<section><h2>해석 범위</h2><p>기존 EYE 대리 라벨과 모델·임계값은 그대로 유지했습니다. SKIN 등급 충돌 2제품도 미해결 상태입니다. 학습 메타데이터에서 명칭/번호가 검색되지 않았어도 독립 표본이라는 증명은 아닙니다.</p>
<p>아래 출처의 일부는 검색 색인만 확인했습니다. 해당 수준과 PDF 접근 실패를 Excel에 명시했습니다. sources 폴더는 웹 발췌 기록이며 원본 PDF 보관본이 아닙니다.</p></section>
<section><h2>원문 출처</h2>__SOURCES__</section></main></html>'''
report=report.replace('__FILES__',' · '.join(link(f,f.split('.')[0]) for f in ['샴푸6제품_원시험연결표.xlsx','자료확보_요청목록.csv','원자료_요청문안_미발송.txt']))
report=report.replace('__BIB__',link(BIB,'EPA 서지 196쪽')).replace('__Z08__',link(url('002724-00792-20080426'),'EPA 승인서 4쪽')).replace('__O18__',link(url('002724-00485-20180405'),'EPA 승인서 1쪽')).replace('__GUIDE__',link(GUIDE,'EPA 자료 요청 안내'))
report=report.replace('__TABLE__',short[['제품','기본 등록번호','원시험 추적 결과','확인한 배합문서 버전','아직 모르는 부분']].to_html(index=False,escape=True)).replace('__SOURCES__',source_links)
(ROOT/'원시험추적_보고서.html').write_text(report,encoding='utf-8')

checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
check('six unique products; five registration families',len(trace)==6 and trace.case_id.is_unique and trace.basic_registration.nunique()==5)
check('no invented gold labels or test conditions',(trace[['eye_epa_gold','skin_epa_gold','test_concentration','test_ph']]=='').all().all())
check('all formal evaluations remain held',not trace.formal_epa_validation_eligible.any())
check('MRID endpoint counts',studies_df.endpoint.value_counts().to_dict()=={'EYE':2,'SKIN':1,'PRODUCT_CHEMISTRY':1})
check('all MRIDs found in retained EPA bibliography evidence',all(m in (ROOT/'sources/shampoo_trace_biblio_find.json').read_text(encoding='utf-8') for m in studies_df.mrid))
check('MRIDs not administrative IDs',all(re.fullmatch(r'\d{8}',x) for x in studies_df.mrid) and all(x.startswith('445391') for x in studies_df.mrid))
check('study citations do not assert report access',not studies_df.full_report_obtained.any() and not studies_df.result_obtained.any())
check('review mentions do not assert report access',not reviews_df.full_review_obtained.any())
check('CSF existence not upgraded to content/match',not formulas_df.csf_contents_obtained.any() and not formulas_df.study_csf_match_verified.any())
check('two explicit dated review leads',(reviews_df.review_date!='').sum()==2)
check('version chronology does not assign 2017 CSF to 2016 SDS','소급 연결하지 않음' in trace.loc[trace.case_id=='SDS_2724_485','formulation_connection'].iloc[0])
doc_ids=set(docs_df.source_id)
check('all product document IDs link',all(set(x.split(' | '))<=doc_ids for x in trace.document_ids))
for item in manifest:check('source unchanged: '+Path(item['path']).name,sha(Path(item['path']))==item['sha256'])
model=BASE/'Eye_cosmetic_validation_20261109/model/eye_A_LR_wmean.joblib'
check('model unchanged',sha(model)=='d71394e691ac5072e4ce900a53b47ddbb65c8685961fe2c9536411acfbbc58a2')
wb=load_workbook(ROOT/'샴푸6제품_원시험연결표.xlsx',data_only=False)
check('saved workbook row counts',wb['제품별연결'].max_row==7 and wb['원시험MRID'].max_row==5)
check('saved workbook no active formulas',not any(c.data_type=='f' for ws in wb for row in ws for c in row))
check('offline report local links resolve',all((ROOT/f).is_file() for f in re.findall(r'href="([^"#]+)"',report) if not f.startswith('http')))
(ROOT/'verification.json').write_text(json.dumps(dict(passed=True,check_count=len(checks),checks=checks,scope='자료 연결과 보존·보고서 구조 확인; 독성 결과를 새로 검증했다는 뜻은 아님'),ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'artifact_checksums.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='artifact_checksums.json'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(summary=summary,checks_passed=len(checks)),ensure_ascii=False,indent=2))
