"""Write a source-based correction addendum without modifying the frozen run."""
from pathlib import Path
import csv
import hashlib
import html
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT / 'Eye_cosmetic_validation_20261109'

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

manual = 'https://www.epa.gov/sites/production/files/documents/label-review-manual.pdf'
labels = {
    '69061-12': ('https://www3.epa.gov/pesticides/chem_search/ppls/069061-00012-20150910.pdf', '2015-09-10', 4),
    '1021-1723': ('https://www3.epa.gov/pesticides/chem_search/ppls/001021-01723-20131224.pdf', '2013-12-24', 4),
    '270-324': ('https://www3.epa.gov/pesticides/chem_search/ppls/000270-00324-20120301.pdf', '2012-03-01', 3),
    '2724-485': ('https://www3.epa.gov/pesticides/chem_search/ppls/002724-00485-20180405.pdf', '2018-04-05', 3),
}
predictions = read_csv(OLD / 'outputs/shampoo_predictions.csv')
audited = []
for p in predictions:
    group = p['registration_group']
    url, date, page = labels[group]
    positive = p['label'] == '1'
    audited.append({
        'product': p['product'], 'epa_reg': p['epa_reg'], 'registration_family': group,
        'original_category': p['source_epa_category'],
        'reviewed_label_based_category': 'II-consistent; study not verified' if positive else 'III or IV; exact category unresolved',
        'binary_label_proxy': int(positive), 'binary_label_changed': False,
        'evidence_type': 'EPA eye-specific labeling; not an original study result',
        'eye_statement': 'Causes substantial but temporary eye injury.' if positive else 'Causes moderate eye irritation.',
        'official_label_url': url, 'label_date': date, 'pdf_page_1_based': page,
        'guidance_url': manual, 'guidance_location': 'Chapter 7, Table 6; PDF page 76',
        'exact_study_category_verified': False, 'formulation_and_study_version_match_verified': False,
        'strict_validation_eligible': False, 'represented_pct_midpoint': p['represented_pct'],
        'score_original': p['score'], 'score_low_original': p['score_low'], 'score_high_original': p['score_high'],
        'note': 'III로 확정한 종전 표현 정정. III/IV 모두 모델 음성 범주이며 무자극을 뜻하지 않음.' if not positive else 'II에 대응하는 공식 문구 확인. 실제 시험 원자료로 등급을 재구성한 것은 아님.',
    })
write_csv('EPA_라벨_재검토.csv', audited)

candidates = [
    dict(id='P01', name='샴푸 배합 특허 US4992266A', count='7 시험 배합', source='https://patents.google.com/patent/US4992266A/en', extra='',
         available='Table I 배합; Table II 24/48/72시간 및 7일 평균 Draize 점수',
         limitation='7일 추적·평균 총점만으로 EPA 세부 등급 확정 불가. 향료·색소·고분자·상업용 계면활성제 구조와 함량 해석 필요.',
         independence='학습자료와 원시험 중복 미확인; 한 특허의 조건들', priority='배합과 실측값이 함께 있어 우선 검토',
         exclusions='Examples 8–11은 예상 결과만 제시하므로 실측 표본에서 제외'),
    dict(id='P02', name='4급 암모늄 혼합물 특허 US6583181B1', count='21 시험 조건: 표1 2개 + 표2 8개 + 표3 5개 + 표4 6개', source='https://patents.google.com/patent/US6583181B1/en', extra='',
         available='Tables 1–4 배합·pH·최대 21일까지 관찰한 Draize 시계열',
         limitation='상업용 원료 용매·중화 후 화학종 확인 필요. 평균 총점으로 EPA 등급 자동 부여 금지. 표1 대조군 1일값이 표 41.7, 본문 47.1로 불일치.',
         independence='학습자료 중복 미확인; 21개의 독립 연구를 뜻하지 않음', priority='회복 시계열과 배합이 있어 우선 검토',
         exclusions='Tables 5–6의 살균 결과는 눈자극 결과가 아니므로 제외'),
    dict(id='S01', name='Zodiac Flea & Tick Shampoo for Dogs & Cats — 2724-792', count='기존 목록과 다른 기본 등록번호 1개',
         source='https://www3.epa.gov/pesticides/chem_search/ppls/002724-00792-20150129.pdf',
         extra='https://www.domyown.com/msds/Zodiac_Flea_and_Tick_Shampoo_for_Dogs_and_Cats_SDS_2020.pdf',
         available='EPA 눈자극 문구; SDS 일부 성분 및 함량 범위',
         limitation='전체 조성·시험 원자료·SDS와 승인 제형의 정확한 일치 미확인. III/IV 라벨 대리값만 가능.',
         independence='기존 2724-485 Oatmeal과 다른 등록계열. 학습자료·시험자료 독립성은 미확인', priority='보조 사례 후보',
         exclusions='이 등록번호의 다른 브랜드를 각각 독립 연구로 세지 않음'),
    dict(id='D01', name='Enforcer Flea & Tick Shampoo Pets II — 1021-1723-40849', count='추가 독립 사례로 세지 않음',
         source='https://labelsds.com/images/user_uploads/Enforcer%20Flea%20Tick%20Shampoo%20Pets%20II%20SDS%208-3-18.pdf', extra='',
         available='제조사 SDS의 성분 범위', limitation='No-Bite와 기본 등록번호 동일; 제형·문서 버전 일치 전 성분 전용 금지',
         independence='1021-1723 계열 중복', priority='기존 계열의 조성 확인 보조', exclusions='새 독립 표본 수에서 제외'),
    dict(id='D02', name='Vet-Kem Ovitrol Plus Flea & Tick Shampoo — 2724-485', count='추가 독립 사례로 세지 않음',
         source='https://www.prnpharmacal.com/wp-content/uploads/2019/12/SDS-Vet-Kem-Ovitrol-Plus-Flea-Tick-Shampoo-for-Dogs-Cats-07052016.pdf', extra='',
         available='제조사 SDS의 성분 범위', limitation='Zodiac Oatmeal 및 Adams Plus와 등록계열 공유; 동일 제형·시험이라고 단정 불가',
         independence='2724-485 계열 중복', priority='기존 계열의 조성 확인 보조', exclusions='새 독립 표본 수에서 제외'),
    dict(id='D03', name='Pro Vita Flea & Tick Shampoo — 69061-12-84459', count='추가 독립 사례로 세지 않음',
         source='https://provitapet.com/wp-content/uploads/2016/02/SDS-FLEA-TICK.pdf', extra='',
         available='제조사 SDS의 살충 성분 함량', limitation='Davis와 등록계열 공유; 전체 조성은 미공개',
         independence='69061-12 계열 중복', priority='기존 계열의 조성 확인 보조', exclusions='새 독립 표본 수에서 제외'),
]
write_csv('추가자료_후보와중복.csv', candidates)

# Numerical observations are preserved on the original endpoint, without inventing EPA labels.
score_values = [
    [1,30.0,21.3,13.0,0.7,7.4], [2,32.0,27.0,20.0,16.7,18.5],
    [3,33.3,27.0,17.7,6.4,12.4], [4,34.0,27.7,19.3,8.0,13.7],
    [5,31.7,19.6,13.0,2.0,8.1], [6,32.0,27.3,23.0,13.0,16.9],
    [7,36.6,28.6,19.1,12.0,16.3],
]
keys = ['example','draize_mean_24h','draize_mean_48h','draize_mean_72h','draize_mean_day7','reported_7day_weighted_average']
scores = [dict(zip(keys, values), epa_category='', binary_label='', source_table='US4992266A Table II') for values in score_values]
write_csv('샴푸특허_7배합_원점수.csv', scores)

summary = '''EPA 라벨 재검토 및 추가 자료 — 2026-11-09

이 문서는 앞선 결과보고서의 등급 확정 표현과 성능 해석을 정정하는 부록입니다.

1. EPA 등급이 뒤집힌 것인가?
현재 확인한 공식 라벨에서는 기존 I/II 대 III/IV 이진 분류를 뒤집을 근거가 발견되지 않았습니다. 그러나 모든 등급을 실제 시험으로 확인했다고 말할 근거도 없습니다.
No-Bite, Adams Cleansing, Davis의 기본 등록번호에 대응하는 EPA 문서에는 II에 해당하는 눈 손상 문구가 있습니다. 이 자료는 라벨 근거이며 시험 동물별 원점수는 아닙니다.
Zodiac Oatmeal과 Adams Plus를 정확히 III로 확정한 표현은 정정합니다. EPA 지침은 IV 제품에도 III 문구를 허용하므로, 문구에서 알 수 있는 범위는 III 또는 IV입니다. 현재 모델은 둘 다 0으로 취급합니다. 0은 무자극·안전 보증이 아닙니다.
GHS 2A/2B 및 WARNING/CAUTION만으로 EPA 눈자극 등급을 확정하지 않습니다. 문서 연도와 실제 평가 제형·시험의 일치도 별도 확인이 필요합니다.

2. 앞선 결과를 어떻게 읽어야 하나?
중간 함량을 입력한 4행이 제공 라벨과 모두 불일치했다는 계산은 그대로입니다. 이것을 모델의 일반적인 정확도 0%로 해석하면 안 됩니다.
모델에 들어간 성분은 제품 중량의 0.779~3.93%뿐입니다. 나머지 96.07~99.221%는 모델 입력에 반영되지 않았습니다. 빠진 부분에는 물뿐 아니라 구조가 해결되지 않은 계면활성제와 비공개 성분 등이 있습니다. 포함된 성분끼리 비율을 다시 정규화했으므로 실제 제품을 충분히 표현하지 못합니다.
4행은 기본 등록번호 3계열입니다. 같은 등록번호 아래 대체 제형이 존재하므로 동일 배합·동일 시험이라고 단정할 수도 없습니다. 계열별 집계는 중복을 보수적으로 처리한 기술 통계일 뿐입니다.
두 제품은 공개된 함량 범위 안에서 판정이 바뀌었습니다. 모델 문제와 입력 부족 문제를 이 결과만으로 분리할 수 없습니다.
CIR 9/22 일치도는 미검증 출처 라벨과의 산술 비교입니다. 모든 EPA 등급을 원시험에서 재판정하지 않았으므로 검증 성능으로 사용하지 않습니다. mild/moderate라는 표현이나 평균 Draize 총점만으로 EPA 등급을 정하면 안 됩니다.
현재 검토한 기존 사례 중 전체 조성과 독립적인 원시험 등급이 함께 확인된 엄격한 외부검증 표본은 0개입니다.

3. 추가 자료
샴푸 특허 US4992266A: 실측 7배합. Table II 수치를 CSV로 옮겼습니다. 평균값과 EPA 등급은 별개라 등급 칸은 비워 두었습니다.
혼합물 특허 US6583181B1: Tables 1–4에 21개 시험 조건. 배합과 회복 시계열이 있어 추가 검토 가치가 있습니다. 화장품 완제품 21종이나 독립 연구 21개를 뜻하지 않습니다.
Zodiac 2724-792: 기존 Oatmeal 2724-485와 다른 등록계열의 추가 후보입니다. 일부 성분과 공식 라벨만 확인되어 엄격한 검증에는 아직 부족합니다.
Enforcer, Vet-Kem, Pro Vita 자료는 기존 등록계열과 겹쳐 추가 독립 사례로 세지 않았습니다.
모든 추가 후보는 학습자료와의 원시험 중복 확인이 남아 있습니다. 이번 부록에서 새 모델 예측이나 임계값 조정은 수행하지 않았습니다.

4. 다음 검증에 적용할 기준
공식 시험 등급 또는 충분한 원시험 기록, 시험한 배합·농도·세척 조건, 성분 구조와 함량, 기존 학습자료와의 중복 여부를 함께 확인해야 합니다. 기준에 부족한 자료는 참고 사례로 남깁니다.
기존 모델과 예측값은 보존하며, 라벨을 예측에 맞추어 뒤집거나 임계값을 이 자료에 맞추지 않습니다.

파일 안내
EPA_라벨_재검토.csv: 기존 5제품의 원래 라벨과 수정된 근거·해석
추가자료_후보와중복.csv: 신규 후보 3자료와 중복 계열 3자료의 출처·한계
샴푸특허_7배합_원점수.csv: 원문 표의 실측 점수. EPA 분류는 비워 둠
원본 보고서·예측·모델 파일은 수정하지 않았습니다. 이 재검토 문서의 해석을 우선합니다.
웹 원문은 조회했으나 PDF 파일 자체를 이 폴더에 내려받아 보관한 것은 아닙니다.
'''
(HERE / '정정_및_추가자료.txt').write_text(summary, encoding='utf-8-sig')

def table(rows, columns):
    def cell(value):
        s = str(value)
        return f'<a href="{html.escape(s, quote=True)}">원문</a>' if s.startswith('https://') else html.escape(s)
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(title)}</th>' for _,title in columns) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{cell(r[k])}</td>' for k,_ in columns) + '</tr>' for r in rows) + '</tbody></table>'

body = '<h1>EPA 라벨 재검토와 추가 자료</h1><p class="lead">종전의 EPA III 확정 표현을 정정합니다. 4행 불일치는 계산 결과이지만, 모델의 일반적인 정확도 0%를 입증하지 않습니다.</p>'
body += '<div class="text">' + html.escape(summary) + '</div>'
body += '<h2>기존 제품의 공식 라벨 근거</h2>' + table(audited,[('product','제품'),('epa_reg','등록번호'),('reviewed_label_based_category','재검토'),('official_label_url','공식 라벨'),('pdf_page_1_based','PDF 페이지')])
body += f'<p><a href="{manual}">EPA Label Review Manual, Chapter 7, Table 6</a>: IV에서도 III 문구 사용 가능.</p>'
body += '<h2>추가 후보와 중복 계열</h2>' + table(candidates,[('name','자료'),('count','수량'),('available','확인한 내용'),('limitation','한계'),('source','출처'),('extra','추가 출처')])
body += '<h2>받아볼 파일</h2><ul>' + ''.join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in ['EPA_라벨_재검토.csv','추가자료_후보와중복.csv','샴푸특허_7배합_원점수.csv','정정_및_추가자료.txt']) + '</ul>'
(HERE / 'EPA_재검토_결과.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>EPA 재검토 결과</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;line-height:1.75;color:#172337}.lead{background:#fff1d6;border-left:5px solid #d18b00;padding:18px;font-weight:600}.text{white-space:pre-wrap}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d3dbe5;padding:10px;text-align:left;vertical-align:top}th{background:#edf2f8}a{color:#0759b5}</style>' + body + '</html>', encoding='utf-8')

metadata = {'date':'2026-11-09','prior_run':str(OLD),'guidance_url':manual,
    'binary_label_proxy_changes':0,'exact_category_uncertainty_corrected':['2724-485','2724-485-270'],
    'new_predictions_run':False,'new_EPA_labels_assigned':False,'strict_validated_external_cases':0,
    'new_candidate_sources':3,'duplicate_family_sources':3,'extracted_measured_shampoo_rows':7,
    'additional_data_training_overlap_checked':False,'pdf_files_downloaded':False,
    'priority':'This addendum supersedes category certainty and performance interpretation in the prior report; preserves its arithmetic.'}
(HERE / 'review_summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

pointer = ROOT / '검증결과_안내.txt'
previous = pointer.read_text(encoding='utf-8-sig')
marker = '[최신 EPA 재검토 부록: 2026-11-09]'
if not previous.startswith(marker):
    pointer.write_text(marker + '\nEPA_재검토_추가자료_20261109/EPA_재검토_결과.html\nEPA III 확정 표현과 기존 성능 해석은 이 부록의 정정을 우선 적용하세요.\n\n--- 최초 계산 결과 안내 (보존) ---\n' + previous, encoding='utf-8-sig')

# Check the actual arithmetic being discussed; do not rerun or refit the model.
scored = [p for p in predictions if p['score']]
assert len(scored) == 4 and sum(float(p['correct']) for p in scored) == 0
assert len({p['registration_group'] for p in scored}) == 3
assert len(audited) == 5 and len(scores) == 7
assert all(r['epa_category'] == '' and r['binary_label'] == '' for r in scores)
assert [p['label'] for p in predictions] == [str(p['binary_label_proxy']) for p in audited]
assert min(float(p['represented_pct']) for p in scored) > 0.778
assert max(float(p['represented_pct']) for p in scored) == 3.93
checksums = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.is_file() and p.name != 'artifact_checksums.json'}
(HERE / 'artifact_checksums.json').write_text(json.dumps(checksums, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'report':str(HERE / 'EPA_재검토_결과.html'),'files':len(checksums)+1,'review_checks':'passed'}, ensure_ascii=False))
