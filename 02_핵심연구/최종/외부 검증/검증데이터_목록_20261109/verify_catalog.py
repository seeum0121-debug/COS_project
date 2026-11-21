"""Check the saved inventory against frozen inputs and retained source extracts."""
from pathlib import Path
import hashlib, json, re, shutil, subprocess, tempfile
import pandas as pd
from openpyxl import load_workbook

P=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)

d=pd.read_csv(P/'검증용_데이터목록.csv')
c=pd.read_csv(P/'성분별_함량근거.csv')
e=pd.read_csv(P/'사례별_근거목록.csv')
s=pd.read_csv(P/'출처목록.csv')
check('112 unique candidate records',len(d)==112 and d.case_id.is_unique)
check('source counts',d.dataset.value_counts().to_dict()=={'CIR':78,'4급암모늄 특허':21,'샴푸 특허':7,'제품 SDS':6})
check('endpoint counts include 6 shared product records',d.endpoints.str.contains('EYE').sum()==65 and d.endpoints.str.contains('SKIN').sum()==53)
check('no unverified EPA labels promoted to gold',d.eye_epa_gold.isna().all() and d.skin_epa_gold.isna().all() and not d.strict_epa_eligible.any())
check('no invented exposure dilution',d.verified_test_concentration_pct.isna().all())
check('pH scope separated',d.ph_status.str.startswith('원문 확인').sum()==32 and (d.ph_status=='원문 확인—제품 pH').sum()==5 and (d.ph_status=='원문 확인—시험 배합 pH').sum()==27 and (d.ph_status=='CSV 기재—원문 미확인').sum()==1)
check('component/evidence keys link to cases',set(c.case_id)<=set(d.case_id) and set(e.case_id)==set(d.case_id))
check('all evidence source IDs exist',all(set(v.split(' | '))<=set(s.source_id) for v in d.evidence_ids))
check('all cases have concrete next action',d.next_action.notna().all())
orig=pd.read_csv(P/'input_snapshot/cir_ingredients.csv')
for i,r in orig.iterrows():
    row=d[d.case_id==f'CIR_row_{i+1}'].iloc[0];ep=r['endpoint(skin/eye)']
    check(f'CIR {i+1}: preserve original label and concentration',row['original_'+ep+'_epa']==r['EPA등급(환산)'] and (pd.isna(row.original_concentration_pct) and pd.isna(r['농도(%)']) or row.original_concentration_pct==r['농도(%)']))
check('all four SLS records flag salt mismatch',(d[d.cas_or_registration=='151-21-3'].structure_status.str.contains('나트륨염')).sum()==4)
check('unverified CAPB pH not promoted',d.loc[d.case_id=='CIR_row_3','ph_status'].iloc[0]=='CSV 기재—원문 미확인')
check('two skin conflicts retained without overwriting original category',len(d[d.conflict_flags.fillna('').str.contains('SKIN 원래')])==2 and (d[d.conflict_flags.fillna('').str.contains('SKIN 원래')].original_skin_epa=='III').all())
check('official EPA source URL attached to product evidence',d.loc[d.dataset=='제품 SDS','epa_source_url'].str.startswith('https://www3.epa.gov/').all())
check('six product records grouped into five basic registrations',d[d.dataset=='제품 SDS'].grouping_key.nunique()==5)
q=pd.read_csv(P/'input_snapshot/quat_formulations.csv')
for _,r in q.iterrows():
    row=d[d.case_id==r.formulation_id].iloc[0]
    check(f'{r.formulation_id}: pH and representation copied unchanged',row.ph_low==r.pH and abs(row.represented_pct_mean-r.represented_nonwater_pct)<1e-9)
source=(P/'sources/catalog_source_2.txt').read_text(encoding='utf-8')
line=next(l for l in source.splitlines() if l.startswith('L606:') and 'pH' in l)
source_ph=[float(x.strip()) for x in line.split('|')[-6:]]
check('table 4 pH transcribed by column',d[d.case_id.str.startswith('Q_T4_')].ph_low.tolist()==source_ph)
check('table 4 structures remain held',d[d.case_id.str.startswith('Q_T4_')].structure_status.str.startswith('보류').all())
check('all seven shampoo recipes sum to 100 as-supplied wt%',all(abs(v-100)<1e-8 for v in c[c.case_id.str.startswith('SH_')].groupby('case_id').pct_low.sum()))
check('shampoo example 1 missing pH retained',pd.isna(d.loc[d.case_id=='SH_Example1','ph_low'].iloc[0]))
check('patent eye data not reused for skin',d[d.dataset.str.contains('특허')].skin_use.str.contains('SKIN 결과 없음').all())
for item in json.loads((P/'input_manifest.json').read_text(encoding='utf-8')):
    original=Path(item['original_path'])
    check('original source unchanged: '+item['snapshot'],digest(original)==item['sha256'])
    if item['snapshot']!='training_metadata_read_only':check('frozen input matches original: '+item['snapshot'],digest(P/'input_snapshot'/item['snapshot'])==item['sha256'])
model=P.parent/'Eye_cosmetic_validation_20261109/model/eye_A_LR_wmean.joblib'
check('frozen model unchanged',digest(model)=='d71394e691ac5072e4ce900a53b47ddbb65c8685961fe2c9536411acfbbc58a2')
wb=load_workbook(P/'검증용_데이터목록.xlsx',data_only=False)
check('workbook opens with expected rows',wb['사례목록'].max_row==113 and wb['사례상세'].max_row==113 and wb['Zein_별도보관'].max_row==206 and wb['중복출처_3건'].max_row==4)
check('spreadsheet contains data rather than formulas',not any(cell.data_type=='f' for ws in wb for row in ws for cell in row))
page=(P/'검증목록_요약.html').read_text(encoding='utf-8')
data=json.loads(re.search(r'const data=(.*?);const labels=',page,re.S)[1])
check('standalone HTML includes every case',len(data)==112 and {r['case_id'] for r in data}==set(d.case_id))
for href in re.findall(r'href="([^"]+)"',page):check('HTML local link resolves: '+href,(P/href).is_file())
node=shutil.which('node')
if node:
    script=re.search(r'<script>(.*?)</script>',page,re.S)[1]
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'catalog.js';f.write_text(script,encoding='utf-8')
        r=subprocess.run([node,'--check',str(f)],capture_output=True,text=True)
        check('HTML filter JavaScript syntax',r.returncode==0)
result={'passed':True,'check_count':len(checks),'checks':checks,'verification_scope':'Counts, original values, key joins, selected primary-source transcription, source/model hashes, Excel and offline HTML structure. Does not upgrade incomplete scientific evidence.'}
(P/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
artifacts={str(f.relative_to(P)):digest(f) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='artifact_checksums.json' and '__pycache__' not in f.parts}
(P/'artifact_checksums.json').write_text(json.dumps(artifacts,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'checks':len(checks),'artifacts':len(artifacts)},ensure_ascii=False))
