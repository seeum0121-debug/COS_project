"""Freeze source adjudication and analysis choices before new predictions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'Eye_cosmetic_validation_20261109'
for folder in ['inputs','model','outputs']:
    (ROOT/folder).mkdir(exist_ok=True)
for name in ['eye_A_LR_wmean.joblib','predict.py','metadata.json','feature_names.json','smoke_test_cases.json']:
    shutil.copy2(OLD/'model'/name,ROOT/'model'/name)

PATENT='https://patents.google.com/patent/US6583181B1/en'
def bac(n): return 'C'*n+'[N+](C)(C)Cc1ccccc1.[Cl-]'
registry={
    'DDAC':('didecyldimethylammonium chloride','7173-51-5','CCCCCCCCCC[N+](C)(C)CCCCCCCCCC.[Cl-]','C22H48ClN'),
    'BAC12':('benzyl dimethyl dodecyl ammonium chloride','139-07-1',bac(12),'C21H38ClN'),
    'BAC14':('benzyl dimethyl tetradecyl ammonium chloride','139-08-2',bac(14),'C23H42ClN'),
    'BAC16':('benzyl dimethyl hexadecyl ammonium chloride','122-18-9',bac(16),'C25H46ClN'),
    'BAC18':('benzyl dimethyl octadecyl ammonium chloride','122-19-0',bac(18),'C27H50ClN'),
    'EDTA':('ethylenediaminetetraacetic acid','60-00-4','O=C(O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O','C10H16N2O8'),
    'NaOH':('sodium hydroxide','1310-73-2','[Na+].[OH-]','HNaO'),
    'NaCl':('sodium chloride','7647-14-5','[Na+].[Cl-]','ClNa'),
    'glycolic':('glycolic acid','79-14-1','OCC(=O)O','C2H4O3'),
    'water':('water','7732-18-5','O','H2O'),
    'ethanol':('ethanol','64-17-5','CCO','C2H6O'),
}
# Reuse the structure resolved for the identical CAS in the frozen prior run.
old_components=pd.read_csv(OLD/'sources/cosmetic_validation/flea_tick_shampoo.csv')
pbo=old_components.loc[old_components.cas=='51-03-6','canonical_smiles'].dropna().unique()
assert len(pbo)==1
registry['PBO']=('piperonyl butoxide; technical mixture approximation','51-03-6',pbo[0],'C19H30O5')
reg=[]
for key,(name,cas,smiles,formula) in registry.items():
    m=Chem.MolFromSmiles(smiles)
    assert m is not None and Chem.GetFormalCharge(m)==0
    actual=rdMolDescriptors.CalcMolFormula(m)
    assert actual==formula,(key,actual,formula)
    reg.append(dict(component_id=key,name=name,cas=cas,smiles=Chem.MolToSmiles(m),formula=actual,molecular_weight=Descriptors.MolWt(m),
                    structure_basis='Explicit molecular name/homologue in source; salts retain counterions; RDKit formula and charge checked' if key!='PBO' else 'Frozen previous CAS resolution; technical related compounds remain unresolved'))
pd.DataFrame(reg).to_csv(ROOT/'inputs/structure_registry.csv',index=False,encoding='utf-8-sig')

families={
    'FMB':{'DDAC':.6,'BAC12':.4*.4,'BAC14':.4*.5,'BAC16':.4*.1},
    'DDAC':{'DDAC':1.0},
    'MB':{'BAC12':.4,'BAC14':.5,'BAC16':.1},
    'MX':{'BAC12':.05,'BAC14':.6,'BAC16':.3,'BAC18':.05},
}
assert all(abs(sum(x.values())-1)<1e-12 for x in families.values())
formulations=[]; components=[]; observations=[]
def add(fid,table,col,family,quat,edta,naoh,nacl,glycolic,ph,days,values,stock_fraction=None):
    amounts={key:quat*value for key,value in families[family].items()}
    for key,value in [('EDTA',edta),('NaOH',naoh),('NaCl',nacl),('glycolic',glycolic)]:
        if value: amounts[key]=value
    additives=edta+naoh+nacl+glycolic
    water=None if stock_fraction is None else 100-quat/stock_fraction-additives
    stock_unknown=None if stock_fraction is None else quat/stock_fraction-quat
    note='Nominal ingredients before mixing; pH not a feature; neutralization and stock solvent not modeled.'
    if fid=='Q_T1_C2': note+=' Day-1 table value 41.7 retained; conflicting prose 47.1 not substituted.'
    formulations.append(dict(formulation_id=fid,dataset='quat_patent',source_url=PATENT,table=table,column=col,quat_family=family,
        quat_active_pct=quat,EDTA_pct=edta,NaOH_pct=naoh,NaCl_pct=nacl,glycolic_pct=glycolic,pH=ph,
        represented_nonwater_pct=sum(amounts.values()),explicit_added_water_pct=water,unresolved_stock_pct=stock_unknown,
        epa_category=None,binary_reference=None,reference_type='reported Draize composite scores; EPA category unresolved',
        strict_validation_eligible=False,notes=note))
    for key,value in amounts.items(): components.append(dict(formulation_id=fid,component_id=key,pct=value,role='primary_disclosed_nonwater'))
    if water is not None: components.append(dict(formulation_id=fid,component_id='water',pct=water,role='sensitivity_only_explicit_added_water'))
    for day,value in zip(days,values):
        observations.append(dict(formulation_id=fid,day=day,draize_score=value,source_table=f'Table {table}',source_column=col,missing_reason='dash in source; not zero' if value is None else ''))

add('Q_T1_C1',1,1,'FMB',1.2,.90,.33,0,0,7.1,[1,2,3,4,7,10,14,21],[26.3,23.2,20,15,7.7,3.7,0,None],.8)
add('Q_T1_C2',1,2,'FMB',1.2,0,0,0,0,6.9,[1,2,3,4,7,10,14,21],[41.7,36.5,31.8,32,30,None,19.7,19],.8)
t2={
    1:[29.2,15.5,18.3,12.8,18,8.7,25.5,16.5],2:[24.5,11.2,17.8,7.7,12.7,3.7,20.5,9.2],
    3:[23.8,7.2,14.3,6.3,9.2,4.2,24.5,6.8],4:[22.5,8.5,10.8,7,7,3,20.3,10.8],
    7:[17.7,1.7,6.3,2,3,.3,12.3,5],10:[17.3,1.7,5.7,0,2,0,11.7,1.7],
    14:[16,1,.3,None,1.7,None,4.7,1],18:[9.7,0,0,None,1.3,None,.3,0],21:[4.3,None,None,None,1,None,0,0]}
for i in range(8):
    add(f'Q_T2_C{i+1}',2,i+1,['DDAC','DDAC','MB','MB','MB','MB','MX','MX'][i],
        [1,1,1,1,1.28,1.28,2,2.1][i],[0,1,0,1,0,1.2,2,4][i],[0,.4,0,.4,0,.48,.8,1.6][i],0,0,
        [7.1,7.5,7.5,7.4,7,6.9,7,7][i],list(t2),[v[i] for v in t2.values()])
t3={1:[33.5,16.2,15,27.4,18],2:[29.3,10.7,13.2,12.7,17.7],3:[29.7,7.8,8.3,7.8,11.8],4:[24.5,4.5,7.8,6.5,15],
    7:[21,1,3.7,15.3,8.3],10:[14.3,3.3,3.7,0,0],14:[12.7,0,3.3,None,None],18:[7.6,None,3.7,None,None],21:[4.7,None,0,None,None]}
for i in range(5):
    add(f'Q_T3_C{i+1}',3,i+1,'MB',[2,1.9,1.9,1.9,1.9][i],[0,6,8,0,0][i],[0,2.4,3.2,0,3.2][i],
        [0,0,0,6,0][i],[0,0,0,0,6][i],[7,7,7.2,7,7][i],list(t3),[v[i] for v in t3.values()],.8)
pd.DataFrame(formulations).to_csv(ROOT/'inputs/quat_formulations.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(components).to_csv(ROOT/'inputs/quat_components.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(observations).to_csv(ROOT/'inputs/quat_observations.csv',index=False,encoding='utf-8-sig')

pairs=[('Q_T1_C2','Q_T1_C1'),('Q_T2_C1','Q_T2_C2'),('Q_T2_C3','Q_T2_C4'),('Q_T2_C5','Q_T2_C6')]
pd.DataFrame([dict(pair_id=f'P{i+1}',control=a,treatment=b,selection='same quaternary identity and nominal active concentration; EDTA/NaOH addition') for i,(a,b) in enumerate(pairs)]).to_csv(ROOT/'inputs/prespecified_pairs.csv',index=False,encoding='utf-8-sig')

exclusions=[]
for i in range(1,7):
    exclusions.append(dict(formulation_id=f'Q_T4_C{i}',source_url=PATENT,reason='Table 4 merged commercial-product headers could not be visually verified from accessible rendering. Withhold predictions rather than guess column assignment.',epa_status='unresolved; composite scores only'))
for i in range(1,8):
    exclusions.append(dict(formulation_id=f'SH_Example{i}',source_url='https://patents.google.com/patent/US4992266A/en',reason='Important commercial surfactant/homopolymer structures unresolved; fragrances/dyes unspecified. No representative coco/ethoxylate molecule substituted.',epa_status='unresolved; mean scores to day 7 only'))
pd.DataFrame(exclusions).to_csv(ROOT/'inputs/excluded_cases.csv',index=False,encoding='utf-8-sig')
shutil.copy2(ROOT.parent/'EPA_재검토_추가자료_20261109/샴푸특허_7배합_원점수.csv',ROOT/'inputs/shampoo_patent_observations.csv')

sds=dict(formulation_id='Zodiac_2724_792',product='Zodiac Flea & Tick Shampoo for Dogs & Cats',epa_reg='2724-792',
    label_proxy=0,label_proxy_category='III or IV inferred from FIFRA eye-specific statement; not a study-confirmed class',
    sds_date='2016-04-07',sds_url='https://www.domyown.com/msds/Zodiac_Flea_and_Tick_Shampoo_for_Dogs_and_Cats_SDS_2020.pdf',
    label_url='https://www3.epa.gov/pesticides/chem_search/ppls/002724-00792-20150129.pdf',
    resolved=[dict(component_id='PBO',low=.5,high=.5),dict(component_id='ethanol',low=1.69,high=1.95)],
    unresolved=[dict(name='pyrethrins',pct=.05),dict(name='coconut fatty acid monoethanolamide',low=1.08,high=1.2),dict(name='other ingredients',pct='balance')],
    strict_validation_eligible=False)
(ROOT/'inputs/zodiac_source.json').write_text(json.dumps(sds,ensure_ascii=False,indent=2),encoding='utf-8')

protocol=dict(created_before_predictions_utc=datetime.now(timezone.utc).isoformat(),
    model_sha256=hashlib.sha256((ROOT/'model/eye_A_LR_wmean.joblib').read_bytes()).hexdigest(),threshold=.4814867474694138,
    refit=False,threshold_tuning=False,new_EPA_labels_assigned=False,
    candidates_screened=29,quat_cases_predicted=15,quat_cases_held=6,shampoo_cases_held=7,additional_SDS_case=1,
    primary_input='As-added explicit nonwater molecular ingredients; patent homolog weight ratios; normalized using frozen predictor. Water excluded: none in 673-row training linkage. Stock solvents and unreported compounds not invented.',
    primary_diagnostic='Spearman association with day-7 original Draize composite; descriptive, not EPA accuracy. All 15 cases and deduplicated/nontraining-feature cases reported.',
    secondary_diagnostics=['Trapezoid area days 1-7 / 6; no imputation required','Four prespecified equal-quat-concentration pairs: direction of day-7 change','Seven cases with known stock strength: add only explicitly computed added water as out-of-training-representation sensitivity'],
    duplicate_policy='Hash rounded feature vectors; report mean observed endpoint per unique vector and remove matches to final-fit training inputs in separate analysis.',
    source_independence='Check metadata, chemistry and exact feature matches; missing original-study identifiers prevents proof of study independence.',
    metrics_forbidden=['EPA accuracy or AUC from patent composite scores','pooled accuracy over patents and SDS','independent-sample p-values for related formulations'],
    source_limits=['Patents are not independent peer-reviewed replications','Table 1 control day1 text/table conflict retained','No raw animal/tissue scores','pH/neutralization absent in model','Inorganic/EDTA ingredients may be outside training chemical space'],
    source_download_note='Direct local download failed with WinError 10013; web-tool extracted text saved with URLs. These are not original PDF files.')
(ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
freeze={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['inputs','model','sources'] for p in (ROOT/folder).glob('*') if p.is_file()}
freeze['protocol.json']=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
(ROOT/'input_manifest.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(formulations)==15 and len(exclusions)==13 and len(observations)==133
print(json.dumps({'frozen_formulations':len(formulations),'held':len(exclusions),'observation_rows':len(observations),'files_hashed':len(freeze)},ensure_ascii=False))
