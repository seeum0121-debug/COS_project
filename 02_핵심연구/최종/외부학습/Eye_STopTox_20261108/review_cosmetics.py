"""Observation-level source audit; no EPA labels or invented recipe balances."""
from common import *
import pandas as pd
import numpy as np

BABY='https://pmc.ncbi.nlm.nih.gov/articles/PMC12656482/'
IIVS='https://iivs.org/wp-content/uploads/2016/09/iivs-poster_eye-irritation-potential-of-prototype-cosmetic-formulations-containing-salicylic-acid-glycolic-acid-and-ethanol.pdf'
# Transcribed against publisher PDF pages 7-9 and PMC Table 2, including
# visually checked spanning cells. Values remain raw reported percent and pH.
TABLE2='''1|Decyl Glucoside|1|5.00|5.5–6.0|S
2|Coco-Glucoside (And) Glyceryl Oleate|1|3.00|5.5–6.0|N
3|Lauryl Glucoside|1|3.00|5.5–6.0|N
4|Capryl/Capramidopropyl Betaine|2|2.00|5.5–6.0|N
5|Capryl/Capramidopropyl Betaine|2|5.00|5.5–6.0|S
6|Cocamidopropyl Hydroxysultaine|3|1.80|5.5–6.0|N
7|Lauryl Hydroxysultaine|4|2.40|5.5–6.0|M
8|Lauryl Hydroxysultaine|5|2.40|5.5–6.0|M
9|Cocamide Methyl MEA|5|2.00|5.5–6.0|N
10|Sodium PEG-7 Olive Oil Carboxylate|6|1.44|5.5–6.0|N
11|Sodium Trideceth Sulfate|7|3.25|5.5–6.0|M
12|TAE-Cocoyl Glutamate|8|5.00|5.5–6.0|N
13|Sodium Taurine Cocoyl Methyltaurate|9|3.00|5.5–6.0|N
14|Sodium Taurine Cocoyl Methyltaurate|9|5.00|5.5–6.0|S
15|Laureth-5 Carboxylic Acid|10|3.00|5.5–6.0|N
16|Laureth-5 Carboxylic Acid|10|5.00|5.5–6.0|S
17|Ammonium Laureth Sulfate|5|5.00|5.5–6.0|M
18|Sodium Lauroyl Sarcosinate|4|5.00|5.5–6.0|M
19|PPG-5-Ceteth-20|11|5.00|6.0–6.5|N
20|Polyglyceryl-10 Isostearate|12|5.00|6.0–6.5|N
21|Polyglyceryl-10 Laurate|13|3.00|6.0–6.5|N
22|PEG-120 Methyl Glucose Dioleate|14|1.00|6.0–6.5|N
23|Cocamidopropyl Betaine|15|2.00|6.0–6.5|M
24|Cocamidopropyl Betaine|1|2.00|6.0–6.5|S
25|Cocamidopropyl Betaine|16|2.00|6.0–6.5|S
26|Cocamidopropyl Betaine|2|2.00|6.0–6.5|N
27|Sodium Cocoamphoacetate|2|1.90|6.0–6.5|N
28|Sodium Cocoamphoacetate|2|3.20|6.0–6.5|S
29|Sodium Lauroamphoacetate|3|1.50|6.0–6.5|N
30|Sodium Lauroamphoacetate|3|3.20|6.0–6.5|N
31|Sodium Lauroamphoacetate|15|5.00|6.0–6.5|M
32|Sodium Lauroamphoacetate|1|5.00|6.0–6.5|S
33|Disodium Cocoamphodiacetate|17|1.90|6.0–6.5|N
34|Disodium Cocoamphodiacetate|2|3.20|6.0–6.5|N
35|Disodium Cocoamphodiacetate|2|5.00|6.0–6.5|N
36|Polyquaternium-10|18|0.30|6.0–6.5|N
37|Sodium Methyl Cocoyl Taurate|19|1.50|6.0–6.5|N
38|Sodium Methyl Cocoyl Taurate|19|3.00|6.0–6.5|N
39|Sodium Cocoyl Alaninate|8|1.50|6.0–6.5|S
40|Disodium Cocoyl Glutamate|4|4.50|6.0–6.5|N
41|Disodium Cocoyl Glutamate|4|8.00|6.0–6.5|N
42|Sodium Laureth Sulfate|5|5.00|6.0–6.5|S
43|Lauramidopropyl Hydroxysultaine|4|3.50|6.0–7.0|M
44|Lauramidopropyl Hydroxysultaine|15|3.50|6.0–7.0|M
45|Lauramidopropyl Hydroxysultaine|3|3.50|6.0–7.0|M
46|Sodium Lauroyl Glutamate|4|5.00|6.5–7.0|S
47|Potassium Cocoyl Glycinate|20|4.50|7.0–8.0|N
48|Sodium Cocoyl Glycinate|17|4.50|7.0–8.0|S
49|Phenoxyethanol|1|0.50|No adjustment|N
50|Phenoxyethanol|1|1.00|No adjustment|N
51|Phenoxyethanol|1|1.00|No adjustment|N
52|Hydroxyacetophenone|21|0.50|No adjustment|N
53|Caprylhydroxamic Acid|22|1.50|No adjustment|N
54|Orange Blossom Fragrance|23|0.10|No adjustment|S
55|Red Mandarin Essential Oil|24|0.10|No adjustment|N
56|Fragrance 369|25|0.20|No adjustment|M
57|Fragrance 440095|21|0.20|No adjustment|N
58|Dipropylene Glycol|26|0.02|No adjustment|N'''

def main():
    labels={'N':'Non-irritating','S':'Slightly irritating','M':'Mildly irritating'}
    simple={'Phenoxyethanol','Hydroxyacetophenone','Caprylhydroxamic Acid','Sodium Lauroyl Sarcosinate','Sodium Lauroyl Glutamate'}
    rows=[]
    for line in TABLE2.splitlines():
        n,name,company,pct,ph,label=line.split('|');n=int(n)
        rows.append({'sample_id':f'baby_table2_{n:02d}','test_number':n,'ingredient_reported':name,
            'supplier_reported':f'Company {company}','concentration_reported':pct+'%',
            'concentration_percent_numeric':float(pct),'concentration_basis':'percent; w/w vs w/v and commercial raw material vs active basis not resolved',
            'ph_reported':ph,'ph_midpoint':None,'original_draize_label':labels[label],
            'rinse_condition_reported':'Do not rinse off' if n==51 else 'not individually specified in Table 2',
            'assay':'modified rabbit Draize; 3 rabbits/sample; 0.1 mL; source supplement',
            'observation_times':'1,24,48,72 h; days 4 and 7',
            'ingredient_group':name,'unit_of_observation':'raw-material supplier/concentration/test condition, not independent marketed product',
            'structure_status':'potential discrete molecule; actual commercial grade/isomer/purity still needs confirmation' if name in simple else 'UVCB/polymer/mixture or identity unresolved; no single structure assigned',
            'complete_recipe_available':False,'epa_A':None,'epa_B':None,
            'quantitative_external_validation_allowed':False,'training_allowed':False,
            'blocking_reasons':'no complete tested recipe/vehicle/pH adjusters and concentration basis; no individual tissue/time scores or verified EPA label mapping; independent study linkage unverified',
            'source_url':BABY,'publisher_pdf_page':7 if n<=24 else 8 if n<=49 else 9,
            'verification':'PMC Table 2 text + publisher PDF visual inspection, spanning cells resolved',
            'notes':'Source spells TAE-Cocoyl; not silently corrected to TEA' if n==12 else ''})
    table2=pd.DataFrame(rows);assert len(table2)==58 and table2.test_number.is_unique
    assert table2.ingredient_group.nunique()==39
    csv(HERE/'baby_table2_observation_audit.csv',table2)
    t3=pd.read_csv(SEARCH/'baby_bath_2025_table3_review_only.csv',keep_default_na=False)
    expected=[('', '', '6.0–6.5','Slightly irritating'),('Panthenol','1','6.0–7.0','Non-irritating'),
        ('Polyquaternium-51','0.007','6.0–7.0','Slightly irritating'),
        ('Maltooligosyl glucoside/hydrogenated starch hydrolysate','1.48','6.0–7.0','Non-irritating'),
        ('PEG-80 sorbitan laurate','3.5','5.5–6.0','Non-irritating'),('Disodium cocoyl glutamate','3','5.5–6.0','Mildly irritating')]
    for i,r in t3.iterrows():
        a,p,h,l=expected[i]
        assert r.additional_ingredient==a and str(r.additional_pct_reported)==p
        assert r.ph_range_reported==h and r.original_draize_label==l
        assert float(r.main_pct_reported)==5 and r.epa_label==''
    t3['source_page']=10;t3['numeric_and_label_transcription_verified']=True
    t3['concentration_basis']='percent; mass/volume and commercial-active basis unresolved'
    t3['new_compound_condition_not_additional_product']=[False,True,True,True,True,True]
    t3['control_link']='Table 2 test 42; count once; no separate study identifier provided'
    t3['quantitative_external_validation_allowed']=False;t3['training_allowed']=False
    t3['assay']='rabbit Draize; do not join to differently diluted SIRC cell observations'
    csv(HERE/'baby_table3_observation_audit.csv',t3)
    iivs=pd.read_csv(SEARCH/'iivs_bcop_table2_review_only.csv',keep_default_na=False)
    expected_values=[(1,4,'----','----','2',3.2),(2,4,'----','----','2',1.6),(3,3.5,'----','----','2',3.7),
        (4,4.5,'10','----','----',.17),(5,8.5,'50','----','----',21.3),(6,4.5,'10','----','2',41.9),
        (7,4,'15','----','2',44.6),(8,4,'30','----','2',79),(9,4,'----','3.9','----',4.6),
        (10,3,'5','3.9','----',3.1),(11,4.5,'10','3.2','----',3.2),(12,4.5,'15','3.2','----',3.6),
        (13,4.5,'20','1.4','1.5',85),(14,4,'30','3.9','2',119.4)]
    op=[3.2,1.7,3.7,.3,18.5,41.9,44.4,78.1,4.6,2.9,3.2,3.4,66.1,117.2]
    perm=[.001,-.004,0,-.009,.189,0,.01,.064,-.001,.012,0,.01,1.059,.147]
    for r,e in zip(iivs.itertuples(),expected_values):
        assert (r.formulation_id,r.ph,str(r.ethanol_pct_raw),str(r.glycolic_acid_pct_raw),str(r.salicylic_acid_pct_raw),r.bcop_ivis_reported)==e
        assert r.epa_label==''
    iivs['mean_opacity_reported']=op;iivs['mean_permeability_reported']=perm
    iivs['ivis_calculated_from_rounded_means']=np.array(op)+15*np.array(perm)
    iivs['ivis_reported_minus_calculated']=iivs.bcop_ivis_reported-iivs.ivis_calculated_from_rounded_means
    iivs['large_arithmetic_discrepancy']=iivs.ivis_reported_minus_calculated.abs()>.2
    iivs['verification']='Original PDF text via web reader; direct download HTTP403 and screenshot unavailable'
    iivs['concentration_basis']='percent; w/w vs w/v not resolved in poster'
    iivs['tested_dilution_exposure_duration']='not specified sufficiently for recipe-level reconstruction in poster'
    iivs['quantitative_external_validation_allowed']=False;iivs['training_allowed']=False
    csv(HERE/'iivs_table2_observation_audit.csv',iivs)
    source_review={
        'stoptx':{'primary_paper':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8863177/',
            'doi':'10.1289/EHP9341','observed_label':'GHS categories 1 through 2B are positive; not classified is negative',
            'original_sources':'REACH study results plus literature; OECD 405 rabbit lesion/reversibility data',
            'original_curation':'Reject nonexperimental/nonprotocol data; remove mixtures, inorganics, organometallics; clean/neutralize salts; normalize chemotypes; consolidate concordant replicates and remove discordant eye replicates using standardized structures.',
            'our_standardization':'RDKit cleanup, largest parent, uncharge, remove stereochemistry, canonical tautomer; conservative screening implementation, not claimed to exactly reproduce authors software.',
            'source_warning':'Paper has inconsistent balanced-set counts/skin wording. Training uses verified Table S4 3545 observed rows, not those prose counts or model-prediction sheets.'},
        'baby':{'source':BABY,'doi':'10.3390/toxics13110948','market_products':39,
            'raw_material_names':39,'table2_observation_conditions':58,'table3_display_rows':6,
            'new_compound_conditions_table3':5,'total_test_samples_per_supplement':102,'rabbits_per_sample':3,
            'n_rabbits_per_supplement':306,'labels_by_table2_condition':table2.original_draize_label.value_counts().to_dict(),
            'label_compatibility':'Supplement Table S1 recovered and visually inspected. Grades use response/recovery criteria and two-of-three animal averages; individual tissue/time records needed to derive EPA A/B. No automatic grade mapping performed.',
            'protocol_separation':'Animal Table 3 uses 5% SLES. SIRC cell experiment uses 0.01% SLES and separately diluted additives; six cell replicates are not six independent products.',
            'case_findings':['Same CAPB concentration and pH, supplier-dependent different grades (tests 23-26).','Table 3 composition effects are qualitative cases; UVCB grade and pH/vehicle prevent isolated causal attribution.'],
            'data_availability':'On reasonable request to corresponding authors; no author contacted.',
            'missing':['complete per-product recipe keyed to study/sample ID','all vehicle and pH-adjuster identities/amounts','percent basis and active-content basis','commercial grade, CAS/structure distribution, purity, EO/chain distribution for SLES and polymers','raw animal-by-tissue scores at each time and reversibility/rinse details','study IDs/batch IDs and overlap references']},
        'iivs':{'source':IIVS,'numbered_formulations':14,'methods_claimed_formulations':15,
            'transcription_verified_rows':14,'epa_mapping_performed':False,
            'issues':['Full vehicles absent; dashes not treated as zero; concentrations not converted to mass fractions.',
                'Formulation 13: reported IVIS 85 versus 66.1+15*1.059=81.985 from printed means; retained as reported.',
                'Formulation 8 IVIS 79 is severe under poster Table 1, but narrative groups 6-8 as moderate.',
                'Abstract and table differ on formulation 13 active concentrations; table values preserved.',
                '14 numbered observations do not resolve methods mention of 15; positive control is not counted as an extra cosmetic product.'],
            'missing':['complete vehicle recipe per formulation ID','concentration units/basis','raw replicates and test exposure/dilution details','clarification of table/narrative discrepancies','EPA-compatible in vivo observations or validated endpoint bridge']},
        'decision':{'eligible_independent_EPA_external_samples':0,'external_performance_computed':False,
            'case_analysis_only':True,'cosmetic_rows_used_for_learning_tuning_threshold':0,
            'chemexpo':'composition/unlabeled candidate only; no observed external Eye gold labels; no 10% rule accuracy computed',
            'author_contacted':False}}
    write(HERE/'external_validation_audit.json',source_review)
    print('COSMETIC AUDIT: 58 raw-material conditions, 6 displayed mixture rows (5 new), 14 IIVS; EPA eligible=0')

if __name__=='__main__':main()
