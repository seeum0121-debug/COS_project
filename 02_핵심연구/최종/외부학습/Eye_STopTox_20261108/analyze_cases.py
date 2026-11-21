"""Descriptive original-endpoint cases only; no model evaluation or EPA mapping."""
from common import *
import pandas as pd

def main():
    d=pd.read_csv(HERE/'baby_table2_observation_audit.csv',keep_default_na=False)
    supplier=[]
    keys=['ingredient_reported','concentration_reported','ph_reported','rinse_condition_reported']
    for key,g in d.groupby(keys,sort=False):
        if g.supplier_reported.nunique()<2:continue
        supplier.append(dict(zip(keys,key),n_test_conditions=len(g),n_supplier_codes=g.supplier_reported.nunique(),
            test_numbers='|'.join(g.test_number.astype(str)),
            supplier_to_original_grade='; '.join(f'{r.supplier_reported}: {r.original_draize_label}' for r in g.itertuples()),
            grade_varies_across_suppliers=g.original_draize_label.nunique()>1,
            interpretation='descriptive raw-material condition comparison; full grade/purity/test-replicate details unavailable; not independent products or causal supplier effect'))
    concentration=[]
    keys=['ingredient_reported','supplier_reported','ph_reported','rinse_condition_reported']
    for key,g in d.groupby(keys,sort=False):
        if g.concentration_percent_numeric.nunique()<2:continue
        g=g.sort_values('concentration_percent_numeric')
        concentration.append(dict(zip(keys,key),n_test_conditions=len(g),
            test_numbers='|'.join(g.test_number.astype(str)),
            concentration_to_original_grade='; '.join(f'{r.concentration_reported}: {r.original_draize_label}' for r in g.itertuples()),
            grade_changes_across_reported_concentrations=g.original_draize_label.nunique()>1,
            interpretation='descriptive contrast with same ingredient/supplier/reported pH entry; No adjustment does not establish equal measured pH; no invented concentration or EPA conversion'))
    csv(HERE/'baby_supplier_case_comparisons.csv',pd.DataFrame(supplier))
    csv(HERE/'baby_concentration_case_comparisons.csv',pd.DataFrame(concentration))
    iivs=pd.read_csv(HERE/'iivs_table2_observation_audit.csv',keep_default_na=False)
    def category(v):
        if 0<=v<=3:return 'Minimally irritating'
        if 3.1<=v<=25:return 'Mildly irritating'
        if 25.1<=v<=55:return 'Moderately irritating'
        if v>=55.1:return 'Severely irritating'
        return 'Not assigned: gap in printed intervals'
    iivs['poster_table1_category_only']=iivs.bcop_ivis_reported.map(category)
    iivs['epa_conversion_performed']=False
    csv(HERE/'iivs_original_endpoint_cases.csv',iivs)
    write(HERE/'descriptive_case_summary.json',{
        'purpose':'source observation case analysis only; NOT predictive accuracy or independent EPA validation',
        'baby_supplier_contrast_groups':len(supplier),
        'baby_supplier_groups_with_different_original_grades':sum(r['grade_varies_across_suppliers'] for r in supplier),
        'baby_concentration_contrast_groups':len(concentration),
        'baby_concentration_groups_with_changed_original_grades':sum(r['grade_changes_across_reported_concentrations'] for r in concentration),
        'iivs_poster_original_category_counts':iivs.poster_table1_category_only.value_counts().to_dict(),
        'iivs_large_arithmetic_discrepancy_formulations':iivs.loc[iivs.large_arithmetic_discrepancy,'formulation_id'].tolist(),
        'cautions':['Do not count contrasts as additional independent products.','Original BCOP category intervals are not EPA categories.','These comparisons do not isolate causal ingredient/supplier effects.','Animal and SIRC cell exposures are kept separate.']})
    print('DESCRIPTIVE CASE ANALYSIS COMPLETE')

if __name__=='__main__':main()
