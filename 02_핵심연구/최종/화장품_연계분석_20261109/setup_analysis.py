"""Snapshot read-only inputs; all analysis output stays in this new folder."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
FINAL=ROOT.parent
COMPARISON=FINAL/'모델 실험/eye/모델 최종/aggregation_compare_20261109'
MODEL=FINAL/'최종모델'
CHEM=FINAL/'준비/학습용데이터'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    result={}
    for path in FINAL.rglob('*'):
        if path.is_relative_to(ROOT) or not path.is_file():
            continue
        stat=path.stat()
        result[str(path.relative_to(FINAL))]=[stat.st_size,stat.st_mtime_ns]
    return result


def main():
    assert not (ROOT/'input_manifest.json').exists(),'Setup already complete'
    for name in ['inputs','inference','outputs','figures','evidence','cache','cache/matplotlib','cache/temp']:
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    (ROOT/'original_file_inventory.json').write_text(json.dumps(inventory(),ensure_ascii=False),encoding='utf-8')
    sources={
        'inputs/chemexpo_personal_care.csv':CHEM/'ChemExpo_personal_care_full_composition_108.csv',
        'inputs/chemexpo_eye.csv':CHEM/'eye_personal_care_full_composition_21.csv',
        'inputs/production_rows.csv':COMPARISON/'input_snapshot.csv',
        'inputs/production_components.csv':COMPARISON/'component_linkage.csv',
        'inputs/production_wmean.npy':COMPARISON/'wmean.npy',
        'inputs/comparison_oof.csv':COMPARISON/'oof_predictions.csv',
        'inputs/comparison_config.json':COMPARISON/'config.json',
        'inference/eye_A_LR_wmean.joblib':MODEL/'eye_A_LR_wmean.joblib',
        'inference/predict.py':MODEL/'predict.py',
        'inference/metadata.json':MODEL/'metadata.json',
        'inference/feature_names.json':MODEL/'feature_names.json',
        'inputs/prior_external_audit.json':FINAL/'외부학습/Eye_STopTox_20261108/external_validation_audit.json',
    }
    manifest={}
    for relative,path in sources.items():
        shutil.copyfile(path,ROOT/relative)
        manifest[relative]={'source':str(path),'sha256':sha(path)}
    repo=Path('C:/Users/DS/COS/github_final_model_20261109')
    revision=subprocess.check_output(['git','rev-parse','origin/main'],cwd=repo,text=True).strip()
    data=subprocess.check_output(['git','show',revision+':cosmetic_validation/cir_ingredients.csv'],cwd=repo)
    (ROOT/'inputs/cir_ingredients_candidate_list.csv').write_bytes(data)
    manifest['inputs/cir_ingredients_candidate_list.csv']={'source':f'git:{revision}:cosmetic_validation/cir_ingredients.csv','sha256':hashlib.sha256(data).hexdigest()}
    (ROOT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    protocol={
        'created_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Cosmetics relevance and input-composition uncertainty, not external cosmetic performance validation.',
        'write_scope':str(ROOT),'existing_files':'read only; no model refit, threshold retune, Git commit/push, or changes outside this folder',
        'model':'Frozen Eye A LR weighted mean; threshold and 2265-feature recipe unchanged',
        'cosmetics_evidence':'Review product category together with product/document title. Exclude industrial misclassification and medical/oral-care scope; separate unclassified plausible cosmetics.',
        'identity':'Exact checksum-valid CAS matching; no automatic correction of malformed CAS. CIR converted labels are never used as ground truth.',
        'subgroups':[
            'original 606 overall',
            'at least one represented non-water ingredient CAS also reported in screened cosmetic documents',
            'at least 50% of represented ingredient weight from those non-water cosmetic CAS',
            'no represented non-water cosmetic CAS overlap',
        ],
        'subgroup_selection':'Rules use composition and product identity, never scores/labels. Metrics use saved original nested-CV held-out predictions and original fold thresholds; no retraining.',
        'subgroup_reporting':'Report unique rows, groups, positive/negative counts and three seed results. AUC undefined for a single class. Flag <20 rows, <10 groups, or <5 of either class as too sparse for stable comparison.',
        'case_eligibility':'Select before scores: screened cosmetic product, unambiguous discrete structures, usable reported weight bounds, at least two distinct represented structures, feasible sum of disclosed fractions <=1. All omitted/unknown fractions remain unknown.',
        'case_scope':'If recipe completeness is not independently established, scores describe the normalized disclosed and resolved component representation, not an experimentally characterized complete product.',
        'uncertainty':'Use only source lower/upper bounds or exact central values; never treat midpoint as measured. Enforce nonnegative mass fractions and disclosed sum <=1. Do not invent water/vehicle amounts.',
        'bounds':'For all-finite component vectors, frozen LR logit is a linear-fractional function of component amounts. Use constrained linear programming for exact representation score extrema and verify against the model.',
        'scenarios':'Seeded uniform-within-reported-bounds scenarios conditioned on disclosed sum <=1. Scenario proportions are not biological risk probabilities or statistical confidence intervals.',
        'limitations':['No cosmetic assay gold labels','EPA III/IV negative does not mean no eye irritation','Source mixture/grade and pH can be unresolved','No ingredient causal claims or safe formulation optimization'],
    }
    (ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SETUP COMPLETE',ROOT,'snapshots',len(manifest),flush=True)


if __name__=='__main__':main()
