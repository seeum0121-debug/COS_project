"""Research-only Eye EPA inference; not a Zein or cosmetic-safety predictor."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
import joblib
from rdkit import Chem

OUT=Path(__file__).resolve().parent
PACKAGE=OUT.parent.parent/'검증/skin_team_audit_20261108/source/pretraining'

def validate(payload):
    components=payload.get('components')
    if not isinstance(components,list) or not components: raise ValueError('Nonempty components list required')
    smiles=[];weights=[]
    for item in components:
        s=item.get('smiles');w=item.get('weight')
        if not isinstance(s,str) or not s.strip() or Chem.MolFromSmiles(s) is None: raise ValueError('Valid source-supported SMILES required')
        if isinstance(w,bool) or not isinstance(w,(int,float)) or not np.isfinite(w) or w<0: raise ValueError('Finite nonnegative relative weight required')
        smiles.append(s);weights.append(w)
    if sum(weights)<=0: raise ValueError('Positive total weight required')
    return smiles,weights

def predict(payload):
    smiles,weights=validate(payload)
    bundle=joblib.load(OUT/'eye_chemberta_head.joblib')
    if bundle['format']!='eye-chemberta-frozen-epa-v1': raise ValueError('Wrong head format')
    checksum=hashlib.sha256((PACKAGE/'best/model.safetensors').read_bytes()).hexdigest()
    if checksum!=bundle['encoder_sha256']: raise ValueError('Encoder differs from training')
    sys.path.insert(0,str(PACKAGE))
    from mixture_predictor import MixturePredictor
    encoder=MixturePredictor(PACKAGE)
    vectors=encoder.encode_components(smiles)
    x=np.average(vectors,axis=0,weights=weights).astype(np.float32).astype(np.float64).reshape(1,-1)
    score=float(bundle['pipeline'].predict_proba(x)[0,1])
    return {'endpoint':'eye_epa','label_definition':'1=EPA I/II; 0=EPA III/IV, not no irritation','score':score,'prediction':int(score>=bundle['threshold']),'threshold':bundle['threshold'],'validation_scope':'internal_group_holdout_only','warning':'Research only; disclose all represented components; do not infer missing structure or treat score as calibrated risk'}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists(): raise FileExistsError(args.output)
    result=predict(json.loads(args.input.read_text(encoding='utf-8-sig')))
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output: args.output.write_text(text,encoding='utf-8')
    print(text)
