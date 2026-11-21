from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM,AutoTokenizer

OUT=Path(__file__).resolve().parent
BASE=OUT.parent.parent/'외부 검증/Eye_cosmetic_validation_20261109'
ENC=OUT.parent.parent/'검증/skin_team_audit_20261108/source/pretraining/best'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x): (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
refs=[BASE/'training_reference/input_snapshot.csv',BASE/'training_reference/component_linkage.csv',BASE/'training_reference/wmean.npy',BASE/'model/eye_A_LR_wmean.joblib',ENC/'model.safetensors',ENC/'config.json',ENC/'tokenizer.json',OUT/'protocol.json']
manifest={str(p):sha(p) for p in refs}
save('input_manifest.json',manifest)
link=pd.read_csv(BASE/'training_reference/component_linkage.csv')
n=len(pd.read_csv(BASE/'training_reference/input_snapshot.csv',low_memory=False))
assert link.smiles.notna().all() and np.isfinite(link.weight).all() and link.weight.ge(0).all()
smiles=sorted(link.smiles.unique())
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
tok=AutoTokenizer.from_pretrained(ENC,local_files_only=True)
model=AutoModelForMaskedLM.from_pretrained(ENC,local_files_only=True).roberta.cpu().eval()
for p in model.parameters(): p.requires_grad_(False)
assert model.config.hidden_size==384
special=set(tok.all_special_ids)
valid=[];tokens=[];audit=[]
for s in smiles:
    t=tok(s,truncation=False,return_attention_mask=True);ids=t['input_ids']
    reason='token_length_gt128' if len(ids)>128 else 'unknown_token' if tok.unk_token_id in ids else 'no_content' if not any(i not in special for i in ids) else ''
    audit.append({'smiles':s,'tokens':len(ids),'eligible':not reason,'reason':reason})
    if not reason: valid.append(s);tokens.append(t)
pd.DataFrame(audit).to_csv(OUT/'token_audit.csv',index=False,encoding='utf-8-sig')
emb=[]
for start in range(0,len(valid),16):
    batch=tok.pad(tokens[start:start+16],padding=True,return_tensors='pt')
    keep=batch['attention_mask'].bool()
    for sid in special: keep=keep & batch['input_ids'].ne(sid)
    with torch.inference_mode():
        hidden=model(**batch).last_hidden_state
        pooled=(hidden*keep.unsqueeze(-1)).sum(1)/keep.sum(1).unsqueeze(-1)
    emb.append(pooled.numpy())
    print(f'Encoded {min(start+16,len(valid))}/{len(valid)}',flush=True)
E=np.concatenate(emb);assert E.shape==(len(valid),384) and np.isfinite(E).all()
np.save(OUT/'component_embeddings.npy',E)
save('component_order.json',valid)
lookup={s:E[i] for i,s in enumerate(valid)}
X=np.full((n,384),np.nan,dtype=np.float32)
rows=[]
for idx,group in link.groupby('row_index'):
    reason='unsupported_component' if any(s not in lookup for s in group.smiles) else ''
    if group.weight.sum()<=0: reason='zero_weight_sum'
    if not reason: X[int(idx)]=np.average(np.stack([lookup[s] for s in group.smiles]),axis=0,weights=group.weight)
    rows.append({'row_index':int(idx),'formulation_id':group.formulation_id.iloc[0],'eligible':not reason,'reason':reason,'n_components':len(group),'weight_sum':float(group.weight.sum())})
assert len(rows)==n
np.save(OUT/'chemberta_wmean.npy',X)
pd.DataFrame(rows).to_csv(OUT/'encoding_rows.csv',index=False,encoding='utf-8-sig')
assert all(sha(Path(p))==h for p,h in manifest.items())
save('encoding_summary.json',{'rows':n,'eligible_rows':int(np.isfinite(X).all(1).sum()),'unique_smiles':len(smiles),'encoded_smiles':len(valid),'encoder_frozen':True,'encoder_dim':384,'torch':torch.__version__,'encoder_sha256':manifest[str(ENC/'model.safetensors')],'upstream_pretraining_overlap':'unknown'})
print('Encoding complete',flush=True)
