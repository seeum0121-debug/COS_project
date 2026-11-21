"""Document-backed, preselected partial-composition sensitivity cases."""
from pathlib import Path
import json, hashlib, urllib.request, time
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    # Selection uses identities, number of components and amount availability only.
    # No case scores have been computed when this file/plan is written.
    cases = [
        dict(document_id=277977, title='Body Spray', tier='primary_cosmetic',
             represented_cas=['64-17-5', '7732-18-5'],
             omitted_reason='Ethoxylated hydrogenated castor oil is a distributed mixture; no single structure is assigned.',
             reason='Three reported components; two discrete structures; source amount ranges for all components.'),
        dict(document_id=281340, title='Nail Polish', tier='primary_cosmetic',
             represented_cas=['123-86-4', '141-78-6', '67-63-0'],
             omitted_reason='Nitrocellulose is a polymer of unspecified chain length/nitration; no surrogate structure.',
             reason='Four reported components; three discrete solvents; source amount ranges for all components.'),
        dict(document_id=277611, title='Bath Fizz', tier='supplementary_title_only',
             represented_cas=['77-92-9', '144-55-8', '7757-82-6'],
             omitted_reason='Nonoxynol-10 is a distributed ethoxylated mixture; no single structure is assigned.',
             reason='Four reported components; three discrete identities; source amount ranges for all components. Supplementary title-only cosmetics screen.'),
    ]
    plan = dict(cases=cases, selection='Purposive demonstrations, not a representative product sample or score-based selection.',
        score_scope='ONLY normalized resolved ingredient subset. No whole-product safety or irritation classification.',
        mass_constraint='Sum of represented amounts plus omitted component lower bounds <= 1. Unreported remainder stays unknown.',
        salt_representation='Retain full ionic SMILES for Morgan bits; saved model uses largest fragment descriptors. Do not simulate solution chemistry.',
        interactions='Model has no explicit pH, reaction, aerosol exposure, polymer grade or molecular interaction inputs.',
        completeness='None of these three cases has a fully structure-resolved product representation; all are conditional sensitivity cases.')
    (ROOT / 'case_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    structures = []
    for cas in sorted(set(c for case in cases for c in case['represented_cas'])):
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/SMILES,InChIKey,MolecularFormula/JSON'
        target = ROOT / 'evidence' / f'pubchem_{cas}.json'
        if target.exists():
            data = target.read_bytes()
        else:
            request = urllib.request.Request(url, headers={'User-Agent': 'COS-composition-research/1.0'})
            with urllib.request.urlopen(request, timeout=40) as response:
                data = response.read()
            target.write_bytes(data)
            time.sleep(.25)
        records = json.loads(data)['PropertyTable']['Properties']
        if len(records) != 1: raise ValueError(f'Ambiguous CAS response {cas}: {len(records)} records')
        r = records[0]
        synonyms_url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{r["CID"]}/synonyms/JSON'
        synonyms_file = ROOT / 'evidence' / f'pubchem_{cas}_synonyms.json'
        if synonyms_file.exists():
            syn_data = synonyms_file.read_bytes()
        else:
            req = urllib.request.Request(synonyms_url, headers={'User-Agent': 'COS-composition-research/1.0'})
            with urllib.request.urlopen(req, timeout=40) as response:
                syn_data = response.read()
            synonyms_file.write_bytes(syn_data)
            time.sleep(.25)
        syn = json.loads(syn_data)['InformationList']['Information'][0]['Synonym']
        assert cas in syn, f'Exact CAS synonym not found: {cas}'
        structures.append(dict(cas=cas, cid=r['CID'], smiles=r['SMILES'], inchi_key=r['InChIKey'],
            molecular_formula=r['MolecularFormula'], exact_cas_synonym=True, property_url=url,
            synonym_url=synonyms_url, property_sha256=hashlib.sha256(data).hexdigest(),
            synonym_sha256=hashlib.sha256(syn_data).hexdigest(),
            retrieved_utc=datetime.now(timezone.utc).isoformat()))
        print(cas, r['CID'], r['MolecularFormula'], r['SMILES'], flush=True)
    pd.DataFrame(structures).to_csv(ROOT / 'outputs/structure_registry.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__': main()
