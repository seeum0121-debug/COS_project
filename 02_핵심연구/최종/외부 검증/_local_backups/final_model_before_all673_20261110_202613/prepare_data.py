"""Build the 16-row target cohort from archived recipes and original D5 labels."""
from pathlib import Path
import csv
from collections import defaultdict
import re

import pymupdf

from common import read_json, write_json, sha256

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    for name, info in read_json(ROOT/'source_manifest.json').items():
        if sha256(ROOT/name) != info['sha256']:
            raise ValueError(f'Source modified: {name}')
    protocol = read_json(ROOT/'protocol.json')
    assert sha256(ROOT/'mapping.json') == protocol['mapping_sha256']
    original_labels = {}
    with pymupdf.open(ROOT/'data/sources/HETCAM_appendixD5.pdf') as document:
        for page_index in (8, 9, 10):
            for table in document[page_index].find_tables().tables:
                for row in table.extract():
                    text = ' '.join(str(value) for value in row)
                    if 'Gettings' not in text or '1996' not in text or not re.search(r'HZ[A-Z]', text):
                        continue
                    code = re.search(r'HZ[A-Z]', row[0]).group()
                    category = re.sub(r'\s+', '', row[10]).removeprefix('Category')
                    assert category in ('I', 'II', 'III', 'IV')
                    assert row[3].casefold() == 'undiluted'
                    assert code not in original_labels
                    original_labels[code] = {'epa': category, 'pdf_page': page_index + 1,
                                            'printed_page': f'D-{page_index+33}',
                                            'in_vivo_concentration': row[3],
                                            'in_vitro_concentration_not_target': row[2]}
    assert len(original_labels) == 25
    registry = {row['formulation_id']: row for row in
                read_csv(ROOT/'data/sources/CTFA53_formulation_registry.csv')}
    grouped = defaultdict(list)
    for row in read_csv(ROOT/'data/sources/CTFA53_material_inputs.csv'):
        if row['formulation_id'].startswith('CTFA1996_'):
            grouped[row['formulation_id']].append(row)
    assert len(grouped) == 25
    cohort, candidate_audit, components, payloads = [], [], [], []
    for fid, rows in sorted(grouped.items()):
        code = fid.split('_')[1]
        label = original_labels[code]
        assert label['epa'] == registry[fid]['epa_neat']
        numeric = all(row['amount_kind'] == 'exact_as_supplied_wt_percent' for row in rows)
        total = sum(float(row['wt_percent_as_supplied']) for row in rows) if numeric else None
        included = numeric and abs(total - 100.) <= protocol['quantity_tolerance_percent_points']
        candidate_audit.append({'formulation_id': fid, 'included': included,
                                'numeric_amounts': numeric, 'total_pct': total,
                                'epa': label['epa'], 'reason': 'included' if included else
                                ('nonnumeric_quantity' if not numeric else 'mass_balance_outside_0.11')})
        if not included:
            continue
        payload = {'formulation_id': fid, 'components': [
            {'name': row['ingredient_as_reported'], 'amount_pct': float(row['wt_percent_as_supplied'])}
            for row in rows]}
        payloads.append(payload)
        cohort.append({'formulation_id': fid, 'product_name': registry[fid]['product_name'],
                       'epa': label['epa'], 'label': int(label['epa'] in ('I', 'II')),
                       'total_pct': total, 'label_pdf_page': label['pdf_page'],
                       'recipe_pdf_pages': ';'.join(sorted({row['source_pdf_page'] for row in rows}))})
        for row in rows:
            components.append({'formulation_id': fid, 'ingredient_order': row['ingredient_order'],
                               'name': row['ingredient_as_reported'],
                               'amount_pct': float(row['wt_percent_as_supplied']),
                               'source_pdf_page': row['source_pdf_page']})
    assert len(cohort) == 16 and sum(row['label'] for row in cohort) == 11
    write_csv(ROOT/'data/formulations.csv', cohort)
    write_csv(ROOT/'data/candidate_audit.csv', candidate_audit)
    write_csv(ROOT/'data/components.csv', components)
    write_json(ROOT/'data/payloads.json', payloads)
    write_json(ROOT/'data/original_D5_labels.json', original_labels)
    write_json(ROOT/'examples/CTFA1996_HZT.json', next(p for p in payloads if p['formulation_id'].endswith('_HZT')))
    write_json(ROOT/'data/preparation_verification.json', {
        'candidate_rows': 25, 'included_rows': 16, 'positive': 11, 'negative': 5,
        'labels_reextracted_from_original_D5': True, 'labels_match_archived_registry': True,
        'source_hashes_match': True, 'recipe_table': 'archived extraction from Appendix B4',
        'independence_from_source_training': 'unresolved'})
    print('Prepared 16 rows (11 positive, 5 negative); 25 labels checked against original D5.', flush=True)


if __name__ == '__main__':
    main()
