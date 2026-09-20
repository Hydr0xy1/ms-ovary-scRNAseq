"""Extract small, immutable evidence tables from the existing Project A archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bee-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.bee_root, args.output
    out.mkdir(parents=True, exist_ok=True)
    archive = 'local_reports/项目综合归档_20260722_v1/'
    deep = 'local_reports/deep_mining_20260718/'
    branch = 'server_backup/branches/branch_human_fibro_sctenifoldknk_pilot_20260324/results/'
    sources = {
        'apis_bulk_evidence.tsv.gz': deep + 'server_shutdown_results/tables/integration/Apis_activation_shutdown_gene_integration.tsv.gz',
        'bombus_gene_evidence.tsv.gz': archive + '05_蜂端关键证据/Bombus单核/tables/core_gene_evidence_matrix.tsv',
        'apis_fly_human_mapping.tsv.gz': deep + 'server_extension/results/deep_mining_20260719/bee_human_program_projection/Apis_Dmel_human_ortholog_bridge.tsv.gz',
        'human_mouse_strict_pairs.tsv.gz': deep + 'next_round_20260719/mouse_ovary_age_bridge/human_mouse_strict_1to1_pairs.tsv',
        'apis_human_alternative_mappings.tsv.gz': deep + 'next_round_20260719/p0_orthology_projection/all_mapping_tier_pairs.tsv.gz',
        'human_readout_evidence.tsv.gz': branch + 'step30_final_node_state_program_model_20260505_193854/tables/final_program_readout_summary.tsv',
        'bee_prior_registry.tsv.gz': branch + 'step29_genelevel_gsr_v32_forced_leiden_20260505_184854/01_clean_bee_registry/01_clean_bee_registry_long.tsv',
        'core159_mapping_review.tsv.gz': 'server_backup/branches/branch_bee_human_ovary_atlas_20260311/results/bee/step07_manual_pilot_capture/scaleup_core159_20260315_015743/final_core159_release_20260317_162435/tables/step07_core159_full.consensus.tsv',
    }
    manifest = []
    for name, source in sources.items():
        path = root / source
        raw = path.read_bytes()
        table = pd.read_csv(path, sep='\t', dtype=str).fillna('')
        table.insert(0, 'source_row_1based', range(2, len(table) + 2))
        destination = out / name
        if destination.exists():
            raise FileExistsError(f'Frozen source exists: {destination}')
        table.to_csv(destination, sep='\t', index=False, compression={'method': 'gzip', 'mtime': 0})
        manifest.append({'file': name, 'source_project': 'A', 'source_relative_path': source,
                         'source_sha256': hashlib.sha256(raw).hexdigest(),
                         'frozen_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                         'source_rows': len(table), 'bytes': destination.stat().st_size})
    pd.DataFrame(manifest).to_csv(out / 'SOURCE_MANIFEST.tsv', sep='\t', index=False)
    (out / 'SOURCE_FREEZE.json').write_text(json.dumps({
        'status': 'frozen_before_mouse_effect_inspection',
        'source': 'existing Project A local archive and server backup',
        'formal_analysis_server': 'connect.bjb1.seetacloud.com:28949',
        'raw_matrices_transferred': False, 'new_public_expression_data': False,
        'note': 'Local operation extracts evidence only. Formal Stage 25 calculations run on the main server.'
    }, indent=2), encoding='utf-8')
    print(pd.DataFrame(manifest)[['file', 'source_rows', 'bytes']].to_string(index=False))


if __name__ == '__main__':
    main()
