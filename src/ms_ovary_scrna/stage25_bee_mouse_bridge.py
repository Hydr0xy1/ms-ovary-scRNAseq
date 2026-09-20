"""Independent Stage 25 bridge. No prior-stage writes or cell-level inference."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

MODULE_RULES = {
    'mitochondrial_energy': r'mitochond|oxidative phosphorylation|atp synth|nad[hp].*dehydrogenase|cytochrome|respiratory.*chain',
    'RNA_processing_translation': r'ribosom|translat|splic|rna.binding|rna processing|ribonucleoprotein|rna helicase|nucleolar|eukaryotic initiation|signal recognition particle',
    'proteostasis_autophagy': r'autophag|proteasom|ubiquitin|protein folding|chaperon|heat.shock|lysosom|endoplasmic reticulum|protein disulfide|peptidase|cathepsin',
    'lipid_sterol_redox': r'lipid|sterol|cholesterol|fatty.acid|acyl.coa|elongation of.*fatty|glutathione|thioredoxin|peroxiredoxin|superoxide dismutase|lipase|lipoprotein|redox|steroid',
    'membrane_structure_transport': r'membrane|transmembrane|ion channel|solute carrier|vesicle|annexin|aquaporin|exocyt|endocyt|traffick|transport|spectrin|ankyrin',
    'ECM_stromal_support': r'extracellular.matrix|collagen|laminin|integrin|fibronectin|proteoglycan|matrix.metallo|basement.membrane|sialophosphoprotein|tenascin|thrombospondin',
    'stress_inflammation': r'stress|inflamm|immune|immunity|heat.shock|apopt|caspase|dna.damage|toll.like|defensin|antimicrobial|cytokine|injury|hypoxia',
    'reproductive_support_secretion': r'oogen|ovary|ovarian|vitellogen|yolk|germline|follic|ecdys|juvenile.hormone|broad.complex|\bbroad\b|\bhr4\b|secreted|secretory|secretion',
}
MODULES = list(MODULE_RULES)
POPS = ['Granulosa', 'Stromal_fibroblast']
LIBRARIES = [f'{g}_{n}' for g in ['Y', 'OC', 'OT'] for n in range(1, 4)]
CONTRASTS = [('OC', 'Y'), ('OT', 'OC'), ('OT', 'Y')]
TARGETS = ['Hif1a', 'Smad3', 'Igfbp2', 'Pak3', 'Abca1']
SEED = 20260925


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def write_tsv(frame, path):
    pd.DataFrame(frame).to_csv(path, sep='\t', index=False)


def bh(values):
    p = np.asarray(values, float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = np.argsort(p[valid])
    if len(valid):
        ranked = p[valid][order] * len(valid) / np.arange(1, len(valid) + 1)
        out[valid[order]] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return out


def exact_test(values, labels, test, reference):
    values, labels = np.asarray(values, float), np.asarray(labels)
    keep = np.isfinite(values) & np.isin(labels, [test, reference])
    v, lab = values[keep], labels[keep]
    n = int(np.sum(lab == test))
    if min(n, len(v) - n) < 2:
        return {'effect': np.nan, 'p_exact_two_sided': np.nan, 'n_permutations': 0}
    observed = v[lab == test].mean() - v[lab == reference].mean()
    null = []
    for chosen in itertools.combinations(range(len(v)), n):
        mask = np.zeros(len(v), bool)
        mask[list(chosen)] = True
        null.append(v[mask].mean() - v[~mask].mean())
    return {'effect': float(observed),
            'p_exact_two_sided': float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
            'n_permutations': len(null)}


def geometry(age, treatment):
    a, t = np.asarray(age, float), np.asarray(treatment, float)
    an, tn = np.linalg.norm(a), np.linalg.norm(t)
    if an < 1e-10 or tn < 1e-10:
        return dict(age_norm=an, treatment_norm=tn, cosine=np.nan,
                    age_projection_coefficient=np.nan, orthogonal_fraction=np.nan)
    cosine = float(np.clip(np.dot(a, t) / (an * tn), -1, 1))
    return dict(age_norm=float(an), treatment_norm=float(tn), cosine=cosine,
                age_projection_coefficient=float(np.dot(a, t) / np.dot(a, a)),
                orthogonal_fraction=float(np.sqrt(max(0, 1 - cosine ** 2))))


def base_classification(g, age_mean, treatment_mean):
    c = g['cosine']
    if not np.isfinite(c):
        return 'inconsistent'
    if abs(c) <= 0.3:
        return 'treatment_specific_orthogonal'
    if c < -0.3 and age_mean * treatment_mean < 0:
        return 'age_opposite'
    if c > 0.3 and age_mean * treatment_mean > 0:
        return 'age_parallel'
    return 'inconsistent'


def _source(root, name):
    return pd.read_csv(root / 'resources/cross_project_bee_bridge' / name,
                       sep='\t', dtype=str).fillna('')


def _id(value):
    value = str(value)
    return value[:-2] if value.endswith('.0') else value


def create_ledger(root, out):
    apis = _source(root, 'apis_bulk_evidence.tsv.gz')
    bombus = _source(root, 'bombus_gene_evidence.tsv.gz')
    human = _source(root, 'human_readout_evidence.tsv.gz')
    registry = _source(root, 'bee_prior_registry.tsv.gz')
    annotation = {r.bee_gene_id: f'{r.bee_symbol_display} {r.bee_description_display}'
                  for r in apis.itertuples()}
    rows = []

    def add(gene, species, dataset, state, tier, direct, mammalian, hypothesis,
            text, filename, source_row, apis_id='', forced_modules=None, note=''):
        modules = forced_modules or [m for m, regex in MODULE_RULES.items()
                                    if re.search(regex, text.lower())]
        for module in modules or ['unassigned']:
            match = re.search(MODULE_RULES.get(module, r'(?!)'), text.lower())
            rows.append(dict(source_gene=gene, module=module, source_project='A',
                             source_dataset=dataset, source_species=species,
                             source_state_or_program=state, evidence_tier=tier,
                             direct_bee_derived=direct, mammalian_supported=mammalian,
                             hypothesis_only=hypothesis, source_annotation=text,
                             assignment_match=match.group(0) if match else 'frozen_axis',
                             source_file=filename, source_row_1based=source_row,
                             apis_gene_id=_id(apis_id), evidence_note=note))
    for r in apis.itertuples():
        ns = int(float(r.n_significant_contexts or 0))
        if ns < 1:
            continue
        npos, nneg = int(float(r.n_positive_contexts or 0)), int(float(r.n_negative_contexts or 0))
        tier = 'Tier1_multi_context' if ns >= 2 and min(npos, nneg) == 0 else 'Tier2_context_specific'
        add(r.bee_gene_id, 'Apis_mellifera', 'GSE120561;GSE76164',
            'caste_and_activation_context;not_age', tier, True, False, False,
            annotation[r.bee_gene_id], 'apis_bulk_evidence.tsv.gz', r.source_row_1based,
            r.bee_gene_id, note='Existing significant-context membership; unsigned functional module.')
    for r in bombus.itertuples():
        apis_id = _id(r.apis_gene_id_1to1)
        text = f'{r.dmel_symbols} {r.dmel_descriptions} {annotation.get(apis_id, "")}'
        add(r.bombus_gene, 'Bombus_terrestris', 'ProjectA_Bombus_snRNA_Stage29',
            f'cluster_{r.cluster};{r.signature_modules}', r.evidence_tier,
            True, False, r.evidence_tier == 'Tier3_supporting', text,
            'bombus_gene_evidence.tsv.gz', r.source_row_1based, apis_id,
            note=f'Apis mapping: {r.apis_mapping_status}; upstream species label retained from project.')
    axis_map = {'lipid_metabolism_redox': 'lipid_sterol_redox',
                'stress_proteostasis': 'proteostasis_autophagy'}
    for r in human.itertuples():
        axis = axis_map.get(r.V3_program_axis, r.V3_program_axis)
        if r.gene == 'COL4A2':
            axis = 'ECM_stromal_support'
        if axis not in MODULES:
            axis = 'unassigned'
        supported = r.V3_human_genelevel_supported == 'True'
        add(r.gene, 'Homo_sapiens', 'ProjectA_human_state_readout_analysis', r.state,
            r.evidence_tier, False, supported, not supported,
            r.V3_program_axis, 'human_readout_evidence.tsv.gz', r.source_row_1based,
            forced_modules=[axis], note='Human state readout; bee-prior rescue alone is hypothesis-only.')
    for r in registry.itertuples():
        axis = axis_map.get(r.program_axis, r.program_axis)
        if axis not in MODULES:
            continue
        add(r.gene, 'Homo_sapiens', 'ProjectA_prior_registry', r.source_type,
            'hypothesis_prior_without_bee_ID', False, False, True, r.program_axis,
            'bee_prior_registry.tsv.gz', r.source_row_1based, forced_modules=[axis],
            note='Registry label alone does not establish direct bee derivation.')
    frame = pd.DataFrame(rows)
    frame.insert(0, 'evidence_id', [f'E{i:06d}' for i in range(1, len(frame) + 1)])
    write_tsv(frame, out / 'BRIDGE_EVIDENCE_LEDGER.tsv')
    write_json(MODULE_RULES, out / 'FROZEN_MODULE_ASSIGNMENT_RULES.json')
    return frame


def mapping_audit(root, out, ledger):
    bridge = _source(root, 'apis_fly_human_mapping.tsv.gz')
    hm = _source(root, 'human_mouse_strict_pairs.tsv.gz')
    # Verify the frozen strict reference at both gene-ID and symbol level.
    if hm.groupby('human_gene_id').mouse_gene_id.nunique().max() != 1:
        raise ValueError('Human–mouse strict table has ambiguous human IDs')
    if hm.groupby('mouse_gene_id').human_gene_id.nunique().max() != 1:
        raise ValueError('Human–mouse strict table has ambiguous mouse IDs')
    hmap = hm.groupby('human_symbol').mouse_symbol.apply(lambda x: sorted(set(x)))
    amap = {str(k): d for k, d in bridge.groupby('apis_ncbi_gene_id')}
    rows = []
    for r in ledger.itertuples():
        candidates, valid_chain, human_candidates = set(), False, set()
        route = 'human_to_mouse_frozen_NCBI'
        if r.source_species == 'Homo_sapiens':
            human_candidates = {r.source_gene}
            valid_chain = True
        elif r.apis_gene_id in amap:
            b = amap[r.apis_gene_id]
            human_candidates = set(b.human_symbol) - {''}
            valid_chain = bool(len(human_candidates) == 1
                               and b.apis_dmel_orthology_type.eq('ortholog_one2one').all()
                               and b.dmel_human_orthology_type.eq('ortholog_one2one').all()
                               and b.both_orthology_confident.eq('True').all())
            route = 'Apis_fly_human_mouse_indirect'
            if r.source_species.startswith('Bombus'):
                route = 'Bombus_Apis_annotation_fly_human_mouse_indirect'
        else:
            route = 'bee_mapping_unavailable_in_frozen_reference'
        for h in human_candidates:
            if h in hmap:
                candidates.update(hmap[h])
        strict = valid_chain and len(candidates) == 1 and len(human_candidates) == 1
        status = ('one_to_one' if strict else 'one_to_many' if len(candidates) > 1
                  else 'non_strict_single_candidate' if candidates else 'unmapped')
        for mouse in sorted(candidates) or ['']:
            rows.append(dict(evidence_id=r.evidence_id, module=r.module,
                             source_species=r.source_species, source_gene=r.source_gene,
                             mouse_gene=mouse, human_candidates=';'.join(sorted(human_candidates)),
                             n_mouse_candidates=len(candidates), mapping_status=status,
                             strict_one_to_one=strict, mapping_route=route,
                             direct_bee_derived=r.direct_bee_derived,
                             mammalian_supported=r.mammalian_supported,
                             hypothesis_only=r.hypothesis_only, evidence_tier=r.evidence_tier))
    detail = pd.DataFrame(rows)
    write_tsv(detail, out / 'ORTHOLOG_MAPPING_DETAIL.tsv')
    audit = []
    for (species, module), d in detail.groupby(['source_species', 'module']):
        d = d.drop_duplicates('source_gene')
        status = d.mapping_status
        audit.append(dict(source_species=species, module=module, original_gene_count=len(d),
                          successfully_mapped=int(status.ne('unmapped').sum()),
                          one_to_one=int(status.eq('one_to_one').sum()),
                          one_to_many=int(status.eq('one_to_many').sum()),
                          non_strict_single_candidate=int(status.eq('non_strict_single_candidate').sum()),
                          unmapped=int(status.eq('unmapped').sum()),
                          mapping_coverage=float(status.ne('unmapped').mean()),
                          strict_mapping_coverage=float(status.eq('one_to_one').mean()),
                          limitation='Frozen strict human-mouse table does not expose discarded alternative pairs.'))
    write_tsv(audit, out / 'ORTHOLOG_MAPPING_AUDIT.tsv')
    eligible = detail.strict_one_to_one & ~detail.hypothesis_only & detail.module.isin(MODULES)
    definitions = []
    for scope, mask in {
        'bridge_supported': eligible,
        'bee_direct': eligible & detail.direct_bee_derived,
        'mammalian_readout': eligible & detail.mammalian_supported,
        'source_tier1': eligible & detail.evidence_tier.str.startswith('Tier1'),
        'hypothesis_only': detail.strict_one_to_one & detail.hypothesis_only & detail.module.isin(MODULES),
    }.items():
        for (module, gene), d in detail[mask].groupby(['module', 'mouse_gene']):
            definitions.append(dict(scope=scope, module=module, mouse_gene=gene,
                                    n_source_genes=d.source_gene.nunique(),
                                    evidence_ids=';'.join(sorted(set(d.evidence_id)))))
    definitions = pd.DataFrame(definitions)
    write_tsv(definitions, out / 'FROZEN_MOUSE_MODULES.tsv')
    return definitions


def project_modules(root, out, definitions):
    import h5py
    from anndata.io import read_elem, sparse_dataset

    input_path = root / 'results/06_annotation_v2.h5ad'
    before = input_path.stat()
    score_rows, coverage, pb_rows, subtype_rows, count_rows, tech_rows = [], [], [], [], [], []
    groups = {lib: lib.rsplit('_', 1)[0] for lib in LIBRARIES}
    modules = {(s, m): sorted(set(d.mouse_gene))
               for (s, m), d in definitions.groupby(['scope', 'module'])}
    with h5py.File(input_path, 'r') as f:
        obs, var = read_elem(f['obs']), read_elem(f['var'])
        if len(obs) != 105763 or not var.index.is_unique or not obs.index.is_unique:
            raise ValueError('Frozen object count or unique feature/cell index audit failed')
        if set(obs.library_id.astype(str)) != set(LIBRARIES):
            raise ValueError('Unexpected library IDs')
        write_tsv(var.assign(feature_name=var.index)[['feature_name', 'gene_ids']], out / 'MOUSE_FEATURE_ID_AUDIT.tsv')
        counts = sparse_dataset(f['layers/counts'])
        all_genes = sorted(set(definitions.mouse_gene) & set(var.index))
        positions = var.index.get_indexer(all_genes)
        for pop in POPS:
            for tier, tiers in [('Tier1', ['Tier1_primary']),
                                ('Tier1_Tier2', ['Tier1_primary', 'Tier2_sensitivity'])]:
                select = obs.cell_type_broad_v2.astype(str).eq(pop) & obs.analysis_tier_v2.astype(str).isin(tiers)
                o = obs[select].copy()
                if not len(o):
                    raise ValueError(f'Empty population: {pop}/{tier}')
                idx = np.flatnonzero(select.to_numpy())
                expr_parts, raw_parts, total_parts = [], [], []
                for start in range(0, len(idx), 2000):
                    x = counts[idx[start:start+2000]].tocsr()
                    total = np.asarray(x.sum(axis=1)).ravel()
                    raw = x[:, positions].toarray().astype(np.float32)
                    expr_parts.append(np.log1p(raw * (1e4 / np.maximum(total, 1))[:, None]))
                    raw_parts.append(raw)
                    total_parts.append(total)
                expression = np.vstack(expr_parts)
                raw = np.vstack(raw_parts)
                totals = np.concatenate(total_parts)
                lib = o.library_id.astype(str).to_numpy()
                sub = o.cell_type_subtype_v2.astype(str).to_numpy()
                if set(lib) != set(LIBRARIES):
                    raise ValueError(f'Missing library in {pop}/{tier}')
                pb = []
                for library in LIBRARIES:
                    loc = lib == library
                    pb.append(np.log2(1 + raw[loc].sum(axis=0, dtype=np.float64) * 1e6 / totals[loc].sum()))
                    count_rows.append(dict(cell_type=pop, cell_tier=tier, library_id=library,
                                           group=groups[library], n_cells=int(loc.sum())))
                    tech_rows.append(dict(cell_type=pop, cell_tier=tier, library_id=library,
                                          median_umi=float(np.median(totals[loc])),
                                          median_pct_mt=float(o.loc[loc, 'pct_counts_mt'].median())))
                pb = np.asarray(pb)
                for i, library in enumerate(LIBRARIES):
                    for j, gene in enumerate(all_genes):
                        pb_rows.append(dict(cell_type=pop, cell_tier=tier, library_id=library,
                                            group=groups[library], gene=gene, log2cpm=pb[i, j]))
                sub_counts = pd.crosstab(lib, sub).reindex(LIBRARIES).fillna(0)
                common = sub_counts.columns[(sub_counts >= 10).all(axis=0)]
                weights = sub_counts[common].div(sub_counts[common].sum(axis=1), axis=0).mean(axis=0)
                for (scope, module), requested in modules.items():
                    present = [g for g in requested if g in all_genes]
                    cols = [all_genes.index(g) for g in present]
                    expressed = [j for j in cols if (raw[:, j] > 0).any()]
                    coverage.append(dict(cell_type=pop, cell_tier=tier, scope=scope, module=module,
                                         n_requested=len(requested), n_present=len(present),
                                         n_expressed=len(expressed), evaluable=len(expressed) >= 3,
                                         genes=';'.join(all_genes[j] for j in expressed)))
                    if len(expressed) < 3:
                        continue
                    cellscore = expression[:, expressed].mean(axis=1)
                    for i, library in enumerate(LIBRARIES):
                        loc = lib == library
                        values = cellscore[loc]
                        common_fraction = float(np.isin(sub[loc], common).mean())
                        standardized = sum(float(cellscore[loc & (sub == s)].mean()) * w
                                           for s, w in weights.items()) if len(common) else np.nan
                        metrics = {'pseudobulk_logcpm': float(pb[i, expressed].mean()),
                                   'cell_mean': float(values.mean()), 'cell_median': float(np.median(values)),
                                   'cell_q90': float(np.quantile(values, 0.9)),
                                   'subtype_standardized_mean': standardized}
                        for metric, value in metrics.items():
                            score_rows.append(dict(cell_type=pop, cell_tier=tier, scope=scope, module=module,
                                                   library_id=library, group=groups[library], metric=metric,
                                                   score=value, n_cells=int(loc.sum()), n_genes=len(expressed),
                                                   common_subtype_fraction=common_fraction))
                        for s in sub_counts.columns:
                            ss = loc & (sub == s)
                            subtype_rows.append(dict(cell_type=pop, cell_tier=tier, scope=scope,
                                                     module=module, library_id=library, subtype=s,
                                                     n_cells=int(ss.sum()), fraction=float(ss.sum()/loc.sum()),
                                                     score=float(cellscore[ss].mean()) if ss.any() else np.nan,
                                                     common_weight=float(weights.get(s, 0))))
                print(f'Projection complete: {pop}/{tier}: {len(o)} cells', flush=True)
    after = input_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Input changed during read-only projection')
    for name, table in [('MODULE_LIBRARY_SCORES.tsv', score_rows), ('MODULE_GENE_COVERAGE.tsv', coverage),
                        ('MODULE_GENE_LIBRARY_PSEUDOBULK.tsv', pb_rows),
                        ('MODULE_SUBTYPE_SENSITIVITY.tsv', subtype_rows), ('LIBRARY_CELL_COUNTS.tsv', count_rows),
                        ('LIBRARY_TECHNICAL_AUDIT.tsv', tech_rows)]:
        write_tsv(table, out / name)
    write_json({'n_cells': len(obs), 'n_features': len(var), 'file_bytes': before.st_size,
                'mtime_ns': before.st_mtime_ns, 'unchanged_after_read': True,
                'read_mode': 'h5py r / AnnData sparse_dataset; raw counts only'}, out / 'INPUT_OBJECT_AUDIT.json')


def infer_modules(out):
    scores = pd.read_csv(out / 'MODULE_LIBRARY_SCORES.tsv', sep='\t')
    rows, loos = [], []
    keys = ['cell_type', 'cell_tier', 'scope', 'module', 'metric']
    for key, d in scores.groupby(keys):
        fields = dict(zip(keys, key))
        for test, ref in CONTRASTS:
            sub = d[d.group.isin([test, ref])]
            observed = exact_test(sub.score, sub.group, test, ref)
            direction = np.sign(observed['effect'])
            pairs = np.subtract.outer(sub[sub.group.eq(test)].score.to_numpy(),
                                      sub[sub.group.eq(ref)].score.to_numpy()).ravel()
            fraction = float(np.mean(np.sign(pairs) == direction))
            loo_signs = []
            for removed in sub.library_id:
                ss = sub[sub.library_id.ne(removed)]
                stat = exact_test(ss.score, ss.group, test, ref)
                stable = np.sign(stat['effect']) == direction
                loo_signs.append(stable)
                loos.append({**fields, 'contrast': f'{test}_vs_{ref}', 'omitted_library': removed,
                             **stat, 'direction_preserved': stable})
            rows.append({**fields, 'contrast': f'{test}_vs_{ref}', **observed,
                         'pairwise_direction_fraction': fraction,
                         'loo_direction_fraction': float(np.mean(loo_signs)),
                         'n_test': int(sub.group.eq(test).sum()), 'n_reference': int(sub.group.eq(ref).sum())})
    tests = pd.DataFrame(rows)
    tests['q_bh_within_scope_tier_metric_contrast'] = tests.groupby(
        ['scope', 'cell_tier', 'metric', 'contrast']).p_exact_two_sided.transform(lambda p: bh(p))
    write_tsv(tests, out / 'MODULE_EXACT_PERMUTATION.tsv')
    write_tsv(loos, out / 'MODULE_LOO_STABILITY.tsv')
    pb = pd.read_csv(out / 'MODULE_GENE_LIBRARY_PSEUDOBULK.tsv', sep='\t')
    coverage = pd.read_csv(out / 'MODULE_GENE_COVERAGE.tsv', sep='\t').fillna('')
    geo_rows, geo_loo, split_rows = [], [], []
    for r in coverage[coverage.evaluable.eq(True)].itertuples():
        genes = r.genes.split(';')
        matrix = pb[pb.cell_type.eq(r.cell_type) & pb.cell_tier.eq(r.cell_tier)].pivot(
            index='library_id', columns='gene', values='log2cpm').reindex(LIBRARIES)[genes]
        fields = dict(cell_type=r.cell_type, cell_tier=r.cell_tier, scope=r.scope, module=r.module)

        def estimate(mat):
            means = {g: mat.loc[mat.index.str.startswith(g + '_')].mean(axis=0).to_numpy()
                     for g in ['Y', 'OC', 'OT']}
            age, treatment = means['OC'] - means['Y'], means['OT'] - means['OC']
            g = geometry(age, treatment)
            label = base_classification(g, age.mean(), treatment.mean())
            return g, label, float(age.mean()), float(treatment.mean())
        g, label, am, tm = estimate(matrix)
        same, tsame = [], []
        for lib in LIBRARIES:
            gg, ll, aa, tt = estimate(matrix.drop(index=lib))
            same.append(ll == label)
            tsame.append(np.sign(tt) == np.sign(tm))
            geo_loo.append({**fields, 'omitted_library': lib, **gg, 'classification': ll,
                            'age_mean': aa, 'treatment_mean': tt})
        for oc in ['OC_1', 'OC_2', 'OC_3']:
            age = matrix.loc[oc].to_numpy() - matrix.loc[['Y_1', 'Y_2', 'Y_3']].mean().to_numpy()
            treatment = matrix.loc[['OT_1', 'OT_2', 'OT_3']].mean().to_numpy() - matrix.loc[
                [x for x in ['OC_1', 'OC_2', 'OC_3'] if x != oc]].mean().to_numpy()
            split_rows.append({**fields, 'age_OC_library': oc, **geometry(age, treatment),
                               'classification': base_classification(geometry(age, treatment), age.mean(), treatment.mean())})
        final = 'library_sensitive' if np.mean(same) < 7/9 or np.mean(tsame) < 8/9 else label
        geo_rows.append({**fields, **g, 'base_classification': label, 'classification': final,
                         'loo_class_fraction': np.mean(same), 'loo_treatment_direction_fraction': np.mean(tsame),
                         'age_mean': am, 'treatment_mean': tm, 'n_genes': len(genes)})
    write_tsv(geo_rows, out / 'MODULE_AGE_TREATMENT_GEOMETRY.tsv')
    write_tsv(geo_loo, out / 'MODULE_GEOMETRY_LOO.tsv')
    write_tsv(split_rows, out / 'MODULE_SPLIT_OC_GEOMETRY.tsv')


def matched_sets(features, requested, excluded, n_random, rng, neighbors=40):
    """Joint nearest-neighbor matching, no replacement within any random set."""
    all_names = features.index.to_numpy()
    x = features.to_numpy(float)
    scale = x.std(axis=0, ddof=1)
    scale[scale < 1e-12] = 1
    z = (x - x.mean(axis=0)) / scale
    lookup = {g: i for i, g in enumerate(all_names)}
    target = np.array([lookup[g] for g in requested], int)
    pool = np.array([i for i, g in enumerate(all_names) if g not in excluded], int)
    if len(pool) < len(target):
        raise ValueError('Insufficient matched-random pool')
    nearest = []
    for j in target:
        dist = np.square(z[pool] - z[j]).sum(axis=1)
        nearest.append(pool[np.argsort(dist, kind='stable')[:neighbors]])
    result = np.empty((n_random, len(target)), dtype=int)
    for rep in range(n_random):
        used = set()
        for j in rng.permutation(len(target)):
            choices = [int(i) for i in nearest[j] if int(i) not in used]
            if not choices:
                raise ValueError('No distinct matched gene remains in nearest-neighbor pool')
            chosen = int(rng.choice(choices))
            used.add(chosen)
            result[rep, j] = chosen
    delta = z[result].mean(axis=1) - z[target].mean(axis=0)
    diag = {'n_matched_genes': len(target), 'n_random': n_random,
            'fraction_draws_all_abs_smd_lt_0_5': float((np.abs(delta).max(axis=1) < 0.5).mean()),
            'max_abs_mean_smd': float(np.abs(delta.mean(axis=0)).max())}
    for j, col in enumerate(features.columns):
        diag[f'mean_smd_{col}'] = float(delta[:, j].mean())
    diag['matching_pass'] = diag['max_abs_mean_smd'] <= 0.25
    return all_names[result], diag


def integrate_vko(root, out, n_random=2000):
    stage = root / 'results/deep_dive_stage24_virtual_knockout'
    universe = pd.read_csv(stage / '00_preparation/gene_universe.tsv', sep='\t').set_index('gene')
    refs = pd.read_csv(stage / '00_preparation/expression_matched_reference_genes.tsv', sep='\t')
    reference_map = dict(zip(refs.target_gene, refs.reference_gene))
    excluded_nodes = set(TARGETS) | set(refs.reference_gene)
    gene_real = pd.read_csv(stage / '02_summary/gene_stability_with_real_transcriptomic_evidence.tsv.gz', sep='\t')
    real = gene_real.drop_duplicates('gene').set_index('gene')
    definitions = pd.read_csv(out / 'FROZEN_MOUSE_MODULES.tsv', sep='\t')
    modules = {(s, m): set(d.mouse_gene) for (s, m), d in definitions.groupby(['scope', 'module'])}
    result_rows, compare_rows, match_rows, null_rows, real_rows, degree_rows = [], [], [], [], [], []
    rng = np.random.default_rng(SEED)
    for scenario_dir in sorted((stage / '01_runs').iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name
        runs, degrees = [], []
        for seed_dir in sorted(scenario_dir.glob('seed_*')):
            with np.load(seed_dir / 'WT_TENSOR.npz') as archive:
                genes = archive['genes'].astype(str)
                wt = np.abs(archive['values']).astype(np.float64)
            if len(set(genes)) != len(genes):
                raise ValueError('Duplicate WT network genes')
            np.fill_diagonal(wt, 0)
            weight = wt.sum(axis=0) + wt.sum(axis=1)
            cutoff = float(np.quantile(wt[np.triu_indices(len(wt), 1)], 0.9))
            adj = wt > cutoff
            np.fill_diagonal(adj, False)
            degree = adj.sum(axis=0) + adj.sum(axis=1)
            deg = pd.DataFrame({'gene': genes, 'weighted_degree': weight, 'edge_degree': degree})
            degrees.append(deg.set_index('gene'))
            degree_rows.extend(deg.assign(scenario=scenario, seed=seed_dir.name).to_dict('records'))
            ranks = {p.name.removesuffix('.tsv.gz'): pd.read_csv(p, sep='\t')
                     for p in sorted((seed_dir / 'knockouts').glob('*.tsv.gz'))}
            if set(ranks) != excluded_nodes:
                raise ValueError(f'Incomplete candidate/reference runs in {seed_dir}')
            runs.append((seed_dir.name, ranks))
            del wt, adj
        features = pd.concat(degrees).groupby(level=0).mean().join(universe[['mean_umi', 'detection_fraction']])
        common = set(features.index) - excluded_nodes
        features = features.loc[sorted(common)]
        features = pd.DataFrame({'log_mean_umi': np.log1p(features.mean_umi),
                                 'detection': features.detection_fraction,
                                 'log_weighted_degree': np.log1p(features.weighted_degree),
                                 'log_edge_degree': np.log1p(features.edge_degree)}, index=features.index)
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError('Missing random-matching features')
        selections = {}
        for seed, ranks in runs:
            selections[seed] = {target: set(d.loc[d.rank_fraction.le(0.05), 'gene']) & common
                                for target, d in ranks.items()}
        if len(runs) > 1:
            selections['strict_all_seeds'] = {target: set.intersection(
                *(selections[seed][target] for seed, _ in runs)) for target in excluded_nodes}
            selections['recurrent_2_of_3'] = {target: set(g for g in common if sum(
                g in selections[seed][target] for seed, _ in runs) >= 2) for target in excluded_nodes}
        for (scope, module), requested in modules.items():
            members = sorted(requested & common)
            if len(members) < 3:
                result_rows.append(dict(scenario=scenario, selection='not_evaluable', scope=scope,
                                        module=module, ko_target='', n_module=len(members),
                                        status='fewer_than_3_module_genes_in_network'))
                continue
            draws, diag = matched_sets(features, members, set(members), n_random, rng)
            match_rows.append(dict(scenario=scenario, scope=scope, module=module, **diag))
            for selection, target_hits in selections.items():
                per_target = {}
                for target, hits in sorted(target_hits.items()):
                    overlap = sorted(set(members) & hits)
                    observed = len(overlap)
                    null = np.isin(draws, list(hits)).sum(axis=1)
                    table = [[observed, len(members) - observed],
                             [len(hits) - observed, len(common) - len(members) - len(hits) + observed]]
                    odds, fisher = fisher_exact(table, alternative='greater')
                    empirical = (1 + np.sum(null >= observed)) / (n_random + 1)
                    per_target[target] = (observed, null)
                    hreal = real.reindex(overlap)
                    age = hreal.aging_effect.to_numpy(float)
                    treatment = hreal.treatment_effect.to_numpy(float)
                    valid = np.isfinite(age) & np.isfinite(treatment)
                    result_rows.append(dict(scenario=scenario, selection=selection, scope=scope, module=module,
                                            ko_target=target, ko_role='candidate' if target in TARGETS else 'reference',
                                            n_module=len(members), n_universe=len(common), n_hits=len(hits),
                                            n_overlap=observed, overlap_genes=';'.join(overlap), odds_ratio=odds,
                                            p_fisher_competitive=fisher, p_matched_random=empirical,
                                            null_mean=float(null.mean()), null_q95=float(np.quantile(null, .95)),
                                            matching_pass=diag['matching_pass'], n_real_effect_genes=int(valid.sum()),
                                            observed_age_opposite_fraction=float(np.mean(age[valid]*treatment[valid]<0)) if valid.any() else np.nan,
                                            median_real_age_effect=float(np.nanmedian(age)) if valid.any() else np.nan,
                                            median_real_treatment_effect=float(np.nanmedian(treatment)) if valid.any() else np.nan,
                                            vko_direction='unsigned_not_estimable', status='evaluated'))
                    for gene in overlap:
                        rr = real.loc[gene] if gene in real.index else None
                        real_rows.append(dict(scenario=scenario, selection=selection, scope=scope, module=module,
                                              ko_target=target, gene=gene,
                                              aging_effect=rr.aging_effect if rr is not None else np.nan,
                                              treatment_effect=rr.treatment_effect if rr is not None else np.nan,
                                              residual_effect=rr.residual_effect if rr is not None else np.nan,
                                              interpretation='Observed readout; VKO has no predicted sign.'))
                    null_rows.append(dict(scenario=scenario, selection=selection, scope=scope, module=module,
                                          ko_target=target, statistic='overlap_count',
                                          null_histogram=json.dumps(dict(zip(*[x.tolist() for x in np.unique(null, return_counts=True)]))),
                                          n_random=n_random, random_seed=SEED))
                for target, reference in reference_map.items():
                    hit, null = per_target[target]
                    rhit, rnull = per_target[reference]
                    delta, delta_null = hit - rhit, null - rnull
                    compare_rows.append(dict(scenario=scenario, selection=selection, scope=scope, module=module,
                                             target_gene=target, reference_gene=reference,
                                             candidate_overlap=hit, reference_overlap=rhit, overlap_difference=delta,
                                             p_candidate_over_reference=(1+np.sum(delta_null >= delta))/(n_random+1),
                                             delta_null_mean=float(delta_null.mean()), matching_pass=diag['matching_pass'],
                                             interpretation='Paired matched-gene-set competitive comparison; no biological replication.'))
        print(f'VKO enrichment complete: {scenario}', flush=True)
    result = pd.DataFrame(result_rows)
    result['q_matched_random'] = result.groupby(['scenario', 'selection', 'scope']).p_matched_random.transform(bh)
    compare = pd.DataFrame(compare_rows)
    compare['q_candidate_over_reference'] = compare.groupby(['scenario', 'selection', 'scope']).p_candidate_over_reference.transform(bh)
    for name, table in [('VKO_MODULE_ENRICHMENT.tsv', result), ('VKO_REFERENCE_COMPARISON.tsv', compare),
                        ('VKO_MATCHING_DIAGNOSTICS.tsv', match_rows), ('VKO_RANDOM_NULL_HISTOGRAMS.tsv', null_rows),
                        ('VKO_REAL_TRANSCRIPTOMIC_READOUTS.tsv', real_rows), ('VKO_NETWORK_DEGREES.tsv', degree_rows)]:
        write_tsv(table, out / name)


def main():
    import argparse
    import traceback
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--steps', nargs='+', default=['inputs', 'projection', 'inference', 'vko'])
    parser.add_argument('--random-sets', type=int, default=2000)
    args = parser.parse_args()
    root = args.root.resolve()
    if str(root) != '/root/autodl-tmp/ovary_scRNAseq':
        raise RuntimeError('Formal analysis must run in the authorized main-server project')
    out = root / 'results/deep_dive_stage25_bee_mouse_bridge'
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'RUN_STATE.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'steps': {}}
    state.update(stage='25', analysis_unit='library/pool', updated_at=now(),
                 git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                 python=platform.python_version(), seed=SEED, random_sets=args.random_sets)
    for step in args.steps:
        state['overall'] = 'running'
        state['steps'][step] = {'status': 'running', 'started': now()}
        write_json(state, state_path)
        try:
            if step == 'inputs':
                manifest = _source(root, 'SOURCE_MANIFEST.tsv')
                for row in manifest.itertuples():
                    if sha(root / 'resources/cross_project_bee_bridge' / row.file) != row.frozen_sha256:
                        raise ValueError(f'Frozen source hash mismatch: {row.file}')
                mapping_audit(root, out, create_ledger(root, out))
            elif step == 'projection':
                project_modules(root, out, pd.read_csv(out / 'FROZEN_MOUSE_MODULES.tsv', sep='\t'))
            elif step == 'inference':
                infer_modules(out)
            elif step == 'vko':
                integrate_vko(root, out, args.random_sets)
            else:
                raise ValueError(step)
            state['steps'][step].update(status='complete', finished=now())
            state['overall'] = 'analysis_in_progress'
        except Exception as exc:
            state['steps'][step].update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            state['overall'] = 'failed'
            write_json(state, state_path)
            raise
        write_json(state, state_path)
