"""Interpret, visualize and validate Stage 25 without altering upstream results."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stage25_bee_mouse_bridge import LIBRARIES, MODULES, POPS, TARGETS, now, sha, write_json, write_tsv

SHORT = {
    'mitochondrial_energy': 'Mitochondrial energy',
    'RNA_processing_translation': 'RNA processing / translation',
    'proteostasis_autophagy': 'Proteostasis / autophagy',
    'lipid_sterol_redox': 'Lipid / sterol / redox',
    'membrane_structure_transport': 'Membrane / transport',
    'ECM_stromal_support': 'ECM / stromal support',
    'stress_inflammation': 'Stress / inflammation',
    'reproductive_support_secretion': 'Reproductive support / secretion',
}
CN = dict(zip(MODULES, ['线粒体能量', 'RNA加工/翻译', '蛋白稳态/自噬', '脂质/甾醇/氧化还原',
                       '膜结构/运输', 'ECM/基质支持', '应激/炎症', '生殖支持/分泌']))


def load(out, name):
    return pd.read_csv(out / name, sep='\t')


def module_synthesis(out):
    tests = load(out, 'MODULE_EXACT_PERMUTATION.tsv')
    geo = load(out, 'MODULE_AGE_TREATMENT_GEOMETRY.tsv')
    loo = load(out, 'MODULE_LOO_STABILITY.tsv')
    split = load(out, 'MODULE_SPLIT_OC_GEOMETRY.tsv')
    coverage = load(out, 'MODULE_GENE_COVERAGE.tsv')
    rows = []
    for pop, scope, module in itertools.product(POPS, ['bridge_supported', 'bee_direct', 'bee_RBH_sensitivity'], MODULES):
        d = tests[tests.cell_type.eq(pop) & tests.scope.eq(scope) & tests.module.eq(module)]
        g = geo[geo.cell_type.eq(pop) & geo.scope.eq(scope) & geo.module.eq(module) & geo.cell_tier.eq('Tier1')]
        c = coverage[coverage.cell_type.eq(pop) & coverage.scope.eq(scope) & coverage.module.eq(module) & coverage.cell_tier.eq('Tier1')]
        row = dict(cell_type=pop, scope=scope, module=module, n_genes=int(c.n_expressed.iloc[0]) if len(c) else 0)
        if g.empty:
            rows.append({**row, 'classification': 'not_evaluable', 'support_level': 'insufficient_gene_coverage'})
            continue
        g = g.iloc[0]
        for contrast, prefix in [('OC_vs_Y', 'age'), ('OT_vs_OC', 'treatment'), ('OT_vs_Y', 'residual')]:
            t = d[d.cell_tier.eq('Tier1') & d.metric.eq('pseudobulk_logcpm') & d.contrast.eq(contrast)].iloc[0]
            row.update({prefix + '_effect': t.effect, prefix + '_p_exact': t.p_exact_two_sided,
                        prefix + '_loo_fraction': t.loo_direction_fraction,
                        prefix + '_pairwise_fraction': t.pairwise_direction_fraction})
        checks = [('Tier1_Tier2', 'pseudobulk_logcpm', 'tier_direction_agrees'),
                  ('Tier1', 'cell_mean', 'center_direction_agrees'),
                  ('Tier1', 'cell_median', 'median_direction_agrees'),
                  ('Tier1', 'cell_q90', 'tail_direction_agrees'),
                  ('Tier1', 'subtype_standardized_mean', 'subtype_direction_agrees')]
        for tier, metric, field in checks:
            t = d[d.cell_tier.eq(tier) & d.metric.eq(metric) & d.contrast.eq('OT_vs_OC')]
            row[field] = bool(np.sign(t.effect.iloc[0]) == np.sign(row['treatment_effect'])) if len(t) and np.isfinite(t.effect.iloc[0]) else None
            row[field + '_effect'] = float(t.effect.iloc[0]) if len(t) else np.nan
        ss = split[split.cell_type.eq(pop) & split.scope.eq(scope) & split.module.eq(module) & split.cell_tier.eq('Tier1')]
        row['split_OC_same_geometry_fraction'] = float(ss.classification.eq(g.base_classification).mean())
        row['cosine'] = g.cosine
        row['orthogonal_fraction'] = g.orthogonal_fraction
        row['pseudobulk_geometry_class'] = g.classification
        bad = loo[loo.cell_type.eq(pop) & loo.scope.eq(scope) & loo.module.eq(module)
                  & loo.cell_tier.eq('Tier1') & loo.metric.eq('pseudobulk_logcpm')
                  & loo.contrast.eq('OT_vs_OC') & ~loo.direction_preserved]
        row['treatment_sign_flip_libraries'] = ';'.join(bad.omitted_library)
        row['classification'] = g.classification
        if row['treatment_loo_fraction'] < 1:
            row['classification'] = 'library_sensitive'
        elif not row['tier_direction_agrees'] or not row['center_direction_agrees']:
            row['classification'] = 'inconsistent'
        row['support_level'] = ('consistent_descriptive_treatment_shift' if row['treatment_loo_fraction'] == 1
                                and row['tier_direction_agrees'] and row['center_direction_agrees']
                                and row['subtype_direction_agrees'] else 'sensitivity_limited')
        row['functional_protection_demonstrated'] = False
        rows.append(row)
    result = pd.DataFrame(rows)
    write_tsv(result, out / 'MODULE_SYNTHESIS.tsv')
    # Coupling is descriptive: nine units and treatment-group confounding preclude
    # a claim that MRJP1 restored a functional inter-cellular interaction.
    scores = load(out, 'MODULE_LIBRARY_SCORES.tsv')
    s = scores[scores.scope.eq('bee_RBH_sensitivity') & scores.cell_tier.eq('Tier1')
               & scores.metric.eq('pseudobulk_logcpm')].copy()
    s['axis'] = s.cell_type + '::' + s.module
    matrix = s.pivot(index='library_id', columns='axis', values='score').reindex(LIBRARIES)
    groups = pd.Series([x.rsplit('_', 1)[0] for x in matrix.index], index=matrix.index)
    residual = matrix - matrix.groupby(groups).transform('mean')
    pairs = []
    for a, b in itertools.combinations(matrix.columns, 2):
        pa, ma = a.split('::'); pb, mb = b.split('::')
        if pa != pb and ma != mb:
            continue
        r = float(spearmanr(matrix[a], matrix[b]).statistic)
        rr = float(spearmanr(residual[a], residual[b]).statistic)
        lv = [float(spearmanr(residual.drop(index=lib)[a], residual.drop(index=lib)[b]).statistic) for lib in LIBRARIES]
        pairs.append(dict(axis_a=a, axis_b=b, n_libraries=9, spearman_raw=r,
                          spearman_group_centered=rr, loo_residual_correlation_min=min(lv),
                          loo_residual_correlation_max=max(lv),
                          inference='descriptive_only; group means removed; not a functional-coupling test'))
    write_tsv(pairs, out / 'MODULE_COUPLING_DESCRIPTIVE.tsv')
    tech = load(out, 'LIBRARY_TECHNICAL_AUDIT.tsv')
    confound=[]
    for (pop,tier,scope,module,metric), d in scores.groupby(['cell_type','cell_tier','scope','module','metric']):
        d=d.merge(tech[tech.cell_type.eq(pop)&tech.cell_tier.eq(tier)],on='library_id')
        for feature in ['median_umi','median_pct_mt']:
            if d.score.notna().all():
                confound.append(dict(cell_type=pop,cell_tier=tier,scope=scope,module=module,metric=metric,
                                     technical_feature=feature,n_libraries=len(d),
                                     spearman=float(spearmanr(d.score,d[feature]).statistic),
                                     interpretation='descriptive_library_correlation; technical and biological effects may be confounded'))
    write_tsv(confound,out/'MODULE_TECHNICAL_CORRELATIONS.tsv')
    return result


def node_model(root, out):
    v = load(out, 'VKO_MODULE_ENRICHMENT.tsv')
    ref = load(out, 'VKO_REFERENCE_COMPARISON.tsv')
    old = root / 'results/deep_dive_stage24_virtual_knockout/02_summary'
    stability = load(old, 'knockout_target_stability.tsv')
    loo = load(old, 'leave_one_library_out_stability.tsv')
    pathways = {
        'Smad3': ('TGFbeta/ECM', 'follicular/stromal support', 'p-SMAD3; COL1A1/FN1; follicle survival and E2'),
        'Hif1a': ('hypoxia/mitochondrial stress capacity', 'stress tolerance', 'HIF1A protein; ATP; mitochondrial potential; challenge survival'),
        'Abca1': ('cholesterol/membrane homeostasis', 'lipid handling', 'ABCA1 protein; cholesterol efflux; filipin; membrane integrity'),
        'Igfbp2': ('IGF/secreted support', 'conditional support hypothesis', 'secreted IGFBP2; IGF signaling; only after stability passes'),
        'Pak3': ('state-dependent remodeling', 'conditional state transition', 'PAK3 activation; morphology; subtype-resolved response'),
    }
    rows = []
    robust_rows = []
    for target in TARGETS:
        d = v[v.ko_target.eq(target) & v.selection.eq('strict_all_seeds') & v.scenario.eq('tier1_primary')]
        rr = ref[ref.target_gene.eq(target) & ref.selection.eq('strict_all_seeds') & ref.scenario.eq('tier1_primary')]
        joined = d.merge(rr[['scope','module','reference_gene','overlap_difference','p_candidate_over_reference',
                             'q_candidate_over_reference']], on=['scope','module'])
        candidates = joined[joined.scope.isin(['bridge_supported', 'bee_RBH_sensitivity'])]
        for r in candidates.itertuples():
            seed = v[v.ko_target.eq(target) & v.scope.eq(r.scope) & v.module.eq(r.module)
                     & v.scenario.eq('tier1_primary') & v.selection.str.startswith('seed_', na=False)]
            sens = v[v.ko_target.eq(target) & v.scope.eq(r.scope) & v.module.eq(r.module)
                     & v.scenario.eq('tier1_plus_tier2') & v.selection.eq('strict_all_seeds')]
            l = v[v.ko_target.eq(target) & v.scope.eq(r.scope) & v.module.eq(r.module)
                  & v.scenario.str.startswith('loo_')]
            robust_rows.append(dict(target_gene=target, scope=r.scope, module=r.module,
                                    primary_random_p=r.p_matched_random, primary_random_q=r.q_matched_random,
                                    reference_gene=r.reference_gene, primary_ref_p=r.p_candidate_over_reference,
                                    primary_ref_q=r.q_candidate_over_reference, overlap_difference=r.overlap_difference,
                                    primary_matching_pass=r.matching_pass,
                                    n_seed_overlap=int(seed.n_overlap.gt(0).sum()),
                                    n_loo_overlap=int(l.n_overlap.gt(0).sum()),
                                    sensitivity_tier_overlap=int(sens.n_overlap.iloc[0]) if len(sens) else 0,
                                    all_seed_random_p_le_005=bool(len(seed)==3 and seed.p_matched_random.le(.05).all()),
                                    all_loo_random_p_le_005=bool(len(l)==3 and l.p_matched_random.le(.05).all())))
        specific = candidates[(candidates.q_matched_random <= .05)
                              & (candidates.q_candidate_over_reference <= .05)
                              & candidates.matching_pass.eq(True) & candidates.overlap_difference.gt(0)]
        s = stability[stability.scenario.eq('tier1_primary') & stability.ko_target.eq(target)].iloc[0]
        l = loo[loo.ko_target.eq(target)]
        role = ('Stage24_stable_candidate; bridge_specificity_unproven' if target == 'Smad3'
                else 'secondary_candidate' if target in ['Hif1a','Abca1']
                else 'deprioritized_unstable' if target == 'Igfbp2' else 'conditional_state_candidate')
        pathway, function, endpoint = pathways[target]
        observed=candidates[candidates.scope.eq('bee_RBH_sensitivity')&candidates.n_overlap.gt(0)]
        actual_programs=';'.join(observed.module)
        actual_genes=';'.join(sorted(set(';'.join(observed.overlap_genes.dropna()).split(';'))-{''}))
        rows.append(dict(candidate_node=target, candidate_status=role,
                         network_perturbed_program=pathway,
                         network_program_evidence='Stage24 context / prespecified hypothesis; Stage25 specificity assessed separately',
                         stage25_overlapping_RBH_programs=actual_programs,
                         observed_transcriptomic_readout=actual_genes,
                         readout_interpretation='Real age/treatment signs in VKO_REAL_TRANSCRIPTOMIC_READOUTS; unsigned VKO cannot predict sign.',
                         hypothesized_function=function, functional_validation_endpoint=endpoint,
                         stage24_seed_jaccard=s.median_pairwise_top_fraction_jaccard,
                         stage24_loo_jaccard_min=float(l.jaccard_vs_full_seed_intersection.min()),
                         n_specific_primary_modules=len(specific),
                         specific_modules=';'.join(specific.scope+'::'+specific.module),
                         causality_established=False, conservation_claim_allowed=False,
                         family_equivalence_allowed=False))
    write_tsv(rows, out / 'NODE_STATE_FUNCTION_MODEL.tsv')
    write_tsv(robust_rows, out / 'VKO_CANDIDATE_ROBUSTNESS.tsv')
    return pd.DataFrame(rows)


def figures(root, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    dest = root / 'figures/deep_dive_stage25_bee_mouse_bridge'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(fig, name, data, claim, legend):
        # Every file is one standalone chart. Tight layout is applied before export.
        fig.tight_layout()
        for ext in ['svg', 'pdf', 'png']:
            fig.savefig(dest / f'{name}.{ext}', dpi=300, facecolor='white', bbox_inches='tight')
        write_tsv(data, dest / f'{name}_source_data.tsv')
        manifest.append(dict(figure=name, claim=claim, legend=legend, backend='Python/matplotlib',
                             source_data=f'{name}_source_data.tsv'))
        plt.close(fig)

    audit = load(out, 'ORTHOLOG_MAPPING_AUDIT.tsv')
    d = audit[audit.source_species.eq('Apis_mellifera') & audit.module.isin(MODULES)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(MODULES))
    for j, (ref, color, label) in enumerate([('Ensembl_strict_chain', '#8b98a6', 'Strict chain'),
                                           ('protein_RBH_reciprocal_unique_sensitivity', '#5d8e9e', 'Protein RBH sensitivity')]):
        block = d[d.mapping_reference.eq(ref)].set_index('module').reindex(MODULES)
        ax.barh(y + (j-.5)*.32, 100*block.one_to_one/block.original_gene_count, height=.3, color=color, label=label)
    ax.set_yticks(y, [SHORT[x] for x in MODULES]); ax.invert_yaxis()
    ax.set_xlabel('One-to-one source genes within each reference (%)'); ax.legend(loc='lower right', fontsize=6)
    ax.set_title('Frozen Apis mapping coverage')
    save(fig, '01_mapping_coverage', d, 'Mapping reference changes usable coverage.',
         'Denominator: frozen source genes per module. Bars use one_to_one/original_gene_count. RBH cardinality is algorithmic, not proven evolutionary 1:1 orthology.')

    scores = load(out, 'MODULE_LIBRARY_SCORES.tsv')
    syn = load(out, 'MODULE_SYNTHESIS.tsv')
    palette = {'Y': '#869bb3', 'OC': '#c48c86', 'OT': '#689f91'}
    for pop in POPS:
        d = scores[scores.scope.eq('bridge_supported') & scores.cell_tier.eq('Tier1')
                   & scores.cell_type.eq(pop) & scores.metric.eq('pseudobulk_logcpm')].copy()
        d['centered_score'] = d.score - d.groupby('module').score.transform('mean')
        modules = [m for m in MODULES if m in set(d.module)]
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for i, module in enumerate(modules):
            for j, group in enumerate(['Y','OC','OT']):
                b = d[d.module.eq(module) & d.group.eq(group)].sort_values('library_id')
                yy = i + (j-1)*.2
                ax.scatter(b.centered_score, yy + np.array([-.035,0,.035]),
                           color=palette[group], s=17, label=group if i==0 else None, zorder=3)
                ax.plot([b.centered_score.mean()]*2, [yy-.065,yy+.065], color=palette[group], lw=1.5)
        ax.axvline(0, color='.8', lw=.7)
        ax.set_yticks(range(len(modules)), [SHORT[m] for m in modules]); ax.invert_yaxis()
        ax.set_xlabel('Module log2(CPM + 1), centered across nine libraries')
        ax.set_title(f'{pop}: strict bridge modules\nn = 3 independent libraries per group'); ax.legend(ncol=3, loc='best')
        save(fig, f'02_library_scores_{pop}', d, 'Individual libraries reveal within-population module movement.',
             'Each dot is one independent library/pool (n=3/group); short ticks are group means. No cell-level significance tests.')

        d = syn[syn.cell_type.eq(pop) & syn.scope.isin(['bridge_supported','bee_RBH_sensitivity'])
                & syn.classification.ne('not_evaluable')]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for j,(scope,color,marker) in enumerate([('bridge_supported','#718ba7','o'),('bee_RBH_sensitivity','#ba9a74','s')]):
            b = d[d.scope.eq(scope)].set_index('module').reindex(MODULES)
            ax.scatter(b.cosine, y+(j-.5)*.22, s=25, color=color, marker=marker,
                       label='Strict bridge' if j==0 else 'RBH sensitivity')
        ax.axvspan(-.3,.3,color='#eef0f1',zorder=0); ax.axvline(0,color='.7',lw=.7)
        ax.set_xlim(-1.05,1.05); ax.set_yticks(y,[SHORT[m] for m in MODULES]); ax.invert_yaxis()
        ax.set_xlabel('Cosine(age vector, treatment vector)'); ax.set_title(f'{pop}: multigene age–treatment direction')
        ax.legend(loc='lower right',fontsize=6)
        save(fig, f'03_geometry_{pop}', d, 'Age-opposite and orthogonal components require separate interpretation.',
             'Gene-space directions from equal-library group means; gray band |cosine|<=0.3. Descriptive geometry shares OC; split-OC and LOO sensitivities are in analysis tables. Orthogonality does not imply protection.')

        sensitivity=load(out,'MODULE_EXACT_PERMUTATION.tsv')
        s=sensitivity[sensitivity.scope.eq('bridge_supported')&sensitivity.cell_type.eq(pop)
                      &sensitivity.contrast.eq('OT_vs_OC')].copy()
        choices=[('Tier1','pseudobulk_logcpm','Pseudobulk'),('Tier1','cell_mean','Cell mean'),
                 ('Tier1','cell_median','Cell median'),('Tier1','cell_q90','Cell Q90'),
                 ('Tier1','subtype_standardized_mean','Subtype fixed'),
                 ('Tier1_Tier2','pseudobulk_logcpm','Tier1 + Tier2')]
        values=np.full((len(MODULES),len(choices)),np.nan)
        for i,module in enumerate(MODULES):
            for j,(tier,metric,_) in enumerate(choices):
                row=s[s.module.eq(module)&s.cell_tier.eq(tier)&s.metric.eq(metric)]
                if len(row): values[i,j]=row.effect.iloc[0]
        fig,ax=plt.subplots(figsize=(7.2,3.8))
        cmap=mpl.colors.ListedColormap(['#c79790','#eeeeee','#7ca497']); cmap.set_bad('#f7f7f7')
        ax.imshow(np.sign(values),vmin=-1,vmax=1,cmap=cmap,aspect='auto')
        for i,j in itertools.product(range(len(MODULES)),range(len(choices))):
            ax.text(j,i,f'{values[i,j]:+.3f}' if np.isfinite(values[i,j]) else 'NA',ha='center',va='center',fontsize=6)
        ax.set_xticks(range(len(choices)),[x[2] for x in choices],rotation=25,ha='right')
        ax.set_yticks(y,[SHORT[m] for m in MODULES]); ax.set_title(f'{pop}: treatment direction sensitivity')
        save(fig,f'06_sensitivity_{pop}',s,'Different score summaries can change the treatment direction.',
             'OT minus OC, n=3 libraries/group. Color denotes effect sign only; numbers retain metric-specific units, which must not be compared across columns. NA: fewer than three genes. Cell scores are log1p(10,000-normalized counts); pseudobulk scores are log2(CPM+1).')

    v = load(out, 'VKO_MODULE_ENRICHMENT.tsv')
    for scope, suffix in [('bridge_supported','strict'),('bee_RBH_sensitivity','rbh')]:
        d = v[v.scope.eq(scope) & v.scenario.eq('tier1_primary') & v.selection.eq('strict_all_seeds') & v.ko_target.isin(TARGETS)]
        if d.empty:
            continue
        fig,ax=plt.subplots(figsize=(7.2,3.2))
        matrix=d.pivot(index='module',columns='ko_target',values='n_overlap').reindex(index=MODULES,columns=TARGETS)
        ax.imshow(matrix.to_numpy(),cmap='Blues',vmin=0,aspect='auto')
        for i,j in itertools.product(range(len(MODULES)),range(len(TARGETS))):
            value=matrix.iloc[i,j]
            ax.text(j,i,str(int(value)) if np.isfinite(value) else 'NA',ha='center',va='center',fontsize=7,
                    color='white' if np.isfinite(value) and value>max(1,np.nanmax(matrix.to_numpy())*.6) else '#343b43')
        ax.set_xticks(range(len(TARGETS)),TARGETS); ax.set_yticks(y,[SHORT[m] for m in MODULES])
        ax.set_title(f'Stable network hits overlapping {suffix} modules')
        save(fig,f'04_vko_overlap_{suffix}',d,'Stable-hit overlap and candidate specificity are distinct quantities.',
             'All-three-seed intersection of top-5% perturbed genes, Tier1 OC Granulosa. Numbers are overlap counts; NA: fewer than three module genes in frozen network. Competitive matched-random P/q values are in source data; counts are not significance.')

    r = load(out,'VKO_CANDIDATE_ROBUSTNESS.tsv')
    d=r[r.scope.eq('bee_RBH_sensitivity')]
    if len(d):
        fig,ax=plt.subplots(figsize=(7.2,3.4))
        for j,target in enumerate(TARGETS):
            b=d[d.target_gene.eq(target)].set_index('module').reindex(MODULES)
            ax.scatter(b.overlap_difference,y+(j-2)*.12,s=17,marker=['o','s','^','D','v'][j],
                       color=['#7f97aa','#b5a08c','#9aaa96','#ae94a6','#8daaab'][j],label=target)
        ax.axvline(0,color='.65',lw=.8)
        ax.set_yticks(y,[SHORT[m] for m in MODULES]); ax.invert_yaxis()
        ax.set_xlabel('Stable overlap: candidate minus matched reference')
        ax.set_title('Candidate specificity under RBH sensitivity'); ax.legend(ncol=5,fontsize=6,loc='upper center',bbox_to_anchor=(.5,-.16))
        save(fig,'05_candidate_reference',d,'Matched references test whether candidate enrichment is selective.',
             'Same frozen OC Granulosa network and strict all-seed hit criterion. Each candidate has one expression-matched reference. Positive count differences alone do not establish specificity; paired null P/q and matching diagnostics are supplied.')
    write_tsv(manifest,dest/'FIGURE_MANIFEST.tsv')
    return manifest


def report(root, out, synthesis, nodes, figure_manifest):
    v = load(out,'VKO_MODULE_ENRICHMENT.tsv')
    rb = load(out,'VKO_CANDIDATE_ROBUSTNESS.tsv')
    ledger=load(out,'BRIDGE_EVIDENCE_LEDGER.tsv')
    mapping=load(out,'ORTHOLOG_MAPPING_AUDIT.tsv')
    diagnostics=load(out,'VKO_MATCHING_DIAGNOSTICS.tsv')
    tests=load(out,'MODULE_EXACT_PERMUTATION.tsv')
    lines=['# Stage 25 蜂—小鼠卵巢功能韧性桥接报告', '',
           '**主要结论：跨项目证据支持有限的年龄反向转录移动，以 RNA 加工/翻译最一致；'
           '尚未建立 ECM—线粒体—蛋白稳态—脂质耦合改善、保护性正交重塑或节点特异的保守机制。**', '',
           '- Granulosa：RNA 加工/翻译和线粒体模块的治疗方向较一致；RNA 主模块治疗 exact P=0.10。'
           '线粒体自然年龄中心效应弱（OC−Y P=0.80），不能据治疗上调直接称为恢复。',
           '- Stromal：RNA 加工/翻译跨 Tier、中心/尾部及 subtype 标准化较一致。'
           '严格线粒体与膜模块的 pseudobulk 和细胞均值方向不同，不能称为稳健恢复。',
           '- ECM 在严格链映射下只有2个支持基因，不能正式评分；RBH 敏感性下 Stromal ECM 有13个表达基因，'
           '呈年龄反向的描述性变化（治疗 P=0.20），尚不足以证明 ECM 功能改善。',
           '- 五节点均未通过新增的匹配随机集＋reference 特异性标准。Smad3 保留 Stage24 稳定性优先级；'
           'Igfbp2 暂降级。',
           '- 主分析及 RBH 敏感性均未得到稳健的 treatment_specific_orthogonal 模块分类。'
           '这限制本次桥接对 H2 的支持，不否定之前全转录组分析中的治疗特异分量。', '',
           '## 结论范围', '',
           '本阶段检验冻结的蜂启发功能模块在 MRJP1 小鼠卵巢中的转录读数与候选网络扰动。'
           '所有生物学比较均以 library/pool 为单位，每组 n=3；结果不能证明卵巢功能恢复、保护性因果机制或整体年轻化。', '',
           f'输入：105,763 个已注释细胞；证据台账 {len(ledger):,} 条来源—模块记录。'
           '沿用 Stage 24 的 9 个已有网络，不重新构网，不重新 QC、聚类或差异分析。', '',
           '严格映射为主分析；互为最佳蛋白匹配（RBH）仅作为单列敏感性分析。'
           'RBH 的一对一是冻结匹配表内的关系，并非已经证明的一对一进化同源。', '',
           '## 1. 哪些模块得到支持', '']
    for pop in POPS:
        lines += [f'### {pop}', '', '|模块|基因数|年龄效应|治疗效应|治疗 exact P|治疗 LOO|综合分类|',
                  '|---|---:|---:|---:|---:|---:|---|']
        for r in synthesis[synthesis.cell_type.eq(pop)&synthesis.scope.eq('bridge_supported')].itertuples():
            if r.classification=='not_evaluable':
                lines.append(f'|{CN[r.module]}|{r.n_genes}|—|—|—|—|严格映射覆盖不足|')
            else:
                lines.append(f'|{CN[r.module]}|{r.n_genes}|{r.age_effect:.3f}|{r.treatment_effect:.3f}|{r.treatment_p_exact:.2f}|{r.treatment_loo_fraction:.2f}|{r.classification}|')
        lines.append('')
    lines += ['效应为等权基因模块的平均 log2(CPM+1) 差：年龄=OC−Y，治疗=OT−OC。'
              '每个 library 内先汇总 raw counts，再等权汇总 library。负/正效应本身不能表示损伤/保护。', '',
              '完整三比较、残余偏离、Tier1 与 Tier1+Tier2、细胞均值/中位数/90% 分位数、'
              '固定 subtype 权重的结果见 MODULE_EXACT_PERMUTATION.tsv 和 MODULE_SYNTHESIS.tsv。', '',
              '严格生殖支持/分泌模块仅含 Cadps2、Sec13、Wls 三基因，主要反映运输/分泌相关 readout，'
              '不能从模块名称推断卵泡成熟或生殖功能恢复。Granulosa RNA 治疗效应超过年龄中心下降量，'
              'OT−Y 仍有正向残余偏离，也不等同于精确回到年轻状态。', '',
              '## 2. 年龄回移与治疗特异重塑', '',
              '每个模块在多基因空间中分解治疗方向相对年龄方向的投影与正交分量。'
              '年龄与治疗共用 OC，可能产生负相关偏差，因此另做 split-OC 描述性检验；'
              '高正交分量不等价于保护，年龄效应小也不能用“年龄比较不显著”推断正交。', '',
              '|细胞类型|参考|模块|cosine|正交范数比例|split-OC 分类一致比例|综合分类|',
              '|---|---|---|---:|---:|---:|---|']
    for r in synthesis[synthesis.scope.isin(['bridge_supported','bee_RBH_sensitivity'])&synthesis.classification.ne('not_evaluable')].itertuples():
        lines.append(f'|{r.cell_type}|{r.scope}|{CN[r.module]}|{r.cosine:.2f}|{r.orthogonal_fraction:.2f}|{r.split_OC_same_geometry_fraction:.2f}|{r.classification}|')
    lines += ['', '分类为预设的描述性决策规则，不是确证性检验结果。'
              'MODULE_AGE_TREATMENT_GEOMETRY.tsv 保留原始几何分类；综合分类同时检查 library、Tier 和中心分数敏感性。', '',
              '## 3–4. 候选特异性及 reference/随机对照', '',
              '|节点|Stage24 seed Jaccard|Stage24 最差 LOO Jaccard|Stage25 特异模块数|当前定位|',
              '|---|---:|---:|---:|---|']
    for r in nodes.itertuples():
        lines.append(f'|{r.candidate_node}|{r.stage24_seed_jaccard:.3f}|{r.stage24_loo_jaccard_min:.3f}|{r.n_specific_primary_modules}|{r.candidate_status}|')
    lines += ['', '“特异模块”要求 Tier1 三 seed 共同 top-5% 命中集：优于匹配随机集和候选对应 reference，'
              '两项竞争性 BH q 均≤0.05、候选重叠更多且匹配质量通过。'
              '这仍是网络模型内的特异性标准，不能升级为 library 级因果证据。', '',
              '随机基因集每次保持模块大小，联合匹配 OC 平均表达、检出率、WT 加权网络度数和边度数；'
              '每组合 2,000 个随机集，不放回抽取。基因集合、随机种子、匹配标准化偏差及空分布频数均可追溯。'
              '为保证候选/reference 使用相同背景，10 个被敲除节点均从共同背景和模块中排除。', '',
              f'匹配诊断共 {len(diagnostics)} 个场景×模块×来源组合，其中 '
              f'{int(diagnostics.matching_pass.eq(False).sum())} 个平均绝对标准化偏差超过 0.25；'
              '这些条目的随机对照比较标为匹配不足，不能据此确认候选特异性。', '',
              'VKO_CANDIDATE_ROBUSTNESS.tsv 同时记录每个 seed、Tier1+Tier2 和三个 OC LOO 场景。'
              'scTenifold 距离无方向；VKO_REAL_TRANSCRIPTOMIC_READOUTS.tsv 中的年龄/治疗正负号来自已有真实转录比较，'
              '不能解释为虚拟敲除预测了这些方向。', '',
              '严格映射模块进入冻结网络后，仅 RNA、膜和蛋白稳态分别保留4、3、4个可检验基因；'
              '其他模块低于3基因阈值。因此“无特异富集”包含明显的网络背景覆盖限制。'
              'RBH 敏感性扩大后，主候选的最小匹配随机 P 仍为约0.122，未带来独立的候选特异支持。'
              'Smad3 的 RBH 膜模块命中 Mctp1/Tanc2、脂质模块命中 Acsbg1/Elovl6；'
              '这些观察重叠可以设计 readout，但不能据重叠本身认定选择性机制。', '',
              '## 5. Library、分布和细胞组成敏感性', '']
    bad=synthesis[synthesis.scope.isin(['bridge_supported','bee_RBH_sensitivity'])&synthesis.treatment_sign_flip_libraries.fillna('').ne('')]
    if len(bad):
        for r in bad.itertuples():
            lines.append(f'- {r.cell_type} / {r.scope} / {CN[r.module]}：剔除 {r.treatment_sign_flip_libraries} 后治疗中心效应变号。')
    else:
        lines.append('可评估主模块的治疗中心效应未出现单库剔除变号；这不等于精确估计或无混杂。')
    lines += ['', '细胞内总体 readout 与高分尾部可能方向不同，二者在 MODULE_SYNTHESIS.tsv 单列。'
              '固定 subtype 权重仅用于所有九个 library 各有≥10个细胞的共同 subtype，'
              '排除比例在 MODULE_LIBRARY_SCORES.tsv 中披露。它削弱已注释 subtype 组成解释，'
              '不能排除未知状态混合或选择性存活。捕获比例不能解释为绝对细胞数。', '',
              '## H1–H4 竞争解释', '',
              '- H1：对通过方向/LOO 敏感性的模块，可称为部分 age-opposite transcriptional movement；'
              '残余偏离及没有年龄下降的模块不能称为恢复。',
              '- H2：正交分量及 RBH 敏感性结果提供治疗特异重塑的描述性候选；'
              '保护性仍需压力挑战和功能终点验证。',
              '- H3：一般应激、代谢转录变化和选择性存活无法从本实验的单次细胞捕获中排除；'
              '中心/尾部与 Tier 分析只提供敏感性信息。',
              '- H4：在 Granulosa 和 Stromal 内部分别分析，且重复 subtype 权重标准化；'
              '这降低大类细胞比例混合的影响，不能证明绝对细胞数稳定。', '',
              '模块耦合另以九个 library 的相关性及去除组均值后的相关性描述（MODULE_COUPLING_DESCRIPTIVE.tsv）。'
              '每组只有三库，不将这些相关性解释为细胞间支持、通讯或功能韧性已恢复。', '',
              'MODULE_TECHNICAL_CORRELATIONS.tsv 提供 library 中位 UMI/线粒体比例与各评分的相关性，'
              '用于识别深度或细胞状态混杂；它不构成消除混杂后的因果估计。', '',
              '## 6. 未支持或无法建立的跨物种联系', '',
              '- 不存在本阶段证明的“蜂王型年轻化基因”或蜂王—年轻/工蜂—衰老对应。',
              '- 严格链映射覆盖不足的模块属于证据不可判定，不是生物学上不存在；完整一对多候选保留于映射明细。',
              '- Bombus cluster 15、br/Hr4 不能凭分泌状态或家族相似建立与小鼠节点的一对一机制联系。',
              '- 人端 bee-prior rescued readout 如果缺少直接蜂 ID 或人端独立支持，仅标记 hypothesis_only。',
              '- IGFBP2 与 IGFBP5 不是同一基因，同属家族不能证明机制保守；Igfbp2 的 Stage24 稳定性不足，暂不纳入最小验证主线。',
              '- Smad3 在 Stage24 较稳定与 Smad3 对蜂模块具有选择性是两个不同问题。'
              '本阶段根据 reference 和匹配随机对照单独评价后者。', '',
              '## 7. 最小湿实验组合', '',
              '优先采用老龄原代 Granulosa 与 Stromal 共培养/卵泡培养，先完成 MRJP1 与 vehicle 的压力挑战比较，'
              '再聚焦一个节点。建议最小设计：MRJP1×Smad3 干预的 2×2 因子实验（vehicle/对照干预、'
              'MRJP1/对照干预、vehicle/Smad3干预、MRJP1/Smad3干预）；以独立动物/独立制备为重复，'
              '避免把培养孔作为生物学重复。干预方向由蛋白/活性实测确定，不从无符号 VKO 推断。', '',
              '最少同时检测：(1) p-SMAD3 与 COL1A1/FN1 或基底膜 ECM；(2) ATP/线粒体膜电位加细胞死亡；'
              '(3) 卵泡存活/成熟或 E2 分泌。加入等强度压力条件，可区分保护性韧性、一般应激和幸存细胞选择。'
              'MRJP1 的节点依赖性需考察交互效应，并用第二种正交干预或救援确认。'
              'Hif1a 和 Abca1 作为第二轮候选；如首轮脂质读数最突出再加入胆固醇外排/filipin。', '',
              '## 方法与可复现性', '',
              '两组比较穷举 C(6,3)=20 种标签排列，双侧最小 P=0.10，不能产生 P<0.05 的正式 library 级结果。'
              'BH 按来源×Tier×metric×contrast 的预定比较族校正；LOO 仅作方向敏感性，逐条输出。'
              '模块至少有3个已表达基因才评分。主表中的证据 Tier 与小鼠细胞质量 Tier 是两个独立字段。', '',
              '模块按冻结功能注释规则定义，可重复分配到多个功能模块；不把激活上/下调符号移植为鼠年龄或保护方向。'
              '蛋白稳态模块含泛素/蛋白酶/内质网相关注释，不能单凭模块名解释为自噬通量增加。', '',
              f'图件共 {len(figure_manifest)} 张，全部独立 SVG/PDF/PNG 与 source data，无拼接 panel。'
              '代码、来源清单、输入哈希、分类规则和运行状态保留；大型结果不进入 Git。', '',
              '分析至此停止，不新增公共表达数据或机器学习模型。']
    (out/'STAGE25_REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def finalize(root):
    out=root/'results/deep_dive_stage25_bee_mouse_bridge'
    synthesis=module_synthesis(out)
    nodes=node_model(root,out)
    fig_manifest=figures(root,out)
    report(root,out,synthesis,nodes,fig_manifest)
    # Validate biological unit, exact-test resolution, identities and deliverable integrity.
    scores=load(out,'MODULE_LIBRARY_SCORES.tsv')
    tests=load(out,'MODULE_EXACT_PERMUTATION.tsv')
    counts=scores.groupby(['cell_type','cell_tier','scope','module','metric']).library_id.nunique()
    assert counts.eq(9).all()
    finite=tests[tests.p_exact_two_sided.notna()]
    assert finite.n_permutations.eq(20).all()
    assert finite.p_exact_two_sided.ge(.1-1e-10).all()
    required=['STAGE25_REPORT_CN.md','BRIDGE_EVIDENCE_LEDGER.tsv','ORTHOLOG_MAPPING_AUDIT.tsv',
              'MODULE_LIBRARY_SCORES.tsv','MODULE_EXACT_PERMUTATION.tsv','MODULE_LOO_STABILITY.tsv',
              'VKO_MODULE_ENRICHMENT.tsv','NODE_STATE_FUNCTION_MODEL.tsv','RUN_STATE.json']
    assert all((out/name).exists() and (out/name).stat().st_size for name in required)
    dest=root/'figures/deep_dive_stage25_bee_mouse_bridge'
    for row in fig_manifest:
        for ext in ['svg','pdf','png']:
            assert (dest/f'{row["figure"]}.{ext}').stat().st_size > 1000
        assert '<text ' in (dest/f'{row["figure"]}.svg').read_text()
    write_json({'status':'passed','biological_unit':'library/pool','all_score_groups_have_nine_libraries':True,
                'exact_minimum_p':float(finite.p_exact_two_sided.min()),'n_figures':len(fig_manifest),
                'source_data_for_every_figure':True,'svg_editable_text':True,
                'visual_review':'pending'},out/'VALIDATION.json')
    manifest=[]
    for directory in [out,dest]:
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.name not in ['OUTPUT_MANIFEST.tsv','RUN_STATE.json']:
                manifest.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    write_tsv(manifest,out/'OUTPUT_MANIFEST.tsv')
    state=json.loads((out/'RUN_STATE.json').read_text())
    state['steps']['report_figures_validation']={'status':'complete','finished':now()}
    state['overall']='awaiting_visual_review'
    write_json(state,out/'RUN_STATE.json')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    if str(a.root.resolve())!='/root/autodl-tmp/ovary_scRNAseq':
        raise RuntimeError('Formal reporting runs only on the main server')
    finalize(a.root)
