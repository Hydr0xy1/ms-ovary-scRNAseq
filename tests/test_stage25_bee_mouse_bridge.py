import numpy as np
import pandas as pd

from ms_ovary_scrna.stage25_bee_mouse_bridge import (
    base_classification, exact_test, geometry, matched_sets, mapping_audit,
)


def test_exact_permutation_respects_six_library_resolution():
    result = exact_test([0, 1, 2, 10, 11, 12], ['Y']*3 + ['OC']*3, 'OC', 'Y')
    assert result['n_permutations'] == 20
    assert result['p_exact_two_sided'] == 0.1
    assert result['effect'] == 10
    reverse = exact_test([0, 1, 2, 10, 11, 12], ['Y']*3 + ['OC']*3, 'Y', 'OC')
    assert reverse['effect'] == -10
    assert reverse['p_exact_two_sided'] == 0.1


def test_geometry_distinguishes_rotation_from_scalar_sign():
    g = geometry([1, 0], [0, 2])
    assert g['orthogonal_fraction'] == 1
    assert base_classification(g, 0.5, 1) == 'treatment_specific_orthogonal'
    g = geometry([1, 2], [-0.5, -1])
    assert g['age_projection_coefficient'] == -0.5
    assert base_classification(g, 1.5, -0.75) == 'age_opposite'
    assert np.isnan(geometry([0, 0], [1, 1])['cosine'])


def test_matching_preserves_size_and_excludes_original_members():
    rng = np.random.default_rng(13)
    features = pd.DataFrame(rng.normal(size=(90, 4)), index=[f'g{x}' for x in range(90)])
    original = ['g1', 'g2', 'g3']
    draws, diagnostic = matched_sets(features, original, set(original), 80, rng)
    assert draws.shape == (80, 3)
    assert not np.isin(draws, original).any()
    assert all(len(set(row)) == 3 for row in draws)
    assert diagnostic['n_random'] == 80


def test_ambiguous_homologs_are_retained_and_hypotheses_excluded(tmp_path):
    resource=tmp_path/'resources/cross_project_bee_bridge'
    resource.mkdir(parents=True)
    out=tmp_path/'out'; out.mkdir()
    hm=pd.DataFrame({'human_gene_id':['1','2','3'],'human_symbol':['H1','H2','H3'],
                     'mouse_gene_id':['11','12','13'],'mouse_symbol':['M1','M2','M3']})
    hm.to_csv(resource/'human_mouse_strict_pairs.tsv.gz',sep='\t',index=False)
    bridge=pd.DataFrame({'apis_ncbi_gene_id':['A1','A1','A2'],'human_symbol':['H1','H2','H3'],
                         'apis_dmel_orthology_type':['ortholog_one2one']*3,
                         'dmel_human_orthology_type':['ortholog_one2many','ortholog_one2many','ortholog_one2one'],
                         'both_orthology_confident':[True]*3})
    bridge.to_csv(resource/'apis_fly_human_mapping.tsv.gz',sep='\t',index=False)
    rb=pd.DataFrame({'mapping_tier':['diamond_rbh_direct_1to1']*2,
                     'apis_gene_id':['A1','A2'],'human_symbol':['H1','H3']})
    rb.to_csv(resource/'apis_human_alternative_mappings.tsv.gz',sep='\t',index=False)
    ledger=pd.DataFrame({'evidence_id':['E1','E2'],'source_gene':['A1','A2'],
                         'source_species':['Apis_mellifera']*2,'apis_gene_id':['A1','A2'],
                         'module':['mitochondrial_energy']*2,'direct_bee_derived':[True]*2,
                         'mammalian_supported':[False]*2,'hypothesis_only':[False,True],
                         'evidence_tier':['Tier1_multi_context']*2})
    definitions=mapping_audit(tmp_path,out,ledger)
    detail=pd.read_csv(out/'ORTHOLOG_MAPPING_DETAIL.tsv',sep='\t')
    ambiguous=detail[detail.evidence_id.eq('E1')]
    assert set(ambiguous.mouse_gene)=={'M1','M2'}
    assert ambiguous.mapping_status.eq('one_to_many').all()
    assert not definitions.scope.eq('bridge_supported').any()
    assert 'M3' not in set(definitions[definitions.scope.eq('bee_RBH_sensitivity')].mouse_gene)
