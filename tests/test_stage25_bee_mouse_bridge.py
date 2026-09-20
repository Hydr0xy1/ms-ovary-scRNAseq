import numpy as np
import pandas as pd

from ms_ovary_scrna.stage25_bee_mouse_bridge import (
    base_classification, exact_test, geometry, matched_sets,
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
