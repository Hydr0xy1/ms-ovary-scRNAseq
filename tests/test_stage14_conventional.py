from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

from ms_ovary_scrna.stage14_conventional import (
    _marker_rows_for_labels,
    _plot_volcano,
    assign_cell_cycle_phase,
    compute_broad_abundance,
    compute_subtype_abundance,
    ora_enrichment,
    validate_enrichment_schema,
    validate_gene_set_mapping,
    validate_output_manifest,
)


def _obs() -> pd.DataFrame:
    rows = []
    for library in ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]:
        group = library.split("_", 1)[0]
        rows.extend(
            [
                (library, group, "Granulosa", "Granulosa_antral_like"),
                (library, group, "Granulosa", "Granulosa_cycling"),
                (library, group, "Immune", "T_cell_candidate"),
            ]
        )
    return pd.DataFrame(
        rows,
        columns=["library_id", "group", "cell_type_broad_v2", "cell_type_subtype_v2"],
    )


def test_broad_abundance_uses_library_denominator() -> None:
    counts, long = compute_broad_abundance(_obs())
    assert counts.loc[counts["library_id"].eq("Y_1"), "Granulosa"].iloc[0] == 2
    y1 = long.loc[long["library_id"].eq("Y_1")]
    assert np.isclose(y1["fraction_of_total_cells"].sum(), 1)
    assert np.isclose(
        y1.loc[y1["cell_type"].eq("Granulosa"), "fraction_of_total_cells"].iloc[0],
        2 / 3,
    )


def test_subtype_fraction_uses_parent_and_whole_ovary_denominators() -> None:
    result = compute_subtype_abundance(_obs(), "Granulosa")
    row = result.loc[
        result["library_id"].eq("OC_1")
        & result["subtype"].eq("Granulosa_antral_like")
    ].iloc[0]
    assert row["fraction_of_parent"] == 0.5
    assert np.isclose(row["fraction_of_whole_ovary"], 1 / 3)


def test_gene_set_identifier_mapping_requires_mouse_overlap_and_unique_universe() -> None:
    result = validate_gene_set_mapping(
        {"Mouse_set": [f"Gene{i}" for i in range(150)]},
        [f"Gene{i}" for i in range(200)],
    )
    assert result["n_overlapping_genes"] == 150


def test_ora_uses_provided_tested_gene_background() -> None:
    result = ora_enrichment(
        selected_genes=["A", "B"],
        tested_genes=["A", "B", "C", "D"],
        gene_sets={"set1": ["A", "B", "outside"]},
    )
    row = result.iloc[0]
    assert row["background_size"] == 4
    assert row["term_size"] == 2
    assert np.isclose(row["pvalue"], hypergeom.sf(1, 4, 2, 2))


def test_enrichment_result_schema() -> None:
    table = pd.DataFrame(
        {
            "population": ["Granulosa"],
            "contrast": ["OC_vs_Y"],
            "database": ["GO_BP"],
            "term": ["GOBP_TEST"],
            "NES": [1.5],
            "nominal_p": [0.01],
            "FDR_q": [0.04],
            "leading_edge": ["A;B"],
            "ranking_metric": ["unified_9_library_DE_Wald_statistic"],
            "statistical_unit": ["library"],
        }
    )
    validate_enrichment_schema(table)


def test_cell_cycle_phase_assignment_matches_scanpy_rule() -> None:
    phases = assign_cell_cycle_phase([-1, 2, 0.1, -0.2], [-2, 1, 3, 0.4])
    assert phases.tolist() == ["G1", "S", "G2M", "G2M"]


def test_marker_catalogue_is_unique_per_label_gene() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [4.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [0.0, 4.0, 1.0],
                [0.0, 3.0, 1.0],
            ]
        )
    )
    markers = _marker_rows_for_labels(
        matrix,
        pd.Series(["A", "A", "B", "B"]),
        np.array(["GeneA", "GeneB", "Shared"]),
        {"A": {"GeneA"}, "B": {"GeneB"}},
        top_n=2,
    )
    assert not markers.duplicated(["label", "gene"]).any()
    assert set(markers.loc[markers["canonical_marker_flag"], "gene"]) == {"GeneA", "GeneB"}


def test_reused_subtype_name_is_scoped_to_parent_broad_type() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [4.0, 0.0],
                [0.0, 2.0],
                [0.0, 4.0],
                [2.0, 0.0],
            ]
        )
    )
    result = _marker_rows_for_labels(
        matrix,
        pd.Series(["Shared", "Other", "Shared", "Other"]),
        np.array(["GeneA", "GeneB"]),
        {},
        top_n=1,
        parent_labels=pd.Series(["Parent1", "Parent1", "Parent2", "Parent2"]),
    )
    assert set(result.loc[result["label"].eq("Shared"), "parent_label"]) == {
        "Parent1",
        "Parent2",
    }


def test_de_visualization_source_preserves_na_padj_and_lfc() -> None:
    table = pd.DataFrame(
        {
            "population": ["Granulosa", "Granulosa"],
            "contrast": ["OC_vs_Y", "OC_vs_Y"],
            "canonical_mouse_symbol": ["A", "B"],
            "log2FoldChange": [1.0, -0.5],
            "pvalue": [0.001, 0.2],
            "padj": [0.01, np.nan],
            "fdr_supported": [True, False],
            "effect_size_supported": [True, True],
        }
    )
    fig, source = _plot_volcano(table, "Granulosa", "OC_vs_Y")
    plt.close(fig)
    assert source.loc[source["canonical_mouse_symbol"].eq("B"), "padj"].isna().all()
    assert source.set_index("canonical_mouse_symbol").loc["A", "log2FoldChange"] == 1.0


def test_output_manifest_completeness(tmp_path: Path) -> None:
    path = tmp_path / "result.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = pd.DataFrame(
        [{"path": "result.tsv", "bytes": path.stat().st_size, "sha256": digest}]
    )
    validate_output_manifest(manifest, tmp_path)
