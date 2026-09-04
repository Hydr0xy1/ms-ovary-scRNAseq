import pandas as pd

from ms_ovary_scrna.de_stage1 import build_eligibility, prefilter_genes


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pops = ["Granulosa", "Rare_luteal_candidate"]
    libs = ["Y_1", "Y_2", "Y_3", "OC_1", "OC_2", "OC_3", "OT_1", "OT_2", "OT_3"]
    rows = []
    qc_rows = []
    for pop in pops:
        for lib in libs:
            group = lib.split("_")[0]
            n = 60 if pop == "Granulosa" else 10
            rows.append((pop, lib, [1, 2, 20]))
            qc_rows.append({"population": pop, "library": lib, "n_cells": n, "total_umi": 1000, "n_expressed_genes": 3})
    idx = pd.MultiIndex.from_tuples([(p, l) for p, l, _ in rows], names=["population", "library"])
    counts = pd.DataFrame([v for _, _, v in rows], index=idx, columns=["g1", "g2", "g3"])
    metadata = pd.DataFrame(
        [{"population": p, "library": l, "group": l.split("_")[0]} for p, l, _ in rows]
    )
    return counts, metadata, pd.DataFrame(qc_rows)


def test_prefilter_requires_three_samples_at_count_ten() -> None:
    counts = pd.DataFrame([[10, 10, 1], [10, 2, 1], [10, 1, 1]], columns=["a", "b", "c"])
    assert list(prefilter_genes(counts).columns) == ["a"]


def test_eligibility_is_contrast_specific_and_descriptive_populations_are_not_tested() -> None:
    counts, metadata, qc = _tables()
    table = build_eligibility(counts, metadata, qc)
    granulosa = table[table.population == "Granulosa"]
    assert granulosa["final_de_status"].eq("Primary_DE_ready").all()
    rare = table[table.population == "Rare_luteal_candidate"]
    assert rare["final_de_status"].eq("Descriptive_only").all()

