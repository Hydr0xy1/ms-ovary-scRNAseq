from ms_ovary_scrna.project import DEFAULT_CONFIG, REPOSITORY_ROOT, load_config, project_paths


def test_default_paths_are_repository_relative() -> None:
    paths = project_paths(load_config(DEFAULT_CONFIG))
    assert paths["root"] == REPOSITORY_ROOT
    assert paths["input"] == REPOSITORY_ROOT / "data" / "Summary"
    assert paths["metadata"] == REPOSITORY_ROOT / "metadata" / "sample_metadata.tsv"
    assert paths["markers"] == (
        REPOSITORY_ROOT / "resources" / "markers" / "ovary_markers.yaml"
    )
