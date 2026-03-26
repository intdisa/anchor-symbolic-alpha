from pathlib import Path

from knowledge_guided_symbolic_alpha.dataio.us_equities import (
    WRDS_REQUIRED_DATASETS,
    build_wrds_query,
    default_output_paths,
    load_us_equities_config,
    validate_us_equities_layout,
)


def test_load_us_equities_config_includes_required_wrds_specs() -> None:
    config = load_us_equities_config("configs/us_equities_data.yaml")

    assert {spec.dataset_name for spec in config.wrds_specs} == set(WRDS_REQUIRED_DATASETS)
    assert config.start_date == "2000-01-01"
    assert config.end_date == "2025-12-31"


def test_build_wrds_query_applies_date_and_filters() -> None:
    config = load_us_equities_config("configs/us_equities_data.yaml")
    spec = next(item for item in config.wrds_specs if item.dataset_name == "crsp_daily")

    sql = build_wrds_query(spec)

    assert "from crsp.dsf" in sql
    assert "shrcd in (10, 11)" in sql
    assert "date >= '2000-01-01'" in sql
    assert "date <= '2025-12-31'" in sql
    assert "order by permno, date" in sql


def test_default_output_paths_match_us_equities_layout() -> None:
    config = load_us_equities_config("configs/us_equities_data.yaml")

    paths = default_output_paths(config)

    assert paths["crsp_daily"] == Path("data/raw/us_equities/wrds/crsp_daily.csv.gz")
    assert paths["fama_french_daily"] == Path("data/raw/us_equities/public/fama_french_daily.csv.gz")


def test_validate_us_equities_layout_reports_missing_files(tmp_path: Path) -> None:
    result = validate_us_equities_layout(tmp_path)

    assert "crsp_daily" in result["missing_wrds"]
    assert "fama_french_daily" in result["missing_public"]
