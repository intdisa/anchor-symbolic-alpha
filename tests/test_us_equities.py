from pathlib import Path

from knowledge_guided_symbolic_alpha.dataio.us_equities import (
    WRDS_REQUIRED_DATASETS,
    build_wrds_query,
    default_output_paths,
    load_route_b_config,
    validate_route_b_layout,
)


def test_load_route_b_config_includes_required_wrds_specs() -> None:
    config = load_route_b_config("configs/route_b_data.yaml")

    assert {spec.dataset_name for spec in config.wrds_specs} == set(WRDS_REQUIRED_DATASETS)
    assert config.start_date == "2000-01-01"
    assert config.end_date == "2025-12-31"


def test_build_wrds_query_applies_date_and_filters() -> None:
    config = load_route_b_config("configs/route_b_data.yaml")
    spec = next(item for item in config.wrds_specs if item.dataset_name == "crsp_daily")

    sql = build_wrds_query(spec)

    assert "from crsp.dsf" in sql
    assert "shrcd in (10, 11)" in sql
    assert "date >= '2000-01-01'" in sql
    assert "date <= '2025-12-31'" in sql
    assert "order by permno, date" in sql


def test_default_output_paths_match_route_b_layout() -> None:
    config = load_route_b_config("configs/route_b_data.yaml")

    paths = default_output_paths(config)

    assert paths["crsp_daily"] == Path("data/raw/route_b/wrds/crsp_daily.parquet")
    assert paths["fama_french_daily"] == Path("data/raw/route_b/public/fama_french_daily.parquet")


def test_validate_route_b_layout_reports_missing_files(tmp_path: Path) -> None:
    result = validate_route_b_layout(tmp_path)

    assert "crsp_daily" in result["missing_wrds"]
    assert "fama_french_daily" in result["missing_public"]
