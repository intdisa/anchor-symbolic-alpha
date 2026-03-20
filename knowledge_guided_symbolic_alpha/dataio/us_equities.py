from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


WRDS_REQUIRED_DATASETS = (
    "crsp_daily",
    "crsp_names",
    "ccm_link",
    "compustat_quarterly",
    "compustat_annual",
)

PUBLIC_REQUIRED_DATASETS = (
    "fama_french_daily",
    "fred_macro_daily",
)


@dataclass(frozen=True)
class WRDSExtractSpec:
    dataset_name: str
    schema: str
    table: str
    columns: tuple[str, ...]
    date_column: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    filters: tuple[str, ...] = ()
    order_by: tuple[str, ...] = ()
    output_file: str | None = None

    @property
    def qualified_table(self) -> str:
        return f"{self.schema}.{self.table}"


@dataclass(frozen=True)
class RouteBConfig:
    output_root: Path
    start_date: str
    end_date: str
    universe_name: str
    wrds_specs: tuple[WRDSExtractSpec, ...]
    public_series: dict[str, dict[str, Any]]


def load_route_b_config(path: str | Path) -> RouteBConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = payload.get("us_equities") or payload.get("route_b", {})
    output_root = Path(config.get("output_root", "data/raw/route_b"))
    start_date = str(config["start_date"])
    end_date = str(config["end_date"])
    universe_name = str(config.get("universe_name", "us_equities"))
    wrds_specs = tuple(_parse_wrds_spec(name, config["wrds"][name], start_date, end_date) for name in WRDS_REQUIRED_DATASETS)
    public_series = {str(name): dict(spec) for name, spec in config.get("public_sources", {}).items()}
    return RouteBConfig(
        output_root=output_root,
        start_date=start_date,
        end_date=end_date,
        universe_name=universe_name,
        wrds_specs=wrds_specs,
        public_series=public_series,
    )


def _parse_wrds_spec(dataset_name: str, payload: dict[str, Any], start_date: str, end_date: str) -> WRDSExtractSpec:
    return WRDSExtractSpec(
        dataset_name=dataset_name,
        schema=str(payload["schema"]),
        table=str(payload["table"]),
        columns=tuple(str(item) for item in payload["columns"]),
        date_column=str(payload["date_column"]) if payload.get("date_column") else None,
        start_date=str(payload.get("start_date", start_date)) if payload.get("date_column") else None,
        end_date=str(payload.get("end_date", end_date)) if payload.get("date_column") else None,
        filters=tuple(str(item) for item in payload.get("filters", ())),
        order_by=tuple(str(item) for item in payload.get("order_by", ())),
        output_file=str(payload.get("output_file", f"{dataset_name}.parquet")),
    )


def build_wrds_query(spec: WRDSExtractSpec) -> str:
    select_clause = ",\n       ".join(spec.columns)
    where_clauses = list(spec.filters)
    if spec.date_column and spec.start_date:
        where_clauses.append(f"{spec.date_column} >= '{spec.start_date}'")
    if spec.date_column and spec.end_date:
        where_clauses.append(f"{spec.date_column} <= '{spec.end_date}'")
    sql = [
        "select",
        f"       {select_clause}",
        f"from {spec.qualified_table}",
    ]
    if where_clauses:
        sql.append("where " + "\n  and ".join(where_clauses))
    if spec.order_by:
        sql.append("order by " + ", ".join(spec.order_by))
    return "\n".join(sql)


def default_output_paths(config: RouteBConfig) -> dict[str, Path]:
    wrds_root = config.output_root / "wrds"
    public_root = config.output_root / "public"
    paths = {
        spec.dataset_name: wrds_root / str(spec.output_file)
        for spec in config.wrds_specs
    }
    paths.update(
        {
            dataset_name: public_root / str(spec.get("output_file", f"{dataset_name}.parquet"))
            for dataset_name, spec in config.public_series.items()
        }
    )
    return paths


def validate_route_b_layout(root: str | Path) -> dict[str, list[str]]:
    root_path = Path(root)
    missing_wrds = [name for name in WRDS_REQUIRED_DATASETS if not any(root_path.glob(f"wrds/{name}.*"))]
    missing_public = [name for name in PUBLIC_REQUIRED_DATASETS if not any(root_path.glob(f"public/{name}.*"))]
    return {
        "missing_wrds": missing_wrds,
        "missing_public": missing_public,
    }


load_us_equities_config = load_route_b_config
validate_us_equities_layout = validate_route_b_layout
