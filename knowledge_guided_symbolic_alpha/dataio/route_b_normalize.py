from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DEFAULT_XSC_SOURCE_ROOT = Path("xsc数据")
DEFAULT_ROUTE_B_WRDS_ROOT = Path("data/raw/route_b/wrds")

CRSP_SOURCE_FILE = "crsp0025.csv"
CCM_SOURCE_FILE = "ccm_link_2025.dta"
QUARTERLY_SOURCE_FILE = "Comp_Quarterly6126.csv"
ANNUAL_SOURCE_FILE = "fwn4guf3dvyp5rls.csv"

CRSP_DAILY_OUTPUT = "crsp_daily.csv.gz"
CRSP_NAMES_OUTPUT = "crsp_names.csv.gz"
CCM_OUTPUT = "ccm_link.csv.gz"
QUARTERLY_OUTPUT = "compustat_quarterly.csv.gz"
ANNUAL_OUTPUT = "compustat_annual.csv.gz"

CRSP_USECOLS = [
    "PERMNO",
    "PERMCO",
    "PrimaryExch",
    "Ticker",
    "TradingSymbol",
    "CUSIP",
    "HdrCUSIP",
    "SecurityNm",
    "IssuerNm",
    "SICCD",
    "USIncFlg",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "DlyCalDt",
    "DlyRet",
    "DlyRetx",
    "DlyPrc",
    "DlyVol",
    "DlyBid",
    "DlyAsk",
    "DlyLow",
    "DlyHigh",
    "DlyFacPrc",
    "DisFacPr",
    "DisFacShr",
    "ShrOut",
]

CRSP_DAILY_COLUMNS = [
    "permno",
    "permco",
    "date",
    "ret",
    "retx",
    "dlret",
    "prc",
    "vol",
    "shrout",
    "bidlo",
    "askhi",
    "cfacpr",
    "cfacshr",
    "exchcd",
    "shrcd",
]

CRSP_NAMES_COLUMNS = [
    "permno",
    "ticker",
    "ncusip",
    "comnam",
    "namedt",
    "nameendt",
    "exchcd",
    "shrcd",
    "siccd",
]

QUARTERLY_USECOLS = [
    "gvkey",
    "datadate",
    "rdq",
    "fyearq",
    "fqtr",
    "atq",
    "ltq",
    "ceqq",
    "seqq",
    "saleq",
    "niq",
    "oiadpq",
    "cheq",
    "dlcq",
    "dlttq",
    "actq",
    "lctq",
    "rectq",
    "invtq",
    "cogsq",
    "xsgaq",
]

ANNUAL_USECOLS = [
    "gvkey",
    "datadate",
    "fyear",
    "at",
    "lt",
    "ceq",
    "seq",
    "sale",
    "ni",
    "oiadp",
    "capx",
    "txditc",
    "pstkrv",
    "pstkl",
    "pstk",
]

CCM_USECOLS = ["gvkey", "LPERMNO", "LINKDT", "LINKENDDT", "LINKTYPE", "LINKPRIM"]

PRIMARY_EXCHANGE_MAP = {
    "N": 1,
    "P": 1,
    "A": 2,
    "Q": 3,
    "X": 3,
    "R": 3,
    "Z": 3,
}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class RouteBNormalizationSummary:
    crsp_daily_rows: int
    crsp_names_rows: int
    ccm_rows: int
    quarterly_rows: int
    annual_rows: int
    outputs: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, indent=2)


def normalize_xsc_route_b(
    source_root: str | Path = DEFAULT_XSC_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_ROUTE_B_WRDS_ROOT,
    chunksize: int = 250_000,
    datasets: tuple[str, ...] | None = None,
) -> RouteBNormalizationSummary:
    source_root = Path(source_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = set(datasets or ("crsp", "ccm", "quarterly", "annual"))
    unknown = selected.difference({"crsp", "ccm", "quarterly", "annual"})
    if unknown:
        raise ValueError(f"Unknown dataset names: {sorted(unknown)!r}")

    daily_rows = 0
    names_rows = 0
    ccm_rows = 0
    quarterly_rows = 0
    annual_rows = 0

    if "crsp" in selected:
        daily_rows, names_rows = _normalize_crsp(
            source_root / CRSP_SOURCE_FILE,
            output_root / CRSP_DAILY_OUTPUT,
            output_root / CRSP_NAMES_OUTPUT,
            chunksize=chunksize,
        )
    if "ccm" in selected:
        ccm_rows = _normalize_ccm(source_root / CCM_SOURCE_FILE, output_root / CCM_OUTPUT)
    if "quarterly" in selected:
        quarterly_rows = _normalize_quarterly(
            source_root / QUARTERLY_SOURCE_FILE,
            output_root / QUARTERLY_OUTPUT,
            chunksize=chunksize,
        )
    if "annual" in selected:
        annual_rows = _normalize_annual(
            source_root / ANNUAL_SOURCE_FILE,
            output_root / ANNUAL_OUTPUT,
            chunksize=chunksize,
        )

    return RouteBNormalizationSummary(
        crsp_daily_rows=daily_rows,
        crsp_names_rows=names_rows,
        ccm_rows=ccm_rows,
        quarterly_rows=quarterly_rows,
        annual_rows=annual_rows,
        outputs={
            "crsp_daily": str(output_root / CRSP_DAILY_OUTPUT),
            "crsp_names": str(output_root / CRSP_NAMES_OUTPUT),
            "ccm_link": str(output_root / CCM_OUTPUT),
            "compustat_quarterly": str(output_root / QUARTERLY_OUTPUT),
            "compustat_annual": str(output_root / ANNUAL_OUTPUT),
        },
    )


def _normalize_crsp(
    source_path: Path,
    daily_output_path: Path,
    names_output_path: Path,
    chunksize: int,
) -> tuple[int, int]:
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    name_aggregates: list[pd.DataFrame] = []
    total_daily_rows = 0
    header_written = False
    _log(f"[route-b] start crsp source={source_path} chunksize={chunksize}")

    with gzip.open(daily_output_path, "wt", encoding="utf-8", newline="") as handle:
        for chunk_index, chunk in enumerate(
            pd.read_csv(source_path, usecols=CRSP_USECOLS, chunksize=chunksize, low_memory=False),
            start=1,
        ):
            filtered = _prepare_crsp_chunk(chunk)
            if filtered.empty:
                continue
            filtered[CRSP_DAILY_COLUMNS].to_csv(handle, index=False, header=not header_written)
            header_written = True
            total_daily_rows += len(filtered)
            name_aggregates.append(_aggregate_names_chunk(filtered))
            if chunk_index % 20 == 0:
                _log(f"[route-b] crsp chunks={chunk_index} rows_written={total_daily_rows}")

    if not header_written:
        raise ValueError(f"No qualifying common-stock rows found in {source_path}.")

    names = pd.concat(name_aggregates, ignore_index=True)
    names = (
        names.groupby(
            ["permno", "ticker", "ncusip", "comnam", "exchcd", "shrcd", "siccd"],
            dropna=False,
            as_index=False,
        )
        .agg(namedt=("namedt", "min"), nameendt=("nameendt", "max"))
        .sort_values(["permno", "namedt"])
        .reset_index(drop=True)
    )
    names[CRSP_NAMES_COLUMNS].to_csv(names_output_path, index=False, compression="gzip")
    _log(f"[route-b] finish crsp rows_written={total_daily_rows} names_rows={len(names)}")
    return total_daily_rows, len(names)


def _prepare_crsp_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    frame = chunk.copy()
    frame["USIncFlg"] = frame["USIncFlg"].astype(str).str.upper()
    frame["SecurityType"] = frame["SecurityType"].astype(str).str.upper()
    frame["SecuritySubType"] = frame["SecuritySubType"].astype(str).str.upper()
    frame["ShareType"] = frame["ShareType"].astype(str).str.upper()
    frame = frame.loc[
        (frame["USIncFlg"] == "Y")
        & (frame["SecurityType"] == "EQTY")
        & (frame["SecuritySubType"] == "COM")
        & (frame["ShareType"] == "NS")
    ].copy()
    if frame.empty:
        return frame

    frame["exchcd"] = frame["PrimaryExch"].astype(str).str.upper().map(PRIMARY_EXCHANGE_MAP)
    frame = frame.dropna(subset=["exchcd"]).copy()
    if frame.empty:
        return frame

    frame["permno"] = pd.to_numeric(frame["PERMNO"], errors="coerce").astype("Int64")
    frame["permco"] = pd.to_numeric(frame["PERMCO"], errors="coerce").astype("Int64")
    frame["date"] = pd.to_datetime(frame["DlyCalDt"], errors="coerce")
    frame["ret"] = pd.to_numeric(frame["DlyRet"], errors="coerce")
    frame["retx"] = pd.to_numeric(frame["DlyRetx"], errors="coerce")
    frame["dlret"] = np.nan
    frame["prc"] = pd.to_numeric(frame["DlyPrc"], errors="coerce")
    frame["vol"] = pd.to_numeric(frame["DlyVol"], errors="coerce")
    frame["shrout"] = pd.to_numeric(frame["ShrOut"], errors="coerce")
    frame["bidlo"] = pd.to_numeric(frame["DlyBid"], errors="coerce").fillna(pd.to_numeric(frame["DlyLow"], errors="coerce"))
    frame["askhi"] = pd.to_numeric(frame["DlyAsk"], errors="coerce").fillna(pd.to_numeric(frame["DlyHigh"], errors="coerce"))
    frame["cfacpr"] = pd.to_numeric(frame["DlyFacPrc"], errors="coerce").fillna(pd.to_numeric(frame["DisFacPr"], errors="coerce")).fillna(1.0)
    frame["cfacshr"] = pd.to_numeric(frame["DisFacShr"], errors="coerce").fillna(1.0)
    frame["shrcd"] = 10
    frame["ticker"] = _coalesce_strings(frame["Ticker"], frame["TradingSymbol"])
    frame["ncusip"] = _coalesce_strings(frame["CUSIP"], frame["HdrCUSIP"])
    frame["comnam"] = _coalesce_strings(frame["SecurityNm"], frame["IssuerNm"])
    frame["siccd"] = pd.to_numeric(frame["SICCD"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["permno", "permco", "date", "prc"])
    frame["permno"] = frame["permno"].astype("int64")
    frame["permco"] = frame["permco"].astype("int64")
    frame["exchcd"] = frame["exchcd"].astype("int64")
    return frame


def _aggregate_names_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    names = frame[["permno", "ticker", "ncusip", "comnam", "siccd", "exchcd", "shrcd", "date"]].copy()
    names = (
        names.groupby(
            ["permno", "ticker", "ncusip", "comnam", "siccd", "exchcd", "shrcd"],
            dropna=False,
            as_index=False,
        )
        .agg(namedt=("date", "min"), nameendt=("date", "max"))
        .sort_values(["permno", "namedt"])
    )
    return names


def _normalize_ccm(source_path: Path, output_path: Path) -> int:
    _log(f"[route-b] start ccm source={source_path}")
    frame = pd.read_stata(source_path, columns=CCM_USECOLS)
    frame = frame.rename(
        columns={
            "LPERMNO": "lpermno",
            "LINKDT": "linkdt",
            "LINKENDDT": "linkenddt",
            "LINKTYPE": "linktype",
            "LINKPRIM": "linkprim",
        }
    )
    frame["gvkey"] = _normalize_string_id(frame["gvkey"], width=6)
    frame["lpermno"] = pd.to_numeric(frame["lpermno"], errors="coerce").astype("Int64")
    frame["linkdt"] = pd.to_datetime(frame["linkdt"], errors="coerce")
    frame["linkenddt"] = pd.to_datetime(frame["linkenddt"], errors="coerce")
    frame = frame.dropna(subset=["gvkey", "lpermno", "linkdt"]).copy()
    frame["lpermno"] = frame["lpermno"].astype("int64")
    frame = frame[["gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"]].sort_values(["gvkey", "lpermno", "linkdt"])
    frame.to_csv(output_path, index=False, compression="gzip")
    _log(f"[route-b] finish ccm rows_written={len(frame)}")
    return len(frame)


def _normalize_quarterly(source_path: Path, output_path: Path, chunksize: int) -> int:
    ordered_columns = [
        "gvkey",
        "datadate",
        "rdq",
        "fyearq",
        "fqtr",
        "atq",
        "ltq",
        "ceqq",
        "seq",
        "saleq",
        "niq",
        "oiadpq",
        "cheq",
        "dlcq",
        "dlttq",
        "actq",
        "lctq",
        "rectq",
        "invtq",
        "cogsq",
        "xsgaq",
    ]
    numeric_columns = [column for column in ordered_columns if column not in {"gvkey", "datadate", "rdq"}]
    return _normalize_compustat_csv(
        source_path=source_path,
        output_path=output_path,
        usecols=QUARTERLY_USECOLS,
        ordered_columns=ordered_columns,
        numeric_columns=numeric_columns,
        chunksize=chunksize,
        rename_columns={"seqq": "seq"},
    )


def _normalize_annual(source_path: Path, output_path: Path, chunksize: int) -> int:
    ordered_columns = ANNUAL_USECOLS.copy()
    numeric_columns = [column for column in ordered_columns if column not in {"gvkey", "datadate"}]
    return _normalize_compustat_csv(
        source_path=source_path,
        output_path=output_path,
        usecols=ANNUAL_USECOLS,
        ordered_columns=ordered_columns,
        numeric_columns=numeric_columns,
        chunksize=chunksize,
    )


def _normalize_compustat_csv(
    source_path: Path,
    output_path: Path,
    usecols: list[str],
    ordered_columns: list[str],
    numeric_columns: list[str],
    chunksize: int,
    rename_columns: dict[str, str] | None = None,
) -> int:
    total_rows = 0
    header_written = False
    rename_columns = rename_columns or {}
    _log(f"[route-b] start compustat source={source_path} chunksize={chunksize}")
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        for chunk_index, chunk in enumerate(
            pd.read_csv(source_path, usecols=usecols, chunksize=chunksize, low_memory=False),
            start=1,
        ):
            frame = chunk.rename(columns=rename_columns).copy()
            frame["gvkey"] = _normalize_string_id(frame["gvkey"], width=6)
            frame["datadate"] = pd.to_datetime(frame["datadate"], errors="coerce")
            if "rdq" in frame.columns:
                frame["rdq"] = pd.to_datetime(frame["rdq"], errors="coerce")
            for column in numeric_columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.dropna(subset=["gvkey", "datadate"]).copy()
            if frame.empty:
                continue
            frame = frame[ordered_columns].sort_values(["gvkey", "datadate"])
            frame.to_csv(handle, index=False, header=not header_written)
            header_written = True
            total_rows += len(frame)
            if chunk_index % 10 == 0:
                _log(f"[route-b] compustat chunks={chunk_index} rows_written={total_rows} source={source_path.name}")
    if not header_written:
        raise ValueError(f"No rows written for {source_path}.")
    _log(f"[route-b] finish compustat rows_written={total_rows} source={source_path.name}")
    return total_rows


def _coalesce_strings(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    primary_norm = _normalize_string_id(primary)
    fallback_norm = _normalize_string_id(fallback)
    return primary_norm.fillna(fallback_norm)


def _normalize_string_id(series: pd.Series, width: int | None = None) -> pd.Series:
    normalized = series.astype(str).str.strip()
    normalized = normalized.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "<NA>": np.nan})
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    if width is not None:
        normalized = normalized.where(normalized.isna(), normalized.str.zfill(width))
    return normalized
