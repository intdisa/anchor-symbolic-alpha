# Route B Data Checklist

Route B targets a U.S. equities cross-sectional discovery setup. The recommended
base stack is:

- WRDS `CRSP` daily stock data
- WRDS `CRSP/Compustat Merged` link table
- WRDS `Compustat North America` quarterly and annual fundamentals
- Public `Ken French` daily factors
- Public `FRED` macro/risk context

## Required WRDS exports

The project now ships `configs/route_b_data.yaml` and `scripts/export_wrds_route_b.py`.
The minimum WRDS extracts are:

- `crsp_daily`
- `crsp_names`
- `ccm_link`
- `compustat_quarterly`
- `compustat_annual`

Default output layout:

```text
data/raw/route_b/
├── wrds/
│   ├── crsp_daily.parquet
│   ├── crsp_names.parquet
│   ├── ccm_link.parquet
│   ├── compustat_quarterly.parquet
│   └── compustat_annual.parquet
├── public/
│   ├── fama_french_daily.parquet
│   └── fred_macro_daily.parquet
└── wrds_export_manifest.json
```

## Export command

The WRDS Python client officially targets Python 3.8-3.12. If your main project
environment is Python 3.13, create a separate 3.12 virtualenv for export.

```bash
python3.12 -m venv .venv-wrds
. .venv-wrds/bin/activate
pip install wrds pandas pyarrow pyyaml
python scripts/export_wrds_route_b.py --dry-run
python scripts/export_wrds_route_b.py
```

The dry run prints the SQL it will execute so you can adjust table names if your
WRDS account exposes the CIZ 2.0 layout instead of the legacy CRSP tables.

## Public data to add after WRDS export

- Ken French daily factors:
  - `mktrf`, `smb`, `hml`, `rmw`, `cma`, `mom`, `rf`
- FRED daily macro context:
  - `VIXCLS`, `DGS2`, `DGS10`, `DFII10`, `DTWEXBGS`

These are listed in `configs/route_b_data.yaml` and are intentionally kept
outside the WRDS export step.

## Point-in-time notes

- Restrict CRSP to common stocks: `shrcd in (10, 11)`
- Restrict to primary U.S. exchanges: `exchcd in (1, 2, 3)`
- Use `CCM` link validity windows during CRSP/Compustat merges
- Preserve `dlret` or an equivalent delisting return field
- Do not replace the historical universe with a present-day ticker list
