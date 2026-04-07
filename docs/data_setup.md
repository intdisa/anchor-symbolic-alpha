# Data Setup

The canonical mainline uses the `us_equities` layout only.

## Raw data roots

```text
data/raw/us_equities/
├── wrds/
│   ├── crsp_daily.csv.gz
│   ├── crsp_delisting.csv.gz
│   ├── crsp_names.csv.gz
│   ├── ccm_link.csv.gz
│   ├── compustat_quarterly.csv.gz
│   └── compustat_annual.csv.gz
├── public/
│   ├── fama_french_daily.csv.gz
│   └── fred_macro_daily.csv.gz
└── wrds_export_manifest.json
```

Processed outputs:

- panel: `data/processed/us_equities/us_equities_panel.parquet`
- splits: `data/processed/us_equities/splits/`
- subsets: `data/processed/us_equities/subsets/<subset_name>/`

## Dependency tiers

- WRDS export: `pip install -e ".[wrds]"`
- panel/subset builders: `pip install -e ".[eval]"`
- training: `pip install -e ".[train]"`

Validated train env:

```bash
python3.11 -m venv .venv-train
. .venv-train/bin/activate
pip install --no-build-isolation -e ".[train,dev]"
```

Use preflight before each tier:

```bash
.venv-wrds/bin/python scripts/run_preflight.py --profile wrds
.venv/bin/python scripts/run_preflight.py --profile eval
.venv-train/bin/python scripts/run_preflight.py --profile train
```

## WRDS export

```bash
. .venv-wrds/bin/activate
python scripts/export_wrds_us_equities.py --dry-run
python scripts/export_wrds_us_equities.py
```

Minimum WRDS datasets:

- `crsp_daily`
- `crsp_delisting`
- `crsp_names`
- `ccm_link`
- `compustat_quarterly`
- `compustat_annual`

## Public data

```bash
. .venv/bin/activate
python scripts/fetch_public_market_data.py
```

This writes:

- `data/raw/us_equities/public/fama_french_daily.csv.gz`
- `data/raw/us_equities/public/fred_macro_daily.csv.gz`

## Panel and subset build

```bash
. .venv/bin/activate
python scripts/build_us_equities_panel.py
python scripts/build_us_equities_subset.py --name liquid500_2010_2025 --max-permnos 500
python scripts/export_us_equities_subset_csv.py --subset-root data/processed/us_equities/subsets/liquid500_2010_2025
```

## Point-in-time rules

- CRSP common stocks only: `shrcd in (10, 11)`
- primary U.S. exchanges only: `exchcd in (1, 2, 3)`
- CCM link validity windows must be applied
- delisting returns must be preserved
- do not rebuild history from a present-day ticker list
