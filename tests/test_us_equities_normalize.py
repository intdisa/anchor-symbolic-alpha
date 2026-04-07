from __future__ import annotations

from pathlib import Path

import pandas as pd

from knowledge_guided_symbolic_alpha.dataio.us_equities_normalize import normalize_xsc_us_equities


def test_normalize_xsc_us_equities_creates_expected_raw_files(tmp_path: Path) -> None:
    source_root = tmp_path / "xsc"
    source_root.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "PERMNO": 10001,
                "PERMCO": 90001,
                "PrimaryExch": "Q",
                "Ticker": "AAA",
                "TradingSymbol": "AAA",
                "CUSIP": "12345678",
                "HdrCUSIP": "12345678",
                "SecurityNm": "Alpha Inc",
                "IssuerNm": "Alpha Inc",
                "SICCD": 3571,
                "USIncFlg": "Y",
                "SecurityType": "EQTY",
                "SecuritySubType": "COM",
                "ShareType": "NS",
                "DlyCalDt": "2020-01-02",
                "DlyRet": 0.01,
                "DlyRetx": 0.01,
                "DlyDelRet": 0.0,
                "DlyPrc": 20.0,
                "DlyVol": 100000,
                "DlyBid": 19.9,
                "DlyAsk": 20.1,
                "DlyLow": 19.5,
                "DlyHigh": 20.5,
                "DlyFacPrc": 1.0,
                "DisFacPr": 1.0,
                "DisFacShr": 1.0,
                "ShrOut": 1000,
            },
            {
                "PERMNO": 10001,
                "PERMCO": 90001,
                "PrimaryExch": "Q",
                "Ticker": "AAA",
                "TradingSymbol": "AAA",
                "CUSIP": "12345678",
                "HdrCUSIP": "12345678",
                "SecurityNm": "Alpha Inc",
                "IssuerNm": "Alpha Inc",
                "SICCD": 3571,
                "USIncFlg": "Y",
                "SecurityType": "EQTY",
                "SecuritySubType": "COM",
                "ShareType": "NS",
                "DlyCalDt": "2020-01-03",
                "DlyRet": -0.02,
                "DlyRetx": -0.02,
                "DlyDelRet": -0.50,
                "DlyPrc": 19.6,
                "DlyVol": 120000,
                "DlyBid": 19.5,
                "DlyAsk": 19.7,
                "DlyLow": 19.2,
                "DlyHigh": 20.0,
                "DlyFacPrc": 1.0,
                "DisFacPr": 1.0,
                "DisFacShr": 1.0,
                "ShrOut": 1000,
            },
            {
                "PERMNO": 10002,
                "PERMCO": 90002,
                "PrimaryExch": "Q",
                "Ticker": "ETF1",
                "TradingSymbol": "ETF1",
                "CUSIP": "87654321",
                "HdrCUSIP": "87654321",
                "SecurityNm": "Filtered ETF",
                "IssuerNm": "Filtered ETF",
                "SICCD": 6726,
                "USIncFlg": "Y",
                "SecurityType": "FUND",
                "SecuritySubType": "ETF",
                "ShareType": "NS",
                "DlyCalDt": "2020-01-02",
                "DlyRet": 0.03,
                "DlyRetx": 0.03,
                "DlyDelRet": 0.0,
                "DlyPrc": 30.0,
                "DlyVol": 150000,
                "DlyBid": 29.9,
                "DlyAsk": 30.1,
                "DlyLow": 29.5,
                "DlyHigh": 30.5,
                "DlyFacPrc": 1.0,
                "DisFacPr": 1.0,
                "DisFacShr": 1.0,
                "ShrOut": 2000,
            },
        ]
    ).to_csv(source_root / "crsp0025.csv", index=False)

    pd.DataFrame(
        [
            {
                "PERMNO": 10001,
                "DelistingDt": "2020-01-03",
                "DelRet": -0.5,
                "DelStatusType": "500",
            }
        ]
    ).to_csv(source_root / "crsp_delist.csv", index=False)

    pd.DataFrame(
        [
            {
                "gvkey": "001000",
                "LPERMNO": 10001,
                "LINKDT": pd.Timestamp("2010-01-01"),
                "LINKENDDT": pd.NaT,
                "LINKTYPE": "LC",
                "LINKPRIM": "P",
            }
        ]
    ).to_stata(source_root / "ccm_link_2025.dta", write_index=False)

    pd.DataFrame(
        [
            {
                "gvkey": "001000",
                "datadate": "2019-12-31",
                "rdq": "2020-02-15",
                "fyearq": 2019,
                "fqtr": 4,
                "atq": 500.0,
                "ltq": 200.0,
                "ceqq": 300.0,
                "seqq": 300.0,
                "saleq": 150.0,
                "niq": 15.0,
                "oiadpq": 20.0,
                "cheq": 60.0,
                "dlcq": 20.0,
                "dlttq": 40.0,
                "actq": 80.0,
                "lctq": 50.0,
                "rectq": 25.0,
                "invtq": 20.0,
                "cogsq": 70.0,
                "xsgaq": 15.0,
            }
        ]
    ).to_csv(source_root / "Comp_Quarterly6126.csv", index=False)

    pd.DataFrame(
        [
            {
                "gvkey": "001000",
                "datadate": "2019-12-31",
                "fyear": 2019,
                "at": 480.0,
                "lt": 190.0,
                "ceq": 290.0,
                "seq": 290.0,
                "sale": 580.0,
                "ni": 48.0,
                "oiadp": 55.0,
                "capx": 20.0,
                "txditc": 5.0,
                "pstkrv": 0.0,
                "pstkl": 0.0,
                "pstk": 0.0,
            }
        ]
    ).to_csv(source_root / "fwn4guf3dvyp5rls.csv", index=False)

    output_root = tmp_path / "us_equities" / "wrds"
    summary = normalize_xsc_us_equities(source_root=source_root, output_root=output_root, chunksize=2, datasets=("crsp", "delisting", "ccm", "quarterly", "annual"))

    assert summary.crsp_daily_rows == 2
    assert summary.crsp_delisting_rows == 1
    assert summary.crsp_names_rows == 1
    assert summary.ccm_rows == 1
    assert summary.quarterly_rows == 1
    assert summary.annual_rows == 1

    crsp_daily = pd.read_csv(output_root / "crsp_daily.csv.gz")
    crsp_names = pd.read_csv(output_root / "crsp_names.csv.gz")
    crsp_delisting = pd.read_csv(output_root / "crsp_delisting.csv.gz")
    ccm = pd.read_csv(output_root / "ccm_link.csv.gz")
    quarterly = pd.read_csv(output_root / "compustat_quarterly.csv.gz")

    assert list(crsp_daily.columns) == [
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
    assert crsp_daily["permno"].tolist() == [10001, 10001]
    assert crsp_daily["exchcd"].tolist() == [3, 3]
    assert crsp_daily["shrcd"].tolist() == [10, 10]
    assert crsp_daily["dlret"].tolist() == [0.0, -0.5]
    assert crsp_delisting["permno"].tolist() == [10001]
    assert crsp_delisting["dlret"].tolist() == [-0.5]
    assert crsp_names.loc[0, "ticker"] == "AAA"
    assert str(ccm.loc[0, "gvkey"]).zfill(6) == "001000"
    assert "seq" in quarterly.columns
