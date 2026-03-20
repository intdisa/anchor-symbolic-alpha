# Route B Access Status

## Confirmed accessible
- `comp.fundq`
- `comp.funda`
- public benchmark/macro sources prepared via `scripts/fetch_route_b_public.py`

## Confirmed missing permissions
- `crsp.dsf` -> missing access to schema `crsp_a_stock`
- `crsp.dsenames` -> missing access to schema `crsp_a_stock`
- `crsp.ccmxpf_linktable` -> missing access to schema `crsp_a_ccm`

## Consequence
The research-grade Route B pipeline based on `CRSP + CCM + Compustat` is blocked until WRDS enables:
- CRSP daily stock/security data
- CRSP/Compustat Merged (CCM)

## Next local checks
Run `scripts/check_wrds_access.py` again to verify whether the Compustat-only fallback tables below are available:
- `comp.secd`
- `comp.security`
- `comp.company`
- `comp.sec_history`
- `comp.sec_idhist`
