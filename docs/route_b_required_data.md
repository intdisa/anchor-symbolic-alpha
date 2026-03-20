# Route B 所需数据清单

## 目标

本项目的 Route B 主任务是：
- 美国股票横截面日频因子发现
- 时间范围建议：`2000-01-01` 到最新
- 研究级主数据方案：`CRSP + CCM + Compustat + 公共因子/宏观`

该文档用于：
- 直接发给帮助导数的人
- 明确哪些数据已经准备好，哪些数据还缺
- 统一字段、过滤条件、交付格式，避免重复返工

---

## 当前状态

### 已准备好
- 公共 Fama/French 日频因子：
  - `/Users/xieshangchen/Documents/New project/data/raw/route_b/public/fama_french_daily.csv.gz`
- 公共 FRED 日频宏观：
  - `/Users/xieshangchen/Documents/New project/data/raw/route_b/public/fred_macro_daily.csv.gz`
- WRDS 权限已确认可访问：
  - `comp.fundq`
  - `comp.funda`

### 当前缺失权限的数据
- `crsp.dsf`
  - 报错：`permission denied for schema crsp_a_stock`
- `crsp.dsenames`
  - 报错：`permission denied for schema crsp_a_stock`
- `crsp.ccmxpf_linktable`
  - 报错：`permission denied for schema crsp_a_ccm`

结论：
- 满血版 Route B 仍缺 `CRSP stock/security` 和 `CCM`
- `Compustat` 权限本身是有的

---

## 必须给我的 WRDS 数据

### 1. CRSP 日频股票主表

- 推荐表：`crsp.dsf`
- 时间范围：`2000-01-01` 到最新
- 需要字段：
  - `permno`
  - `permco`
  - `date`
  - `ret`
  - `retx`
  - `prc`
  - `vol`
  - `shrout`
  - `bidlo`
  - `askhi`
  - `cfacpr`
  - `cfacshr`
  - `exchcd`
  - `shrcd`
- 过滤条件：
  - `shrcd in (10, 11)`
  - `exchcd in (1, 2, 3)`
- 用途：
  - 构建股票日频收益、价格、成交量、市值、流动性、波动率等横截面特征

### 2. CRSP 股票名称/历史标识表

- 推荐表：`crsp.dsenames`
- 需要字段：
  - `permno`
  - `ticker`
  - `ncusip`
  - `comnam`
  - `namedt`
  - `nameendt`
  - `exchcd`
  - `shrcd`
  - `siccd`
- 用途：
  - 历史 ticker/cusip 映射
  - 证券身份识别
  - 行业与结果解释

### 3. CRSP-Compustat 链接表

- 推荐表：`crsp.ccmxpf_linktable`
- 需要字段：
  - `gvkey`
  - `lpermno`
  - `linkdt`
  - `linkenddt`
  - `linktype`
  - `linkprim`
- 过滤条件：
  - `linktype in ('LC', 'LU', 'LS')`
  - `linkprim in ('P', 'C')`
- 用途：
  - 将 CRSP 市场数据与 Compustat 基本面做点时点映射

### 4. Compustat Quarterly Fundamentals

- 推荐表：`comp.fundq`
- 时间范围：`2000-01-01` 到最新
- 需要字段：
  - `gvkey`
  - `datadate`
  - `rdq`
  - `fyearq`
  - `fqtr`
  - `atq`
  - `ltq`
  - `ceqq`
  - `seqq`
  - `saleq`
  - `niq`
  - `oiadpq`
  - `cheq`
  - `dlcq`
  - `dlttq`
  - `actq`
  - `lctq`
  - `rectq`
  - `invtq`
  - `cogsq`
  - `xsgaq`
- 过滤条件：
  - `indfmt = 'INDL'`
  - `datafmt = 'STD'`
  - `popsrc = 'D'`
  - `consol = 'C'`
- 说明：
  - 请明确导出 `seqq`
  - 不要只给 `seq`
- 用途：
  - 点时点季度基本面特征

### 5. Compustat Annual Fundamentals

- 推荐表：`comp.funda`
- 时间范围：`2000-01-01` 到最新
- 需要字段：
  - `gvkey`
  - `datadate`
  - `fyear`
  - `at`
  - `lt`
  - `ceq`
  - `seq`
  - `sale`
  - `ni`
  - `oiadp`
  - `capx`
  - `txditc`
  - `pstkrv`
  - `pstkl`
  - `pstk`
- 过滤条件：
  - `indfmt = 'INDL'`
  - `datafmt = 'STD'`
  - `popsrc = 'D'`
  - `consol = 'C'`
- 用途：
  - 年度价值、盈利、投资、资本结构等慢变量特征

---

## 强烈建议额外给我的 WRDS 数据

### 6. CRSP Delisting 数据

如果主表里没有完整退市收益，请单独导出。

- 建议字段：
  - `permno`
  - `dlstdt`
  - `dlret`
  - `dlstcd`
- 用途：
  - 避免 survivorship bias
  - 保证研究级回测可信度

### 7. Compustat Company / Industry 元数据

- 推荐表：`comp.company`
- 建议字段：
  - `gvkey`
  - `conm`
  - `sic`
  - `gsector`
  - `ggroup`
  - `gind`
  - `gsubind`
- 用途：
  - 行业中性化
  - 横截面 context / family 设计

---

## 交付格式要求

请直接给“原始导出结果”，不要预处理成因子。

### 推荐格式
- 首选：`csv.gz`
- 次选：`parquet`

### 要求
- 编码：UTF-8
- 日期字段保持原始日期格式
- 不要删缺失值
- 不要前向填充
- 不要自己先做 train/valid/test 切分
- 不要自己先算特征
- 不要手工复权二次处理

### 推荐文件名
- `crsp_daily.csv.gz`
- `crsp_names.csv.gz`
- `ccm_link.csv.gz`
- `compustat_quarterly.csv.gz`
- `compustat_annual.csv.gz`
- `crsp_delist.csv.gz`（如有）
- `compustat_company.csv.gz`（如有）

---

## 建议导出 SQL 口径

### CRSP 日频股票主表
```sql
select
    permno,
    permco,
    date,
    ret,
    retx,
    prc,
    vol,
    shrout,
    bidlo,
    askhi,
    cfacpr,
    cfacshr,
    exchcd,
    shrcd
from crsp.dsf
where shrcd in (10, 11)
  and exchcd in (1, 2, 3)
  and date >= '2000-01-01'
order by permno, date;
```

### CRSP 名称历史表
```sql
select
    permno,
    ticker,
    ncusip,
    comnam,
    namedt,
    nameendt,
    exchcd,
    shrcd,
    siccd
from crsp.dsenames
where shrcd in (10, 11)
  and exchcd in (1, 2, 3)
order by permno, namedt;
```

### CCM 链接表
```sql
select
    gvkey,
    lpermno,
    linkdt,
    linkenddt,
    linktype,
    linkprim
from crsp.ccmxpf_linktable
where linktype in ('LC', 'LU', 'LS')
  and linkprim in ('P', 'C')
order by gvkey, lpermno, linkdt;
```

### Compustat Quarterly Fundamentals
```sql
select
    gvkey,
    datadate,
    rdq,
    fyearq,
    fqtr,
    atq,
    ltq,
    ceqq,
    seqq,
    saleq,
    niq,
    oiadpq,
    cheq,
    dlcq,
    dlttq,
    actq,
    lctq,
    rectq,
    invtq,
    cogsq,
    xsgaq
from comp.fundq
where indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
  and datadate >= '2000-01-01'
order by gvkey, datadate;
```

### Compustat Annual Fundamentals
```sql
select
    gvkey,
    datadate,
    fyear,
    at,
    lt,
    ceq,
    seq,
    sale,
    ni,
    oiadp,
    capx,
    txditc,
    pstkrv,
    pstkl,
    pstk
from comp.funda
where indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
  and datadate >= '2000-01-01'
order by gvkey, datadate;
```

---

## 可直接转发给帮忙导数的人

请帮我导出 WRDS 原始研究数据，时间范围 `2000-01-01` 到最新，交付格式优先 `csv.gz`：

1. `crsp.dsf`
   - 字段：`permno, permco, date, ret, retx, prc, vol, shrout, bidlo, askhi, cfacpr, cfacshr, exchcd, shrcd`
   - 过滤：`shrcd in (10,11)`，`exchcd in (1,2,3)`

2. `crsp.dsenames`
   - 字段：`permno, ticker, ncusip, comnam, namedt, nameendt, exchcd, shrcd, siccd`

3. `crsp.ccmxpf_linktable`
   - 字段：`gvkey, lpermno, linkdt, linkenddt, linktype, linkprim`
   - 过滤：`linktype in ('LC','LU','LS') and linkprim in ('P','C')`

4. `comp.fundq`
   - 字段：`gvkey, datadate, rdq, fyearq, fqtr, atq, ltq, ceqq, seqq, saleq, niq, oiadpq, cheq, dlcq, dlttq, actq, lctq, rectq, invtq, cogsq, xsgaq`
   - 过滤：`indfmt='INDL', datafmt='STD', popsrc='D', consol='C'`

5. `comp.funda`
   - 字段：`gvkey, datadate, fyear, at, lt, ceq, seq, sale, ni, oiadp, capx, txditc, pstkrv, pstkl, pstk`
   - 过滤：`indfmt='INDL', datafmt='STD', popsrc='D', consol='C'`

6. 最好再加：
   - `CRSP delisting`：`permno, dlstdt, dlret, dlstcd`
   - `comp.company`：`gvkey, conm, sic, gsector, ggroup, gind, gsubind`

注意：
- 不要预处理
- 不要删缺失值
- 不要自己先算因子
- 不要自己先做切分
- 直接给原始导出结果即可
