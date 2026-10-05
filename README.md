# QuantLab

QuantLab is a small quantitative research framework for testing systematic portfolio ideas with explicit assumptions around execution, transaction costs, risk and reproducibility.

The current experiment, **H001**, studies whether a time-series momentum filter improves the behaviour of an inverse-volatility ETF portfolio.

## H001 — Time-Series Momentum Risk Gate

H001 uses a diversified ETF universe:

`SPY`, `IWM`, `EFA`, `EEM`, `IEF`, `LQD`, `GLD`, `DBC`

The baseline portfolio is fully invested using inverse-volatility weights.

H001 applies an additional 252-session momentum filter:

- positive excess momentum → keep the ETF allocation;
- non-positive excess momentum → set the ETF weight to zero;
- removed allocation moves to cash;
- surviving weights are not renormalized.

Volatility is estimated using a 60-session rolling window.

The portfolio is rebalanced monthly.

Signals calculated at `close(t)` are executed at the close of the next observed session, so the backtest does not capture the return immediately following information used to form the signal.

## Results

Primary results use **10 bps transaction costs per unit of gross ETF notional traded**.

| Strategy | CAGR | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Baseline | 5.20% | 8.74% | 0.459 | -25.96% |
| ExposureMatched | 4.42% | 5.47% | 0.554 | -13.89% |
| H001 | 4.89% | 5.40% | 0.645 | -11.01% |

The fully invested baseline produces the highest absolute CAGR.

H001 instead shows a substantially lower volatility and drawdown profile, together with a higher Sharpe ratio.

### Exposure-matched control

Because H001 frequently holds cash, I also compare it with an `ExposureMatched` portfolio.

This control keeps the baseline inverse-volatility composition but scales total ETF exposure to match H001 at every rebalance.

Average target ETF exposure for H001 is approximately **65.2%**.

At 10 bps, H001 still improves on the matched control:

- CAGR: **4.89% vs 4.42%**
- Sharpe: **0.645 vs 0.554**
- Max drawdown: **-11.01% vs -13.89%**

This suggests that the result is not explained only by holding more cash.

It is an attribution control, not an independent trading strategy.

## Robustness

Transaction costs were evaluated at:

`0`, `5`, `10`, and `20` bps.

The main risk-adjusted result remains present across those scenarios.

The strategy was also evaluated over four predefined subperiods. H001 outperformed the exposure-matched control in CAGR and Sharpe in three of the four periods.

The main exception was **2018–2022**, where the matched portfolio produced higher CAGR and Sharpe, although H001 still had the smaller maximum drawdown.

A local parameter grid was also evaluated:

- volatility windows: `40`, `60`, `90`
- momentum lookbacks: `189`, `252`, `315`

Across all 9 combinations, H001 had higher CAGR, higher Sharpe and a better maximum drawdown than its corresponding exposure-matched control.

The original `60 / 252` specification remains unchanged. The grid is used as a robustness check rather than to select the best historical parameters.

## Figures

### Wealth

![Portfolio wealth](figures/h001/wealth_10bps.png)

### Drawdown

![Drawdowns](figures/h001/drawdowns_10bps.png)

## Project structure

```text
src/quantlab/
├── data/
├── evaluation/
├── features/
├── portfolio/
├── signals/
└── simulation/

tests/
├── integration/
└── unit/

scripts/
├── download_h001_data.py
├── export_h001_results.py
├── run_h001_backtest.py
└── run_h001_sensitivity.py

results/h001/
figures/h001/
```

The research logic is separated from the simulation and evaluation infrastructure so future experiments can reuse the same components.

## Tests

The test suite covers:

- returns and volatility;
- momentum and cash-return calculations;
- inverse-volatility allocation;
- exposure matching;
- rebalance timing;
- portfolio accounting;
- weight drift;
- transaction costs;
- performance metrics;
- an end-to-end synthetic H001 case.

```bash
pytest -v
```

Current status:

```text
29 passed
```

Tests also run automatically through GitHub Actions.

## Reproducing the experiment

Install the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Raw market-data snapshots are intentionally excluded from the repository.

With the required local Tiingo and FRED data available:

```bash
python scripts/run_h001_backtest.py
python scripts/run_h001_sensitivity.py
python scripts/export_h001_results.py
```

Generated tables are stored in `results/h001/`.

## Limitations

This is a historical research experiment, not evidence of future profitability.

Important limitations include:

- retrospective ETF universe selection;
- no pristine untouched out-of-sample dataset;
- simplified cash-return modelling;
- constant transaction-cost assumptions;
- no explicit market-impact model;
- dependence on adjusted-price data from the provider.

The exposure-matched portfolio is used for attribution and depends on H001's exposure path.

## Status

**H001 complete.**