# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assignment

---

## Setup & How to Run

**Requirements**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
```

**Data files** — place these in the same directory as the notebook:
- `fear_greed_index.csv`
- `historical_data_part1.csv`
- `historical_data_part2.csv`

**Run**
```bash
jupyter notebook trader_analysis.ipynb
```
Execute cells top-to-bottom. All charts render inline.

---

## Methodology

### Data Preparation
- Combined two trade files → **211,224 trades** with zero nulls and zero duplicates
- Parsed `Timestamp IST` (`DD-MM-YYYY HH:MM`) to daily UTC dates
- Mapped 5-class sentiment to 3 classes: **Fear** (F/G < 40), **Neutral** (40–60), **Greed** (> 60)
- Inner-joined on `date` → **211,218 matched trades** across **731 trading days**

### Feature Engineering (Daily per Trader)
| Feature | Description |
|---|---|
| `total_pnl` | Sum of Closed PnL for the day |
| `net_pnl` | PnL minus trading fees |
| `num_trades` | Trade count per day |
| `win_rate` | Wins / closing trades |
| `long_short_ratio` | Directional bias |
| `avg_size_usd` | Median USD position size (leverage proxy) |
| `drawdown` | Cumulative PnL − rolling max (drawdown proxy) |
| `lev_proxy` | Normalized position size relative to trader's own median |

### Statistical Validation
Every key claim is backed by a hypothesis test:
- **Welch t-test + Mann–Whitney U** on PnL: Fear vs Greed (p = 0.689 — not significant)
- **Mann–Whitney U** on trade count: Fear vs Greed (p < 0.001 — significant)
- **Mann–Whitney U** on avg PnL: Frequent vs Infrequent traders

---

## Key Insights

1. **PnL Paradox** — Fear days show higher *mean* PnL ($5,185 vs $4,144) driven by outlier wins. Greed days are more *consistently* profitable (64.3% vs 60.4%). The difference is not statistically significant — trader skill dominates over sentiment.

2. **Fear Triggers Hyperactivity** — Traders make 37% more trades on Fear days (105 vs 77 avg/day, p < 0.001) and shift toward short positions. Overtrading increases fee drag and reflects reactive decision-making.

3. **Quality Over Quantity** — Infrequent traders achieve higher Sharpe ratios and smaller drawdowns. High-leverage traders post larger absolute PnL but absorb significantly worse drawdowns.

4. **Drawdown is Sentiment-Driven** — Tail risk is measurably worse on Fear days. Risk controls should tighten during Fear periods.

5. **Sentiment is a Real Signal** — Fear/Greed index value ranks in the top predictive features for profitability (XGBoost AUC = 0.836).

---

## Strategy Recommendations

### Strategy 1 — Fear-Day Discipline Protocol
> *When F/G < 40, cap daily trade count at 1.5× your 7-day average and limit long exposure to ≤ 40%.*

**Rationale:** Fear days drive statistically significant overtrading (+37%). Constraining trade count reduces fee drag and emotional noise. Target: frequent traders (>80 trades/day).

### Strategy 2 — Greed-Day Position Size Cap
> *When F/G > 60, cap individual position size at 1.5× your 30-day baseline.*

**Rationale:** Traders naturally size up during Greed, creating overconfidence-driven overexposure. A hard size ceiling limits drawdown without sacrificing upside. Target: all traders, especially high-leverage segment.

**Rule of thumb:** During extreme sentiment (F/G < 20 or > 80), infrequent/low-leverage traders should halve position sizes or sit out — their Sharpe advantage is largest at extremes.

---

## Bonus — Predictive Model

| Model | Test AUC | CV AUC (5-fold) | Accuracy |
|---|---|---|---|
| **XGBoost** | **0.836** | **0.817 ± 0.012** | **79%** |
| Random Forest | 0.808 | — | 77% |

**Top predictive features:** rolling win rate, trade count, long/short ratio, Fear/Greed value, lagged PnL — confirming that sentiment is a meaningful predictor alongside behavioral history.
