# futures-tca

Post-trade transaction cost analysis for large-tick futures on synthetic
order-book data where the queue state, the true fair value and the true dealer
skew in the calendar quote are recorded, so every cost estimate can be set next
to the truth.

## Status

The simulator, cost decomposition, policies, roll model and report pipeline are
implemented, including the calibration checked by tests 4 and 5.
Charts (a), (b), (c), (e) and `reports/summary.csv` are generated from seed
20261101. There is no chart (d): a futures-versus-cash chart and its module were
dropped before the first release, and the other charts kept their letters.
Out of scope for v1: sourcing roll parameters from the literature (every roll
parameter stays a labelled assumption, see the table below, because v1 is
strictly synthetic and allows no figures typed in from papers) and the passive
roll (v1 is limited to the quote-based aggressive roll).

## Thesis

For a large-tick contract the spread is one tick almost all the time, so price
is a discrete variable and the first decision is "join the queue or cross". The
outcome of joining is a lottery whose odds depend on queue-ahead and on how the
book leans when the order is posted.

## Charts

| | File | Question |
|---|---|---|
| (a) | `a_queue_lottery.png` | Expected cost in ticks vs queue-ahead at posting; fill probability and time |
| (b) | `b_markouts.png` | Mean signed markout at horizons 1s–300s after passive vs aggressive fill |
| (c) | `c_roll_cost.png` | Roll cost by method vs day relative to expiry |
| (e) | `e_tick_regimes.png` | Share of cost variance explained across large/medium/small tick regimes |

## Key findings (every number is in `reports/summary.csv`)

Mean over 50 one-lot buy orders per regime (`build_summary`), cost in ticks,
positive = paid. Passive is the share of orders filled passively; the rest
crossed at the 30 s timeout.

| Regime | Policy | Total | Spread | Selection | Fees | Passive |
|---|---|---|---|---|---|---|
| large | AGG | 0.56 | 0.50 | 0.00 | 0.06 | 0.00 |
| large | PASS_30 | 0.20 | 0.03 | 0.11 | 0.06 | 0.48 |
| medium | AGG | 0.82 | 0.64 | 0.00 | 0.18 | 0.00 |
| medium | PASS_30 | -0.02 | -0.36 | 0.16 | 0.18 | 0.82 |
| small | AGG | 4.25 | 3.65 | 0.00 | 0.60 | 0.00 |
| small | PASS_30 | -2.41 | -3.13 | 0.12 | 0.60 | 0.94 |

- **Chart (a), large tick:** an order at the front of the queue costs -0.42 ticks
  and fills with probability 0.99. An order with `a` lots ahead exists only when
  at least `a` lots are queued, so each point averages the paths with that many
  lots at posting, and AGG is measured on the same paths. PASS_AT(30 s) costs as
  much as AGG from 360 lots ahead, 1.2 x `q_bar` (+0.02 ticks, 95% CI -0.02 to +0.05).
  It is more expensive with the CI above zero from 420 lots (+0.07, CI +0.03 to
  +0.11) and reaches +0.14 at 600 lots (316 paths, fill probability 0.06). Queue
  size and direction are tied together here: in the engine a long bid queue raises
  the rate of buy market orders and lowers the rate of sell market orders, so the
  price tends to rise while the order waits and the unfilled order crosses at a
  higher price.
- **Chart (b), large tick:** a passive fill with no queue left behind it has a
  markout of +0.50 ticks at 1 s (CI +0.43 to +0.58, 184 fills) and it stays at
  +0.55 at 300 s, because the fill coincides with the level emptying and the mid
  moving against it. A fill with queue behind it has +0.05 at 300 s (1637 fills)
  and an aggressive fill -0.03; neither differs from zero at 300 s. Markouts are
  measured from the mid just before the filling event. Each episode contributes
  one order, so the CIs in the chart treat the samples as independent.
- **Chart (e):** R² of per-order PASS(30 s) cost on the queue size at posting is
  0.26 / 0.40 in the large regime, 0.07 / 0.08 in the medium regime, and 0.01 / 0.05
  in the small regime (queue size / spread plus imbalance). In the large regime the
  lean of the book explains more than queue size alone. In the medium and small
  regimes neither explains much of the variance.
- **Chart (c):** THRESH(k = 0.25) saves 0.21 ticks over FIXED_DAY(-5) with a
  perfect fair-value estimate and 0.12 at sd 0.25. At sd 0.5 the saving is about
  zero. At sd 1.0 it is -0.07 with a 95% CI of -0.16 to +0.02, so it is not
  distinguishable from zero (n = 200 episodes per point).
- **Fee:** 0.064 ticks per lot in the large regime (1.0 currency / 15.6 tick
  value), rescaled to each regime's tick.

## Roll assumptions

Roll cost is half the quote width plus the dealer skew `e` between the market
calendar mid and fair value. The richness `z` moves fair value and market
together, so it cancels and never changes a roll cost in v1; THRESH exploits
`e`. All values below are assumptions, not estimates.

| Parameter | Value | Unit |
|---|---|---|
| carry | 2 | ticks |
| richness half-life | 4 | hours |
| richness sd | 0.4 | ticks |
| skew half-life | 20 | minutes |
| skew sd | 0.25 | ticks |
| migration midpoint | -5 | days to expiry |
| migration width | 2 | days |
| forced roll day | -2 | days to expiry |

The quote-width tables are in `roll.py`. THRESH uses k = 0.25 and the error grid
sd 0, 0.25, 0.5, 1.0 from `report.py`.

## Data statement

Strictly synthetic. No exchange, vendor or employer data. Every parameter is a
stated assumption. Model structure follows the papers in docs/citations.md; no
numeric parameter comes from them.

## Development

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
make lint
make test
make report   # rebuilds reports/ from seed 20261101
```

## Limitations

- v1: aggressive roll only (no passive calendar queue)
- Own-footprint not modelled (valid for n << q_bar)
- No real-data validation
- The engine tracks only the best level on each side. While a better level is in
  front of a resting order, events there do not touch the order, and lots of a
  market order beyond a cleared level are dropped, as the engine drops them. When
  the touch returns to the order's level the engine draws a fresh queue, and the
  order keeps its position up to that queue's size. Sending the order to the back
  instead, or letting the dropped lots reach it, changes the medium-regime PASS_30
  total, so read that row as "about zero". The large-regime conclusions do not
  change.
- A resting order that leads the engine's touch may be quoted through by the far
  side. The replay leaves it resting until the next market order or the timeout.
- Chart (a) uses the controlled `PASS_AT(a, T)` experiment. `a` cannot exceed the
  queue at posting, so points further right rest on fewer paths and on books that
  lean further towards the bid.
- The summary table uses 50 orders per regime, so its PASS_30 means are noisy.
- `Regime.add_scale` is a calibration factor (`scripts/calibrate_add_scale.py`);
  regime spread and volatility targets are checked by tests 4 and 5.

## License

MIT
