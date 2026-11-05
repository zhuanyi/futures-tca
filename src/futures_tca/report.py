"""Report generation: charts (a), (b), (c), (e) and reports/summary.csv.

Builds all figures and the summary table from the fixed seed (20261101).
Entry point: `python -m futures_tca.report --out reports`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from futures_tca import REPORT_SEED
from futures_tca.book import generate_episode
from futures_tca.costs import add_perf_columns, compute_cost, compute_markout
from futures_tca.policies import replay_agg, replay_pass, replay_pass_at
from futures_tca.roll import (
    roll_cal_cross,
    roll_leg_simul,
    simulate_roll_episode,
    thresh_saving,
)
from futures_tca.spec import REGIMES, ContractSpec
from futures_tca.stats import ci_mean, variance_decomposition

# Fewest paths behind a chart (a) point
_CHART_A_MIN_PATHS = 30

# Strip software metadata from saved figures (privacy)
_SAVEFIG_KW = dict(metadata={"Software": None}, dpi=150)


def _ensure_out_dir(out: str) -> Path:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def chart_a_queue_lottery(
    contract: ContractSpec,
    regime_name: str = "large",
    n_episodes: int = 3000,
    seed: int = REPORT_SEED,
    out: str = "reports",
    timeout: float = 30.0,
) -> dict:
    """Chart (a): expected cost in ticks vs queue-ahead at posting.

    For a 1-lot buy: cost of PASS_AT(a, timeout) per queue-ahead value (total
    cost incl. the timeout cross, 95% CI). An order with ``a`` lots ahead exists
    only when at least ``a`` lots are queued, so each point averages the paths
    with ``Qb(t0) >= a`` and AGG is measured on the same paths. Values with fewer
    than ``_CHART_A_MIN_PATHS`` such paths are left out (ahead 0 always stays). Panels for fill
    probability and mean time to fill. The break-even is the smallest queue-ahead
    where PASS_AT costs more than AGG on average (None if it never does);
    ``break_even_ci_ahead`` is the smallest where the paired 95% CI is above zero.
    """
    out_dir = _ensure_out_dir(out)
    regime = REGIMES[regime_name]
    t0 = 120.0

    aheads = list(range(0, int(2 * regime.q_bar) + 1, max(1, int(regime.q_bar // 5))))
    logs = [
        generate_episode(regime, seed=seed + ep, warmup=t0, horizon=timeout + 60.0)
        for ep in range(n_episodes)
    ]
    agg_cost = np.array([
        compute_cost(replay_agg(log, +1, 1, t0), log, contract).total for log in logs
    ])
    queue_t0 = np.array([log.state_at(t0)[2] for log in logs])

    rows = []
    for a_init in aheads:
        paths = np.flatnonzero(queue_t0 >= a_init)
        if len(paths) < min(_CHART_A_MIN_PATHS, n_episodes):
            continue
        costs, filled, times = [], [], []
        for i in paths:
            log = logs[i]
            res = replay_pass_at(log, side=+1, qty=1, t0=t0, ahead_init=a_init, timeout=timeout)
            costs.append(compute_cost(res, log, contract).total)
            passive = [f for f in res.fills if f.fill_type == "passive"]
            filled.append(1.0 if passive else 0.0)
            times.append(passive[-1].t - t0 if passive else timeout)
        mean, lo, hi = ci_mean(np.array(costs))
        diff, diff_lo, diff_hi = ci_mean(np.array(costs) - agg_cost[paths])
        rows.append({
            "ahead": a_init, "n_paths": len(paths),
            "mean_cost": mean, "ci_lo": lo, "ci_hi": hi,
            "agg_cost": float(agg_cost[paths].mean()),
            "diff": diff, "diff_ci_lo": diff_lo, "diff_ci_hi": diff_hi,
            "fill_rate": float(np.mean(filled)), "mean_fill_time": float(np.mean(times)),
        })
    df = pd.DataFrame(rows)
    worse = df[df["diff"] > 0]
    worse_ci = df[df["diff_ci_lo"] > 0]
    break_even = int(worse["ahead"].iloc[0]) if len(worse) else None
    break_even_ci = int(worse_ci["ahead"].iloc[0]) if len(worse_ci) else None

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(df["ahead"], df["mean_cost"], "b-o", markersize=3, label=f"PASS_AT({timeout:g}s)")
    axes[0].fill_between(df["ahead"], df["ci_lo"], df["ci_hi"], alpha=0.25)
    axes[0].plot(df["ahead"], df["agg_cost"], "r--", label="AGG, same paths")
    axes[0].set_ylabel("Mean total cost (ticks)")
    axes[0].set_title("Chart (a): Queue lottery, paths with at least that many lots queued")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["ahead"], df["fill_rate"], "g-o", markersize=3)
    axes[1].set_ylabel("Fill probability")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["ahead"], df["mean_fill_time"], "r-o", markersize=3)
    axes[2].set_ylabel(f"Mean time to fill (s, timeout counted as {timeout:g} s)")
    axes[2].set_xlabel("Queue ahead (lots)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "a_queue_lottery.png", **_SAVEFIG_KW)
    plt.close(fig)

    return {
        "chart_a": df.to_dict("records"),
        "agg_cost": float(agg_cost.mean()),
        "break_even_ahead": break_even,
        "break_even_ci_ahead": break_even_ci,
    }


def chart_b_markouts(
    contract: ContractSpec,
    regime_name: str = "large",
    n_episodes: int = 2000,
    seed: int = REPORT_SEED,
    out: str = "reports",
) -> dict:
    """Chart (b): mean signed markout 1 s to 300 s after a fill, with 95% CI.

    One order per episode at t0, so the samples in a group are independent.
    Passive fills (PASS(120 s) joining the back at the touch) are split by
    whether queue remained behind us at the fill (benign) or we were last
    (toxic), and compared with aggressive fills.
    """
    out_dir = _ensure_out_dir(out)
    regime = REGIMES[regime_name]
    horizons = [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
    t0 = 120.0

    groups: dict[str, dict[float, list[float]]] = {
        "passive, queue behind": {h: [] for h in horizons},
        "passive, last in queue": {h: [] for h in horizons},
        "aggressive": {h: [] for h in horizons},
    }
    for ep in range(n_episodes):
        log = generate_episode(regime, seed=seed + 10000 + ep, warmup=t0, horizon=420.0)
        res = replay_pass(log, side=+1, qty=1, t0=t0, timeout=120.0, complete=False)
        for fill in res.fills:
            key = "passive, queue behind" if fill.queue_behind_at_fill > 0 else (
                "passive, last in queue")
            for h in horizons:
                groups[key][h].append(compute_markout(fill, +1, log, h))
        agg = replay_agg(log, side=+1, qty=1, t0=t0)
        for fill in agg.fills:
            for h in horizons:
                groups["aggressive"][h].append(compute_markout(fill, +1, log, h))

    fig, ax = plt.subplots(figsize=(8, 5))
    _nan = (float("nan"), float("nan"), float("nan"))
    stats = {
        label: [
            ci_mean(np.array(by_h[h])) if len(by_h[h]) else _nan
            for h in horizons
        ]
        for label, by_h in groups.items()
    }
    for label, rows in stats.items():
        means = [r[0] for r in rows]
        ax.plot(horizons, means, "-o", label=label, markersize=4)
        ax.fill_between(horizons, [r[1] for r in rows], [r[2] for r in rows], alpha=0.2)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Horizon (s)")
    ax.set_ylabel("Mean markout cost (ticks, + = against the fill)")
    ax.set_title("Chart (b): Post-fill markouts")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "b_markouts.png", **_SAVEFIG_KW)
    plt.close(fig)

    return {
        "chart_b": {label: [r[0] for r in rows] for label, rows in stats.items()},
        "chart_b_ci": {
            label: [(r[1], r[2]) for r in rows] for label, rows in stats.items()
        },
        "chart_b_n": {k: [len(v[h]) for h in horizons] for k, v in groups.items()},
        "horizons": horizons,
    }


def chart_c_roll_cost(
    n_episodes: int = 200,
    seed: int = REPORT_SEED,
    out: str = "reports",
    k: float = 0.25,  # assumption
) -> dict:
    """Chart (c): roll cost by method vs day, and THRESH saving vs fair-value error.

    Left: mean cost of CAL_CROSS and LEG_SIMUL by day (widths from the
    liquidity-migration tables). Right: saving of THRESH(k, sigma_eps) over
    FIXED_DAY(-5) with 95% CI against the desk's fair-value error ``sigma_eps``.
    """
    out_dir = _ensure_out_dir(out)
    day_grid = list(range(-10, 1))
    cost = {m: {d: [] for d in day_grid} for m in ("CAL_CROSS", "LEG_SIMUL")}
    for ep in range(n_episodes):
        episode = simulate_roll_episode(n_days=11, seed=seed + ep)
        for i, d_float in enumerate(episode["days"]):
            d = round(float(d_float))
            fair, mkt = episode["fair_spread"][i], episode["market_spread"][i]
            cost["CAL_CROSS"][d].append(
                roll_cal_cross(fair, mkt, int(episode["cal_width"][i]), side=1)
            )
            cost["LEG_SIMUL"][d].append(
                roll_leg_simul(
                    fair, mkt, int(episode["front_width"][i]), int(episode["next_width"][i]), 1
                )
            )

    sigmas = [0.0, 0.25, 0.5, 1.0]  # assumption: fair-value error grid
    savings = {}
    for sigma in sigmas:
        mean, lo, hi = ci_mean(thresh_saving(k, sigma, n_episodes, seed=seed, fixed_day=-5.0))
        savings[sigma] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for method, by_day in cost.items():
        ax.plot(day_grid, [np.mean(by_day[d]) for d in day_grid], "o-", label=method)
    ax.set_xlabel("Day relative to expiry")
    ax.set_ylabel("Roll cost vs fair value (ticks)")
    ax.set_title("Chart (c): cost by method and day")
    ax.legend()
    ax.grid(True, alpha=0.3)

    means = [savings[s]["mean"] for s in sigmas]
    err = [[savings[s]["mean"] - savings[s]["ci_lo"] for s in sigmas],
           [savings[s]["ci_hi"] - savings[s]["mean"] for s in sigmas]]
    ax2.errorbar(sigmas, means, yerr=err, fmt="o-", capsize=4)
    ax2.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("sd of fair-value error (ticks)")
    ax2.set_ylabel(f"Saving of THRESH(k={k:g}) over FIXED_DAY(-5) (ticks)")
    ax2.set_title("THRESH saving vs fair-value error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "c_roll_cost.png", **_SAVEFIG_KW)
    plt.close(fig)

    return {
        "chart_c": {m: {d: float(np.mean(v)) for d, v in by_day.items()}
                    for m, by_day in cost.items()},
        "thresh_saving": savings,
    }


def chart_e_tick_regimes(
    contract: ContractSpec,
    n_episodes: int = 100,
    seed: int = REPORT_SEED,
    out: str = "reports",
    timeout: float = 30.0,
) -> dict:
    """Chart (e): share of cross-sectional cost variance explained, by tick regime.

    Per-order cost (ticks) of PASS(timeout) for a 1-lot buy posted at several
    times per episode. R^2 of cost on queue-ahead alone vs on spread plus
    imbalance against the order (both known at posting).
    """
    out_dir = _ensure_out_dir(out)
    t0s = [60.0 + 30.0 * j for j in range(6)]

    results = {}
    for regime_name in ["large", "medium", "small"]:
        regime = REGIMES[regime_name]
        costs, ahead, spread, imb, passive = [], [], [], [], []
        for ep in range(n_episodes):
            log = generate_episode(regime, seed=seed + 20000 + ep, warmup=60.0, horizon=300.0)
            for t0 in t0s:
                res = replay_pass(log, side=+1, qty=1, t0=t0, timeout=timeout)
                costs.append(compute_cost(res, log, contract).total)
                passive.append(res.n_partial > 0)
                b, a, Qb, Qa = log.state_at(t0)
                ahead.append(Qb)
                spread.append(a - b)
                imb.append(-(Qb - Qa) / (Qb + Qa))  # + = ask-heavy = against a buyer
        y = np.array(costs)
        r2_queue = variance_decomposition(y, np.array(ahead, dtype=float)[:, None])["overall_r2"]
        r2_state = variance_decomposition(
            y, np.column_stack([spread, imb]).astype(float)
        )["overall_r2"]
        results[regime_name] = {
            "mean_cost": float(y.mean()),
            "std_cost": float(y.std()),
            "r2_queue_ahead": r2_queue,
            "r2_spread_imbalance": r2_state,
            "passive_share": float(np.mean(passive)),
            "n": len(y),
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(results)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [results[k]["r2_queue_ahead"] for k in labels], 0.4, label="queue-ahead")
    ax.bar(x + 0.2, [results[k]["r2_spread_imbalance"] for k in labels], 0.4,
           label="spread + imbalance")
    ax.set_xticks(x, labels)
    ax.set_ylabel("R² of per-order cost")
    ax.set_title("Chart (e): variance explained by tick regime")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / "e_tick_regimes.png", **_SAVEFIG_KW)
    plt.close(fig)

    return {"chart_e": results}


def build_summary(
    contract: ContractSpec,
    out: str = "reports",
    n_episodes: int = 50,
) -> pd.DataFrame:
    """Build summary.csv: mean cost columns per (regime, policy), 1-lot buy orders.

    One lot keeps the order well inside the own-footprint bound ``n <= 0.1 q_bar``
    for every regime. bps columns use each regime's own tick size and the fee is
    rescaled to that tick.
    """
    out_dir = _ensure_out_dir(out)
    policies = {
        "AGG": lambda log, t0: replay_agg(log, +1, 1, t0),
        "PASS_30": lambda log, t0: replay_pass(log, +1, 1, t0, 30.0),
    }

    rows = []
    for regime_name in ["large", "medium", "small"]:
        regime = REGIMES[regime_name]
        logs = [
            generate_episode(regime, seed=REPORT_SEED + ep, warmup=120.0, horizon=300.0)
            for ep in range(n_episodes)
        ]
        for policy_name, policy_fn in policies.items():
            per_order = []
            for log in logs:
                cost = compute_cost(policy_fn(log, 120.0), log, contract)
                cost.policy = policy_name
                per_order.append(add_perf_columns(cost, regime.tick_bps))
            df_p = pd.DataFrame(per_order)
            row = df_p.mean(numeric_only=True).to_dict()
            row["regime"] = regime_name
            row["policy"] = policy_name
            row["fill_type"] = df_p["fill_type"].mode().iloc[0]
            row["n_orders"] = len(df_p)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df


def chart_metric_rows(a: dict, b: dict, c: dict, e: dict) -> pd.DataFrame:
    """Chart values quoted by the README, one (regime, policy, metric, value) per row."""
    rows: list[tuple[str, str, str, float]] = []
    for r in a["chart_a"]:
        for key in ("n_paths", "mean_cost", "agg_cost", "diff", "diff_ci_lo", "diff_ci_hi",
                    "fill_rate"):
            rows.append(("large", "chart_a", f"{key}_ahead_{r['ahead']}", r[key]))
    for key in ("break_even_ahead", "break_even_ci_ahead"):
        rows.append(("large", "chart_a", key, float("nan") if a[key] is None else a[key]))
    for label, means in b["chart_b"].items():
        slug = label.replace(", ", "_").replace(" ", "_")
        for j, h in enumerate(b["horizons"]):
            lo, hi = b["chart_b_ci"][label][j]
            rows.append(("large", "chart_b", f"{slug}_mean_{h:g}s", means[j]))
            rows.append(("large", "chart_b", f"{slug}_ci_lo_{h:g}s", lo))
            rows.append(("large", "chart_b", f"{slug}_ci_hi_{h:g}s", hi))
            rows.append(("large", "chart_b", f"{slug}_n_{h:g}s", b["chart_b_n"][label][j]))
    for sigma, v in c["thresh_saving"].items():
        for key in ("mean", "ci_lo", "ci_hi"):
            rows.append(("", "chart_c", f"thresh_saving_{key}_sd_{sigma:g}", v[key]))
    for regime, v in e["chart_e"].items():
        for key in ("r2_queue_ahead", "r2_spread_imbalance", "passive_share", "n"):
            rows.append((regime, "chart_e", key, v[key]))
    return pd.DataFrame(rows, columns=["regime", "policy", "metric", "value"])


def write_summary(costs: pd.DataFrame, metrics: pd.DataFrame, out: str = "reports") -> None:
    """Write summary.csv: the cost rows, then the chart metric rows."""
    frame = pd.concat([costs, metrics], ignore_index=True)
    frame.to_csv(_ensure_out_dir(out) / "summary.csv", index=False)


def main() -> None:
    """Generate all charts and summary.csv."""
    parser = argparse.ArgumentParser(description="Generate futures-tca report")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Fast mode (small N)")
    args = parser.parse_args()

    contract = ContractSpec()
    fast = args.fast

    print(f"Generating report to {args.out}/ (fast={fast})")

    a = chart_a_queue_lottery(contract, n_episodes=50 if fast else 3000, out=args.out)
    print("  Chart (a) done")

    b = chart_b_markouts(contract, n_episodes=50 if fast else 2000, out=args.out)
    print("  Chart (b) done")

    c = chart_c_roll_cost(n_episodes=50 if fast else 200, out=args.out)
    print("  Chart (c) done")

    e = chart_e_tick_regimes(contract, n_episodes=50 if fast else 100, out=args.out)
    print("  Chart (e) done")

    costs = build_summary(contract, out=args.out, n_episodes=50)
    write_summary(costs, chart_metric_rows(a, b, c, e), out=args.out)
    print("  summary.csv done")

    print("Report generation complete.")


if __name__ == "__main__":
    main()
