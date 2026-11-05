"""Tests for roll.py: OU richness, roll methods, threshold rule.

Test numbers: 20-23.
"""

from __future__ import annotations

import numpy as np
import pytest

from futures_tca.roll import (
    calendar_quote_width,
    front_leg_width,
    liquidity_migration,
    next_leg_width,
    roll_cal_cross,
    roll_fixed_day,
    roll_leg_seq,
    roll_leg_simul,
    roll_thresh,
    simulate_ou,
    simulate_roll_episode,
    thresh_saving,
)

# ---------------------------------------------------------------------------
# Test 20: Truth recovery
# ---------------------------------------------------------------------------


def _simulated_rolls(n_rolls: int, seed0: int = 20261101):
    """One roll per episode on a random day: (cost_cal, cost_leg, expected_cal, expected_leg)."""
    rng = np.random.default_rng(seed0)
    cal = np.empty(n_rolls)
    leg = np.empty(n_rolls)
    exp_cal = np.empty(n_rolls)
    exp_leg = np.empty(n_rolls)
    for i in range(n_rolls):
        ep = simulate_roll_episode(seed=seed0 + 1 + i)
        d = int(rng.integers(0, len(ep["days"])))
        fair, mkt = ep["fair_spread"][d], ep["market_spread"][d]
        w_cal, w_f, w_n = (int(ep[k][d]) for k in ("cal_width", "front_width", "next_width"))
        cal[i] = roll_cal_cross(fair, mkt, w_cal, side=1)
        leg[i] = roll_leg_simul(fair, mkt, w_f, w_n, side=1)
        exp_cal[i] = w_cal / 2.0
        exp_leg[i] = (w_f + w_n) / 2.0
    return cal, leg, exp_cal, exp_leg


class TestTruthRecovery:
    """Known-answer test on simulated rolls: cost = half the width + the recorded richness."""

    def test_cal_cross_mean_cost(self):
        """CAL_CROSS mean cost = E[w/2] + E[m - s*] within 3 SE on 20000 rolls."""
        cal, _, exp_cal, _ = _simulated_rolls(20000)
        # E[m - s*] = 0 by construction (zero-mean skew), so the target is E[w/2]
        diff = cal - exp_cal
        se = np.std(diff) / np.sqrt(len(diff))
        assert abs(np.mean(diff)) < 3 * se, f"mean(cost - w/2) = {np.mean(diff):.4f} +- {3*se:.4f}"
        # The skew part has the nominal sd
        assert np.std(diff) == pytest.approx(0.25, rel=0.05)

    def test_leg_simul_mean_cost(self):
        """LEG_SIMUL mean cost = E[(w_f + w_n)/2] + E[m - s*] within 3 SE."""
        _, leg, _, exp_leg = _simulated_rolls(20000)
        diff = leg - exp_leg
        se = np.std(diff) / np.sqrt(len(diff))
        assert abs(np.mean(diff)) < 3 * se

    def test_roll_cost_equals_width_plus_richness(self):
        ep = simulate_roll_episode(seed=1)
        for d in range(len(ep["days"])):
            c = roll_cal_cross(ep["fair_spread"][d], ep["market_spread"][d],
                               int(ep["cal_width"][d]), side=1)
            assert c == pytest.approx(ep["cal_width"][d] / 2 + ep["richness"][d])

    def test_seller_richness_sign(self):
        assert roll_cal_cross(2.0, 2.3, 2, side=-1) == pytest.approx(1.0 - 0.3)


# ---------------------------------------------------------------------------
# Test 21: Legging term
# ---------------------------------------------------------------------------


class TestLeggingTerm:
    def test_legging_mean_zero(self):
        """Legging term has mean 0 within 3 SE."""
        rng = np.random.default_rng(20261101)
        n_rolls = 10000
        errors = np.empty(n_rolls)

        for i in range(n_rolls):
            _, err = roll_leg_seq(
                fair_spread=2.0,
                market_spread=2.0,
                front_width=1,
                next_width=1,
                side=1,
                delta=0.2,
                sigma_leg=0.5,
                rng=rng,
            )
            errors[i] = err

        se = np.std(errors) / np.sqrt(n_rolls)
        assert abs(np.mean(errors)) < 3 * se

    def test_legging_sd(self):
        """Legging term sd = sigma_leg * sqrt(delta) within 5%."""
        rng = np.random.default_rng(20261101)
        n_rolls = 10000
        errors = np.empty(n_rolls)
        sigma_leg = 0.5
        delta = 0.2

        for i in range(n_rolls):
            _, err = roll_leg_seq(
                fair_spread=2.0,
                market_spread=2.0,
                front_width=1,
                next_width=1,
                side=1,
                delta=delta,
                sigma_leg=sigma_leg,
                rng=rng,
            )
            errors[i] = err

        expected_sd = sigma_leg * np.sqrt(delta)
        actual_sd = np.std(errors)
        assert abs(actual_sd - expected_sd) / expected_sd < 0.05, (
            f"sd = {actual_sd:.4f}, expected {expected_sd:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 22: Threshold rule
# ---------------------------------------------------------------------------


class TestThresholdRule:
    def test_thresh_logic_known_answer(self):
        """On a hand-built episode THRESH picks the first cheap day, else the forced day."""
        days = np.linspace(-10, 0, 11)
        fair = np.full(11, 2.0)
        richness = np.array([0.1, 0.2, -0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        ep = {
            "days": days, "fair_spread": fair, "market_spread": fair + richness,
            "cal_width": np.full(11, 2),
        }
        rng = np.random.default_rng(0)
        cost, day = roll_thresh(ep, k=0.25, sigma_eps=0.0, rng=rng)
        assert day == -8.0
        assert cost == pytest.approx(1.0 - 0.3)
        # No day cheap enough -> forced at day -2
        cost, day = roll_thresh(ep, k=1.0, sigma_eps=0.0, rng=rng)
        assert day == -2.0
        assert cost == pytest.approx(1.0 + 0.1)
        assert roll_fixed_day(ep, -5.0) == pytest.approx(1.0 + 0.1)

    def test_thresh_beats_fixed_perfect_info(self):
        """sigma_eps = 0: the 95% CI of the saving over FIXED_DAY(-5) excludes 0 (positive)."""
        s = thresh_saving(k=0.25, sigma_eps=0.0, n_episodes=4000)
        half = 1.96 * s.std() / np.sqrt(len(s))
        assert s.mean() - half > 0, f"saving {s.mean():.3f} +- {half:.3f}"

    def test_thresh_fails_with_noise(self):
        """sigma_eps = 1.0: the saving CI covers 0 or is negative."""
        s = thresh_saving(k=0.25, sigma_eps=1.0, n_episodes=4000)
        half = 1.96 * s.std() / np.sqrt(len(s))
        assert s.mean() - half <= 0, f"saving {s.mean():.3f} +- {half:.3f}"

    def test_saving_decreases_with_noise(self):
        means = [
            thresh_saving(0.25, sigma, 3000).mean() for sigma in (0.0, 0.5, 1.0)
        ]
        assert means[0] > means[1] > means[2]


# ---------------------------------------------------------------------------
# Test 23: OU process
# ---------------------------------------------------------------------------


class TestOUProcess:
    def test_stationary_sd(self):
        """OU stationary sd within 5% of nominal."""
        rng = np.random.default_rng(20261101)
        n_steps = 200000
        dt = 1.0
        half_life = 100.0
        sd_nominal = 2.0

        x = simulate_ou(n_steps, dt, mean=0, half_life=half_life,
                        stationary_sd=sd_nominal, rng=rng)

        # Discard first 5000 steps for burn-in
        x_burn = x[5000:]
        sd_actual = np.std(x_burn)

        assert abs(sd_actual - sd_nominal) / sd_nominal < 0.05, (
            f"sd = {sd_actual:.4f}, expected {sd_nominal:.4f}"
        )

    def test_half_life(self):
        """OU half-life estimated from path within 15% of nominal."""
        rng = np.random.default_rng(20261101)
        n_steps = 50000
        dt = 1.0
        half_life_nominal = 100.0

        x = simulate_ou(n_steps, dt, mean=0, half_life=half_life_nominal,
                        stationary_sd=2.0, rng=rng)

        # Estimate half-life from autocorrelation
        x_centered = x - np.mean(x)
        n = len(x_centered)
        spec = np.fft.rfft(x_centered, 2 * n)  # zero-padded: linear, not circular, autocorr
        autocorr = np.fft.irfft(spec * np.conj(spec))[:n]
        autocorr /= autocorr[0]

        # Find lag where autocorr crosses 0.5
        crossing = np.argmax(autocorr < 0.5)
        half_life_est = crossing * dt

        assert abs(half_life_est - half_life_nominal) / half_life_nominal < 0.15, (
            f"half-life = {half_life_est:.1f}, expected {half_life_nominal:.1f}"
        )


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


class TestLiquidityMigration:
    def test_logistic_shape(self):
        """Liquidity migration is logistic."""
        d = np.array([-10, -5, 0])
        rho = liquidity_migration(d, d_roll=-5, w=2)
        assert rho[0] < 0.1  # far from roll
        assert abs(rho[1] - 0.5) < 0.01  # at midpoint
        assert rho[2] > 0.9  # at expiry

    def test_scalar(self):
        rho = liquidity_migration(-5.0, d_roll=-5, w=2)
        assert abs(rho - 0.5) < 0.01


class TestQuoteWidths:
    def test_front_width(self):
        assert front_leg_width() == 1

    def test_next_width_liquid(self):
        rng = np.random.default_rng(42)
        w = next_leg_width(rho=0.9, rng=rng)
        assert w == 1  # liquid next contract

    def test_calendar_width(self):
        rng = np.random.default_rng(42)
        w = calendar_quote_width(rho=0.5, rng=rng)
        assert w in [1, 2, 3]


class TestRollEpisode:
    def test_basic(self):
        episode = simulate_roll_episode(n_days=11, seed=42)
        assert len(episode["days"]) == 11
        assert len(episode["fair_spread"]) == 11
        assert len(episode["market_spread"]) == 11
        assert len(episode["richness"]) == 11
        assert len(episode["rho"]) == 11

    def test_rho_monotone(self):
        episode = simulate_roll_episode(n_days=11, seed=42)
        rho = episode["rho"]
        # rho should increase as we approach expiry
        assert rho[-1] > rho[0]


class TestEpisodeOU:
    def test_episode_uses_exact_ou_stationary_sd(self):
        """Across episodes, richness and skew keep their nominal stationary sd on every day."""
        n = 3000
        z = np.empty((n, 11))
        e = np.empty((n, 11))
        for i in range(n):
            ep = simulate_roll_episode(seed=700_000 + i, carry=2.0)
            z[i] = ep["fair_spread"] - 2.0
            e[i] = ep["richness"]
        # No blow-up and no drift in the sd across days (a stable exact transition)
        assert np.allclose(z.std(axis=0), 0.4, rtol=0.08)
        assert np.allclose(e.std(axis=0), 0.25, rtol=0.08)

    def test_widths_in_range(self):
        ep = simulate_roll_episode(seed=3)
        assert set(np.unique(ep["cal_width"])) <= {1, 2, 3}
        assert set(np.unique(ep["next_width"])) <= {1, 2, 3}
        assert np.all(ep["front_width"] == 1)

    def test_deterministic(self):
        a = simulate_roll_episode(seed=9)
        b = simulate_roll_episode(seed=9)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
