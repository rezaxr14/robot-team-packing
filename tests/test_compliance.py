"""Compliance supervisor on an abstract plant: numpy only (run with --noconftest if torch is absent)."""
import numpy as np
import pytest

from rtp.control.compliance import PROFILES, ComplianceConfig, ComplianceSupervisor, State
from rtp.control.compliance_sim import hold, pulse, settle_time, simulate

K = PROFILES["free_motion"][0]


@pytest.mark.parametrize("force", [5.0, 10.0, 20.0, 30.0])
@pytest.mark.parametrize("axis,sign", [(0, 1), (0, -1), (1, 1), (2, -1)])
def test_steady_deflection_equals_force_over_stiffness(force, axis, sign):
    d = np.zeros(3); d[axis] = sign
    r = simulate(hold, pulse(d, force, 0.2, 2.0), 2.0)
    i = int(1.9 / 0.001)
    delta = np.linalg.norm(r["x"][i, :3])
    assert delta == pytest.approx(force / K, rel=0.10)           # spring law within +-10 %


def test_release_returns_to_plan_with_bounded_speed_and_no_overshoot():
    r = simulate(hold, pulse([1, 0, 0], 20.0, 0.2, 1.2), 8.0)
    cfg = r["cfg"]
    assert r["state"][int(1.1 / 0.001)] == "YIELD"
    ts = settle_time(r, 1.2)
    assert ts is not None and ts <= 2.5                          # back within 5 mm in <= 2.5 s
    after = slice(int(1.2 / 0.001), None)
    speed = np.linalg.norm(r["xd"][after, :3], axis=1)
    assert speed.max() <= 1.1 * cfg.v_return                     # bounded return speed
    assert r["x"][after, 0].min() >= -0.10 * 0.1                 # overshoot <= 10 % of the 0.1 m deflection
    rms = np.sqrt(np.mean(np.sum(r["x"][int(6.0 / 0.001):, :3] ** 2, axis=1)))
    assert rms <= 0.002                                          # no limit cycle
    assert r["state"][-1] == "TRACK"


def test_plan_clock_is_frozen_while_pushed_and_resumes_where_it_paused():
    plan = lambda s: np.array([min(0.1 * s, 0.3), 0, 0, 0, 0, 0])      # 0.1 m/s along x for 3 s
    r = simulate(plan, pulse([-1, 0, 0], 25.0, 1.0, 2.0), 9.0)
    t = r["t"]
    yl = [i for i, st in enumerate(r["state"]) if st in ("YIELD", "RETURN")]
    assert yl, "the push must be detected"
    s_y = r["s"][yl]
    assert np.ptp(s_y) == pytest.approx(0.0, abs=1e-9)           # plan clock exactly frozen
    assert r["state"][-1] == "TRACK"
    assert np.linalg.norm(r["x"][-1, :3] - np.array([0.3, 0, 0])) <= 0.003    # trajectory completed after the pause
    names = [e for _t, e in r["events"]]
    assert "PUSHED" in names and "YIELDED" in names and any(n.startswith("RESUMED") for n in names)


def test_moving_plan_alone_does_not_trigger_a_yield():
    plan = lambda s: np.array([min(0.15 * s, 0.3), 0, 0, 0, 0, 0])     # 0.15 m/s: lag without feed-forward > e_pause
    r = simulate(plan, lambda t: np.zeros(6), 3.0)
    assert r["yield_count"] == 0 and set(r["state"]) == {"TRACK"}


def test_leash_caps_the_spring_force_and_the_reference_follows_the_person():
    r = simulate(hold, pulse([1, 0, 0], 38.0, 0.2, 6.0), 6.0, cfg=ComplianceConfig(f_lin_max=60.0))
    cfg = r["cfg"]
    lin = np.linalg.norm(r["x"][:, :3] - r["x_ref"][:, :3], axis=1)
    assert lin.max() <= cfg.delta_max * 1.02
    assert np.linalg.norm(r["F"][:, :3], axis=1).max() <= 60.0 + 1e-9


def test_invariants_hold_under_random_pushes():
    rng = np.random.default_rng(3)
    # random piecewise-constant 6-D external wrench up to 35 N / 2.6 Nm, changing every 0.3-1.5 s
    segs, t = [], 0.0
    while t < 60.0:                                              # push for 0.3-1.5 s, then an idle gap of 0.5-3 s
        d = rng.uniform(0.3, 1.5)
        v = np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.1])
        v = v / np.linalg.norm(v[:3]) * rng.uniform(0, 35.0)          # plausible pushes, below the arm's own 40 N cap
        v[3:] = np.clip(v[3:], -1.5, 1.5)                        # torques below the arm's own 4 Nm cap
        segs.append((t, t + d, v)); t += d + rng.uniform(0.5, 3.0)
    f = lambda tt: next((v for a, b, v in segs if a <= tt < b), np.zeros(6))
    plan = lambda s: np.array([0.05 * np.sin(0.5 * s), 0, 0, 0, 0, 0])
    r = simulate(plan, f, 60.0)                                  # 60 000 ticks
    assert "FAULT" not in [e for _t, e in r["events"]]
    cfg = r["cfg"]
    assert np.linalg.norm(r["F"][:, :3], axis=1).max() <= cfg.f_lin_max + 1e-9
    assert np.linalg.norm(r["F"][:, 3:], axis=1).max() <= cfg.m_max + 1e-9
    gap = np.linalg.norm(r["x"][:, :3] - r["x_ref"][:, :3], axis=1)
    held = np.array([s in ("YIELD", "RETURN") for s in r["state"]])
    slack = 2 * np.linalg.norm(r["xd"][:, :3], axis=1) * 0.001  # the plant moves one tick after the leash is applied
    assert (gap[held] <= cfg.delta_max + slack[held] + 1e-9).all()   # leash invariant while yielding/returning
    free = np.linalg.norm(r["f_ext"][:, :3], axis=1) < cfg.f_release
    ret = np.array([s == "RETURN" for s in r["state"]]) & free
    if ret.any():
        assert np.linalg.norm(r["xd"][ret, :3], axis=1).max() <= 1.15 * cfg.v_return


@pytest.mark.parametrize("mass", [3.0, 4.5, 6.0])              # arm + 0..3 kg payload against a 3 kg identified mass
def test_payload_variation_still_returns(mass):
    r = simulate(hold, pulse([1, 0, 0], 20.0, 0.2, 1.2), 10.0, mass=mass)
    assert settle_time(r, 1.2) is not None and r["state"][-1] == "TRACK"


def test_extra_latency_and_noisy_wrench_estimate_stay_stable():
    r = simulate(hold, pulse([0, 1, 0], 15.0, 0.2, 1.2), 10.0, delay_ticks=3, est_noise=0.5)
    assert np.isfinite(r["x"]).all() and np.abs(r["x"]).max() < 0.3
    assert settle_time(r, 1.2, tol=0.008) is not None


def test_a_yield_longer_than_the_limit_faults_into_a_controlled_stop():
    cfg = ComplianceConfig(t_yield_max=2.0)
    r = simulate(hold, pulse([1, 0, 0], 20.0, 0.2, 4.0), 8.0, cfg=cfg)
    assert "FAULT" in [e for _t, e in r["events"]] and r["state"][-1] == "FAULT"
    assert np.linalg.norm(r["xd"][-1, :3]) < 0.01                # damped to a stop, not a free spring
    assert np.linalg.norm(r["F"][:, :3], axis=1).max() <= cfg.f_lin_max + 1e-9


def test_an_external_force_beyond_a_push_is_a_collision_and_stops_the_arm():
    r = simulate(hold, pulse([1, 0, 0], 120.0, 0.5, 0.52), 3.0)    # a 20 ms, 120 N impact
    assert "FAULT" in [e for _t, e in r["events"]]
    assert np.linalg.norm(r["F"][:, :3], axis=1).max() <= r["cfg"].f_lin_max + 1e-9
    assert np.linalg.norm(r["xd"][-1, :3]) < 0.01


def test_unknown_stiffness_profile_is_refused_and_scale_is_clipped():
    sup = ComplianceSupervisor()
    with pytest.raises(KeyError):
        sup.set_profile("stiff_as_a_rock")
    for _ in range(3000):
        sup.step(0.001, np.zeros(6), np.zeros(6), np.zeros(6), hold, scale=100.0)
    assert sup.k_lin <= 1.5 * PROFILES["free_motion"][0] + 1e-6  # a learned policy cannot exceed 1.5x the profile


# ---- stiction floor: the Newton FR3 plan_pause failure (measured 3.241 mm, clock never resumes) ----
FRICTION = (0.65, 0.65)                                   # Coulomb, stiction (N): 0.65 N / 200 N/m = 3.25 mm
PLAN_S = 2.0


def _plan_pause_run(cfg=None, post=8.0):
    plan = lambda s: np.array([0.05 * min(s, PLAN_S), 0, 0, 0, 0, 0])
    return simulate(plan, pulse([1, 0, 0], 10.0, 0.5, 1.5), 1.5 + post, cfg=cfg, friction=FRICTION), plan


def test_without_the_fix_the_stiction_floor_deadlocks_the_resume():
    """Documents the defect: old behaviour = no stall-aware resume, no integral."""
    r, plan = _plan_pause_run(ComplianceConfig(eps_stall=0.0, ki_lin=0.0))
    names = [e for _t, e in r["events"]]
    assert not any(n.startswith("RESUMED") for n in names)
    assert r["state"][-1] == "RETURN"                                   # stuck forever
    assert np.ptp(r["s"][int(1.6 / 0.001):]) == 0.0                      # plan clock never advances again
    assert np.linalg.norm(r["x_ref"][-1, :3] - r["x"][-1, :3]) > 0.003  # parked just outside eps_e (~3.2 mm)


def test_stall_aware_resume_and_gated_integral_fix_plan_pause_with_friction():
    r, plan = _plan_pause_run()
    names = [e for _t, e in r["events"]]
    assert any(n.startswith("RESUMED") for n in names)
    held = [i for i, st in enumerate(r["state"]) if st in ("YIELD", "RETURN")]
    assert np.ptp(r["s"][held]) == pytest.approx(0.0, abs=1e-9)          # still frozen while yielding/returning
    assert r["s"][-1] > r["s"][held[-1]] + 1.0                           # the clock advances after resume
    assert r["state"][-1] == "TRACK"
    final_err = np.linalg.norm(r["x"][-1, :3] - plan(r["s"][-1])[:3])
    assert final_err <= 0.003                                            # the pre-registered 3 mm bound, unchanged


def test_integral_never_acts_in_yield_or_return_so_release_return_overshoot_is_unchanged():
    r = simulate(hold, pulse([1, 0, 0], 20.0, 0.2, 1.2), 10.0, friction=FRICTION)
    st = r["state"]
    assert not np.any(r["integral"][(st == "YIELD") | (st == "RETURN")])
    after = slice(int(1.2 / 0.001), None)
    delta = np.linalg.norm(r["x"][int(1.15 / 0.001), :3])
    overshoot = max(0.0, -r["x"][after, 0].min()) / delta
    assert overshoot <= 0.10
    assert np.linalg.norm(r["xd"][after, :3], axis=1).max() <= 1.1 * r["cfg"].v_return
    assert settle_time(r, 1.2, tol=0.005) is not None or r["state"][-1] == "TRACK"


def test_integral_authority_is_capped_so_it_cannot_fight_a_real_push():
    cfg = ComplianceConfig()
    for force, min_defl in ((2.0, 0.004), (3.0, 0.009)):                 # below f_release: integral eligible, still capped at 1 N
        r = simulate(hold, pulse([1, 0, 0], force, 0.5, 20.0), 20.0, friction=None)
        defl = np.linalg.norm(r["x"][-1, :3])
        assert defl >= min_defl, (force, defl)                           # the person still moves it: (F - i_max)/K at least
        assert np.linalg.norm(r["F"][-1, :3] - (-200.0 * r["x"][-1, :3])) <= cfg.i_max_lin + 0.05


def test_no_stick_slip_hunting_after_the_integral_clears_the_floor():
    r, plan = _plan_pause_run(post=14.0)
    last = r["x"][-int(3.0 / 0.001):, 0]                                 # final 3 s of the static hold
    assert last.max() - last.min() < 0.001
