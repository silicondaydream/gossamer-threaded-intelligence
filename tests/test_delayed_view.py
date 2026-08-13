"""The delayed peer view — the mechanism the whole DCC corpus is built on.

These tests moved here with the code. That is the point of the move: while
`DelayedView` lived in the Maneuver.Map runner it was not covered by Gossamer's
tests, not shipped in the wheel the papers pin, and not runnable by anyone
reproducing a DCC result from the public package (DOCS §2). A reader would call
this *the method*, so both it and its tests belong in the package.

The arithmetic is locked far harder than any assertion here can manage, by the two
fingerprints: SHA-256 over the exact `repr()` of every Q the coordination seam
produces, across 12 primitives × 2 delays, plus the HMA market lock. What THIS
file pins is what a digest cannot see — the guards, the absences, and the ordering.
"""
import numpy as np
import pytest

from gossamer.communication import DelayedView, gating_enabled, resolve_delay_steps


def _rng():
    return np.random.default_rng(0)


# --- delay: ms -> steps -------------------------------------------------------

def test_latency_in_ms_converts_to_steps_against_dt():
    assert resolve_delay_steps({"comm_latency_ms": "20000"}, dt=1.0) == 20
    assert resolve_delay_steps({"comm_latency_ms": "20000"}, dt=0.5) == 40


def test_explicit_step_count_is_the_fallback_not_an_override():
    """`comm_latency_ms` wins when set; `coupling_delay_steps` is the other spelling."""
    assert resolve_delay_steps({"coupling_delay_steps": "7"}, dt=1.0) == 7
    assert resolve_delay_steps({"comm_latency_ms": "5000",
                                "coupling_delay_steps": "7"}, dt=1.0) == 5


def test_delay_is_never_negative_and_is_capped():
    assert resolve_delay_steps({"coupling_delay_steps": "-3"}, dt=1.0) == 0
    # The ring is delay+1 float32 frames, so the axis has a memory cost and a cap.
    assert resolve_delay_steps({"coupling_delay_steps": "999999"}, dt=1.0) == 2000


def test_a_delay_cap_override_is_honoured(monkeypatch):
    monkeypatch.setenv("DCC_MAX_DELAY_STEPS", "10")
    assert resolve_delay_steps({"coupling_delay_steps": "500"}, dt=1.0) == 10


# --- gating is one predicate, spelled once ------------------------------------

def test_gating_predicate_is_shared_with_the_caller():
    """The caller has to make the same decision — whether to fetch edges from its
    engine — and two copies of this predicate is how gating ends up on for the view
    and off for the fetch."""
    assert gating_enabled({"comm_collect_edges": "1"})
    for off in ({}, {"comm_collect_edges": "0"}, {"comm_collect_edges": ""},
                {"comm_collect_edges": "false"}, {"comm_collect_edges": "No"}):
        assert not gating_enabled(off), off


# --- the delayed view ---------------------------------------------------------

def test_the_primitive_sees_the_frame_from_delay_steps_ago():
    """THE mechanism of the whole DCC corpus: decide on stale peer state. If the view
    were fresh, Q would be independent of the delay axis and the phase diagram would be
    measuring nothing."""
    view = DelayedView({"coupling_delay_steps": "2"}, dt=1.0, num_agents=1,
                       prediction=None, rng=_rng())
    seen = []
    for step in range(5):
        pos = np.full((1, 3), float(step))
        vp, _vv, _e = view.perceive(pos, np.zeros((1, 3)))
        seen.append(float(vp[0, 0]))
    # Steps 0,1 have no 2-step-old frame yet, so they read the oldest they have —
    # information that was never sent cannot arrive. Then the lag settles at 2.
    assert seen == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_zero_delay_sees_the_truth():
    view = DelayedView({}, dt=1.0, num_agents=1, prediction=None, rng=_rng())
    pos = np.array([[7.0, 0.0, 0.0]])
    vp, _vv, _e = view.perceive(pos, np.zeros((1, 3)))
    assert vp[0, 0] == 7.0


def test_delivered_edges_travel_with_the_frame_they_were_measured_on():
    """An edge transmitted at t must be applied at t + delay_steps, not immediately —
    otherwise the delivery gate leaks information backwards through the delay."""
    view = DelayedView({"coupling_delay_steps": "1", "comm_collect_edges": "1"},
                       dt=1.0, num_agents=1, prediction=None, rng=_rng())
    e0, e1 = [(0, 1)], [(1, 0)]
    _, _, edges = view.perceive(np.zeros((1, 3)), np.zeros((1, 3)), e0)
    assert edges == e0                       # only frame available
    _, _, edges = view.perceive(np.zeros((1, 3)), np.zeros((1, 3)), e1)
    assert edges == e0                       # e1 is still in flight


def test_ungated_runs_discard_edges_they_are_handed():
    """Gating off means the primitive reads the full delayed view. A caller that
    passes edges anyway must not accidentally enable gating."""
    view = DelayedView({}, dt=1.0, num_agents=1, prediction=None, rng=_rng())
    assert not view.gated
    _, _, edges = view.perceive(np.zeros((1, 3)), np.zeros((1, 3)), [(0, 1)])
    assert edges is None


def test_gating_without_edges_raises_rather_than_coordinating_ungated():
    """The per-step half of the stale-`leviathan-base` guard. Silently un-gating turns
    a P5 delivery-gated cell into an ungated one under the same label — and the whole
    P2 cost frontier was degenerate for exactly this reason: 0 bits/s still gave Q=1.0.

    (The other half — does the caller's engine expose `comm_edges` at all — lives in
    the caller, which is the only side that can answer it.)"""
    view = DelayedView({"comm_collect_edges": "1"}, dt=1.0, num_agents=1,
                       prediction=None, rng=_rng())
    with pytest.raises(ValueError, match="delivery gating is enabled"):
        view.perceive(np.zeros((1, 3)), np.zeros((1, 3)), None)


def test_sensing_noise_perturbs_the_perceived_position_not_the_velocity():
    """Sensing noise is the OBSERVATION channel. Perturbing velocity would inject energy
    and break the caller's true-velocity re-actuation; perturbing position cannot."""
    view = DelayedView({"sensing_noise": "5.0"}, dt=1.0, num_agents=4,
                       prediction=None, rng=_rng())
    pos, vel = np.zeros((4, 3)), np.full((4, 3), 2.0)
    vp, vv, _ = view.perceive(pos, vel)
    assert not np.allclose(vp, 0.0)          # position was perturbed
    np.testing.assert_allclose(vv, 2.0)      # velocity is still true
    np.testing.assert_allclose(pos, 0.0)     # and the TRUE state is untouched


def test_sensing_noise_draws_stay_on_the_callers_stream_in_order():
    """The rng is held BY REFERENCE on purpose: the draws share the run's single
    seeded stream. Giving them a stream of their own would renumber every draw after
    them and move published numbers."""
    shared = np.random.default_rng(12345)
    view = DelayedView({"sensing_noise": "1.0"}, dt=1.0, num_agents=2,
                       prediction=None, rng=shared)
    vp, _, _ = view.perceive(np.zeros((2, 3)), np.zeros((2, 3)))

    expected_rng = np.random.default_rng(12345)
    expected = expected_rng.normal(0.0, 1.0, size=(2, 3))
    np.testing.assert_array_equal(vp, expected)
    # And the caller's stream advanced by exactly those draws, so whatever it draws
    # next is what it would have drawn with the noise inline.
    assert shared.normal() == expected_rng.normal()


def test_no_sensing_noise_leaves_the_view_exact():
    """Runs that don't set it (P1/P3/Vicsek) must be byte-identical."""
    view = DelayedView({}, dt=1.0, num_agents=2, prediction=None, rng=_rng())
    pos = np.array([[1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])
    vp, _, _ = view.perceive(pos, np.zeros((2, 3)))
    np.testing.assert_array_equal(vp, pos)


def test_the_true_state_is_never_mutated():
    """The engine integrates the TRUE state from the accel the caller returns; this
    only shapes what the primitive is allowed to see."""
    view = DelayedView({"coupling_delay_steps": "1", "sensing_noise": "3.0"},
                       dt=1.0, num_agents=2, prediction=None, rng=_rng())
    pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    before_pos, before_vel = pos.copy(), vel.copy()
    view.perceive(pos, vel)
    np.testing.assert_array_equal(pos, before_pos)
    np.testing.assert_array_equal(vel, before_vel)


def test_an_unknown_predictor_aborts_rather_than_running_the_baseline():
    """`except Exception: _predictor = None` used to downgrade a P3 'kalman' cell to the
    no-prediction baseline — a different condition, reported under the label asked for."""
    with pytest.raises(ValueError, match="unknown predictor"):
        DelayedView({}, dt=1.0, num_agents=2,
                    prediction={"model": "crystal_ball"}, rng=_rng())


def test_model_none_is_a_legitimate_way_to_ask_for_no_predictor():
    view = DelayedView({}, dt=1.0, num_agents=2,
                       prediction={"model": "none"}, rng=_rng())
    assert view.predictor is None


# --- per-agent delay offsets (roadmap 20) -------------------------------------
# The delay axis stops being a dialled scalar and becomes measured geometry: each
# peer's staleness is its own contact-plan delay, handed across the seam as an
# array of steps (Gossamer never imports Orrery).

def test_a_uniform_per_agent_offset_is_the_scalar_path():
    """THE compatibility proof. Per-agent offsets are a generalisation, not a new
    mechanism, so at a uniform offset they must reproduce the scalar path exactly —
    otherwise landing this would re-pin every published delay number."""
    for d in (0, 1, 3, 7):
        scalar = DelayedView({"coupling_delay_steps": str(d)}, dt=1.0, num_agents=4,
                             prediction=None, rng=_rng())
        per_agent = DelayedView({}, dt=1.0, num_agents=4, prediction=None, rng=_rng(),
                                delay_steps_per_agent=np.full(4, d, dtype=int))
        assert per_agent.delay_steps == d
        for step in range(12):
            pos = np.full((4, 3), float(step)) + np.arange(4)[:, None]
            vel = pos * 0.5
            a_pos, a_vel, _ = scalar.perceive(pos, vel)
            b_pos, b_vel, _ = per_agent.perceive(pos, vel)
            np.testing.assert_array_equal(a_pos, b_pos, err_msg=f"d={d} step={step}")
            np.testing.assert_array_equal(a_vel, b_vel, err_msg=f"d={d} step={step}")


def test_each_agent_is_seen_at_its_own_staleness():
    """Agent i's row comes from the frame i's own offset back. This is the whole
    point: a constellation's delay is not one number (orrery.astro.delay.delay_stats
    is built around exactly that), so the view must not pretend it is."""
    offsets = np.array([0, 1, 3], dtype=int)
    view = DelayedView({}, dt=1.0, num_agents=3, prediction=None, rng=_rng(),
                       delay_steps_per_agent=offsets)
    for step in range(6):
        # Every agent's position IS the step number, so a row's value reads back as
        # the step it was sampled on.
        vp, _, _ = view.perceive(np.full((3, 3), float(step)), np.zeros((3, 3)))
    # After step 5: agent 0 sees step 5, agent 1 sees step 4, agent 2 sees step 2.
    np.testing.assert_array_equal(vp[:, 0], [5.0, 4.0, 2.0])


def test_a_short_ring_reads_the_oldest_frame_it_has():
    """Early in the run a deep offset has no frame that old. Reading the oldest
    available is correct — information that was never sent cannot arrive — and it is
    what the scalar path does too."""
    view = DelayedView({}, dt=1.0, num_agents=2, prediction=None, rng=_rng(),
                       delay_steps_per_agent=np.array([0, 9], dtype=int))
    vp, _, _ = view.perceive(np.full((2, 3), 3.0), np.zeros((2, 3)))
    assert vp[0, 0] == 3.0 and vp[1, 0] == 3.0   # both read the only frame there is


def test_a_gated_edge_surfaces_after_its_SOURCE_delay():
    """An edge (src -> dst) arrives when its bundle does, and the flight time is the
    SOURCE's offset: dst's picture of src is `d_src` stale, so the link that carried
    it is too. Mixing offsets must not leak an edge forward through the delay."""
    offsets = np.array([0, 2], dtype=int)
    view = DelayedView({"comm_collect_edges": "1"}, dt=1.0, num_agents=2,
                       prediction=None, rng=_rng(), delay_steps_per_agent=offsets)
    z = np.zeros((2, 3))
    # Step 0: agent 0 (offset 0) and agent 1 (offset 2) each transmit.
    _, _, e = view.perceive(z, z, np.array([[0, 1], [1, 0]]))
    # 0's edge is instant; 1's is still in flight, and the ring holds only this frame
    # so 1 reads the oldest it has — which is this one.
    assert sorted(map(tuple, e)) == [(0, 1), (1, 0)]
    # Step 1: only agent 0 transmits. Agent 1's offset-2 frame is still step 0.
    _, _, e = view.perceive(z, z, np.array([[0, 1]]))
    assert sorted(map(tuple, e)) == [(0, 1), (1, 0)]  # 0's new edge + 1's step-0 edge
    # Step 2: nobody transmits. 0 contributes nothing; 1's step-0 edge surfaces.
    _, _, e = view.perceive(z, z, np.zeros((0, 2), dtype=int))
    assert sorted(map(tuple, e)) == [(1, 0)]


def test_offsets_must_be_integer_steps_one_per_agent():
    """Silent truncation of a float offset would be a different experiment, and the
    seconds->steps conversion belongs upstream where the oracle can refuse an
    unreachable pair instead of flattening it."""
    with pytest.raises(ValueError, match="one offset per agent"):
        DelayedView({}, dt=1.0, num_agents=3, prediction=None, rng=_rng(),
                    delay_steps_per_agent=np.array([1, 2], dtype=int))
    with pytest.raises(TypeError, match="integer steps"):
        DelayedView({}, dt=1.0, num_agents=2, prediction=None, rng=_rng(),
                    delay_steps_per_agent=np.array([1.5, 2.5]))
    with pytest.raises(ValueError, match="non-negative"):
        DelayedView({}, dt=1.0, num_agents=2, prediction=None, rng=_rng(),
                    delay_steps_per_agent=np.array([1, -2], dtype=int))


def test_per_agent_offsets_refuse_a_predictor_rather_than_guessing_a_horizon():
    """`predict()` takes ONE horizon, so combining them would extrapolate every peer
    by the deepest offset — wrong for all but the slowest agent, and silent."""
    with pytest.raises(ValueError, match="not yet combinable"):
        DelayedView({}, dt=1.0, num_agents=2, prediction={"model": "const_vel"},
                    rng=_rng(), delay_steps_per_agent=np.array([1, 5], dtype=int))
    # `none` stays legitimate.
    v = DelayedView({}, dt=1.0, num_agents=2, prediction={"model": "none"},
                    rng=_rng(), delay_steps_per_agent=np.array([1, 5], dtype=int))
    assert v.predictor is None
