"""A6 conformance smoke for the Newton-backed SimBackend stand-in."""
import pytest

from rtp.simworld import SimWorld


def test_backend_resets_steps_and_reports_objects():
    pytest.importorskip("newton")
    from isaac.newton_packing_sim import NewtonPackingSim

    backend = NewtonPackingSim()
    backend.reset(7)
    assert len(backend.object_states()) == 10
    w = SimWorld(backend, layout=backend.layout, seed=7)
    assert len(w.objs) == 10
    backend.step(2)
    name = backend._items[0][0]
    rep = backend.run_skill("L", "grasp", name, {})
    assert rep.category in ("success", "slip", "other_failure")
    assert "sim_pose_standin" in rep.detail
