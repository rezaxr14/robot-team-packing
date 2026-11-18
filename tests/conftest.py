import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_rng():
    """No test may change another test's random stream.

    Several suites (toy trainers, arm policies, flow sampling) train on CPU with global torch RNG.
    Without this, a new test that calls ``manual_seed`` silently changes the outcome of a later
    policy-rollout test, and the failure appears in a completely unrelated file.
    """
    import random

    import numpy as np
    import torch

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
