import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import parallel_extras as px  # noqa: E402


def test_planning_formulas_match_ppt4_slides():
    assert px.training_memory_gb(1e9) == pytest.approx(16) and px.training_memory_gb(1e10) == pytest.approx(160)      # slide 8
    assert px.ring_allreduce_bytes_per_worker(400, 8) == pytest.approx(700)                                          # slide 11: 700 MB
    assert px.ring_allreduce_bytes_per_worker(400, 8) / 10_000 == pytest.approx(0.07)                                # 70 ms at 10 GB/s
    assert px.hybrid_layouts(64) == 28                                                                               # slide 16
    assert px.amdahl(0.1, 8) == pytest.approx(4.71, abs=0.01)                                                        # slide 13
    assert px.successive_halving_cost(27, 3, 4) == 108                                                               # slide 18
