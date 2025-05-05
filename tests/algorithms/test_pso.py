import numpy as np

from gossamer.algorithms.optimization.pso import pso


def test_pso_scalar():
    # One-dimensional quadratic function with known minimum at x=3
    def f(x):
        return (x[0] - 3.0) ** 2

    bounds = [(0.0, 10.0)]
    best_pos, best_val = pso(f, bounds, n_particles=20, max_iter=100, w=0.7, c1=1.4, c2=1.4)
    # Best position should be near 3.0
    assert np.isclose(best_pos[0], 3.0, atol=0.1)
    assert best_val < 0.05