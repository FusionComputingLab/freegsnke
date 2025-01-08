import numpy as np

from freegsnke.implicit_euler import implicit_euler_solver


def test_implicit_euler_solver():
    """Tests the implicit_euler_solver class.

    TODO add a test checking the output of the solver against a known solution
    """
    N = 10
    M = np.random.randn(N, N)
    R = np.diag(np.random.randn(N))

    full_timestep = 1
    internal_timestep = 0.1

    solver = implicit_euler_solver(M, R, 1, 0.1)

    assert hasattr(
        solver, "inverse_operator"
    ), "Solver did not pre-calculate inverse operator"

    M_update = np.random.randn(N, N)
    R_update = np.random.randn(N, N)

    solver.set_Lmatrix(M_update)
    assert np.all(solver.Lmatrix == M_update), "Did not set L matrix properly"

    solver.set_Mmatrix(M_update)
    assert np.all(solver.Mmatrix == M_update), "Did not set M matrix properly"

    solver.set_Rmatrix(R_update)

    assert np.all(solver.Rmatrix == R_update), "Did not set R matrix properly"

    solver.set_timesteps(full_timestep, internal_timestep)
    assert solver.full_timestep == full_timestep, "Full timestep not set properly"
    assert (
        solver.internal_timestep <= full_timestep
    ), "Internal timestep larger than full timestep"

    solver.set_timesteps(
        full_timestep, 3 * full_timestep
    )  # test if internal_timestep > full_timestep
    assert (
        solver.internal_timestep == solver.full_timestep and solver.n_steps == 1
    ), "Error when internal timestep larger than the full timestep"

    I = np.random.randn(N)
    F = np.random.randn(N)

    Itpdt = solver.full_stepper(I, F)
