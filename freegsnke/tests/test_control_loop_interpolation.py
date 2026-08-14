"""Tests for inexpensive control-loop waveform evaluation."""

import numpy as np

from freegsnke.control_loop.shape_category import ShapeController
from freegsnke.control_loop.systems_category import SystemsController
from freegsnke.control_loop.useful_functions import (
    ConstantInterpolant,
    interpolate_spline,
    interpolate_step,
)
from freegsnke.control_loop.virtual_circuits_category import VirtualCircuitsController


def waveform(values):
    """Return a three-point waveform for controller test data."""

    return {"times": np.array([0.0, 1.0, 2.0]), "vals": np.asarray(values)}


def test_constant_interpolants_preserve_scalar_and_array_shapes():
    scalar = interpolate_spline(waveform([2.0, 2.0, 2.0]))
    matrix = interpolate_step(waveform([np.eye(2), np.eye(2), np.eye(2)]))

    assert isinstance(scalar, ConstantInterpolant)
    assert isinstance(matrix, ConstantInterpolant)
    np.testing.assert_array_equal(scalar(0.5), np.array(2.0))
    np.testing.assert_array_equal(scalar([0.5, 1.5]), np.array([2.0, 2.0]))
    np.testing.assert_array_equal(matrix(0.5), np.eye(2))
    np.testing.assert_array_equal(matrix([0.5, 1.5]), np.stack([np.eye(2)] * 2))
    np.testing.assert_array_equal(scalar.derivative()(0.5), np.array(0.0))


def test_shape_controller_reuses_spline_derivatives():
    data = {
        "shape": {
            "ff": waveform([0.0, 2.0, 2.0]),
            "ref": waveform([1.0, 1.0, 1.0]),
            "blend": waveform([1.0, 1.0, 1.0]),
            "k_prop": waveform([1.0, 1.0, 1.0]),
            "k_int": waveform([0.0, 0.0, 0.0]),
            "k_deriv": waveform([0.0, 0.0, 0.0]),
            "damping": waveform([1.0, 1.0, 1.0]),
        }
    }
    controller = ShapeController(data=data, ctrl_targets=["shape"])
    derivative = controller.interpolant_derivatives["shape"]["ff"]

    np.testing.assert_allclose(
        controller.extract_values(0.5, ["shape"], "ff", deriv=True), [2.0]
    )
    assert controller.interpolant_derivatives["shape"]["ff"] is derivative


def test_coil_controllers_reuse_spline_derivatives():
    limits = waveform([[-10.0], [-10.0], [-10.0]])
    systems = SystemsController(
        data={
            "coil_pert": waveform([0.0, 2.0, 2.0]),
            "min_coil_curr_lims": limits,
            "max_coil_curr_lims": waveform([[10.0], [10.0], [10.0]]),
            "max_coil_curr_ramp_lims": waveform([[5.0], [5.0], [5.0]]),
        },
        ctrl_coils=["coil"],
    )
    virtual_circuits = VirtualCircuitsController(
        data={
            "coil_order": ["coil"],
            "coil_ref": waveform([0.0, 2.0, 2.0]),
            "shape": waveform([[1.0], [1.0], [1.0]]),
            "plasma": waveform([[1.0], [1.0], [1.0]]),
        },
        ctrl_coils=["coil"],
        ctrl_targets=["shape"],
        plasma_target=["plasma"],
    )

    np.testing.assert_allclose(systems.extract_values(0.5, ["coil"], True), [2.0])
    np.testing.assert_allclose(
        virtual_circuits.extract_values(0.5, ["coil_ref"], True), [2.0]
    )
