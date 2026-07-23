import numpy as np

from freegsnke.control_loop.plasma_category import PlasmaController
from freegsnke.control_loop.shape_category import ShapeController


def _waveform(value):
    return {"times": np.array([0.0, 1.0]), "vals": np.array([value, value])}


def test_plasma_controller_defaults_missing_derivative_gain_to_zero():
    data = {
        "ip_ref": _waveform(1.0),
        "ip_blend": _waveform(1.0),
        "vloop_ff": _waveform(0.0),
        "k_prop": _waveform(0.0),
        "k_int": _waveform(0.0),
        "M_solenoid": _waveform(1.0),
    }

    controller = PlasmaController(data)

    assert "k_deriv" in controller.data
    assert controller.interpolants["k_deriv"](-1.0) == 0.0
    assert controller.interpolants["k_deriv"](0.0) == 0.0
    assert controller.interpolants["k_deriv"](1.0) == 0.0
    assert "k_deriv" not in data


def test_shape_controller_defaults_missing_derivative_gain_to_zero():
    data = {
        "shape_target": {
            "ff": {"times": np.array([0.0, 1.0]), "vals": np.array([0.0, 0.0])},
            "ref": {"times": np.array([0.0, 1.0]), "vals": np.array([0.0, 0.0])},
            "blend": {"times": np.array([0.0, 1.0]), "vals": np.array([1.0, 1.0])},
            "k_prop": _waveform(0.0),
            "k_int": _waveform(0.0),
            "damping": _waveform(1.0),
        }
    }

    controller = ShapeController(data=data, ctrl_targets=["shape_target"])

    assert "k_deriv" in controller.data["shape_target"]
    assert controller.interpolants["shape_target"]["k_deriv"](-1.0) == 0.0
    assert controller.interpolants["shape_target"]["k_deriv"](0.0) == 0.0
    assert controller.interpolants["shape_target"]["k_deriv"](1.0) == 0.0
    assert "k_deriv" not in data["shape_target"]
