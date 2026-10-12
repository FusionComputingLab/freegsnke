"""Check that notebook evidence survives overwrites and rejects numerical drift."""

import json
from argparse import Namespace
from types import SimpleNamespace

import nbformat
import numpy as np
import pytest
from migration_notebooks import compare_notebooks, snapshot, source_digest


def test_snapshots_preserve_histories_before_later_scenarios_overwrite_them(tmp_path):
    flux = np.arange(12.0).reshape(3, 4)
    state = {"history_equilibria": [SimpleNamespace(plasma_psi=flux)]}
    snapshot(state, tmp_path, 1)
    first = json.loads((tmp_path / "cell-001.json").read_text())
    flux[:] = -100
    state["history_equilibria"] = [SimpleNamespace(plasma_psi=np.ones((3, 4)))]
    snapshot(state, tmp_path, 2)
    second = json.loads((tmp_path / "cell-002.json").read_text())
    key = "history_equilibria/0/plasma_psi"
    assert first[key] != second[key]
    retained = np.load(tmp_path / "arrays" / (first[key] + ".npy"))
    np.testing.assert_array_equal(retained, np.arange(12.0).reshape(3, 4))


def prepare_comparison(tmp_path, candidate):
    """Build a small pair of independently retained notebook result sets."""
    for name, values in (("reference", [1.0, 2.0]), ("candidate", candidate)):
        directory = tmp_path / name / "example00"
        snapshot({"flux": np.asarray(values)}, directory, 1)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "source_digest": "same-source",
                    "tool_digest": "same-tool",
                    "environment": {"threads": "1"},
                }
            )
        )
        (directory / "packages.json").write_text(json.dumps({"numpy": "same"}))
    return Namespace(
        reference=tmp_path / "reference",
        candidate=tmp_path / "candidate",
        report=tmp_path / "comparison.json",
        notebooks=["example00"],
    )


def test_comparison_accepts_roundoff_and_reports_larger_numerical_drift(tmp_path):
    args = prepare_comparison(tmp_path, [1.0, np.nextafter(2.0, 3.0)])
    compare_notebooks(args)
    assert json.loads(args.report.read_text())["passed"]
    directory = args.candidate / "example00"
    snapshot({"flux": np.array([1.0, 2.0 + 1e-9])}, directory, 1)
    with pytest.raises(SystemExit, match="1"):
        compare_notebooks(args)
    report = json.loads(args.report.read_text())
    assert not report["passed"]
    failure = report["notebooks"]["example00"]["failures"][0]
    assert failure["variable"] == "flux"
    assert failure["max_absolute_error"] > 1e-10


def test_comparison_rejects_changed_workloads_even_when_numbers_agree(tmp_path):
    args = prepare_comparison(tmp_path, [1.0, 2.0])
    path = args.candidate / "example00" / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["source_digest"] = "shortened-workload"
    path.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit, match="1"):
        compare_notebooks(args)
    assert not json.loads(args.report.read_text())["passed"]


def test_source_digest_allows_prose_updates_but_requires_exact_code_cells():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("Original explanation"),
            nbformat.v4.new_code_cell("import freegsnke\nsteps = 50"),
        ]
    )
    original = source_digest(notebook)
    notebook.cells[0].source = "Updated standalone package explanation"
    assert source_digest(notebook) == original
    notebook.cells[1].source = "import freegsnke\nsteps = 5"
    assert source_digest(notebook) != original
    notebook.cells[1].source = "import another_package\nsteps = 50"
    assert source_digest(notebook) != original


def test_comparison_rejects_any_dependency_mismatch(tmp_path):
    args = prepare_comparison(tmp_path, [1.0, 2.0])
    packages = args.candidate / "example00" / "packages.json"
    packages.write_text(json.dumps({"numpy": "same", "unexpected-package": "1.0"}))
    with pytest.raises(SystemExit, match="1"):
        compare_notebooks(args)
    failures = json.loads(args.report.read_text())["notebooks"]["example00"]["failures"]
    assert "Pinned dependency versions do not match" in failures
