"""Execute migration notebooks and compare retained numerical checkpoints.

Run the same tool with separate reference and candidate Python environments.
Original notebook cells are preserved; added cells seed random generators and
save numerical state after every source code cell, before later cells overwrite
histories. Neither execution errors nor numerical differences are suppressed.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NOTEBOOKS = ("example00", "example01a", "example01b", "example02", "example05a")
PRECISION_ULPS = 256
OBJECT_ATTRIBUTES = {
    "equilibrium": (
        "R",
        "Z",
        "plasma_psi",
        "tokamak_psi",
        "psi_axis",
        "psi_bndry",
        "opt",
        "xpt",
        "mask_inside_limiter",
        "mask_outside_limiter",
        "flag_limiter",
        "has_relevant_xpoint",
        "current_vec",
        "_current",
    ),
    "machine": (
        "coils_dict",
        "coils_list",
        "n_active_coils",
        "n_passive_coils",
        "coil_resist",
        "coil_self_ind",
        "wall",
        "limiter",
    ),
    "profile": (
        "jtor",
        "Ip",
        "paxis",
        "betap",
        "fvac",
        "alpha_m",
        "alpha_n",
        "Beta0",
        "alpha",
        "beta",
        "opt",
        "xpt",
        "psi_axis",
        "psi_bndry",
    ),
    "stepper": (
        "currents_vec",
        "vessel_currents_vec",
        "dt_step",
        "full_timestep",
        "plasma_norm_factor",
        "plasma_resistivity",
        "eq1",
        "eq2",
        "linearised_sol",
    ),
    "linear": ("growth_rates", "instability_timescale", "M", "R"),
    "solver": (
        "gs_operator_order",
        "rhs",
        "psi_boundary",
        "jtor_at_start",
        "relative_norm_residual",
        "relative_del_psit",
    ),
    "constraint": ("b", "A", "control_mask", "plasma_vals", "coil_vals"),
}


def numeric_state(namespace):
    """Extract numerical values without invoking physics or mutating objects."""
    result = {}

    def collect(key, value, depth=0):
        if depth > 8 or callable(value):
            return
        if isinstance(value, (bool, int, float, complex, np.number, np.ndarray)):
            array = np.asarray(value)
            if array.dtype.kind in "biufc" and array.size:
                result[key] = array
            return
        if isinstance(value, dict):
            for name, item in sorted(value.items(), key=lambda entry: str(entry[0])):
                collect(f"{key}/{name}", item, depth + 1)
            return
        if isinstance(value, (list, tuple)):
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                array = None
            if array is not None and array.dtype.kind in "biufc":
                collect(key, array, depth + 1)
            else:
                for index, item in enumerate(value):
                    collect(f"{key}/{index}", item, depth + 1)
            return
        attributes = vars(value) if hasattr(value, "__dict__") else {}
        if "plasma_psi" in attributes:
            kind = "equilibrium"
        elif "coils_dict" in attributes:
            kind = "machine"
        elif "currents_vec" in attributes and "eq1" in attributes:
            kind = "stepper"
        elif "instability_timescale" in attributes:
            kind = "linear"
        elif "gs_operator_order" in attributes:
            kind = "solver"
        elif "n_control_coils" in attributes:
            kind = "constraint"
        elif "Ip" in attributes or "jtor" in attributes:
            kind = "profile"
        elif "R" in attributes and "Z" in attributes:
            for name in ("R", "Z"):
                collect(f"{key}/{name}", attributes[name], depth + 1)
            return
        else:
            return
        for name in OBJECT_ATTRIBUTES[kind]:
            if name in attributes:
                collect(f"{key}/{name}", attributes[name], depth + 1)
        if kind == "machine":
            collect(f"{key}/currents", value.getCurrents(), depth + 1)

    for name, value in sorted(namespace.items()):
        if not name.startswith("_") and name not in {"t_per_step"}:
            collect(name, value)
    return result


def snapshot(namespace, directory, cell_index):
    """Persist full-precision content-addressed arrays and a cell manifest."""
    directory = Path(directory)
    arrays_dir = directory / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, array in numeric_state(namespace).items():
        array = np.ascontiguousarray(array) if array.ndim else array
        content = array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
        digest = hashlib.sha256(content).hexdigest()
        target = arrays_dir / f"{digest}.npy"
        if not target.exists():
            np.save(target, array, allow_pickle=False)
        manifest[name] = digest
    (directory / f"cell-{cell_index:03d}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def source_digest(notebook):
    """Normalize only the backend namespace for notebook provenance matching."""
    source = [
        (c.cell_type, c.source.replace("freegs4e", "freegsnke")) for c in notebook.cells
    ]
    return hashlib.sha256(json.dumps(source).encode()).hexdigest()


def tree_digest(repo):
    """Fingerprint tracked and new source files, including uncommitted changes."""
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=repo
        )
        .decode()
        .split("\0")
    )
    manifest = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in sorted(set(paths))
        if name and (repo / name).is_file()
    }
    return manifest


def run_notebooks(args):
    """Execute complete notebooks in fresh, explicitly selected kernels."""
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tool = Path(__file__).resolve()
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMEXPR_MAX_THREADS": "32",
            "PYTHONHASHSEED": "0",
            "MPLBACKEND": "Agg",
            "JUPYTER_RUNTIME_DIR": str(output / "runtime"),
            "MPLCONFIGDIR": str(output / "matplotlib"),
            "PYTHONPATH": str(repo),
        }
    )
    specs = output / "kernels"
    spec_dir = specs / "migration"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    str(args.python.absolute()),
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Migration validation",
                "language": "python",
            }
        )
    )
    for prefix in args.notebooks:
        matches = list((repo / "examples").glob(f"{prefix} - *.ipynb"))
        if len(matches) != 1:
            raise ValueError(f"Expected one notebook for {prefix}, found {matches}")
        source = nbformat.read(matches[0], as_version=4)
        directory = output / prefix
        if directory.exists():
            raise FileExistsError(
                f"Refusing to overwrite notebook evidence: {directory}"
            )
        directory.mkdir()
        metadata = {
            "notebook": matches[0].name,
            "source_digest": source_digest(source),
            "tool_digest": hashlib.sha256(tool.read_bytes()).hexdigest(),
            "repo_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "repo_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
            ).hexdigest(),
            "source_files": tree_digest(repo),
            "python": str(args.python.absolute()),
            "precision_ulps": PRECISION_ULPS,
            "environment": {
                k: environment[k]
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "NUMEXPR_MAX_THREADS",
                    "PYTHONHASHSEED",
                )
            },
        }
        setup = nbformat.v4.new_code_cell(
            "import random as _eav_random, numpy as _eav_np\n"
            "_eav_random.seed(42); _eav_np.random.seed(42)\n"
            "import sys as _eav_sys, json as _eav_json, importlib.metadata as _eav_metadata\n"
            f"_eav_sys.path.insert(0, {str(tool.parent)!r})\n"
            "from migration_notebooks import snapshot as _eav_snapshot\n"
            f"from pathlib import Path as _eav_Path\n_eav_written = _eav_Path({str(directory / 'packages.json')!r}).write_text(_eav_json.dumps({{d.metadata['Name']: d.version for d in _eav_metadata.distributions()}}, sort_keys=True, indent=2))\n"
        )
        executed = nbformat.v4.new_notebook(metadata=source.metadata, cells=[setup])
        for index, cell in enumerate(source.cells):
            executed.cells.append(cell)
            if cell.cell_type == "code" and cell.source.strip():
                executed.cells.append(
                    nbformat.v4.new_code_cell(
                        f"_eav_snapshot(get_ipython().user_ns, {str(directory)!r}, {index})",
                        metadata={"tags": ["migration-checkpoint"]},
                    )
                )
        client = NotebookClient(
            executed, kernel_name="migration", timeout=args.timeout, allow_errors=False
        )
        client.km = client.create_kernel_manager()
        client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(specs)])
        client.on_cell_start = lambda cell, cell_index: print(
            f"CELL {prefix} {cell_index}/{len(executed.cells)}", flush=True
        )
        started = time.monotonic()
        print(f"START {prefix}", flush=True)
        try:
            client.execute(cwd=str(repo / "examples"), env=environment)
            metadata["status"] = "passed"
        except BaseException:
            metadata["status"] = "failed"
            raise
        finally:
            metadata["elapsed_seconds"] = time.monotonic() - started
            nbformat.write(executed, directory / "executed.ipynb")
            (directory / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        print(f"PASS {prefix} ({metadata['elapsed_seconds']:.1f}s)", flush=True)


def compare_notebooks(args):
    """Require precision agreement at every recorded numerical checkpoint."""
    report = {"precision_ulps": PRECISION_ULPS, "notebooks": {}, "passed": True}
    for prefix in args.notebooks:
        ref_dir, got_dir = args.reference / prefix, args.candidate / prefix
        ref_meta = json.loads((ref_dir / "metadata.json").read_text())
        got_meta = json.loads((got_dir / "metadata.json").read_text())
        result = {
            "passed": True,
            "arrays_compared": 0,
            "failures": [],
            "max_relative_scale_error": 0.0,
        }
        if (
            ref_meta["status"] != "passed"
            or got_meta["status"] != "passed"
            or ref_meta["source_digest"] != got_meta["source_digest"]
        ):
            result["failures"].append("Execution status or normalized source mismatch")
        if (
            ref_meta["tool_digest"] != got_meta["tool_digest"]
            or ref_meta["environment"] != got_meta["environment"]
        ):
            result["failures"].append(
                "Validation tool or execution environment mismatch"
            )
        ref_packages = json.loads((ref_dir / "packages.json").read_text())
        got_packages = json.loads((got_dir / "packages.json").read_text())
        ref_packages.pop("freegs4e", None)
        got_packages.pop("freegs4e", None)
        if ref_packages != got_packages:
            result["failures"].append("Pinned dependency versions do not match")
        manifests = sorted(ref_dir.glob("cell-*.json"))
        if [p.name for p in manifests] != [
            p.name for p in sorted(got_dir.glob("cell-*.json"))
        ]:
            result["failures"].append("Checkpoint cell set mismatch")
        for manifest in manifests:
            ref = json.loads(manifest.read_text())
            got = json.loads((got_dir / manifest.name).read_text())
            if ref.keys() != got.keys():
                result["failures"].append(
                    {
                        "cell": manifest.name,
                        "missing": sorted(ref.keys() - got.keys()),
                        "extra": sorted(got.keys() - ref.keys()),
                    }
                )
            for name in sorted(ref.keys() & got.keys()):
                expected = np.load(
                    ref_dir / "arrays" / f"{ref[name]}.npy", allow_pickle=False
                )
                actual = np.load(
                    got_dir / "arrays" / f"{got[name]}.npy", allow_pickle=False
                )
                result["arrays_compared"] += 1
                if expected.shape != actual.shape or expected.dtype != actual.dtype:
                    result["failures"].append(
                        {
                            "cell": manifest.name,
                            "variable": name,
                            "error": "shape or dtype mismatch",
                        }
                    )
                    continue
                if expected.dtype.kind in "biu":
                    passed = np.array_equal(expected, actual)
                    error, scale = 0.0 if passed else 1.0, 1.0
                else:
                    finite = np.isfinite(expected) & np.isfinite(actual)
                    matching_special = np.array_equal(
                        np.isfinite(expected), np.isfinite(actual)
                    ) and np.array_equal(
                        expected[~finite], actual[~finite], equal_nan=True
                    )
                    error = (
                        float(np.max(np.abs(actual[finite] - expected[finite])))
                        if finite.any()
                        else 0.0
                    )
                    scale = (
                        float(np.max(np.abs(expected[finite]))) if finite.any() else 0.0
                    )
                    epsilon = np.finfo(expected.real.dtype).eps
                    passed = (
                        matching_special and error <= PRECISION_ULPS * epsilon * scale
                    )
                scaled_error = error / scale if scale else error
                result["max_relative_scale_error"] = max(
                    result["max_relative_scale_error"], scaled_error
                )
                if not passed:
                    result["failures"].append(
                        {
                            "cell": manifest.name,
                            "variable": name,
                            "max_absolute_error": error,
                            "reference_scale": scale,
                            "relative_scale_error": scaled_error,
                        }
                    )
        result["passed"] = not result["failures"]
        report["notebooks"][prefix] = result
        report["passed"] &= result["passed"]
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"passed": report["passed"], "report": str(args.report)}, indent=2)
    )
    if not report["passed"]:
        raise SystemExit(1)


def main():
    """Parse the execution or comparison command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--repo", type=Path, required=True)
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--timeout", type=int, default=3600)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    for command in (run, compare):
        command.add_argument(
            "--notebooks", nargs="+", choices=NOTEBOOKS, default=list(NOTEBOOKS)
        )
    args = parser.parse_args()
    (run_notebooks if args.command == "run" else compare_notebooks)(args)


if __name__ == "__main__":
    main()
