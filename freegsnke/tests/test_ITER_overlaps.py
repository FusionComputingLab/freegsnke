"""Check the ITER configuration for overlapping active and passive elements.

Run from the repository root with::

    python -m freegsnke.tests.test_ITER_overlaps

Or run as a pytest check::

    python -m pytest freegsnke/tests/test_ITER_overlaps.py

Each active winding is checked using its rectangular cross-section; passive
structures use their configured polygons or filament rectangles. Edge contact
is allowed, and intersection areas up to 1e-12 m^2 are ignored. The limiter and
wall are loaded as machine boundaries, not treated as solid coil elements.
Running as a script also displays all overlapping polygons on one R-Z plot.
"""

from pathlib import Path

import numpy as np

from freegsnke import build_machine

ITER_CONFIG_DIR = Path(__file__).resolve().parents[2] / "machine_configs" / "ITER"


def _load_iter_elements():
    """Load active and passive ITER descriptions with scalar dimensions."""
    active, passive, _, _ = build_machine.load_data_dicts(
        active_coils_path=ITER_CONFIG_DIR / "ITER_active_coils.pickle",
        passive_coils_path=ITER_CONFIG_DIR / "ITER_passive_coils.pickle",
        limiter_path=ITER_CONFIG_DIR / "ITER_limiter.pickle",
        wall_path=ITER_CONFIG_DIR / "ITER_wall.pickle",
    )
    # ITER stores some dimensions as one-element arrays. The overlap helper
    # expects scalar dimensions (float(array) also fails with NumPy 2.4+).
    for circuit in active.values():
        elements = [circuit] if "R" in circuit else circuit.values()
        for element in elements:
            for dimension in ("dR", "dZ"):
                element[dimension] = np.asarray(element[dimension]).item()
    return active, passive


def find_iter_overlaps():
    """Return every overlapping element pair in the checked-in ITER geometry."""
    active, passive = _load_iter_elements()
    # Check the descriptions directly: tokamak() rejects overlapping geometry
    # before it can return a machine, and passive refinement is unnecessary here.
    return build_machine.check_for_overlapping_elements(active, passive)


def print_iter_overlap_coordinates():
    """Print each overlapping pair's cross-section vertices as (R, Z) in metres.

    Active windings and passive filaments are shown as rectangle corners;
    extended passive structures are shown using their polygon vertices.
    """
    active, passive = _load_iter_elements()
    overlaps = build_machine.check_for_overlapping_elements(active, passive)
    shapes = dict(
        [
            *build_machine._active_element_shapes(active),
            *build_machine._passive_element_shapes(passive),
        ]
    )
    if not overlaps:
        print("No overlapping ITER elements found.")
        return

    for first, second in overlaps:
        print(f"{first} <-> {second} (R, Z in metres):")
        for name in (first, second):
            print(f"  {name}:")
            # Omit the closing vertex, which repeats the first corner.
            for r, z in list(shapes[name].exterior.coords)[:-1]:
                print(f"    ({r:.17g}, {z:.17g})")


def test_iter_elements_do_not_overlap():
    """Fail with all offending labels when the ITER elements overlap."""
    overlaps = find_iter_overlaps()
    assert not overlaps, "Overlapping ITER elements:\n" + "\n".join(
        f"  {first} <-> {second}" for first, second in overlaps
    )


def plot_iter_overlaps(save_path=None, show=True):
    """Plot all overlapping elements together in the (R, Z) poloidal plane.

    The first polygon of each pair is solid and the second is dashed, with
    both drawn in black. Labels are placed inside the polygons. Optionally
    save the figure to ``save_path``; use ``show=False`` for noninteractive use.
    Returns the Matplotlib figure and axes for further adjustment.
    """
    return _plot_iter_overlap_pairs(find_iter_overlaps(), save_path, show)


def plot_iter_overlap(overlaps, index, save_path=None, show=True):
    """Plot only ``overlaps[index]``, using a zero-based pair index.

    Example::

        overlaps = find_iter_overlaps()
        fig, ax = plot_iter_overlap(overlaps, index=0)

    The view fits the selected pair. Solid/dashed outlines follow the pair's
    order. Returns the figure and axes; optional saving works as in
    :func:`plot_iter_overlaps`.
    """
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer.")
    if not 0 <= index < len(overlaps):
        raise IndexError(
            f"Overlap index {index} is out of range for {len(overlaps)} pairs."
        )
    first, second = overlaps[index]
    return _plot_iter_overlap_pairs(
        [(first, second)],
        save_path,
        show,
        title=f"ITER overlap [{index}]: {first} / {second}",
    )


def _plot_iter_overlap_pairs(overlaps, save_path, show, title=None):
    """Draw the supplied overlapping pairs using shared polygon styling."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    active, passive = _load_iter_elements()
    shapes = dict(
        [
            *build_machine._active_element_shapes(active),
            *build_machine._passive_element_shapes(passive),
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 12), layout="constrained")
    labelled = set()
    for first, second in overlaps:
        for name, linestyle in ((first, "-"), (second, "--")):
            shape = shapes[name]
            vertices = np.asarray(shape.exterior.coords)
            ax.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.3,
                    linestyle=linestyle,
                )
            )
            if name not in labelled:
                # Align the text with the longest edge to fit narrow elements.
                edges = np.diff(vertices, axis=0)
                edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
                angle = np.degrees(np.arctan2(edge[1], edge[0]))
                angle = (angle + 90) % 180 - 90
                point = shape.representative_point()
                ax.text(
                    point.x,
                    point.y,
                    name,
                    ha="center",
                    va="center",
                    fontsize=7,
                    rotation=angle,
                    rotation_mode="anchor",
                )
                labelled.add(name)

    ax.autoscale_view()
    ax.margins(0.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title or f"ITER overlapping elements ({len(overlaps)} pairs)")
    ax.grid(True, alpha=0.2)
    if overlaps:
        ax.legend(
            handles=[
                Line2D([], [], color="black", linestyle="-", label="First in pair"),
                Line2D([], [], color="black", linestyle="--", label="Second in pair"),
            ],
            loc="upper right",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No overlapping elements",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def main():
    """Print overlapping pairs and return a nonzero status if any are found."""
    overlaps = find_iter_overlaps()
    if not overlaps:
        print("No overlapping ITER elements found.")
        return 0

    print(f"Found {len(overlaps)} overlapping ITER element pair(s):")
    for first, second in overlaps:
        print(f"  {first} <-> {second}")
    print_iter_overlap_coordinates()
    plot_iter_overlaps()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
