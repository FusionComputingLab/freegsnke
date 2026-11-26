"""
Prepare an approximately up-down symmetric machine for symmetric evolution.

The routines in this module retain the electrically even part of a
user-provided machine description as the external current basis. They identify
reflected element pairs, report odd active circuits, average accepted pairs
onto an exactly reflected geometry, and provide explicit current transforms.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and
README files.

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from shapely.affinity import scale
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

_SIDE_PATTERN = re.compile(r"(^|_)(upper|lower)(?=_|$)", re.IGNORECASE)
_TRAILING_INDEX_PATTERN = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class ElementPair:
    """Record one upper/lower pairing and its pre-symmetrisation mismatch."""

    upper_name: str
    lower_name: str
    group: str
    reflected_rms: float
    matching_numeric_suffix: bool


@dataclass(frozen=True)
class GeometryDiscrepancy:
    """Record one pre-symmetrisation reflected-geometry discrepancy."""

    component: str
    upper_name: str
    lower_name: str
    reflected_rms: float


@dataclass
class PreparedUpDownMachine:
    """
    Store symmetric machine descriptions, pairings, and current transforms.

    ``active_coils_data`` preserves every retained original active-current
    label. Electrically odd source circuits are rejected by default or recorded
    in ``excluded_odd_active_names`` when explicitly omitted.
    ``even_active_coils_data`` combines each upper/lower pair into one series
    circuit for the existing strict symmetric evolutive-solver path.
    ``passive_coils_data`` preserves every original passive label.
    """

    active_coils_data: dict
    even_active_coils_data: dict
    passive_coils_data: list
    limiter_data: list
    wall_data: list
    original_active_names: tuple
    even_active_names: tuple
    passive_names: tuple
    active_pairs: tuple
    passive_pairs: tuple
    self_symmetric_active_names: tuple
    self_symmetric_passive_names: tuple
    source_z_midplane: float
    z_shift: float
    midplane_fit_rms: float
    midplane_fit_samples: int
    geometry_discrepancies: tuple
    excluded_odd_active_names: tuple

    def largest_geometry_discrepancies(
        self, count: int | None = None
    ) -> tuple[GeometryDiscrepancy, ...]:
        """Return pre-symmetrisation discrepancies in descending RMS order."""
        records = tuple(
            sorted(
                self.geometry_discrepancies,
                key=lambda record: record.reflected_rms,
                reverse=True,
            )
        )
        if count is None:
            return records
        if count < 0:
            raise ValueError("'count' must be non-negative or None.")
        return records[:count]

    @property
    def maximum_geometry_discrepancy(self) -> GeometryDiscrepancy | None:
        """Return the largest pre-symmetrisation discrepancy, or None."""
        largest = self.largest_geometry_discrepancies(count=1)
        return None if not largest else largest[0]

    def check_geometry_tolerance(self, max_reflected_rms: float) -> None:
        """Raise if any component exceeds a reflected RMS tolerance in metres."""
        max_reflected_rms = float(max_reflected_rms)
        if max_reflected_rms < 0:
            raise ValueError("'max_reflected_rms' must be non-negative.")
        failures = [
            record
            for record in self.largest_geometry_discrepancies()
            if record.reflected_rms > max_reflected_rms
        ]
        if not failures:
            return
        summary = "; ".join(
            f"{record.component} '{record.upper_name}' / "
            f"'{record.lower_name}': {record.reflected_rms:.4g} m"
            for record in failures[:5]
        )
        if len(failures) > 5:
            summary += f"; and {len(failures) - 5} more"
        raise ValueError(
            "Machine geometry exceeds the permitted reflected RMS mismatch "
            f"of {max_reflected_rms:.4g} m. Largest failures: {summary}."
        )

    @property
    def active_reflection_operator(self) -> np.ndarray:
        """Return the reflection permutation in the original active basis."""
        return _reflection_operator(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )

    @property
    def passive_reflection_operator(self) -> np.ndarray:
        """Return the reflection permutation in the original passive basis."""
        return _reflection_operator(
            self.passive_names,
            self.passive_pairs,
            self.self_symmetric_passive_names,
        )

    @property
    def reflection_operator(self) -> np.ndarray:
        """Return the block reflection permutation for all metal currents."""
        active = self.active_reflection_operator
        passive = self.passive_reflection_operator
        # Active and passive currents occupy separate contiguous blocks in the
        # machine-current vector, so reflection cannot mix the two populations.
        reflection = np.zeros(
            (
                len(self.original_active_names) + len(self.passive_names),
                len(self.original_active_names) + len(self.passive_names),
            )
        )
        reflection[: len(active), : len(active)] = active
        reflection[len(active) :, len(active) :] = passive
        return reflection

    @property
    def even_machine_reflection_operator(self) -> np.ndarray:
        """
        Return reflection in the reduced machine basis used for even evolution.

        Every reduced active circuit is self-symmetric, while passive elements
        retain their original labels and are exchanged with their partners.
        """
        active_count = len(self.even_active_names)
        passive = self.passive_reflection_operator
        reflection = np.eye(active_count + len(passive))
        reflection[active_count:, active_count:] = passive
        return reflection

    @property
    def active_original_to_even(self) -> np.ndarray:
        """
        Return the map from original active currents to pair-average currents.

        The reduced even current of an upper/lower pair is
        ``(I_upper + I_lower) / 2``. A self-symmetric circuit is unchanged.
        """
        forward, _, _, _, _ = _parity_transforms(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )
        return forward

    @property
    def active_original_to_odd(self) -> np.ndarray:
        """
        Return the map from original active currents to pair-difference currents.

        The odd current of an upper/lower pair is
        ``(I_upper - I_lower) / 2``.
        """
        _, odd_forward, _, _, _ = _parity_transforms(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )
        return odd_forward

    @property
    def active_even_to_original(self) -> np.ndarray:
        """Return the map from reduced even currents to original active currents."""
        _, _, backward, _, _ = _parity_transforms(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )
        return backward

    @property
    def active_odd_to_original(self) -> np.ndarray:
        """Return the map from odd pair currents to original active currents."""
        _, _, _, odd_backward, _ = _parity_transforms(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )
        return odd_backward

    @property
    def active_odd_names(self) -> tuple[str, ...]:
        """Return labels for the active odd-current coordinates."""
        _, _, _, _, odd_names = _parity_transforms(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )
        return odd_names

    def split_active_currents(
        self, currents: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split original active currents into even and odd coordinates.

        Parameters
        ----------
        currents : array-like
            Active currents in ``original_active_names`` order.

        Returns
        -------
        even_currents, odd_currents : ndarray
            Pair-average and pair-difference currents. Both retain ampere units.
        """
        currents = np.asarray(currents)
        if currents.shape[0] != len(self.original_active_names):
            raise ValueError(
                "Active current vector is incompatible with original_active_names."
            )
        even, odd, _, _, _ = _parity_transforms(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )
        return even @ currents, odd @ currents

    def combine_active_currents(
        self,
        even_currents: npt.ArrayLike,
        odd_currents: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Reconstruct original active currents from even and odd coordinates.

        Omitting ``odd_currents`` returns the exactly even projection.
        """
        even_currents = np.asarray(even_currents)
        if even_currents.shape[0] != len(self.even_active_names):
            raise ValueError(
                "Even current vector is incompatible with even_active_names."
            )
        _, _, even, odd, odd_names = _parity_transforms(
            self.original_active_names,
            self.active_pairs,
            self.self_symmetric_active_names,
        )
        currents = even @ even_currents
        if odd_currents is not None:
            odd_currents = np.asarray(odd_currents)
            if odd_currents.shape[0] != len(odd_names):
                raise ValueError(
                    "Odd current vector is incompatible with active_odd_names."
                )
            currents = currents + odd @ odd_currents
        return currents

    def project_metal_currents_even(self, currents: npt.ArrayLike) -> np.ndarray:
        """Project all original-basis metal currents onto the even subspace."""
        currents = np.asarray(currents)
        if currents.shape[0] != self.reflection_operator.shape[0]:
            raise ValueError("Metal current vector has the wrong length.")
        # For an involution R, (I + R) / 2 is the projector onto vectors that
        # are unchanged by reflection. This removes the odd current component.
        return 0.5 * (currents + self.reflection_operator @ currents)

    def symmetrise_square_operator(self, operator: npt.ArrayLike) -> np.ndarray:
        """
        Average a full metal-current operator with its reflected counterpart.

        This is useful for auditing operators generated before the geometric
        description itself was made exactly symmetric.
        """
        return _symmetrise_square_operator(operator, self.reflection_operator)

    def symmetrise_even_machine_resistances(
        self, resistances: npt.ArrayLike
    ) -> np.ndarray:
        """Reflection-average resistances in the reduced even-machine basis."""
        resistances = np.asarray(resistances)
        reflection = self.even_machine_reflection_operator
        if resistances.shape != (len(reflection),):
            raise ValueError("Resistance vector is incompatible with the machine.")
        return 0.5 * (resistances + reflection @ resistances)

    def symmetrise_even_machine_square_operator(
        self, operator: npt.ArrayLike
    ) -> np.ndarray:
        """Reflection-average an operator in the reduced even-machine basis."""
        return _symmetrise_square_operator(
            operator, self.even_machine_reflection_operator
        )

    def symmetrise_greens(self, greens: npt.ArrayLike, z_axis: int = -1) -> np.ndarray:
        """
        Average coil Green functions with their reflected partners.

        Parameters
        ----------
        greens : ndarray
            Green functions with the original metal-current index on axis zero.
        z_axis : int, default=-1
            Axis corresponding to an exactly symmetric Z grid.
        """
        return _symmetrise_greens(greens, self.reflection_operator, z_axis)

    def symmetrise_even_machine_greens(
        self, greens: npt.ArrayLike, z_axis: int = -1
    ) -> np.ndarray:
        """Reflection-average Green functions in the reduced machine basis."""
        return _symmetrise_greens(greens, self.even_machine_reflection_operator, z_axis)


def _symmetrise_square_operator(
    operator: npt.ArrayLike, reflection: np.ndarray
) -> np.ndarray:
    """Average a metal operator with its reflected input/output coordinates."""
    operator = np.asarray(operator)
    if operator.shape != reflection.shape:
        raise ValueError("Operator shape is incompatible with the machine.")
    return 0.5 * (operator + reflection @ operator @ reflection)


def _symmetrise_greens(
    greens: npt.ArrayLike, reflection: np.ndarray, z_axis: int = -1
) -> np.ndarray:
    """Reflect both the observation grid and the source-current coordinates."""
    greens = np.asarray(greens)
    if greens.shape[0] != len(reflection):
        raise ValueError("Green-function coil axis has the wrong length.")
    reflected = np.einsum("ij,j...->i...", reflection, np.flip(greens, axis=z_axis))
    return 0.5 * (greens + reflected)


def prepare_up_down_symmetric_machine(
    active_coils_data: dict[str, Any],
    passive_coils_data: Sequence[dict[str, Any]] | None = None,
    limiter_data: Sequence[dict[str, float]] | None = None,
    wall_data: Sequence[dict[str, float]] | None = None,
    z_midplane: float | str = 0.0,
    active_pairs: Sequence[tuple[str, str]] | None = None,
    passive_pairs: Sequence[tuple[str, str]] | None = None,
    max_pair_mismatch: float | None = None,
    exclude_odd_active: bool = False,
) -> PreparedUpDownMachine:
    """
    Identify pairs and construct an exactly up-down symmetric description.

    Pairing is performed independently within structural groups. Names and
    ``element`` metadata are used only to define candidate groups; reflected
    geometry determines the final one-to-one assignment. Optional explicit
    pair lists can be supplied for descriptions whose metadata are ambiguous.

    Parameters
    ----------
    active_coils_data : dict
        Standard FreeGSNKE active-coil description.
    passive_coils_data : list, optional
        Standard FreeGSNKE passive-structure description.
        Outputs carry private refinement-pair metadata so machine construction
        uses reflected filament samples rather than independent meshes.
    limiter_data, wall_data : list, optional
        Standard FreeGSNKE boundary descriptions. Each outline is intersected
        with its reflection, with a warning and comparison plot if its footprint
        changes.
    z_midplane : float or "auto", default=0
        Symmetry-plane coordinate. ``"auto"`` fits the common vertical offset
        from matched machine points, shifts the complete description so that
        the fitted plane is at ``Z=0``, and then performs exact pair averaging.
    active_pairs, passive_pairs : sequence of pairs, optional
        Explicit ``(upper_name, lower_name)`` pairings. When omitted, pairings
        are inferred from names/metadata and reflected geometry.
    max_pair_mismatch : float, optional
        Maximum permitted reflected RMS point mismatch in metres before
        averaging. The limit applies to active and passive pairs, internally
        symmetric elements, limiter, and wall. For boundary outlines this is the
        RMS distance of source vertices to the reflected boundary, without pairing.
    exclude_odd_active : bool, default=False
        If True, omit active circuits whose reflected winding polarity is odd
        from the prepared machine and record them in
        ``excluded_odd_active_names``. By default their presence raises with a
        complete list, requiring explicit user consent before exclusion.

    Returns
    -------
    PreparedUpDownMachine
        Symmetric full and even-active descriptions, reflection maps, current
        transforms, and auditable pair diagnostics.
    """
    # Work on private copies throughout: callers often reuse the original
    # machine description to compare full and symmetry-reduced calculations.
    passive_coils_data = [] if passive_coils_data is None else passive_coils_data
    active_original = deepcopy(active_coils_data)
    passive_original = deepcopy(passive_coils_data)
    limiter_original = None if limiter_data is None else deepcopy(limiter_data)
    wall_original = None if wall_data is None else deepcopy(wall_data)
    for coil in active_original.values():
        for _, part in _active_parts(coil):
            _normalise_coordinates(part)
    for element in passive_original:
        _normalise_coordinates(element)

    active_names = tuple(active_original)
    passive_names = tuple(
        entry.get("name", f"passive_{index}")
        for index, entry in enumerate(passive_original)
    )
    # Passive entries arrive as a list, unlike the name-keyed active mapping.
    # Give unnamed entries stable labels so every later pairing is auditable.
    _check_unique(passive_names, "passive")

    if isinstance(z_midplane, str):
        if z_midplane.lower() != "auto":
            raise ValueError("'z_midplane' must be a float or 'auto'.")
        # Automatic recentring is deliberately a two-pass procedure. A rough
        # plane is sufficient to classify upper/lower candidates, after which
        # matched points provide a much better estimate of the common plane.
        preliminary_midplane = _geometry_bounds_midplane(
            active_original, passive_original
        )
        preliminary_active_pairs, preliminary_active_self = _identify_active_pairs(
            active_original, active_pairs, preliminary_midplane
        )
        preliminary_passive_pairs, preliminary_passive_self = _identify_passive_pairs(
            passive_original,
            passive_names,
            passive_pairs,
            preliminary_midplane,
        )
        source_z_midplane, midplane_fit_rms, midplane_fit_samples = (
            _fit_source_midplane(
                active_original,
                passive_original,
                passive_names,
                preliminary_active_pairs,
                preliminary_active_self,
                preliminary_passive_pairs,
                preliminary_passive_self,
                preliminary_midplane,
            )
        )
        # The reduced solver assumes that reflection is Z -> -Z. Translate all
        # supplied geometry together so the fitted source plane becomes zero.
        z_shift = -source_z_midplane
        target_midplane = 0.0
        _shift_active_geometry(active_original, z_shift)
        _shift_flat_geometry(passive_original, z_shift)
        _shift_flat_geometry(limiter_original, z_shift)
        _shift_flat_geometry(wall_original, z_shift)
    else:
        source_z_midplane = float(z_midplane)
        target_midplane = source_z_midplane
        z_shift = 0.0
        midplane_fit_rms = np.nan
        midplane_fit_samples = 0

    # Pair again after any shift. These are the definitive pairings used to
    # construct geometry, diagnostics, and current-coordinate transforms.
    active_pair_names, active_self = _identify_active_pairs(
        active_original, active_pairs, target_midplane
    )
    passive_pair_names, passive_self = _identify_passive_pairs(
        passive_original, passive_names, passive_pairs, target_midplane
    )

    # Geometric reflection is not enough for active circuits: winding polarity
    # determines whether equal source current produces an even or odd field.
    active_parity = {
        (upper_name, lower_name): _paired_active_parity(
            active_original[upper_name],
            active_original[lower_name],
            target_midplane,
        )
        for upper_name, lower_name, _ in active_pair_names
    }
    active_parity.update(
        {
            (name,): _self_active_parity(active_original[name], target_midplane)
            for name in active_self
        }
    )
    incompatible_active = tuple(
        name for names, parity in active_parity.items() if parity == 0 for name in names
    )
    if incompatible_active:
        raise ValueError(
            "Active circuits have incompatible reflected winding magnitudes or "
            f"mixed parity: {incompatible_active}."
        )
    excluded_odd_active = tuple(
        name for names, parity in active_parity.items() if parity < 0 for name in names
    )
    if excluded_odd_active and not exclude_odd_active:
        raise ValueError(
            "Electrically odd active circuits cannot be included in strict "
            f"even evolution: {excluded_odd_active}. Pass "
            "'exclude_odd_active=True' to omit and record them explicitly."
        )
    if excluded_odd_active:
        # Strictly even evolution has no coordinate for an odd source circuit.
        # Remove it before generating either the full symmetric description or
        # the reduced series-circuit basis, while retaining its name in output.
        excluded = set(excluded_odd_active)
        for name in excluded:
            active_original.pop(name)
        active_names = tuple(active_original)
        active_pair_names = [
            pair
            for pair in active_pair_names
            if pair[0] not in excluded and pair[1] not in excluded
        ]
        active_self = [name for name in active_self if name not in excluded]

    # First make the retained active geometry exactly reflected. Mismatch is
    # measured before averaging so the approximation remains quantitatively
    # visible to the caller.
    symmetric_active = active_original
    active_diagnostics = []
    geometry_discrepancies = []
    for upper_name, lower_name, group in active_pair_names:
        upper, lower, mismatch = _symmetrise_active_pair(
            symmetric_active[upper_name],
            symmetric_active[lower_name],
            target_midplane,
        )
        symmetric_active[upper_name] = upper
        symmetric_active[lower_name] = lower
        active_diagnostics.append(
            _make_diagnostic(upper_name, lower_name, group, mismatch)
        )
        geometry_discrepancies.append(
            GeometryDiscrepancy("active", upper_name, lower_name, mismatch)
        )

    # An unpaired circuit is retained only when it spans the midplane and can
    # therefore be symmetrised internally (for example, a central solenoid).
    for name in active_self:
        mismatch = _self_active_mismatch(symmetric_active[name], target_midplane)
        symmetric_active[name] = _symmetrise_self_active(
            symmetric_active[name], target_midplane
        )
        geometry_discrepancies.append(
            GeometryDiscrepancy("active", name, name, mismatch)
        )

    # Passive elements retain their original one-element-per-current basis.
    # Only their geometry is averaged here; reflected currents remain separate.
    passive_by_name = dict(zip(passive_names, passive_original))
    passive_diagnostics = []
    for upper_name, lower_name, group in passive_pair_names:
        upper, lower, mismatch = _symmetrise_element_pair(
            passive_by_name[upper_name],
            passive_by_name[lower_name],
            target_midplane,
            polygon=True,
        )
        passive_by_name[upper_name] = upper
        passive_by_name[lower_name] = lower
        lower["_up_down_refinement"] = (upper_name, target_midplane)
        passive_diagnostics.append(
            _make_diagnostic(upper_name, lower_name, group, mismatch)
        )
        geometry_discrepancies.append(
            GeometryDiscrepancy("passive", upper_name, lower_name, mismatch)
        )

    for name in passive_self:
        mismatch = _self_element_mismatch(
            passive_by_name[name], target_midplane, polygon=True
        )
        passive_by_name[name] = _symmetrise_self_element(
            passive_by_name[name], target_midplane, polygon=True
        )
        passive_by_name[name]["_up_down_refinement"] = (name, target_midplane)
        geometry_discrepancies.append(
            GeometryDiscrepancy("passive", name, name, mismatch)
        )

    # Restore the original passive ordering because downstream inductance and
    # resistance arrays use positional, rather than name-based, indexing.
    symmetric_passive = [passive_by_name[name] for name in passive_names]

    # Both boundary outlines retain only the region inside their reflections.
    if limiter_original is None:
        symmetric_limiter = None
    else:
        symmetric_limiter, mismatch = _symmetrise_boundary_with_mismatch(
            limiter_original, target_midplane
        )
        geometry_discrepancies.append(
            GeometryDiscrepancy("limiter", "upper path", "lower path", mismatch)
        )
    if wall_original is None:
        symmetric_wall = None
    else:
        symmetric_wall, mismatch = _symmetrise_boundary_with_mismatch(
            wall_original, target_midplane
        )
        geometry_discrepancies.append(
            GeometryDiscrepancy("wall", "upper path", "lower path", mismatch)
        )
    # The strict symmetric evolutive solver uses one active coordinate per
    # reflected pair. Connect each pair as a single series-circuit description.
    even_active, even_names = _build_even_active_description(
        symmetric_active, active_pair_names, active_self
    )

    # Package both geometries and every ordering/transform needed to move
    # currents and operators between the original and reduced representations.
    prepared = PreparedUpDownMachine(
        active_coils_data=symmetric_active,
        even_active_coils_data=even_active,
        passive_coils_data=symmetric_passive,
        limiter_data=symmetric_limiter,
        wall_data=symmetric_wall,
        original_active_names=active_names,
        even_active_names=tuple(even_names),
        passive_names=passive_names,
        active_pairs=tuple(active_diagnostics),
        passive_pairs=tuple(passive_diagnostics),
        self_symmetric_active_names=tuple(active_self),
        self_symmetric_passive_names=tuple(passive_self),
        source_z_midplane=source_z_midplane,
        z_shift=z_shift,
        midplane_fit_rms=midplane_fit_rms,
        midplane_fit_samples=midplane_fit_samples,
        geometry_discrepancies=tuple(geometry_discrepancies),
        excluded_odd_active_names=excluded_odd_active,
    )
    if max_pair_mismatch is not None:
        prepared.check_geometry_tolerance(max_pair_mismatch)
    for original, symmetric, name in (
        (limiter_original, symmetric_limiter, "Limiter"),
        (wall_original, symmetric_wall, "Wall"),
    ):
        if original is not None:
            _warn_and_plot_boundary_change(original, symmetric, target_midplane, name)
    return prepared


def symmetrise_limiter(
    limiter_data: Sequence[dict[str, float]], z_midplane: float = 0.0
) -> list[dict[str, float]]:
    """
    Keep the intersection of a limiter and its reflection about the midplane.

    Retain existing boundary segments and add vertices only at edge crossings.
    Unequal upper/lower vertex counts require no pairing or resampling. An
    already symmetric polygon is returned unchanged. Otherwise, warn and show
    the original, reflected and retained outlines in a nonblocking plot.

    Parameters
    ----------
    limiter_data : sequence of dict
        Ordered polygon vertices with ``R`` and ``Z`` entries.
    z_midplane : float, default=0
        Coordinate of the symmetry plane.

    Returns
    -------
    list of dict
        Symmetric inner boundary, preserving winding and explicit closure.

    Raises
    ------
    ValueError
        If the input is invalid or the intersection is empty, disconnected,
        or contains holes.
    """
    result, _ = _symmetrise_boundary_with_mismatch(limiter_data, z_midplane)
    _warn_and_plot_boundary_change(limiter_data, result, z_midplane, "Limiter")
    return result


def _symmetrise_boundary_with_mismatch(boundary_data, z_midplane):
    """Intersect reflected outlines and measure source-to-reflection RMS distance."""
    points = np.asarray([[p["R"], p["Z"]] for p in boundary_data], dtype=float)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("A boundary requires at least three finite vertices.")
    closed = np.allclose(points[0], points[-1], rtol=0.0, atol=1e-12)
    if closed:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("A boundary requires at least three distinct vertices.")
    source = Polygon(points)
    if not source.is_valid or source.area <= 0:
        raise ValueError("Invalid boundary polygon.")
    reflected = scale(source, xfact=1, yfact=-1, origin=(0, z_midplane))
    if source.equals(reflected):
        return deepcopy(boundary_data), 0.0
    inner = source.intersection(reflected)
    if inner.is_empty or inner.geom_type != "Polygon" or inner.area <= 0:
        raise ValueError("Boundary intersection must be one connected polygon.")
    if not inner.is_valid or len(inner.interiors):
        raise ValueError("Boundary intersection must be valid and without holes.")
    distances = [Point(point).distance(reflected.boundary) for point in points]
    mismatch = float(np.sqrt(np.mean(np.square(distances))))
    inner = orient(inner, sign=1 if source.exterior.is_ccw else -1)
    coords = list(inner.exterior.coords)
    if not closed:
        coords = coords[:-1]
    return [{"R": float(r), "Z": float(z)} for r, z in coords], mismatch


def _warn_and_plot_boundary_change(original, symmetric, z_midplane, name):
    """Warn about a changed footprint and display its original/reflected overlap."""
    points = np.asarray([[p["R"], p["Z"]] for p in original], dtype=float)
    retained = np.asarray([[p["R"], p["Z"]] for p in symmetric], dtype=float)
    source, inner = Polygon(points), Polygon(retained)
    if source.equals(inner):
        return
    removed_area = max(0.0, source.area - inner.area)
    removed_percent = 100 * removed_area / source.area
    warnings.warn(
        f"{name} changed by intersection with its reflection about "
        f"Z={z_midplane:g} m: removed {removed_area:.6g} m² "
        f"({removed_percent:.3g}%); {len(original)} -> {len(symmetric)} vertices. "
        "See the boundary comparison plot.",
        UserWarning,
        stacklevel=3,
    )
    reflected = points.copy()
    reflected[:, 1] = 2 * z_midplane - points[:, 1]
    # Close plotting paths even when the supplied polygon omits its closing vertex.
    paths = [np.vstack((p, p[0])) for p in (points, reflected, retained)]
    candidates = np.vstack((points, reflected))
    tolerance = 1e-12 * max(np.ptp(points, axis=0).max(), 1.0)
    crossings = retained[
        np.min(np.linalg.norm(retained[:, None] - candidates[None, :], axis=-1), axis=1)
        > tolerance
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout="constrained")
    for axis in axes:
        axis.fill(*paths[2].T, color="#00865A", alpha=0.15)
        axis.plot(*paths[0].T, color="#0072B2", label=f"Original {name.lower()}")
        axis.plot(*paths[1].T, "--", color="#D55E00", label=f"Reflected {name.lower()}")
        axis.plot(
            *paths[2].T, color="#00865A", linewidth=2, label="Retained intersection"
        )
        if len(crossings):
            axis.scatter(
                *crossings.T,
                marker="*",
                s=80,
                color="#A51C47",
                zorder=4,
                label="New edge crossings",
            )
        axis.set(xlabel="R [m]", ylabel="Z [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[0].axhline(z_midplane, color="grey", linestyle=":", linewidth=0.8)
    axes[0].set_title(f"Original and retained {name.lower()}")
    axes[0].legend(fontsize="small")
    height = max(np.max(np.abs(points[:, 1] - z_midplane)), 1e-12)
    axes[1].set_ylim(z_midplane - 0.05 * height, z_midplane + 1.1 * height)
    axes[1].set_title("Upper half: retained boundary segments")
    fig.suptitle(f"{name} symmetrisation: {removed_percent:.3g}% of area removed")
    plt.show(block=False)


def symmetrise_boundary(
    boundary_data: Sequence[dict[str, float]], z_midplane: float = 0.0
) -> list[dict[str, float]]:
    """
    Keep the intersection of a boundary and its reflection about the midplane.

    This is the common strategy for limiter and wall outlines. Retain existing
    segments and introduce vertices only at edge crossings, without pairing or
    resampling. Warn and show a comparison plot whenever the footprint changes.

    Parameters
    ----------
    boundary_data : sequence of dict
        Ordered closed-polygon vertices with ``R`` and ``Z`` entries.
    z_midplane : float, default=0
        Coordinate of the symmetry plane.

    Returns
    -------
    list of dict
        Symmetric inner boundary, preserving winding and explicit closure.

    Raises
    ------
    ValueError
        If the input is invalid or the intersection is empty, disconnected,
        or contains holes.
    """
    result, _ = _symmetrise_boundary_with_mismatch(boundary_data, z_midplane)
    _warn_and_plot_boundary_change(boundary_data, result, z_midplane, "Boundary")
    return result


def _point_rms_distance(first, second):
    """Return the root-mean-square Euclidean distance between paired points."""
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=-1))))


def _normalise_coordinates(element):
    """Keep one-dimensional, finite R/Z arrays in private geometry copies."""
    for key in ("R", "Z"):
        element[key] = np.atleast_1d(np.asarray(element[key], dtype=float))
    if element["R"].shape != element["Z"].shape or element["R"].ndim != 1:
        raise ValueError("R and Z must be matching one-dimensional coordinates.")
    if not element["R"].size or not np.isfinite([element["R"], element["Z"]]).all():
        raise ValueError("R and Z must contain finite coordinates.")


def _element_points(element, polygon=False):
    """Return unique polygon vertices or filament points and explicit closure."""
    points = np.column_stack((element["R"], element["Z"])).astype(float)
    closed = polygon and len(points) > 2 and np.array_equal(points[0], points[-1])
    return (points[:-1] if closed else points), closed


def _split_about_plane(values, z_midplane):
    """Partition coordinates into disjoint upper, lower and middle masks."""
    values = np.asarray(values)
    middle = np.isclose(values, z_midplane, rtol=0.0, atol=1e-8)
    return (values > z_midplane) & ~middle, (values < z_midplane) & ~middle, middle


def _active_sides(element, z_midplane):
    """Partition compound parts once for parity and geometry operations."""
    parts = _active_parts(element)
    masks = _split_about_plane([_centroid_z(part) for _, part in parts], z_midplane)
    return tuple([part for part, keep in zip(parts, mask) if keep] for mask in masks)


def _geometry_bounds_midplane(active_data, passive_data):
    """Estimate a preliminary midplane from the complete geometry bounds."""
    z_values = []
    for element in active_data.values():
        for _, part in _active_parts(element):
            z_values.extend(np.asarray(part["Z"], dtype=float))
    for element in passive_data:
        z_values.extend(np.asarray(element["Z"], dtype=float))
    if not z_values:
        raise ValueError("Cannot fit a midplane without machine geometry.")
    # This estimate is used only to classify candidates for the later matched-
    # point fit, so the midpoint of the global vertical extent is sufficient.
    return 0.5 * (np.min(z_values) + np.max(z_values))


def _shift_active_geometry(active_data, z_shift):
    """Apply a uniform vertical shift to an active-coil description in place."""
    for element in active_data.values():
        for _, part in _active_parts(element):
            part["Z"] = np.asarray(part["Z"], dtype=float) + z_shift


def _shift_flat_geometry(elements, z_shift):
    """Apply a uniform vertical shift to flat geometry dictionaries in place."""
    if elements is None:
        return
    for element in elements:
        element["Z"] = np.asarray(element["Z"], dtype=float) + z_shift


def _pair_z_midpoints(upper, lower, z_midplane, polygon):
    """Return fitted-plane samples from one aligned upper/lower point pair."""
    upper_points, _ = _element_points(upper, polygon)
    lower_points, _ = _element_points(lower, polygon)
    reflected_lower = lower_points.copy()
    reflected_lower[:, 1] = 2 * z_midplane - reflected_lower[:, 1]
    # Use the preliminary plane only to establish correspondence. Once points
    # are paired, (Z_upper + Z_lower) / 2 estimates their actual symmetry plane.
    assignment = _best_point_assignment(upper_points, reflected_lower, polygon)
    return 0.5 * (upper_points[:, 1] + lower_points[assignment, 1])


def _active_pair_z_midpoints(upper, lower, z_midplane):
    """Return fitted-plane samples from all parts of an active pair."""
    upper_parts = _active_parts(upper)
    lower_parts = _active_parts(lower)
    # Compound circuits can store filament bundles under unrelated dictionary
    # keys. Match bundles geometrically before matching points within a bundle.
    upper_indices, lower_indices = _active_part_assignment(
        upper_parts, lower_parts, z_midplane
    )
    return np.concatenate(
        [
            _pair_z_midpoints(
                upper_parts[upper_index][1],
                lower_parts[lower_index][1],
                z_midplane,
                False,
            )
            for upper_index, lower_index in zip(upper_indices, lower_indices)
        ]
    )


def _self_z_midpoints(element, z_midplane, polygon=False):
    """Return fitted-plane samples from a circuit or element spanning the plane."""
    points = np.vstack(
        [_element_points(part, polygon)[0] for _, part in _active_parts(element)]
    )
    upper_mask, lower_mask, middle_mask = _split_about_plane(points[:, 1], z_midplane)
    upper = points[upper_mask]
    lower = points[lower_mask]
    if len(upper) != len(lower):
        raise ValueError(
            "Automatic midplane fitting requires equal upper/lower point counts "
            "in each self-symmetric circuit."
        )
    # A self-symmetric element supplies both sides of the fit itself; pairing
    # its upper and reflected-lower points yields the same midpoint samples as
    # an explicitly named upper/lower element pair.
    reflected_lower = lower.copy()
    reflected_lower[:, 1] = 2 * z_midplane - reflected_lower[:, 1]
    assignment = _best_point_assignment(upper, reflected_lower, False)
    return np.r_[0.5 * (upper[:, 1] + lower[assignment, 1]), points[middle_mask, 1]]


def _fit_source_midplane(
    active_data,
    passive_data,
    passive_names,
    active_pairs,
    active_self,
    passive_pairs,
    passive_self,
    preliminary_midplane,
):
    """Fit the common source-plane offset from all paired geometry points."""
    # Pool pointwise midplanes across all conductors. This gives detailed
    # elements proportionally more influence than coarse single-point elements.
    samples = []
    for upper_name, lower_name, _ in active_pairs:
        samples.extend(
            _active_pair_z_midpoints(
                active_data[upper_name],
                active_data[lower_name],
                preliminary_midplane,
            )
        )
    for name in active_self:
        samples.extend(_self_z_midpoints(active_data[name], preliminary_midplane))

    passive_by_name = dict(zip(passive_names, passive_data))
    for upper_name, lower_name, _ in passive_pairs:
        samples.extend(
            _pair_z_midpoints(
                passive_by_name[upper_name],
                passive_by_name[lower_name],
                preliminary_midplane,
                True,
            )
        )
    for name in passive_self:
        samples.extend(
            _self_z_midpoints(passive_by_name[name], preliminary_midplane, True)
        )

    samples = np.asarray(samples, dtype=float)
    if not len(samples):
        raise ValueError("No reflected point pairs are available to fit a midplane.")
    # The mean is the least-squares constant plane through all pair midpoints;
    # the RMS records how well one global plane describes the supplied machine.
    fitted_midplane = float(np.mean(samples))
    fit_rms = float(np.sqrt(np.mean((samples - fitted_midplane) ** 2)))
    return fitted_midplane, fit_rms, len(samples)


def _check_unique(names, description):
    """Raise when current labels are not unique."""
    if len(set(names)) != len(names):
        raise ValueError(f"The {description} element names must be unique.")


def _strip_side(name):
    """Remove upper/lower tokens from a grouping label."""
    stripped = _SIDE_PATTERN.sub(lambda match: match.group(1), str(name))
    return re.sub(r"_+", "_", stripped).strip("_")


def _name_side(name):
    """Return the explicit side encoded in a name, if present."""
    match = _SIDE_PATTERN.search(str(name))
    return None if match is None else match.group(2).lower()


def _centroid_z(element):
    """Return the mean Z coordinate of one flat geometry dictionary."""
    return float(np.mean(np.asarray(element["Z"], dtype=float)))


def _active_parts(element):
    """Return named flat geometry parts from an active-coil entry."""
    if "R" in element and "Z" in element:
        return [(None, element)]
    return list(element.items())


def _active_centroid_z(element):
    """Return the filament-count-weighted active-circuit Z centroid."""
    parts = _active_parts(element)
    counts = np.asarray([np.size(part["Z"]) for _, part in parts])
    means = np.asarray([_centroid_z(part) for _, part in parts])
    return float(np.average(means, weights=counts))


def _identify_active_pairs(active_data, explicit_pairs, z_midplane):
    """Identify active pairs and self-symmetric circuits."""
    names = tuple(active_data)
    _check_unique(names, "active")
    if explicit_pairs is not None:
        # Explicit pairs override naming inference but still pass through all
        # existence, geometry, and winding-parity checks below.
        pairs = [(upper, lower, _strip_side(upper)) for upper, lower in explicit_pairs]
        _check_unique(
            [name for upper, lower, _ in pairs for name in (upper, lower)],
            "active pair",
        )
    else:
        # Active series circuits are normally named in unambiguous upper/lower
        # pairs. Removing the side token supplies the reduced-circuit label.
        by_group = {}
        for name in names:
            side = _name_side(name)
            if side is not None:
                by_group.setdefault(_strip_side(name), {})[side] = name
        pairs = []
        for group, sides in by_group.items():
            if set(sides) != {"upper", "lower"}:
                raise ValueError(
                    f"Active group '{group}' does not contain one upper and one lower circuit."
                )
            pairs.append((sides["upper"], sides["lower"], group))

    paired = {name for upper, lower, _ in pairs for name in (upper, lower)}
    unknown = paired.difference(names)
    if unknown:
        raise ValueError(f"Unknown active pair labels: {sorted(unknown)}")
    # Anything not explicitly paired is interpreted as one circuit that is
    # internally symmetric, rather than silently treating it as an unmatched
    # upper- or lower-only source.
    self_names = [name for name in names if name not in paired]
    raise_error = False
    for name in self_names:
        z_values = np.concatenate(
            [np.asarray(part["Z"]) for _, part in _active_parts(active_data[name])]
        )
        upper, lower, middle = _split_about_plane(z_values, z_midplane)
        if not (np.all(middle) or (np.any(upper) and np.any(lower))):
            print(f"Unpaired active circuit '{name}' does not span the symmetry plane.")
            raise_error = True
    if raise_error:
        raise ValueError("Unpaired active circuits found. Aborting symetrization.")
    return pairs, self_names


def _passive_group(element, name):
    """Return the structural candidate group for one passive element."""
    metadata_group = element.get("element")
    return _strip_side(metadata_group if metadata_group is not None else name)


def _identify_passive_pairs(passive_data, names, explicit_pairs, z_midplane):
    """Identify passive pairs by grouped reflected-geometry assignment."""
    by_name = dict(zip(names, passive_data))
    if explicit_pairs is not None:
        requested = {name for pair in explicit_pairs for name in pair}
        unknown = requested.difference(names)
        if unknown:
            raise ValueError(f"Unknown passive pair labels: {sorted(unknown)}")
        pairs = [
            (upper, lower, _passive_group(by_name[upper], upper))
            for upper, lower in explicit_pairs
        ]
        _check_unique(
            [name for upper, lower, _ in pairs for name in (upper, lower)],
            "passive pair",
        )
        paired = {name for upper, lower, _ in pairs for name in (upper, lower)}
        return pairs, [name for name in names if name not in paired]

    # Passive structures often contain many nominally identical elements. First
    # separate structural groups and sides; only compare geometry within a group.
    grouped = {}
    self_names = []
    for name, element in zip(names, passive_data):
        z_centroid = _centroid_z(element)
        if np.isclose(z_centroid, z_midplane):
            self_names.append(name)
            continue
        side = "upper" if z_centroid > z_midplane else "lower"
        group = _passive_group(element, name)
        grouped.setdefault(group, {"upper": [], "lower": []})[side].append(name)

    pairs = []
    for group, sides in grouped.items():
        upper_names = sides["upper"]
        lower_names = sides["lower"]
        if len(upper_names) != len(lower_names):
            raise ValueError(
                f"Passive group '{group}' has {len(upper_names)} upper and "
                f"{len(lower_names)} lower elements."
            )
        # Find the globally minimum one-to-one reflected assignment. Greedy
        # nearest-neighbour matching can pair one lower element more than once.
        upper_indices, lower_indices = _compatible_assignment(
            [by_name[name] for name in upper_names],
            [by_name[name] for name in lower_names],
            z_midplane,
            polygon=True,
        )
        pairs.extend(
            (
                upper_names[upper_index],
                lower_names[lower_index],
                group,
            )
            for upper_index, lower_index in zip(upper_indices, lower_indices)
        )
    return pairs, self_names


def _best_point_assignment(upper_points, reflected_lower_points, polygon):
    """Return lower-point indices aligned with upper-point indices."""
    if len(upper_points) != len(reflected_lower_points):
        raise ValueError(
            "Reflected element partners must contain the same point count."
        )
    count = len(upper_points)
    if polygon and count > 2:
        # Polygon vertices carry boundary connectivity. Allow cyclic shifts and
        # traversal reversal, but never an arbitrary permutation that would
        # scramble edges or create a self-intersecting conductor polygon.
        candidates = []
        indices = np.arange(count)
        for direction in (indices, indices[::-1]):
            for shift in range(count):
                candidate = np.roll(direction, shift)
                candidates.append(
                    (
                        _point_rms_distance(
                            upper_points, reflected_lower_points[candidate]
                        ),
                        candidate,
                    )
                )
        return min(candidates, key=lambda item: item[0])[1]
    # Filament clouds have no path connectivity, so use unrestricted optimal
    # one-to-one assignment in Euclidean (R, Z) space.
    costs = np.linalg.norm(
        upper_points[:, np.newaxis, :] - reflected_lower_points[np.newaxis, :, :],
        axis=-1,
    )
    upper_indices, lower_indices = linear_sum_assignment(costs)
    assignment = np.empty(count, dtype=int)
    assignment[upper_indices] = lower_indices
    return assignment


def _geometry_mismatch(upper, lower, z_midplane, polygon):
    """Return RMS reflected point mismatch between two flat elements."""
    upper_points, _ = _element_points(upper, polygon)
    lower_points, _ = _element_points(lower, polygon)
    reflected_lower = lower_points.copy()
    reflected_lower[:, 1] = 2 * z_midplane - reflected_lower[:, 1]
    assignment = _best_point_assignment(upper_points, reflected_lower, polygon)
    return _point_rms_distance(upper_points, reflected_lower[assignment])


def _active_part_winding(part):
    """Return the signed current multiplier of one active-coil part."""
    return float(part.get("polarity", 1.0)) * float(part.get("multiplier", 1.0))


def _active_part_assignment(upper_parts, lower_parts, z_midplane):
    """Match geometry only between bundles with compatible filament counts."""
    return _compatible_assignment(
        [part for _, part in upper_parts],
        [part for _, part in lower_parts],
        z_midplane,
        count_label="filament",
    )


def _compatible_assignment(
    upper, lower, z_midplane, polygon=False, count_label="point"
):
    """Match compatible point counts before comparing reflected geometry."""
    costs = np.asarray(
        [
            [
                (
                    _geometry_mismatch(upper_part, lower_part, z_midplane, polygon)
                    if len(_element_points(upper_part, polygon)[0])
                    == len(_element_points(lower_part, polygon)[0])
                    else np.inf
                )
                for lower_part in lower
            ]
            for upper_part in upper
        ]
    ).reshape(len(upper), len(lower))
    try:
        return linear_sum_assignment(costs)
    except ValueError as error:
        raise ValueError(
            f"Reflected elements cannot be matched one-to-one with equal {count_label} counts."
        ) from error


def _matched_active_parts(upper_parts, lower_parts, z_midplane):
    """Return geometrically matched upper/lower active-coil parts."""
    if len(upper_parts) != len(lower_parts):
        return None
    # Part names are not assumed to agree across reflected circuits. Geometry
    # supplies the correspondence used for both mismatch and polarity checks.
    upper_indices, lower_indices = _active_part_assignment(
        upper_parts, lower_parts, z_midplane
    )
    return [
        (upper_parts[upper_index][1], lower_parts[lower_index][1])
        for upper_index, lower_index in zip(upper_indices, lower_indices)
    ]


def _matched_parts_parity(matched_parts):
    """Return +1 for even, -1 for odd, or 0 for incompatible windings."""
    parities = []
    for upper, lower in matched_parts:
        upper_winding = _active_part_winding(upper)
        lower_winding = _active_part_winding(lower)
        if not np.isclose(abs(upper_winding), abs(lower_winding)):
            return 0
        # Equal signed winding gives even magnetic forcing; opposite winding
        # gives odd forcing. Mixed parity across parts cannot form one valid
        # reduced current coordinate.
        product = upper_winding * lower_winding
        if product == 0:
            return 0
        parities.append(int(np.sign(product)))
    return parities[0] if parities and len(set(parities)) == 1 else 0


def _paired_active_parity(upper, lower, z_midplane):
    """Classify the electrical reflection parity of two active circuits."""
    matched = _matched_active_parts(
        _active_parts(upper), _active_parts(lower), z_midplane
    )
    return 0 if matched is None else _matched_parts_parity(matched)


def _self_active_parity(element, z_midplane):
    """Classify a series circuit containing its own reflected filament bundles."""
    if "R" in element and "Z" in element:
        return 1
    upper, lower, middle = _active_sides(element, z_midplane)
    matched = _matched_active_parts(upper, lower, z_midplane)
    parity = 0 if matched is None else _matched_parts_parity(matched)
    # A part lying on the plane maps to itself and is therefore even. It cannot
    # be combined consistently with an otherwise odd compound circuit.
    if middle and parity < 0:
        return 0
    return parity if matched else (1 if middle else 0)


def _self_element_mismatch(element, z_midplane, polygon=False):
    """Return reflected RMS mismatch within one midplane-spanning element."""
    points, _ = _element_points(element, polygon)
    upper_mask, lower_mask, _ = _split_about_plane(points[:, 1], z_midplane)
    upper = points[upper_mask]
    lower = points[lower_mask]
    if len(upper) != len(lower):
        raise ValueError(
            "A self-symmetric element must have equal upper/lower point counts."
        )
    if not len(upper):
        return 0.0
    return _geometry_mismatch(
        {"R": upper[:, 0], "Z": upper[:, 1]},
        {"R": lower[:, 0], "Z": lower[:, 1]},
        z_midplane,
        False,
    )


def _self_active_mismatch(element, z_midplane):
    """Return the largest reflected RMS mismatch within one active circuit."""
    if "R" in element and "Z" in element:
        return _self_element_mismatch(element, z_midplane)
    upper, lower, middle = _active_sides(element, z_midplane)
    matched = _matched_active_parts(upper, lower, z_midplane)
    if matched is None:
        raise ValueError(
            "A self-symmetric active circuit must have equal upper/lower part counts."
        )
    mismatches = [
        _geometry_mismatch(upper_part, lower_part, z_midplane, False)
        for upper_part, lower_part in matched
    ]
    mismatches.extend(_self_element_mismatch(part, z_midplane) for _, part in middle)
    return max(mismatches, default=0.0)


def _average_pair_scalars(upper, lower):
    """Average scalar material/discretisation fields that affect dynamics."""
    fields = (
        "resistivity",
        "dR",
        "dZ",
        "min_refine_per_area",
        "min_refine_per_length",
        "current_multiplier",
    )
    # Geometry averaging alone is insufficient: reflected elements must also
    # have identical material and mesh parameters to produce symmetric dynamics.
    for field in fields:
        if field in upper and field in lower:
            value = 0.5 * (float(upper[field]) + float(lower[field]))
            upper[field] = value
            lower[field] = value
    for field in ("polarity", "multiplier"):
        if field in upper and field in lower:
            if not np.allclose(upper[field], lower[field]):
                raise ValueError(
                    f"Reflected active partners must have matching '{field}'."
                )


def _symmetrise_element_pair(upper, lower, z_midplane, polygon):
    """Average one flat upper/lower pair onto exactly reflected geometries."""
    upper_points, upper_closed = _element_points(upper, polygon)
    lower_points, lower_closed = _element_points(lower, polygon)
    reflected_lower = lower_points.copy()
    reflected_lower[:, 1] = 2 * z_midplane - reflected_lower[:, 1]
    assignment = _best_point_assignment(upper_points, reflected_lower, polygon)
    mismatch = _point_rms_distance(upper_points, reflected_lower[assignment])

    # Average in the upper half-plane, mirror that average exactly, then undo
    # the assignment so the lower element keeps its original point ordering.
    symmetric_upper = 0.5 * (upper_points + reflected_lower[assignment])
    symmetric_lower_aligned = symmetric_upper.copy()
    symmetric_lower_aligned[:, 1] = 2 * z_midplane - symmetric_upper[:, 1]
    symmetric_lower = np.empty_like(symmetric_lower_aligned)
    symmetric_lower[assignment] = symmetric_lower_aligned

    if upper_closed:
        symmetric_upper = np.vstack((symmetric_upper, symmetric_upper[0]))
    if lower_closed:
        symmetric_lower = np.vstack((symmetric_lower, symmetric_lower[0]))

    upper["R"] = symmetric_upper[:, 0]
    upper["Z"] = symmetric_upper[:, 1]
    lower["R"] = symmetric_lower[:, 0]
    lower["Z"] = symmetric_lower[:, 1]
    _average_pair_scalars(upper, lower)
    return upper, lower, mismatch


def _symmetrise_active_pair(upper, lower, z_midplane):
    """Symmetrise all filament groups in a paired active circuit."""
    upper_parts = _active_parts(upper)
    lower_parts = _active_parts(lower)
    if len(upper_parts) != len(lower_parts):
        raise ValueError("Paired active circuits must contain the same part count.")

    # Match compound-circuit parts independently of dictionary insertion order.
    # Each matched flat pair can then use the common element-level operation.
    upper_indices, lower_indices = _active_part_assignment(
        upper_parts, lower_parts, z_midplane
    )
    mismatches = []
    for upper_index, lower_index in zip(upper_indices, lower_indices):
        upper_key, upper_part = upper_parts[upper_index]
        lower_key, lower_part = lower_parts[lower_index]
        new_upper, new_lower, mismatch = _symmetrise_element_pair(
            upper_part, lower_part, z_midplane, polygon=False
        )
        if upper_key is None:
            upper = new_upper
        else:
            upper[upper_key] = new_upper
        if lower_key is None:
            lower = new_lower
        else:
            lower[lower_key] = new_lower
        mismatches.append(mismatch)
    return upper, lower, float(np.max(mismatches))


def _symmetrise_self_element(element, z_midplane, polygon=False):
    """Make one flat element spanning the midplane internally symmetric."""
    points, closed = _element_points(element, polygon)
    upper, lower, middle = _split_about_plane(points[:, 1], z_midplane)
    upper_indices, lower_indices = np.flatnonzero(upper), np.flatnonzero(lower)
    middle_indices = np.flatnonzero(middle)
    if len(upper_indices) != len(lower_indices):
        raise ValueError(
            "A self-symmetric element must have equal upper/lower point counts."
        )
    if len(upper_indices):
        # Reuse pair symmetrisation on the two subsets, then write the averaged
        # points back into their original positions in this single element.
        upper = {"R": points[upper_indices, 0], "Z": points[upper_indices, 1]}
        lower = {"R": points[lower_indices, 0], "Z": points[lower_indices, 1]}
        new_upper, new_lower, _ = _symmetrise_element_pair(
            upper, lower, z_midplane, polygon=False
        )
        points[upper_indices, 0] = new_upper["R"]
        points[upper_indices, 1] = new_upper["Z"]
        points[lower_indices, 0] = new_lower["R"]
        points[lower_indices, 1] = new_lower["Z"]
    # Numerical noise on self-reflecting points is removed explicitly.
    points[middle_indices, 1] = z_midplane
    if closed:
        points = np.vstack((points, points[0]))
    element["R"] = points[:, 0]
    element["Z"] = points[:, 1]
    return element


def _symmetrise_self_active(element, z_midplane):
    """Make a direct or compound active circuit internally symmetric."""
    if "R" in element and "Z" in element:
        return _symmetrise_self_element(element, z_midplane)

    upper, lower, middle = _active_sides(element, z_midplane)
    if len(upper) != len(lower):
        raise ValueError(
            "A self-symmetric active circuit must have equal upper/lower part counts."
        )
    # For compound circuits, pair upper/lower parts geometrically and treat any
    # midplane-spanning part as internally self-symmetric.
    upper_indices, lower_indices = _active_part_assignment(upper, lower, z_midplane)
    for upper_index, lower_index in zip(upper_indices, lower_indices):
        upper_key, upper_part = upper[upper_index]
        lower_key, lower_part = lower[lower_index]
        element[upper_key], element[lower_key], _ = _symmetrise_element_pair(
            upper_part, lower_part, z_midplane, polygon=False
        )
    for key, part in middle:
        element[key] = _symmetrise_self_element(part, z_midplane)
    return element


def _make_diagnostic(upper, lower, group, mismatch):
    """Build a pairing diagnostic."""
    upper_index = _TRAILING_INDEX_PATTERN.search(upper)
    lower_index = _TRAILING_INDEX_PATTERN.search(lower)
    same_suffix = (
        upper_index is not None
        and lower_index is not None
        and upper_index.group(1) == lower_index.group(1)
    )
    return ElementPair(upper, lower, group, mismatch, same_suffix)


def _build_even_active_description(active_data, pair_names, self_names):
    """Combine reflected active pairs into series circuits for even evolution."""
    # Index both labels to the same pair so iteration can preserve the original
    # active-circuit order without requiring upper members to appear first.
    pair_lookup = {}
    for upper, lower, group in pair_names:
        pair_lookup[upper] = (upper, lower, group)
        pair_lookup[lower] = (upper, lower, group)

    even_data = {}
    even_names = []
    consumed = set()
    for name in active_data:
        if name in consumed:
            continue
        if name in self_names:
            even_data[name] = deepcopy(active_data[name])
            even_names.append(name)
            consumed.add(name)
            continue

        upper, lower, group = pair_lookup[name]
        reduced_name = group
        # Distinct source groups can collapse to the same stripped label. Keep
        # both circuits by falling back to an explicit composite name.
        if reduced_name in even_data or reduced_name in self_names:
            reduced_name = f"{upper}__{lower}"
        while reduced_name in even_data or reduced_name in self_names:
            reduced_name += "_pair"
        circuit = {}
        # A single current through this compound entry drives both reflected
        # source circuits, which is precisely the retained even coordinate.
        for side, source_name in (("upper", upper), ("lower", lower)):
            for part_key, part in _active_parts(active_data[source_name]):
                suffix = "coil" if part_key is None else str(part_key)
                circuit[f"_{side}_{suffix}"] = deepcopy(part)
        even_data[reduced_name] = circuit
        even_names.append(reduced_name)
        consumed.update((upper, lower))
    return even_data, even_names


def _reflection_operator(names, pairs, self_names):
    """Construct a current-coordinate reflection permutation."""
    names = tuple(names)
    indices = {name: index for index, name in enumerate(names)}
    reflection = np.zeros((len(names), len(names)))
    # Paired coordinates exchange places under reflection.
    for pair in pairs:
        upper_index = indices[pair.upper_name]
        lower_index = indices[pair.lower_name]
        reflection[upper_index, lower_index] = 1
        reflection[lower_index, upper_index] = 1
    # Internally symmetric coordinates map to themselves.
    for name in self_names:
        reflection[indices[name], indices[name]] = 1
    # Reflection must be an involution. This also catches missing, duplicated,
    # or otherwise incomplete pair assignments before matrices are transformed.
    if not np.allclose(reflection @ reflection, np.eye(len(names))):
        raise ValueError("Pairing does not define a complete reflection operator.")
    return reflection


def _parity_transforms(names, pairs, self_names):
    """Build ampere-preserving average/difference current transforms."""
    names = tuple(names)
    indices = {name: index for index, name in enumerate(names)}
    pair_by_name = {}
    for pair in pairs:
        pair_by_name[pair.upper_name] = pair
        pair_by_name[pair.lower_name] = pair

    # Rows map original currents into parity coordinates; columns perform the
    # inverse reconstruction. The 1/2 appears only in the forward maps so both
    # transformed coordinates retain physical ampere units.
    even_rows = []
    odd_rows = []
    even_columns = []
    odd_columns = []
    odd_names = []
    consumed = set()
    for name in names:
        if name in consumed:
            continue
        if name in self_names:
            # A self-symmetric circuit contributes one even coordinate and no
            # independent odd coordinate.
            row = np.zeros(len(names))
            row[indices[name]] = 1
            even_rows.append(row)
            even_columns.append(row.copy())
            consumed.add(name)
            continue

        pair = pair_by_name[name]
        upper_index = indices[pair.upper_name]
        lower_index = indices[pair.lower_name]
        # I_even=(I_upper+I_lower)/2 and
        # I_odd=(I_upper-I_lower)/2. Reconstruction uses their sum/difference.
        even_row = np.zeros(len(names))
        even_row[[upper_index, lower_index]] = 0.5
        odd_row = np.zeros(len(names))
        odd_row[upper_index] = 0.5
        odd_row[lower_index] = -0.5
        even_column = np.zeros(len(names))
        even_column[[upper_index, lower_index]] = 1
        odd_column = np.zeros(len(names))
        odd_column[upper_index] = 1
        odd_column[lower_index] = -1
        even_rows.append(even_row)
        odd_rows.append(odd_row)
        even_columns.append(even_column)
        odd_columns.append(odd_column)
        odd_names.append(pair.group)
        consumed.update((pair.upper_name, pair.lower_name))

    return (
        np.asarray(even_rows).reshape((-1, len(names))),
        np.asarray(odd_rows).reshape((-1, len(names))),
        np.asarray(even_columns).reshape((-1, len(names))).T,
        np.asarray(odd_columns).reshape((-1, len(names))).T,
        tuple(odd_names),
    )
