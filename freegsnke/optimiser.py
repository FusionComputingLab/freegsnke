"""
Differential Evolution optimisation

Implemented generically, to optimise opaque objects

The optimiser copies objects using deepcopy, and manipulates objects
using supplied control objects.

Copyright 2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import bisect
import copy
import random
from typing import Callable


def mutate(obj: object, controls: list[object]) -> object:
    """
    Create a new object by randomly perturbing the controls of an object.

    Each control value is scaled by a random factor in the range 0.95 to 1.05.

    Parameters
    ----------
    obj : object
        Object to copy. It is deep-copied, not modified.
    controls : list[object]
        Control objects, each providing ``get(obj)`` and ``set(obj, value)``.

    Returns
    -------
    object
        The perturbed deep copy of `obj`.
    """
    new_obj = copy.deepcopy(obj)
    for control in controls:
        # Add 10% variations to controls
        new_value = control.get(obj) * (1.0 + 0.1 * (random.random() - 0.5))
        if abs(new_value) < 1e-10:
            # Don't know what scale, so add ~1 random value to it
            new_value += random.normal(loc=0.0, scale=1.0)
        control.set(new_obj, new_value)
    return new_obj


def pickUnique(N: int, m: int, e: list[int]) -> list[int]:
    """
    Pick random distinct integers from ``0 ... N-1``, excluding some values.

    The returned values are in a random order (not sorted).

    Parameters
    ----------
    N : int
        Size of the range to pick from.
    m : int
        Number of values to pick; must satisfy ``m <= N - len(e)``.
    e : list[int]
        Values to exclude.

    Returns
    -------
    list[int]
        The `m` picked values.
    """
    assert m <= N - len(e)

    inds = sorted(e)  # Sorted list of indices. Used to avoid clashes
    others = []  # The list of three agents
    for i in range(m):
        newind = random.randint(0, N - 1 - i - len(e))
        for ind in inds:
            if newind == ind:
                newind += 1
        bisect.insort(inds, newind)
        others.append(newind)
    return others


def optimise(
    obj: object,
    controls: list[object],
    measure: Callable[[object], float],
    maxgen: int = 10,
    N: int = 10,
    CR: float = 0.3,
    F: float = 1.0,
    monitor: (
        Callable[[int, tuple[float, object], list[tuple[float, object]]], None] | None
    ) = None,
) -> object:
    """
    Use Differential Evolution to optimise an object.

    See https://en.wikipedia.org/wiki/Differential_evolution

    Parameters
    ----------
    obj : object
        Starting solution. Objects are deep-copied and passed to `controls`
        and `measure`.
    controls : list[object]
        Control objects, each with methods ``set(object, value)``, which
        modifies the given object, and ``get(object)``, which returns the
        value from the object.
    measure : Callable[[object], float]
        Returns a score for a given object; the optimiser minimises it.
    maxgen : int, optional
        Maximum number of generations.
    N : int, optional
        Population size (must be >= 4).
    CR : float, optional
        Crossover probability, in [0, 1].
    F : float, optional
        Differential weight, in [0, 2].
    monitor : Callable or None, optional
        Called each generation as ``monitor(generation, best, population)``,
        where `generation` is an int, `best` is a ``(score, object)`` tuple and
        `population` is a list of ``(score, object)`` tuples.

    Returns
    -------
    object
        The object with the lowest score.
    """
    assert N >= 4

    best = (measure(obj), obj)  # Highest score, candidate solution (agent)
    population = [best]  # List of (score, agent)

    for i in range(N - 1):
        agent = mutate(obj, controls)
        score_agent = (measure(agent), agent)
        population.append(score_agent)
        if score_agent[0] > best[0]:
            best = score_agent

    for generation in range(maxgen):
        next_pop = []  # Next generation
        for ai, agent in enumerate(population):
            # Pick three other random agents, all different
            others = [population[index][1] for index in pickUnique(N, 3, [ai])]

            new_obj = copy.deepcopy(agent[1])
            R = random.randint(0, len(controls) - 1)  # Pick a random control to modify
            for i, control in enumerate(controls):
                if i == R or random.random() < CR:
                    control.set(
                        new_obj,
                        control.get(others[0])
                        + F * (control.get(others[1]) - control.get(others[2])),
                    )
            score = measure(new_obj)
            if score < agent[0]:
                # Better than original
                new_agent = (score, new_obj)
                next_pop.append(new_agent)
                if score < best[0]:
                    # Now the best candidate
                    best = new_agent
            else:
                next_pop.append(agent)
        # End of generation. Call monitor with best object, score
        if monitor:
            monitor(generation, best, next_pop)
        # Change to next generation
        population = next_pop
    # Finished, return best candidate
    return best[1]
