"""
Useful functions for copying attributed between FreeGSNKE objects.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

import copy
import logging

import numpy as np

logger = logging.getLogger(__name__)


def copy_into(
    obj, new_obj, attr: str, *, mutable=False, strict=True, allow_deepcopy=False
):
    """
    Copy an attribute from one object into another.

    This function transfers a named attribute from `obj` to `new_obj`,
    optionally performing a deep copy for mutable objects.

    Parameters
    ----------
    obj : object
        Source object to copy the attribute from.

    new_obj : object
        Destination object to copy the attribute into.

    attr : str
        Name of the attribute to copy (equivalent to obj.attr).

    mutable : bool, optional
        If True, treat the attribute as mutable and ensure it is copied
        rather than referenced. For NumPy arrays, a shallow copy is used
        when possible.

    strict : bool, optional
        If True, raise an AttributeError when `obj` does not have `attr`.
        If False, silently return without copying.

    allow_deepcopy : bool, optional
        If True, allow deepcopying of non-NumPy mutable objects when required.
        If False, raise a TypeError when deepcopy would be needed.

    Raises
    ------
    TypeError
        If `mutable=True`, the attribute is not a safe NumPy array copy,
        and `allow_deepcopy=False`.

    Returns
    -------
    None
    """

    if not hasattr(obj, attr) and not strict:
        logger.info(f"{obj.__class__} has no attribute {attr} but not in strict mode")
        # return without an error because we are not strict
        return

    # will error if strict and attribute doesnt exist
    attribute_value = getattr(obj, attr)

    # handle singletons
    if attribute_value is None or attribute_value is True or attribute_value is False:
        setattr(new_obj, attr, attribute_value)
        return

    if mutable:
        if (
            isinstance(attribute_value, np.ndarray)
            and not attribute_value.dtype.hasobject
        ):
            attribute_value = np.copy(attribute_value)

        else:
            if not allow_deepcopy:
                raise TypeError(
                    f"Cannot copy {attribute_value.__class__} without deepcopying"
                )

            logger.info(
                f"Deepcopying {attribute_value.__class__} because it is mutable but not a numpy array"
                "of non-objects"
            )

            attribute_value = copy.deepcopy(attribute_value)

    setattr(new_obj, attr, attribute_value)
