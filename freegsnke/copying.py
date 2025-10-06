import copy
import logging

import numpy as np

logger = logging.getLogger(__name__)


def copy_into(
    obj, new_obj, attr: str, *, mutable=False, strict=True, allow_deepcopy=False
):
    """Copies an attribute from one object to another.

    Parameters
    ==========
    obj : object
        The object to copy the attribute from

    new_obj : object
        The object to copy the attribute to

    attr : str
        The attribute name to copy (e.g. copy `obj.attr` into `new_obj`)

    mutable : bool
        If an attribute is mutable it needs to be copied explicitly between objects,
        a reference is not sufficient.

    strict : bool
        Raise an error if `True` and `attr` is not an attribute of `obj`

    allow_deepcopy : bool
        Raise an error if `True` and a deepcopy is required to copy the attribute across.
    """

    if not hasattr(obj, attr) and not strict:
        logger.info(f"{obj.__class__} has no attribute {attr} but not in strict mode")
        # return without an error because we are not strict
        return

    # will error if strict and attribute doesnt exist
    attribute_value = getattr(obj, attr)

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
