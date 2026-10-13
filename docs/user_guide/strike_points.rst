Native strike-point descriptors
===============================

FreeGSNKE measures a separatrix leg's first contact with an ordered wall
polyline independently of machine-specific ALPD scenarios. The implementation
is in ``freegsnke.shape_descriptors`` and is also exposed by ``Equilibrium``.

``Ss`` is cumulative wall arc length in metres from the first wall vertex,
increasing in vertex order. ``Rs`` is the major radius obtained by evaluating
the same wall at the measured ``Ss``. Reversing the wall reverses the arc
convention but leaves ``Rs`` unchanged. On a closed wall, repeat the first
vertex at the end. In a multiple-null equilibrium, select the X-point
explicitly to avoid changing divertor branches. The magnetic axis is used
to exclude directions toward the core, which could otherwise reach the
opposite divertor on a full wall.

To get both coordinates from a single contour trace::

    contact = eq.strikepoint(wall=ordered_wall, xpoint=selected_xpoint,
                            leg="outer")
    ss, rs = contact.Ss, contact.Rs

Convenience measurements are ``eq.Ss(...)`` and ``eq.Rs(...)`` with the same
keyword arguments. Omitting ``wall`` uses ``eq.tokamak.wall`` in its stored
order; omitting ``xpoint`` selects ``eq.xpt[0, :2]``. Missing wall contact or
missing X-point raises an error rather than choosing a contact near a target.

For shape control and virtual circuits::

    from freegsnke.shape_descriptors import StrikePointDescriptor

    strike = StrikePointDescriptor(
        wall=ordered_wall,
        xpoint=lambda eq: eq.xpt[0, :2],
        leg="outer",
        names=("Ss", "Rs"),
    )
    ss, rs = strike(eq)

The X-point selector can be a fixed pair or a callable. Both outputs share
one measurement per call. Supply this callable as a virtual-circuit target
calculator, or combine its values with other equilibrium measurements in
an application-specific descriptor function. Requested coordinate order
is preserved through the ``names`` argument.

Normally constrain one of ``Ss`` or ``Rs``: they describe the same contact
and using both as independent equality targets adds redundant information.
Wall arc is useful on vertical or non-monotonic targets, where radius alone
cannot uniquely identify a strike position. A radius request can be converted
into an arc request only after choosing the appropriate target section.

