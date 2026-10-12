# Implicit odd GS response

This example differentiates a manufactured symmetric equilibrium under an odd
applied magnetic flux. It creates all inputs and needs no machine files.
FreeGS4E 0.15 supplies a uniform-grid spatial inverse. The response algorithm
accepts any compatible linear inverse callback; it does not inspect grid details.

Write the free-boundary plasma-flux map as $G:j_\phi\mapsto\psi_{pl}$ and the
reflection permutation as $P$. The odd projection is $P_o=(I-P)/2$.
For local source derivative $C=\partial j_\phi/\partial\psi$ and applied flux
$g$ per forcing ampere, the implicit tangent is

\[
(I-P_oGCP_o)v=P_oG(Cg),\qquad j_I=P_o[C(v+g)].
\]

Here $v$ is plasma flux response; the current responds to total flux $v+g$.
`solve_odd_response(inverse, C, g, reflection)` uses matrix-free GMRES and checks
the final equation residual. It requires an even base, even $C$, odd $g$ and a
reflection-equivariant inverse. The caller must establish those physical and
operator assumptions; input checks cannot prove them for an arbitrary callback.

The example source is

\[
j(\psi,I)=j_0+C(\psi+Ig-\psi_0)+\beta(\psi+Ig-\psi_0)^3,
\qquad \psi_0=G j_0.
\]

Two external rings with opposite unit currents provide $g$. The source vanishes
on the outer rectangle. Its boundary flux is obtained by direct Green sums;
FreeGS4E's DST solver then solves $\Delta^*\psi=-\mu_0Rj$ in the interior.
The discrete boundary rows contain physical flux, without an extra minus sign.

Independent nonlinear solves at positive and negative forcing amplitudes give
central differences. Their current-derivative errors should decrease quadratically
as the amplitude shrinks. This verifies differentiation on one fixed grid, not
spatial convergence, a normalized-flux profile model or a stability prediction.
Scalar normalization and critical-flux terms cancel only when the corresponding
symmetry assumptions hold. Sharp interfaces and critical-point switching need
separate treatment.

Flux is in Wb/rad; $g$ is Wb/rad/A; current density is A/m² and the returned
current derivative is A/m²/A. Area weights are $dR\,dZ$, without $2\pi R$.
The illustrative incremental linkage is $K=2\pi\sum_i g_iw_i(j_I)_i$.

From the repository root with its dependencies installed:

```sh
PYTHONPATH=. python examples/odd_gs_response/nonlinear_response.py
```

The script writes `response-output/response.json`. The tests in
`freegsnke/tests/test_gs_response.py` additionally compare the tangent with an
independent dense solve and check rejected forcing and singular equations.
