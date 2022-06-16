#this is the same as the nonlinear solver in freeGS, but does not return an 
#error if tolerances are not met, or if they're not met within the 
#assigned max num of iterations

from numpy import amin, amax, array


def fast_solve(
    eq,
    profiles,
#     constrain=None,
    rtol=3e-3,
    atol=1e-5,
    blend=0.0,
#     show=False,
#     axis=None,
#     pause=0.0001,
    psi_bndry=None,
    maxits=50,
    convergenceInfo=False
):
    """
    Perform Picard iteration to find solution to the Grad-Shafranov equation

    eq       - an Equilibrium object (equilibrium.py)
    profiles - A Profile object for toroidal current (jtor.py)

    rtol     - Relative tolerance (change in psi)/( max(psi) - min(psi) )
    atol     - Absolute tolerance, change in psi
    blend    - Blending of previous and next psi solution
               psi{n+1} <- psi{n+1} * (1-blend) + blend * psi{n}

    show     - If true, plot the plasma equilibrium at each nonlinear step
    axis     - Specify a figure to plot onto. Default (None) creates a new figure
    pause    - Delay between output plots. If negative, waits for window to be closed

    maxits   - Maximum number of iterations. Set to None for no limit.
               If this limit is exceeded then a RuntimeError is raised.
    """

#     if constrain is not None:
#         # Set the coil currents to get X-points in desired locations
#         constrain(eq)

    # Get the total psi = plasma + coils
    psi = eq.psi()

#     if show:
#         import matplotlib.pyplot as plt
#         from .plotting import plotEquilibrium

#         if pause > 0.0 and axis is None:
#             # No axis specified, so create a new figure
#             fig = plt.figure()
#             axis = fig.add_subplot(111)

    iteration = 0  # Count number of iterations
    psi_maxchange_iterations, psi_relchange_iterations = [], []
    # Start main loop
    while True:
#         if show:
#             # Plot state of plasma equilibrium
#             if pause < 0:
#                 fig = plt.figure()
#                 axis = fig.add_subplot(111)
#             else:
#                 axis.clear()

#             plotEquilibrium(eq, axis=axis, show=False)

#             if pause < 0:
#                 # Wait for user to close the window
#                 plt.show()
#             else:
#                 # Update the canvas and pause
#                 # Note, a short pause is needed to force drawing update
#                 axis.figure.canvas.draw()
#                 plt.pause(pause)

        # Copy psi to compare at the end
        psi_last = psi.copy()

        # Solve equilbrium, using the given psi to calculate Jtor
        eq.solve(profiles, psi=psi, psi_bndry=psi_bndry)

        # Get the new psi, including coils
        psi = eq.psi()

        # Compare against last solution
        psi_change = psi_last - psi
        psi_maxchange = amax(abs(psi_change))
        psi_relchange = psi_maxchange / (amax(psi) - amin(psi))
        
        print(psi_maxchange, psi_relchange)

        psi_maxchange_iterations.append(psi_maxchange)
        psi_relchange_iterations.append(psi_relchange)

        # Check if the relative change in psi is small enough
        if (psi_maxchange < atol) or (psi_relchange < rtol):
            break
            
        

#         # Adjust the coil currents
#         if constrain is not None:
#             constrain(eq)

        psi = (1.0 - blend) * eq.psi() + blend * psi_last

        # Check if the maximum iterations has been exceeded
        iteration += 1
        if iteration > 5:
            if psi_relchange > psi_relchange_iterations[-2]:
                break
            if iteration > maxits:
                break
            
    if convergenceInfo: 
        return array(psi_maxchange_iterations),\
               array(psi_relchange_iterations)
    
