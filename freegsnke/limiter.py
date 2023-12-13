import numpy as np

def core_mask_limiter(psi, psi_bndry, core_mask, 
                      limiter_mask_out,
                      limiter_mask_in,
                      linear_coeff=.5):
    """Checks if plasma is in a limiter configuration rather than a diverted configuration.
    This is obtained by checking whether the core mask deriving from the assumption of a diverted configuration
    implies an overlap with the limiter.

    Parameters
    ----------
    psi : np.array
        The flux function, including both plasma and metal components.
        np.shape(psi) = (eq.nx, eq.ny)
    psi_bndry : float
        The value of the flux function at the boundary. 
        This is xpt[0][2] for a diverted configuration, where xpt is the output of critical.find_critical
    core_mask : np.array
        The mask identifying the plasma region under the assumption of a diverted configuration.
        This is the result of FreeGS' critical.core_mask 
        Same size as psi. 
        _description_
    limiter_mask : np.array
        The mask identifying the outside border of the limiter.
        Same size as psi. 
    linear_coeff : float
        Value between 0 and 1 to interpolate the psi_bndry value.
        Defaluts to 0.5.


    Returns
    -------
    psi_bndry : float
        The value of the flux function at the boundary.
    core_mask : np.array
        The core mask after correction
    flag_limiter : bool
        Flag to identify if the plasma is in a diverted or limiter configuration.
    
    """

    offending_mask = (core_mask * limiter_mask_out).astype(bool)
    flag_limiter = np.any(offending_mask)

    if flag_limiter:
        psi_max_out = np.amax(psi[offending_mask])
        psi_max_in = np.amax(psi[(core_mask * limiter_mask_in).astype(bool)])
        psi_bndry = linear_coeff*psi_max_out + (1-linear_coeff)*psi_max_in
        core_mask = (psi > psi_bndry)*core_mask     

    return psi_bndry, core_mask, flag_limiter
