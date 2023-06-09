import numpy as np
from freegs.gradshafranov import Greens
from . import machine_config


def define_reduced_plasma_grid(R, Z):
    """Defines a mask for the plasma grid points inside the limiter

    Parameters
    ----------
    R : ndarray
        (NxM) array of R coordinates of the grid points
    Z : ndarray
        (NxM) array of Z coordinates of the grid points

    Returns
    -------
    plasma_pts : ndarray
        Array with R and Z coordinates of all the points inside the limiter
    mask_inside_limiter : ndarray
        Mask of the grid points that are inside the limiter 
    """
    mask_inside_limiter = np.ones_like(R)
    mask_inside_limiter *= (R>0.265)*(R<1.582)
    mask_inside_limiter *= (Z<.95+1*R)*(Z>-.95-1*R)
    mask_inside_limiter *= (Z<-1.98+9.*R)*(Z>1.98-9.*R)
    mask_inside_limiter *= (Z<2.26-1.1*R)*(Z>-2.26+1.1*R)
    mask_inside_limiter = mask_inside_limiter.astype(bool)

    plasma_pts = np.concatenate((R[mask_inside_limiter][:,np.newaxis],
                                 Z[mask_inside_limiter][:,np.newaxis]), axis=-1)

    return plasma_pts, mask_inside_limiter


def make_layer_mask(mask_inside_limiter, layer_size=3):
    """Creates a mask for the points just outside limiter, with a width=`layer_size`

    Parameters
    ----------
    mask_inside_limiter : np.ndarray
        Mask of the points inside the limiter
    layer_size : int, optional
        Width of the layer outside the limiter, by default 3

    Returns
    -------
    layer_mask : np.ndarray
        Mask of the points outside the limiter within a distance of `layer_size` from the limiter
    """
    nx, ny = np.shape(mask_inside_limiter)
    layer_mask = np.zeros(np.array([nx, ny]) + 2*np.array([layer_size, layer_size]))

    for i in np.arange(-layer_size, layer_size+1)+layer_size:
        for j in np.arange(-layer_size, layer_size+1)+layer_size:
            layer_mask[i:i+nx, j:j+ny] += mask_inside_limiter
    layer_mask = layer_mask[layer_size:layer_size+nx, layer_size:layer_size+ny]
    layer_mask *= (1-mask_inside_limiter)
    layer_mask = (layer_mask>0).astype(bool)
    return layer_mask


def Myy(plasma_pts):
    """Calculates the matrix of mutual inductances between plasma grid points

    Parameters
    ----------
    plasma_pts : np.ndarray
        Array with R and Z coordinates of all the points inside the limiter

    Returns
    -------
    Myy : np.ndarray
        Array of mutual inductances between plasma grid points
    """
    greenm = Greens(plasma_pts[:, np.newaxis, 0], plasma_pts[:, np.newaxis, 1],
                    plasma_pts[np.newaxis, :, 0], plasma_pts[np.newaxis, :, 1])
    return 2*np.pi*greenm


def Mey(plasma_pts):
    """Calculates the matrix of mutual inductances between plasma grid points and all vessel coils

    Parameters
    ----------
    plasma_pts : np.ndarray
        Array with R and Z coordinates of all the points inside the limiter

    Returns
    -------
    Mey : np.ndarray
        Array of mutual inductances between plasma grid points and all vessel coils
    """
    coils_dict = machine_config.coils_dict
    mey = np.zeros((len(coils_dict), len(plasma_pts)))
    for j,labelj in enumerate(coils_dict.keys()):
        greenm = Greens(plasma_pts[:, 0, np.newaxis],
                        plasma_pts[:, 1, np.newaxis],
                        coils_dict[labelj]['coords'][0][np.newaxis, :],
                        coils_dict[labelj]['coords'][1][np.newaxis, :])        
        greenm *= coils_dict[labelj]['polarity'][np.newaxis, :]
        mey[j] = np.sum(greenm, axis=-1)
    return 2*np.pi*mey


