import numpy as np
from freegs.gradshafranov import Greens
from . import MASTU_coils



# extracts points inside the wall
# R = eq.R, Z = eq.Z
def define_reduced_plasma_grid(R, Z):

    mask_inside_limiter = np.ones_like(R)
    mask_inside_limiter *= (R>0.263)*(R<1.584)
    mask_inside_limiter *= (Z<.96+1*R)*(Z>-.96-1*R)
    mask_inside_limiter *= (Z<-2+9.*R)*(Z>2-9.*R)
    mask_inside_limiter *= (Z<2.28-1.1*R)*(Z>-2.28+1.1*R)
    mask_inside_limiter = mask_inside_limiter.astype(bool)

    plasma_pts = np.concatenate((R[mask_inside_limiter][:,np.newaxis],
                                 Z[mask_inside_limiter][:,np.newaxis]), axis=-1)

    return plasma_pts, mask_inside_limiter


# calculate Myy: matrix of mutual inductunces between plasma grid points
def Myy(plasma_pts):
    greenm = Greens(plasma_pts[:, np.newaxis, 0], plasma_pts[:, np.newaxis, 1],
                    plasma_pts[np.newaxis, :, 0], plasma_pts[np.newaxis, :, 1])
    return 2*np.pi*greenm


# calculate Mey: matrix of mutual indicances between plasma grid points and all vessel coils
def Mey(plasma_pts):
    coils_dict = MASTU_coils.coils_dict
    mey = np.zeros((len(coils_dict), len(plasma_pts)))
    for j,labelj in enumerate(coils_dict.keys()):
        greenm = Greens(plasma_pts[:, 0, np.newaxis],
                        plasma_pts[:, 1, np.newaxis],
                        coils_dict[labelj]['coords'][0][np.newaxis, :],
                        coils_dict[labelj]['coords'][1][np.newaxis, :])        
        greenm *= coils_dict[labelj]['polarity'][np.newaxis, :]
        mey[j] = np.sum(greenm, axis=-1)
    return 2*np.pi*mey


