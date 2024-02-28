import numpy as np
from freegs.gradshafranov import Greens

# from . import machine_config


def make_layer_mask(plasma_domain_mask, layer_size=3):
    """Creates a mask for the points just outside the input plasma_domain_mask, with a width=`layer_size`

    Parameters
    ----------
    layer_size : int, optional
        Width of the layer outside the limiter, by default 3

    Returns
    -------
    layer_mask : np.ndarray
        Mask of the points outside the limiter within a distance of `layer_size` from the limiter
    """
    nx, ny = np.shape(plasma_domain_mask)
    layer_mask = np.zeros(np.array([nx, ny]) + 2 * np.array([layer_size, layer_size]))

    for i in np.arange(-layer_size, layer_size + 1) + layer_size:
        for j in np.arange(-layer_size, layer_size + 1) + layer_size:
            layer_mask[i : i + nx, j : j + ny] += plasma_domain_mask
    layer_mask = layer_mask[layer_size : layer_size + nx, layer_size : layer_size + ny]
    layer_mask *= 1 - plasma_domain_mask
    layer_mask = (layer_mask > 0).astype(bool)
    return layer_mask


class Grids:
    """Generates the reduced domain grid on which the plasma is allowed to live.
    This allows for a significant reduction of the size of the matrices containing
    plasma-plasma and plasma-metal induction data.
    Handles the transfromation between full domain grid and reduced grid and viceversa.
    """

    def __init__(self, eq, plasma_domain_mask):
        """Instantiates the class.

        Parameters
        ----------
        eq : freeGS equilibrium object
            Used to extract domain information
        plasma_domain_mask : np.ndarray
            Mask of domain points to be included in the reduced plasma grid.
            Needs to have the same shape as eq.R and eq.Z
        """

        self.R = eq.R
        self.Z = eq.Z
        self.nx, self.ny = np.shape(eq.R)

        # area factor for Iy
        dR = eq.R[1, 0] - eq.R[0, 0]
        dZ = eq.Z[0, 1] - eq.Z[0, 0]
        self.dRdZ = dR * dZ

        self.map2d = np.zeros_like(eq.R)

        # the if statement should be eliminated in favour of actual input
        # The one below is a 'MASTU by-eye'
        # if plasma_domain_mask is None:
        #     plasma_domain_mask = np.ones_like(self.R)
        #     plasma_domain_mask *= (self.R>0.265)*(self.R<1.582)
        #     plasma_domain_mask *= (self.Z<.95+1*self.R)*(self.Z>-.95-1*self.R)
        #     plasma_domain_mask *= (self.Z<-1.98+9.*self.R)*(self.Z>1.98-9.*self.R)
        #     plasma_domain_mask *= (self.Z<2.26-1.1*self.R)*(self.Z>-2.26+1.1*self.R)
        #     plasma_domain_mask = plasma_domain_mask.astype(bool)
        self.plasma_domain_mask = plasma_domain_mask

        # Extracts R and Z coordinates of the grid points in the reduced plasma domain
        self.plasma_pts = np.concatenate(
            (
                self.R[self.plasma_domain_mask][:, np.newaxis],
                self.Z[self.plasma_domain_mask][:, np.newaxis],
            ),
            axis=-1,
        )

        self.idxs_mask = np.mgrid[0 : self.nx, 0 : self.ny][
            np.tile(self.plasma_domain_mask, (2, 1, 1))
        ].reshape(2, -1)

        self.make_layer_mask()

    def Iy_from_jtor(self, jtor):
        """Generates 1d vector of plasma current values at the grid points of the reduced plasma domain.

        Parameters
        ----------
        jtor : np.ndarray
            Plasma current distribution on full domain. np.shape(jtor) = np.shape(eq.R)

        Returns
        -------
        Iy : np.ndarray
            Reduced 1d plasma current vector
        """
        Iy = jtor[self.plasma_domain_mask] * self.dRdZ
        return Iy

    def normalize_sum(self, Iy, epsilon=1e-6):
        """Normalises any vector by the linear sum of its elements.

        Parameters
        ----------
        jtor : np.ndarray
            Plasma current distribution on full domain. np.shape(jtor) = np.shape(eq.R)
        epsilon : float, optional
            avoid divergences, by default 1e-6

        Returns
        -------
        _type_
            _description_
        """
        hat_Iy = Iy / (np.sum(Iy) + epsilon)
        return hat_Iy

    def hat_Iy_from_jtor(self, jtor):
        """Generates 1d vector on reduced plasma domain for the normalised vector
        $$ Jtor*dR*dZ/I_p $$.


        Parameters
        ----------
        jtor : np.ndarray
            Plasma current distribution on full domain. np.shape(jtor) = np.shape(eq.R)
        epsilon : float, optional
            avoid divergences, by default 1e-6

        Returns
        -------
        hat_Iy : np.ndarray
            Reduced 1d plasma current vector, normalized to total plasma current

        """
        hat_Iy = jtor[self.plasma_domain_mask]
        hat_Iy = self.normalize_sum(hat_Iy)
        return hat_Iy

    def check_if_outside_domain(self, jtor):
        return np.sum(jtor[self.layer_mask])

    def rebuild_map2d(self, reduced_vector):
        """Rebuilds 2d map on full domain corresponding to 1d vector
        reduced_vector on smaller plasma domain

        Parameters
        ----------
        reduced_vector : np.ndarray
            1d vector on reduced plasma domain

        Returns
        -------
        self.map2d : np.ndarray
            2d map on full domain. Values on gridpoints outside the
            reduced plasma domain are set to zero.
        """
        self.map2d[self.idxs_mask[0], self.idxs_mask[1]] = reduced_vector
        return self.map2d

    def make_layer_mask(self, layer_size=3):
        """Creates a mask for the points just outside the reduced domain, with a width=`layer_size`

        Parameters
        ----------
        layer_size : int, optional
            Width of the layer outside the limiter, by default 3

        Returns
        -------
        layer_mask : np.ndarray
            Mask of the points outside the limiter within a distance of `layer_size` from the limiter
        """

        layer_mask = np.zeros(
            np.array([self.nx, self.ny]) + 2 * np.array([layer_size, layer_size])
        )

        for i in np.arange(-layer_size, layer_size + 1) + layer_size:
            for j in np.arange(-layer_size, layer_size + 1) + layer_size:
                layer_mask[i : i + self.nx, j : j + self.ny] += self.plasma_domain_mask
        layer_mask = layer_mask[
            layer_size : layer_size + self.nx, layer_size : layer_size + self.ny
        ]
        layer_mask *= 1 - self.plasma_domain_mask
        layer_mask = (layer_mask > 0).astype(bool)
        self.layer_mask = layer_mask

    # def Myy(
    #     self,
    # ):
    #     """Calculates the matrix of mutual inductances between plasma grid points

    #     Parameters
    #     ----------
    #     plasma_pts : np.ndarray
    #         Array with R and Z coordinates of all the points inside the limiter

    #     Returns
    #     -------
    #     Myy : np.ndarray
    #         Array of mutual inductances between plasma grid points
    #     """
    #     greenm = Greens(
    #         self.plasma_pts[:, np.newaxis, 0],
    #         self.plasma_pts[:, np.newaxis, 1],
    #         self.plasma_pts[np.newaxis, :, 0],
    #         self.plasma_pts[np.newaxis, :, 1],
    #     )
    #     return 2 * np.pi * greenm

    # def Mey(
    #     self,
    # ):
    #     """Calculates the matrix of mutual inductances between plasma grid points and all vessel coils

    #     Parameters
    #     ----------
    #     plasma_pts : np.ndarray
    #         Array with R and Z coordinates of all the points inside the limiter

    #     Returns
    #     -------
    #     Mey : np.ndarray
    #         Array of mutual inductances between plasma grid points and all vessel coils
    #     """
    #     coils_dict = machine_config.coils_dict
    #     mey = np.zeros((machine_config.n_coils, len(self.plasma_pts)))
    #     for j, labelj in enumerate(machine_config.coils_order):
    #         greenm = Greens(
    #             self.plasma_pts[:, 0, np.newaxis],
    #             self.plasma_pts[:, 1, np.newaxis],
    #             coils_dict[labelj]["coords"][0][np.newaxis, :],
    #             coils_dict[labelj]["coords"][1][np.newaxis, :],
    #         )
    #         greenm *= coils_dict[labelj]["polarity"][np.newaxis, :]
    #         mey[j] = np.sum(greenm, axis=-1)
    #     return 2 * np.pi * mey
