"""
Defines the plasma_current Object, which handles the lumped parameter model
used as an effective circuit equation for the plasma.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

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

import abc

import numexpr as ne
import numpy as np

from .gradshafranov import Greens
from .parallel import threaded_take


def make_Myy_handler(
    domain_type: str,
    limiter_handler: int,
    layer_size: int | None = None,
    tolerance: int | None = None,
    cache_myy: bool | None = None,
):
    """
    Returns an Myy_handler object of appropriate type given the domain_type specified. Note that
    if `domain_type`=="fft" all optional arguments are ignored.

    Parameters
    ----------
    domain_type: str
        Type of domain to define Myy over. Supported values: "reduced", "fft".
    limiter_handler : FreeGSNKE limiter object, i.e. eq.limiter_handler
        Sets the properties of the domain grid and those of the limiter
    layer_size : int, optional
        (for reduced domain only)
        Used when recalculating myy.
        A layer of layer_size pixels is added to envelop the mask defined by the
        plasma. This 'broadened' mask defines the pixels included in the myy matrix
        By default 5
    tolerance : int, optional
        (for reduced domain only)
        Used to check if myy needs recalculating. Myy is not recalculated if
        the mask defined by the plasma region, broadened by tolerance pixels,
        is fully contained in the domain of the current myy matrix,
        By default 3
    cache_myy : bool, optional
        Determines whether the mutual inductance matrix should be pre-calculated and
        stored, or re-calculated every time a product with Myy is computed
        (reducing memory usage by up to an order of magnitude).
        True by default.

    """

    # TODO: perhaps replace `domain_type` with something more generic that directly distinguishes
    # between cached and otf reduced Myy. This may be a good idea depending on how the GPU version
    # is coupled. (options: "real_cached", "real_otf", "fft", "gpu")

    if domain_type == "reduced":

        # Apply defaults
        layer_size = 5 if layer_size is None else layer_size
        tolerance = 3 if tolerance is None else tolerance
        cache_myy = True if cache_myy is None else cache_myy

        return Reduced_Myy_handler(limiter_handler, layer_size, tolerance, cache_myy)

    elif domain_type == "fft":
        return FFT_Myy_handler(limiter_handler)

    else:
        raise ValueError(f"Domain type {domain_type} not recognized.")


class Myy_handler(abc.ABC):
    """
    Abstract base class for objects handling all operations which involve the Myy matrix,
    i.e. the mututal inductance matrix of all domain grid points.
    """

    def make_1D(self, R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Utility function to obtain a 1D vector representation of R and Z

        Parameters
        ----------
        R : np.ndarray
            2D array with the R values of the grid
        Z : np.ndarray
            2D array with the Z values of the grid

        Returns
        -------
        tuple[np.ndarray, np.ndarray}
            1D vector of R-values and 1D vector of Z-values
        """

        dz = Z[0, 1] - Z[0, 0]
        nZ = Z.shape[1]
        Z_1D = np.arange(0, dz * nZ, dz)
        R_1D = R[:, 0]

        return R_1D, Z_1D

    @abc.abstractmethod
    def grid_greens(self, R_1D: np.ndarray, Z_1D: np.ndarray) -> np.ndarray:
        """
        Calculates and stores the Green's function values in the corresponding Myy domain.

        Parameters
        ----------
        R_1D : np.ndarray
            Array of dim=1 representing the R values of the grid in the appropriate Myy domain.
        Z_1D : np.ndarray
            Array of dim=1 representing the Z values of the grid in the appropriate Myy domain.

        Returns
        -------
        np.ndarray
            Array containing the Greens's function in the appropriate representation.
        """
        pass

    @abc.abstractmethod
    def check_Myy(self, hatIy: np.ndarray) -> int | bool:
        """As defined in Reduced_Myy_handler. Definition enforced for compatibility with
        non_linear_solver"""
        pass

    @abc.abstractmethod
    def force_build_Myy(self, hatIy: np.ndarray) -> None:
        """As defined in Reduced_Myy_handler. Definition enforced for compatibility with
        non_linear_solver"""
        pass

    @abc.abstractmethod
    def dot(self, hatIy: np.ndarray) -> np.ndarray:
        """
        Performs the product with a vector defined on the reduced plasma domain, i.e. inside the
        limiter, independently of the domain of Myy. Returns a vector on the input domain.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on reduced plasma domain, e.g. inside the limiter
        """
        pass

    @abc.abstractmethod
    def _myy_dot(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs the dot product Myy@vector for a vector already projected to the domain
        in which Myy is defined. Returns a vector on the input domain.

        Returns
        -------
        np.ndarray
            1d vector on the input domain.
        """
        pass


# TODO: after merge: rename reduced to adaptive? Might be clearer
class Reduced_Myy_handler(Myy_handler):
    """
    Handles all operations involving the Myy matrix.

    Myy is the mutual inductance matrix of the domain grid points. To reduce
    memory usage, the domain on which Myy is built and stored is set
    adaptively, so as to cover the plasma. This object handles that adaptivity.
    """

    def __init__(
        self,
        limiter_handler: object,
        layer_size: int = 5,
        tolerance: int = 3,
        cache_myy=True,
    ) -> None:
        """
        Instantiate the object.

        Parameters
        ----------
        limiter_handler : object
            FreeGSNKE limiter handler (i.e. ``eq.limiter_handler``). Sets the
            properties of the domain grid and those of the limiter.
        layer_size : int, optional
            Used when recalculating Myy. A layer of `layer_size` pixels is added
            to envelop the mask defined by the plasma. This 'broadened' mask
            defines the pixels included in the Myy matrix.
        tolerance : int, optional
            Used to check if Myy needs recalculating. Myy is not recalculated if
            the mask defined by the plasma region, broadened by `tolerance`
            pixels, is fully contained in the domain of the current Myy matrix.
        """

        limiter_handler.build_reduced_rect_domain()

        self.reduce_rect_domain = limiter_handler.reduce_rect_domain
        self.extract_index_mask = limiter_handler.extract_index_mask
        self.rebuild_map2d = limiter_handler.rebuild_map2d
        self.broaden_mask = limiter_handler.broaden_mask

        self.mask_inside_limiter = limiter_handler.mask_inside_limiter
        self.mask_inside_limiter_red = self.reduce_rect_domain(self.mask_inside_limiter)

        self.idxs_mask_red = self.extract_index_mask(self.mask_inside_limiter_red)

        R_red = self.reduce_rect_domain(limiter_handler.eqR)
        Z_red = self.reduce_rect_domain(limiter_handler.eqZ)
        R_1D, Z_1D = self.make_1D(R_red, Z_red)

        self.gg = self.grid_greens(R_1D, Z_1D)

        self.layer_size = layer_size
        self.tolerance = tolerance

        self.cache_myy = cache_myy

    def grid_greens(self, R_1D, Z_1D):
        """
        Calculate the Green's function values on the reduced rectangular domain.

        The values are computed on the minimal rectangular region that fully
        encompasses the limiter, using that the Green's functions are invariant
        under vertical translations.

        Parameters
        ----------
        R_1D : np.ndarray
            Like eq.R, but on the rectangular reduced domain and only in 1D, i.e.
            ``self.reduce_rect_domain(limiter_handler.eqR)[:, 0]``
        Z_1D : np.ndarray
            Like eq.Z, but on the rectangular reduced domain and only in 1D,
            uniform spacing is assumed.

        Returns
        -------
        np.ndarray
            Array of shape (nR, nR, nZ) with entry ``[i, j, k]`` equal to
            2 pi times the Green's function between ``(R[i, 0], k dz)`` and
            ``(R[j, 0], 0)``.
        """

        ggreens = Greens(
            R_1D[:, np.newaxis, np.newaxis],
            Z_1D[np.newaxis, np.newaxis, :],
            R_1D[np.newaxis, :, np.newaxis],
            0,
            scale_factor=2.0 * np.pi,
        )

        return ggreens

    def build_mask_from_hatIy(self, hatIy: np.ndarray, layer_size: int) -> np.ndarray:
        """
        Build the mask used by `build_Myy_from_mask` from a hatIy map.

        The mask is broadened by `layer_size` pixels, and the limiter mask is
        taken into account.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.
        layer_size : int
            Number of pixels by which the mask is broadened.

        Returns
        -------
        np.ndarray
            Boolean mask on the reduced rectangular domain, of the same shape
            as ``self.mask_inside_limiter_red``.
        """
        hatIy_mask = hatIy > 0
        hatIy_rect_red = self.rebuild_map2d(
            hatIy_mask, self.mask_inside_limiter_red, self.idxs_mask_red
        )
        hatIy_broad_rect_red = self.broaden_mask(hatIy_rect_red, layer_size=layer_size)
        hatIy_broad_rect_red *= self.mask_inside_limiter_red
        return hatIy_broad_rect_red

    def build_Myy_from_mask(self, mask: np.ndarray) -> None:
        """
        Build the Myy matrix only including domain points in the input mask.

        Sets ``myy_mask_red``, ``outside_myy_mask``, ``idxs_myy_mask_red`` and
        ``myy``.

        Parameters
        ----------
        mask : np.ndarray
            Mask of the domain points to include. Defined on the reduced
            rectangular domain grid, i.e. the smallest rectangular domain around
            the limiter mask (same size as ``self.mask_inside_limiter_red``).
        """
        self.myy_mask_red = mask
        self.outside_myy_mask = np.logical_not(mask)

        nmask = np.sum(mask)

        self.idxs_myy_mask_red = self.extract_index_mask(mask)

        if self.cache_myy:
            dz_idxs = self.idxs_myy_mask_red[1]
            r_idxs = self.idxs_myy_mask_red[0]

            self.myy = np.empty((nmask, nmask))

            d1, d2, d3 = self.gg.shape
            d23 = d2 * d3

            # important to keep this as a python loop, do not try to vectorize
            for i in range(nmask):

                idxs1 = r_idxs
                idxs2 = r_idxs[i]

                idxs3 = np.abs(dz_idxs[i] - dz_idxs)
                idcs = idxs1 * d23 + idxs2 * d3 + idxs3

                # same as self.myy[i] = self.gg.reshape(-1)[idcs] but faster
                np.take(self.gg, idcs, out=self.myy[i], mode="wrap")

    def force_build_Myy(self, hatIy: np.ndarray) -> None:
        """
        Build the Myy matrix only including domain points in the input vector.

        The vector is broadened by ``layer_size`` pixels and need not be a mask.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.
        """

        hatIy_broad_rect_red = self.build_mask_from_hatIy(
            hatIy, layer_size=self.layer_size
        )
        self.build_Myy_from_mask(hatIy_broad_rect_red)

    def check_Myy(self, hatIy: np.ndarray) -> int:
        """
        Check whether Myy needs rebuilding for the input hatIy.

        hatIy, broadened by ``tolerance`` pixels, is compared with the current
        Myy mask. Note 1: tolerance should be smaller than ``layer_size`` in
        `build_mask_from_hatIy`. Note 2: tolerance should be larger than the
        number of pixels by which the plasma is expected to move every
        timestep of the evolution.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.

        Returns
        -------
        int
            Number of broadened-hatIy pixels lying outside the current Myy
            mask. Nonzero means Myy should be rebuilt.
        """

        hatIy_broad_rect_red = self.build_mask_from_hatIy(
            hatIy, layer_size=self.tolerance
        )
        flag = np.sum(hatIy_broad_rect_red[self.outside_myy_mask])
        return flag

    def dot(self, hatIy: np.ndarray) -> np.ndarray:
        """
        Multiply Myy by a vector defined on the reduced plasma domain.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on the reduced plasma domain, e.g. inside the limiter.

        Returns
        -------
        np.ndarray
            Resulting 1d vector on the same domain.
        """

        # first bring hatIy from the reduced domain to the current myy domain
        hatIy_rect_red = self.rebuild_map2d(
            hatIy, self.mask_inside_limiter_red, self.idxs_mask_red
        )
        hatIy_myy_red = hatIy_rect_red[self.myy_mask_red]

        # perform the dot product
        result = self._myy_dot(hatIy_myy_red)

        # bring result back to the reduced plasma domain
        result_rect_red = self.rebuild_map2d(
            result, self.mask_inside_limiter_red, self.idxs_myy_mask_red
        )
        result_red = result_rect_red[self.mask_inside_limiter_red]

        return result_red

    def _myy_dot(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs the dot product Myy@vector for a vector already projected onto the reduced
        rectangular domain in which Myy is defined.

        If myy is cached, np.dot is called directly. Otherwise, the product is batch-computed using
        an on-the-fly calculation for blocks of myy (myy calculation is parallelized).

        Returns a vector on the input domain.

        Parameters
        ----------
        vector: np.ndarray
            1d vector projected onto the reduced rectangular domain of Myy
        """

        if self.cache_myy:
            return np.dot(self.myy, vector)

        else:
            nmask = np.sum(self.myy_mask_red)

            inshape = vector.shape

            if len(inshape) > 1:
                outshape = (nmask, *inshape[:-2], inshape[-1])
            else:
                outshape = (nmask,)

            dz_idxs = self.idxs_myy_mask_red[1]
            r_idxs = self.idxs_myy_mask_red[0]

            d1, d2, d3 = self.gg.shape
            d23 = d2 * d3

            num_slices = 20  # TODO: perhaps instead of fixing the number of slices, fix the block size?
            step = (nmask - 1) // num_slices + 1

            idcs = np.empty((step, nmask), dtype=np.int64)
            myy_buff = np.empty(idcs.shape)
            result = np.empty(outshape)

            for i in range(num_slices):

                start = i * step
                end = start + step
                end = min(end, nmask)

                idxs1 = r_idxs[np.newaxis]
                idxs2 = r_idxs[start:end, np.newaxis]
                idxs3a = dz_idxs[start:end, np.newaxis]
                idxs3b = dz_idxs[np.newaxis]

                # idcs is flattened version of (idxs1,idxs2,idxs3)
                # TODO: check why there are casting issues with abs()
                ne.evaluate(
                    "idxs1*d23 + idxs2*d3 + abs(idxs3a-idxs3b)",
                    out=idcs[: end - start],
                    casting="unsafe",
                )

                # same as self.myy_buff[:end-start] = self.gg.reshape(-1)[idcs_slice] but faster
                threaded_take(
                    self.gg,
                    idcs[: end - start],
                    out=myy_buff[: end - start],
                    mode="wrap",
                )

                np.dot(myy_buff[: end - start], vector, out=result[start:end])

            return result


class FFT_Myy_handler(Myy_handler):
    """Object handling all operations which involve the Myy matrix,
    i.e. the mututal inductance matrix of all domain grid points.
    To reduce memory usage, the Green's function is computed in Fourier
    space, which reduces the size scaling from quartic to cubic.
    """

    def __init__(self, limiter_handler: object) -> None:
        """Instantiates the object

        Parameters
        ----------
        limiter_handler : FreeGSNKE limiter object, i.e. eq.limiter_handler
            Sets the properties of the domain grid and those of the limiter
        """

        self.up_project = limiter_handler.up_project
        self.down_project = limiter_handler.down_project

        R_1D, Z_1D = self.make_1D(limiter_handler.eqR, limiter_handler.eqZ)

        self.gg = self.grid_greens(R_1D, Z_1D)  # in Fourier space, (nR, nR, L_fft)

    def grid_greens(self, R_1D: np.ndarray, Z_1D: np.ndarray) -> np.ndarray:
        """
        Calculates the Green's function values on the Fourier space of the full geometric
        domain defined by the limiter. Relies on the fact that Green's functions are invariant
        for vertical translations.

        Parameters
        ----------
        R : np.ndarray
            Like eq.R, but only in 1D
            i.e. self.make_1D(limiter_handler.eqR,...)
        Z : np.ndarray
            Like eq.Z, but only in 1D

        Returns
        -------
        gg_fft: np.ndarray
            Array of shape (nR,nR,L_fft) containing the values of the Green's function in Fourier
            space. (L_fft = (2*nZ - 1) // 2 + 1)
        """

        # TODO: worth evaluating whether L_fft is ever different from nZ

        nR = len(R_1D)
        nZ = len(Z_1D)

        # Linear convolution length: L = 2*nZ - 1
        L = 2 * nZ - 1

        # h[i,j,k] = 2π * Greens(R_i, Z_k, R_j, 0)
        h_full = np.empty((nR, nR, nZ))

        num_slices = 10  # fine-tuned to balance memory vs. compute needs
        step = nR // num_slices

        for i in range(num_slices):

            start = i * step
            end = start + step
            end = end if i != num_slices - 1 else nR  # last slice gets the remainder

            # Fill up slice of h_full in-place. Applies 2π factor automatically.
            Greens(
                R_1D[start:end, np.newaxis, np.newaxis],
                Z_1D[np.newaxis, np.newaxis, :],
                R_1D[np.newaxis, :, np.newaxis],
                0.0,
                scale_factor=2.0 * np.pi,
                out=h_full[start:end],
            )

        # build symmetric kernel g of length L for *linear* Toeplitz convolution:
        #    g[..., k]   = h[..., k]       for k=0..nZ-1
        #    g[..., L-k] = h[..., k]       for k=1..nZ-1
        g = np.zeros((nR, nR, L), dtype=h_full.dtype)
        k = np.arange(1, nZ)

        g[:, :, 0] = h_full[:, :, 0]
        g[:, :, k] = h_full[:, :, k]
        g[:, :, L - k] = h_full[:, :, k]

        # Perform rFFT of g along Z, and return result
        gg_fft = np.fft.rfft(g, axis=2)

        return gg_fft  # (nR, nR, L_fft)

    def check_Myy(self, hatIy: np.ndarray) -> bool:
        """Dummy, always returns False. Defined only for compatibility with non_linear_solver"""
        return False

    def force_build_Myy(self, hatIy: np.ndarray) -> None:
        """Ignored. Defined only for compatibility with non_linear_solver"""
        pass

    def dot(self, hatIy: np.ndarray) -> np.ndarray:
        """
        Performs the product with a vector defined on the reduced domain, i.e. inside the limiter.
        Returns a vector on the input domain.

        Parameters
        ----------
        hatIy: np.ndarray
            1d vector on reduced plasma domain, e.g. inside the limiter

        Returns
        -------
        np.ndarray
            Resulting 1d vector on the same domain.
        """
        up_hatIy = self.up_project(hatIy)
        Myy_hatIy = self._myy_dot(up_hatIy)
        reduced_prod = self.down_project(Myy_hatIy)
        return reduced_prod

    def _myy_dot(self, vector: np.ndarray) -> np.ndarray:
        """
        Performs the dot product Myy@vector for a vector that has already been up-projected onto
        the full grid domain (where Myy is defined), but NOT yet transformed into Fourier space.

        Returns a vector on the input domain.

        Parameters
        ----------
        vector: np.ndarray
            1d vector on the full domain, in real space

        Returns
        -------
        np.ndarray
            Resulting 1d vector on the same domain.
        """

        nR, nZ = self.gg.shape[1], self.gg.shape[2]
        x = vector.reshape(nR, nZ)

        # Zero-pad vec along Z to length L
        L = 2 * nZ - 1
        pad_width = L - nZ  # equals nZ - 1
        x_padded = np.pad(x, ((0, 0), (0, pad_width)))  # (nR, L)

        # rFFT of padded vec
        x_fft = np.fft.rfft(x_padded, axis=1)  # (nR, L_fft)

        gg_fft = self.gg
        x_fft = x_fft[np.newaxis, :, :]
        conv_fft = ne.evaluate("sum(gg_fft*x_fft, axis=1)")

        y_full = np.fft.irfft(conv_fft, n=L, axis=1)
        y_full = y_full[:, :nZ]  # .reshape(-1)

        return y_full
