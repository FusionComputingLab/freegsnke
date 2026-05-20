"""
Defines the plasma_current Object, which handles the lumped parameter models
used as an effective circuit equation for the plasma.

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

import numexpr as ne
import numpy as np
from freegs4e.gradshafranov import Greens
from freegs4e.parallel_funcs import threaded_take


class Myy_handler:
    """Object handling all operations which involve the Myy matrix,
    i.e. the mututal inductance matrix of all domain grid points.
    To reduce memory usage, the domain on which myy is built and stored
    is set adaptively, so to cover the plasma. This object handles this
    adaptive aspect.

    """

    def __init__(self, limiter_handler, layer_size=5, tolerance=3, cache_myy=True):
        """Instantiates the object

        Parameters
        ----------
        limiter_handler : FreeGSNKE limiter object, i.e. eq.limiter_handler
            Sets the properties of the domain grid and those of the limiter
        layer_size : int, optional
            Used when recalculating myy.
            A layer of layer_size pixels is added to envelop the mask defined by the
            plasma. This 'broadened' mask defines the pixels included in the myy matrix
            By default 5
        tolerance : int, optional
            Used to check if myy needs recalculating. Myy is not recalculated if
            the mask defined by the plasma region, broadened by tolerance pixels,
            is fully contained in the domain of the current myy matrix,
            By default 3
        """

        limiter_handler.build_reduced_rect_domain()

        self.reduce_rect_domain = limiter_handler.reduce_rect_domain
        self.extract_index_mask = limiter_handler.extract_index_mask
        self.rebuild_map2d = limiter_handler.rebuild_map2d
        self.broaden_mask = limiter_handler.broaden_mask

        self.mask_inside_limiter = limiter_handler.mask_inside_limiter
        self.mask_inside_limiter_red = self.reduce_rect_domain(self.mask_inside_limiter)

        self.idxs_mask_red = self.extract_index_mask(self.mask_inside_limiter_red)

        self.gg = self.grid_greens(
            self.reduce_rect_domain(limiter_handler.eqR),
            self.reduce_rect_domain(limiter_handler.eqZ),
        )

        self.layer_size = layer_size
        self.tolerance = tolerance

        self.cache_myy = cache_myy

    def grid_greens(self, R, Z):
        """Calculates and stores the green function values on the minimal rectangular
        region that fully encompasses the limiter. Uses that the green functions are invariant
        for vertical translations.

        Parameters
        ----------
        R : np.ndarray
            Like eq.R, but on the rectangular reduced domain,
            i.e. self.reduce_rect_domain(limiter_handler.eqR)
        Z : np.ndarray
            Like eq.Z, but on the rectangular reduced domain
        """

        dz = Z[0, 1] - Z[0, 0]
        nZ = np.shape(Z)[1]

        ggreens = Greens(
            R[:, 0][:, np.newaxis, np.newaxis],
            dz * np.arange(nZ)[np.newaxis, np.newaxis, :],
            R[:, 0][np.newaxis, :, np.newaxis],
            0,
        )

        return 2 * np.pi * ggreens

    def build_mask_from_hatIy(self, hatIy, layer_size):
        """Builds the mask that will be used by build_myy_from_mask
        based on the hatIy map. The mask is broadened by a number of pixels
        equal to layer mask. The limiter mask is taken into account.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on reduced plasma domain, e.g. inside the limiter
        layer_size : int, optional
            _description_, by default 3
        """
        hatIy_mask = hatIy > 0
        hatIy_rect_red = self.rebuild_map2d(
            hatIy_mask, self.mask_inside_limiter_red, self.idxs_mask_red
        )
        hatIy_broad_rect_red = self.broaden_mask(hatIy_rect_red, layer_size=layer_size)
        hatIy_broad_rect_red *= self.mask_inside_limiter_red
        return hatIy_broad_rect_red

    def build_Myy_from_mask(self, mask):
        """Build the Myy matrix only including domain points in the input mask

        Parameters
        ----------
        mask : np.ndarray
            mask of the domain points to include.
            Map is defined on the reduced rectangular domain grid,
            i.e. the smallest rectangular domain around limiter mask
            (same size as self.mask_inside_limiter_red)
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

    def force_build_Myy(self, hatIy):
        """Builds the Myy matrix only including domain points in the input vector (not necessarily a mask)

        Parameters
         ----------
         hatIy : np.ndarray
             1d vector on reduced plasma domain, e.g. inside the limiter
        """

        hatIy_broad_rect_red = self.build_mask_from_hatIy(
            hatIy, layer_size=self.layer_size
        )
        self.build_Myy_from_mask(hatIy_broad_rect_red)

    def check_Myy(self, hatIy):
        """Rebuilds myy when the input hatIy, broadened by a number of pixels
        set by tolerance, is not fully inside the current myy_mask
        Note 1. tolerance should be smaller than 'layer_size' in build_mask_from_hatIy
        Note 2. tolerance should be larger than the number of pixels by which the plasma
        is expected to 'move' every timestep of the evolution.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on reduced plasma domain, e.g. inside the limiter
        tolerance : int
            number of pixels by which hatIy should be 'inside self.myy_mask_red'
        """
        hatIy_broad_rect_red = self.build_mask_from_hatIy(
            hatIy, layer_size=self.tolerance
        )
        flag = np.sum(hatIy_broad_rect_red[self.outside_myy_mask])
        return flag

    def dot(self, hatIy):
        """Performs the product with a vector defined on the reduced domain, i.e. inside the limiter.
        Returns a vector on the same domain.

        Parameters
        ----------
        hatIy : np.ndarray
            1d vector on reduced plasma domain, e.g. inside the limiter
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

    def _myy_dot(self, hatIy_myy_red):

        if self.cache_myy:
            return np.dot(self.myy, hatIy_myy_red)

        else:
            nmask = np.sum(self.myy_mask_red)

            inshape = hatIy_myy_red.shape

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

                np.dot(myy_buff[: end - start], hatIy_myy_red, out=result[start:end])

            return result


class fft_Myy:
    """Object handling all operations which involve the Myy matrix,
    i.e. the mututal inductance matrix of all domain grid points.
    To reduce memory usage, the domain on which myy is built and stored
    is set adaptively, so to cover the plasma. This object handles this
    adaptive aspect.

    """

    def __init__(self, limiter_handler, layer_size=None, tolerance=None):
        """Instantiates the object

        Parameters
        ----------
        limiter_handler : FreeGSNKE limiter object, i.e. eq.limiter_handler
            Sets the properties of the domain grid and those of the limiter
        layer_size : int, optional
            Used when recalculating myy.
            A layer of layer_size pixels is added to envelop the mask defined by the
            plasma. This 'broadened' mask defines the pixels included in the myy matrix
            By default 5
        tolerance : int, optional
            Used to check if myy needs recalculating. Myy is not recalculated if
            the mask defined by the plasma region, broadened by tolerance pixels,
            is fully contained in the domain of the current myy matrix,
            By default 3
        """

        self.up_project = limiter_handler.up_project
        self.down_project = limiter_handler.down_project

        eqR = limiter_handler.eqR
        eqZ = limiter_handler.eqZ

        nR = eqR.shape[0]
        nZ = eqZ.shape[1]

        dz = eqZ[0, 1] - eqZ[0, 0]
        Z_1D = np.arange(0, dz * nZ, dz)
        R_1D = eqR[:, 0]

        # Linear convolution length: L = 2*nZ - 1
        L = 2 * nZ - 1
        # L_fft = L // 2 + 1  # rFFT length

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
                out=h_full[start:end, :, :],
            )

        # build symmetric kernel g of length L for *linear* Toeplitz convolution:
        #    g[..., k]   = h[..., k]       for k=0..nZ-1
        #    g[..., L-k] = h[..., k]       for k=1..nZ-1
        g = np.zeros((nR, nR, L), dtype=h_full.dtype)
        k = np.arange(1, nZ)

        g[:, :, 0] = h_full[:, :, 0]
        g[:, :, k] = h_full[:, :, k]
        g[:, :, L - k] = h_full[:, :, k]

        # Perform rFFT of g along Z, and cache
        self.gg_fft = np.fft.rfft(g, axis=2)  # (nR, nR, L_fft)

    def check_Myy(self, hatIy):
        return False

    def force_build_Myy(self, hatIy):
        pass

    def dot(self, hatIy):
        up_hatIy = self.up_project(hatIy)
        Myy_hatIy = self._myy_dot(up_hatIy)
        reduced_prod = self.down_project(Myy_hatIy)
        return reduced_prod

    def _myy_dot(self, x):

        nR, nZ = self.gg_fft.shape[1], self.gg_fft.shape[2]
        x = x.reshape(nR, nZ)

        # Zero-pad vec along Z to length L
        L = 2 * nZ - 1
        pad_width = L - nZ  # equals nZ - 1
        x_padded = np.pad(x, ((0, 0), (0, pad_width)))  # (nR, L)

        # rFFT of padded vec
        x_fft = np.fft.rfft(x_padded, axis=1)  # (nR, L_fft)

        conv_fft = self.gg_fft * x_fft[np.newaxis, :, :]
        conv_fft = conv_fft.sum(axis=1)
        y_full = np.fft.irfft(conv_fft, n=L, axis=1)
        y_full = y_full[:, :nZ]  # .reshape(-1)

        return y_full
