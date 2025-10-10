import numpy as np

from .copying import copy_into


class Jtor_refiner:
    """Class to allow for the refinement of the toroidal plasma current Jtor.
    Currently applied to the Lao85 profile family when 'refine_flag=True'.
    Only grid cells that are crossed by the separatrix are refined.
    """

    def __init__(self, eq, nnx, nny):
        """Instantiates the object and prepares necessary quantities.

        Parameters
        ----------
        eq : freegs4e Equilibrium object
            Specifies the domain properties
        nnx : even integer
            refinement factor in the R direction
        nny : even integer
            refinement factor in the Z direction
        """

        self.eqR = eq.R
        self.eqZ = eq.Z
        self.dR = self.eqR[1, 0] - self.eqR[0, 0]
        self.dZ = self.eqZ[0, 1] - self.eqZ[0, 0]
        self.dRdZ = self.dR * self.dZ
        self.nx, self.ny = np.shape(eq.R)
        self.nxny = self.nx * self.ny

        self.nnx = nnx
        self.nny = nny
        self.hnnx = nnx // 2
        self.hnny = nny // 2
        self.nnxy = nnx * nny

        self.path = eq.limiter_handler.path
        self.prepare_for_refinement()

        self.edges_mask = np.ones_like(eq.R)
        self.edges_mask[0, :] = 0
        self.edges_mask[:, 0] = 0
        self.edges_mask[-1, :] = 0
        self.edges_mask[:, -1] = 0

    def copy(self):
        obj = type(self).__new__(type(self))

        copy_into(self, obj, "eqR", mutable=True)
        copy_into(self, obj, "eqZ", mutable=True)

        copy_into(self, obj, "dR")
        copy_into(self, obj, "dZ")
        copy_into(self, obj, "dRdZ")
        copy_into(self, obj, "nx")
        copy_into(self, obj, "ny")
        copy_into(self, obj, "nxny")
        copy_into(self, obj, "nnx")
        copy_into(self, obj, "nny")
        copy_into(self, obj, "hnnx")
        copy_into(self, obj, "hnny")
        copy_into(self, obj, "nnxy")

        obj.prepare_for_refinement()

        copy_into(self, obj, "edges_mask", mutable=True)
        copy_into(self, obj, "lcfs_mask", mutable=True, strict=False)
        copy_into(self, obj, "value_mask", mutable=True, strict=False)
        copy_into(self, obj, "gradient_mask", mutable=True, strict=False)
        copy_into(self, obj, "mask_to_refine", mutable=True, strict=False)

        return obj

    def prepare_for_refinement(
        self,
    ):
        """Prepares necessary quantities to operate refinement."""
        self.Ridx = np.tile(np.arange(self.nx), (self.ny, 1)).T
        self.Zidx = np.tile(np.arange(self.ny), (self.nx, 1))

        self.xx = np.linspace(0, 1 - 1 / self.nnx, self.nnx) + 1 / (2 * self.nnx)
        self.yy = np.linspace(0, 1 - 1 / self.nny, self.nny) + 1 / (2 * self.nny)
        self.xxc = self.xx - 0.5
        self.yyc = self.yy - 0.5

        self.xxx = np.concatenate(
            (1 - self.xx[:, np.newaxis], self.xx[:, np.newaxis]), axis=-1
        )
        self.yyy = np.concatenate(
            (1 - self.yy[:, np.newaxis], self.yy[:, np.newaxis]), axis=-1
        )
        self.xxxx = np.concatenate(
            (
                self.xxx[np.newaxis, : self.hnnx],
                self.xxx[np.newaxis, self.hnnx :],
                self.xxx[np.newaxis, self.hnnx :],
                self.xxx[np.newaxis, : self.hnnx],
            ),
            axis=0,
        )
        self.yyyy = np.concatenate(
            (
                self.yyy[np.newaxis, : self.hnny],
                self.yyy[np.newaxis, : self.hnny],
                self.yyy[np.newaxis, self.hnny :],
                self.yyy[np.newaxis, self.hnny :],
            ),
            axis=0,
        )

        fullr = np.tile(
            (
                self.eqR[:, :, np.newaxis]
                + self.dR * self.xxc[np.newaxis, np.newaxis, :]
            )[:, :, :, np.newaxis],
            [1, 1, 1, self.nny],
        )
        fullz = np.tile(
            (
                self.eqZ[:, :, np.newaxis]
                + self.dZ * self.yyc[np.newaxis, np.newaxis, :]
            )[:, :, np.newaxis, :],
            [1, 1, self.nnx, 1],
        )
        fullg = np.concatenate(
            (fullr[:, :, :, :, np.newaxis], fullz[:, :, :, :, np.newaxis]), axis=-1
        )
        full_masks = self.path.contains_points(fullg.reshape(-1, 2))
        # these are the refined masks of points inside the limiter
        self.full_masks = full_masks.reshape(self.nx, self.ny, self.nnx, self.nny)

        srr, szz = np.meshgrid(np.arange(self.nnx), np.arange(self.nny), indexing="ij")
        quartermasks = np.zeros((self.nnx, self.nny, 4))
        quartermasks[:, :, 2] = (srr < (self.nnx / 2)) * (szz < (self.nny / 2))
        quartermasks[:, :, 3] = (srr >= (self.nnx / 2)) * (szz < (self.nny / 2))
        quartermasks[:, :, 1] = (srr < (self.nnx / 2)) * (szz >= (self.nny / 2))
        quartermasks[:, :, 0] = (srr >= (self.nnx / 2)) * (szz >= (self.nny / 2))
        self.quartermasks = quartermasks

    def get_indexes_for_refinement(self, mask_to_refine):
        """Generates the indexes of psi values to be used for bilinear interpolation.

        Parameters
        ----------
        mask_to_refine : np.array
            Mask of all domain cells to be refined

        Returns
        -------
        np.array
            indexes of psi values to be used for bilinear interpolation
            4 points per cell to refine, already set in 2-by-2 matrix for vectorised interpolation
            dimensions = (no of cells to refine, 2, 2)
        """
        RRidxs = np.concatenate(
            (
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
            ),
            axis=1,
        )

        ZZidxs = np.concatenate(
            (
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
            ),
            axis=1,
        )

        return RRidxs, ZZidxs

    def build_jtor_value_mask(self, unrefined_jtor, threshold, quantiles=(0.5, 0.9)):
        """Selects the cells that need to be refined based on their value of jtor.
        Selection is such that it includes cells where jtor exceeds the value calculated
        based on the quantiles and threshold[0].

        Parameters
        ----------
        unrefined_jtor : np.array
            The jtor distribution
        thresholds : float
            the relevant value (in the tuple) used to identify where to apply refinement
        """

        jtor_quantiles = np.quantile(unrefined_jtor.reshape(-1), quantiles)
        mask = (unrefined_jtor - jtor_quantiles[0]) > threshold * (
            jtor_quantiles[1] - jtor_quantiles[0]
        )
        return mask

    def build_jtor_gradient_mask(self, unrefined_jtor, threshold, quantiles=(0.5, 0.9)):
        """Selects the cells that need to be refined based on their value of the gradient of jtor.
        Selection is such that it includes cells where the norm of the gradient exceeds the value calculated
        based on the quantiles and threshold[1].

        Parameters
        ----------
        unrefined_jtor : np.array
            The jtor distribution
        thresholds : float
            the relevant value (in the tuple) used to identify where to apply refinement
        """
        gradient_mask = np.zeros_like(unrefined_jtor)

        # right
        right_gradient = np.abs(unrefined_jtor[:-1, :-1] - unrefined_jtor[1:, :-1])
        right_gradient = self.build_jtor_value_mask(
            right_gradient, threshold, quantiles
        )
        # include both indexes in refinement:
        gradient_mask[:-1, :-1] += right_gradient
        gradient_mask[1:, :-1] += right_gradient

        # up
        up_gradient = np.abs(unrefined_jtor[:-1, :-1] - unrefined_jtor[:-1, 1:])
        up_gradient = self.build_jtor_value_mask(up_gradient, threshold, quantiles)
        # include both indexes in refinement:
        gradient_mask[:-1, :-1] += up_gradient
        gradient_mask[:-1, 1:] += up_gradient

        return gradient_mask > 0

    def build_LCFS_mask(self, core_mask):
        """Builds a mask composed of all gridpoints connected to edges crossed by the LCFS.
        These trigger refinement.

        Parameters
        ----------
        core_mask : np.array
            Plasma core mask on the standard domain (self.nx, self.ny)
        """

        core_mask = core_mask.astype(float)
        lcfs_mask = np.zeros_like(core_mask)
        # right
        right_mask = core_mask[:-1, :] + core_mask[1:, :] == 1
        # include both indexes in refinement:
        lcfs_mask[:-1, :] += right_mask
        lcfs_mask[1:, :] += right_mask
        # # include one more pixel
        # lcfs_mask[:-2, :] += right_mask[1:,:]
        # lcfs_mask[2:, :] += right_mask[:-1,:]
        # up
        up_mask = core_mask[:, :-1] + core_mask[:, 1:] == 1
        # include both indexes in refinement:
        lcfs_mask[:, :-1] += up_mask
        lcfs_mask[:, 1:] += up_mask
        # # include one more pixel
        # lcfs_mask[:, :-2] += up_mask[:,1:]
        # lcfs_mask[:, 2:] += up_mask[:,:-1]
        return lcfs_mask

    def build_mask_to_refine(self, unrefined_jtor, core_mask, thresholds):
        """Selects the cells that need to be refined, using the user-defined thresholds

        Parameters
        ----------
        unrefined_jtor : np.array
            The jtor distribution
        core_mask : np.array
            Plasma core mask on the standard domain (self.nx, self.ny)
        thresholds : tuple (threshold for jtor criterion, threshold for gradient criterion)
            tuple of values used to identify where to apply refinement, by default None
        """

        mask_to_refine = np.zeros_like(unrefined_jtor)

        # include all cells that are crossed by the lcfs:
        self.lcfs_mask = self.build_LCFS_mask(core_mask)
        mask_to_refine += self.lcfs_mask

        # include cells that warrant refinement according to criterion on jtor value:
        self.value_mask = self.build_jtor_value_mask(unrefined_jtor, thresholds[0])
        mask_to_refine += self.value_mask

        # include cells that warrant refinement according to criterion on gradient value:
        self.gradient_mask = self.build_jtor_gradient_mask(
            unrefined_jtor, thresholds[1]
        )
        mask_to_refine += self.gradient_mask

        # remove all edges, as these cannot be refined
        mask_to_refine *= self.edges_mask

        # make bool mask
        self.mask_to_refine = mask_to_refine.astype(bool)

    def build_bilinear_psi_interp(self, psi, core_mask, unrefined_jtor, thresholds):
        """Builds the mask of cells on which to operate refinement.
        Cells that are crossed by the separatrix and cells with large gradient on jtor are considered.
        Refines psi in the same cells.

        Parameters
        ----------
        psi : np.array
            Psi on the standard domain (self.nx, self.ny)
        core_mask : np.array
            Plasma core mask on the standard domain (self.nx, self.ny)
        unrefined_jtor : np.array
            The jtor distribution
        thresholds : tuple (threshold for jtor criterion, threshold for gradient criterion)
            tuple of values used to identify where to apply refinement, by default None
        """

        self.build_mask_to_refine(unrefined_jtor, core_mask, thresholds)

        # this is a vector of R values at the refined calculation points
        refined_R = np.tile(
            (
                self.eqR[self.Ridx[self.mask_to_refine], 0][:, np.newaxis]
                + self.dR * self.xxc[np.newaxis, :]
            )[:, :, np.newaxis],
            (1, 1, self.nny),
        )

        # build refined psi
        # get indexes to build psi for bilinear interp
        RRidxs, ZZidxs = self.get_indexes_for_refinement(self.mask_to_refine)
        # this is psi on the vertices as needed for each grid point to be refined
        psi_where_needed = psi[RRidxs, ZZidxs]
        # this is psi refined at the refined calculation points
        bilinear_psi = np.sum(
            np.sum(
                psi_where_needed[:, :, np.newaxis, :, :]
                * self.yyyy[np.newaxis, :, :, np.newaxis, :],
                -1,
            )[:, :, np.newaxis, :, :]
            * self.xxxx[np.newaxis, :, :, np.newaxis, :],
            axis=-1,
        )
        # reformat so to have same structure as refined_R
        format_bilinear_psi = np.zeros(
            (np.sum(self.mask_to_refine), self.nnx, self.nny)
        )
        format_bilinear_psi[:, self.hnnx :, self.hnny :] = bilinear_psi[:, 0]
        format_bilinear_psi[:, : self.hnnx, self.hnny :] = bilinear_psi[:, 1]
        format_bilinear_psi[:, : self.hnnx :, : self.hnny] = bilinear_psi[:, 2]
        format_bilinear_psi[:, self.hnnx :, : self.hnny] = bilinear_psi[:, 3]

        return format_bilinear_psi, refined_R

    def build_from_refined_jtor(self, unrefined_jtor, refined_jtor):
        """Averages the refined maps to the (nx, ny) domain grid.

        Parameters
        ----------
        unrefined_jtor : np.array
            (nx, ny) jtor map from unresolved method
        refined_jtor : np.array
             maps of the refined jtor, dimension = (no cells to refine, nnx, nny)


        Returns
        -------
        Refined jtor on the (nx, ny) domain grid
        """
        # mask out refinement points that are outside the limiter
        masked_refined_jtor = (
            refined_jtor
            * self.full_masks[
                self.Ridx[self.mask_to_refine], self.Zidx[self.mask_to_refine], :, :
            ]
        )
        # average in each refinement region
        masked_refined_jtor = np.sum(masked_refined_jtor, axis=(1, 2)) / self.nnxy

        # assign to jtor
        jtor = 1.0 * unrefined_jtor
        jtor[self.Ridx[self.mask_to_refine], self.Zidx[self.mask_to_refine]] = (
            masked_refined_jtor
        )

        return jtor
