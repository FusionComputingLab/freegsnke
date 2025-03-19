import numpy as np


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

        self.path = eq.limiter_handler.path
        self.prepare_for_refinement()

    def prepare_for_refinement(
        self,
    ):
        """Prepares necessary quantities to operate refinement."""
        self.Ridx = np.tile(np.arange(self.nx - 1), (self.ny - 1, 1)).T
        self.Zidx = np.tile(np.arange(self.ny - 1), (self.nx - 1, 1))

        self.xx = np.linspace(0, 1 - 1 / self.nnx, self.nnx) + 1 / (2 * self.nnx)
        self.yy = np.linspace(0, 1 - 1 / self.nny, self.nny) + 1 / (2 * self.nny)
        # self.xx = np.linspace(0,1,self.nnx)
        # self.yy = np.linspace(0,1,self.nny)

        self.xxx = np.concatenate(
            (1 - self.xx[:, np.newaxis], self.xx[:, np.newaxis]), axis=-1
        )
        self.yyy = np.concatenate(
            (1 - self.yy[:, np.newaxis], self.yy[:, np.newaxis]), axis=-1
        )

        fullr = np.tile(
            (self.eqR[:, :, np.newaxis] + self.dR * self.xx[np.newaxis, np.newaxis, :])[
                :, :, :, np.newaxis
            ],
            [1, 1, 1, self.nny],
        )
        fullz = np.tile(
            (self.eqZ[:, :, np.newaxis] + self.dZ * self.yy[np.newaxis, np.newaxis, :])[
                :, :, np.newaxis, :
            ],
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
        quartermasks[:, :, 0] = (srr < (self.nnx / 2)) * (szz < (self.nny / 2))
        quartermasks[:, :, 1] = (srr >= (self.nnx / 2)) * (szz < (self.nny / 2))
        quartermasks[:, :, 3] = (srr < (self.nnx / 2)) * (szz >= (self.nny / 2))
        quartermasks[:, :, 2] = (srr >= (self.nnx / 2)) * (szz >= (self.nny / 2))
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
        )
        ZZidxs = np.concatenate(
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
        )
        return RRidxs, ZZidxs

    def build_jtor_value_mask(self, unrefined_jtor, thresholds, quantiles=(0.5, 0.9)):
        """Selects the cells that need to be refined based on their value of jtor.
        Selection is such that it includes cells where jtor exceeds the value calculated
        based on the quantiles and threshold[0].

        Parameters
        ----------
        unrefined_jtor : np.array
            The jtor distribution
        thresholds : tuple (threshold for jtor criterion, threshold for gradient criterion)
            tuple of values used to identify where to apply refinement, by default None
        """

        jtor_quantiles = np.quantile(unrefined_jtor.reshape(-1), quantiles)
        mask = (unrefined_jtor - jtor_quantiles[0]) > thresholds[0] * (
            jtor_quantiles[1] - jtor_quantiles[0]
        )
        return mask

    def build_jtor_gradient_mask(
        self, unrefined_jtor, thresholds, quantiles=(0.5, 0.9)
    ):
        """Selects the cells that need to be refined based on their value of the gradient of jtor.
        Selection is such that it includes cells where the norm of the gradient exceeds the value calculated
        based on the quantiles and threshold[1].

        Parameters
        ----------
        unrefined_jtor : np.array
            The jtor distribution
        thresholds : tuple (threshold for jtor criterion, threshold for gradient criterion)
            tuple of values used to identify where to apply refinement, by default None
        """

        # right
        gradient_mask = (unrefined_jtor[:-1, :-1] - unrefined_jtor[1:, :-1]) ** 2
        # up
        gradient_mask += (unrefined_jtor[:-1, :-1] - unrefined_jtor[:-1, 1:]) ** 2
        # up-right
        gradient_mask += (unrefined_jtor[:-1, :-1] - unrefined_jtor[1:, 1:]) ** 2

        gradient_mask = gradient_mask**0.5

        mask = self.build_jtor_value_mask(gradient_mask, thresholds, quantiles)
        return mask

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

        # include all cells that are crossed by the lcfs:
        core_mask = core_mask.astype(float)
        mask_to_refine = (
            core_mask[:-1, :-1]
            + core_mask[1:, :-1]
            + core_mask[:-1, 1:]
            + core_mask[1:, 1:]
        )
        self.mask_to_refine = (mask_to_refine > 0) * (mask_to_refine < 4)

        # include cells that warrant refinement according to criterion on jtor value:
        value_mask = self.build_jtor_value_mask(unrefined_jtor, thresholds)
        self.mask_to_refine += value_mask[1:-1, 1:-1]

        # include cells that warrant refinement according to criterion on gradient value:
        self.mask_to_refine += self.build_jtor_value_mask(unrefined_jtor, thresholds)

        self.mask_to_refine = self.mask_to_refine.astype(bool)

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
        # this is a vector of Rvalues in the refined cells
        refined_R = np.tile(
            (
                self.eqR[self.Ridx[self.mask_to_refine], 0][:, np.newaxis]
                + self.dR * self.xx[np.newaxis, :]
            )[:, :, np.newaxis],
            (1, 1, self.nny),
        )
        RRidxs, ZZidxs = self.get_indexes_for_refinement(self.mask_to_refine)
        # this is psi on the 4 vertices of the grids to refine
        psi_where_needed = psi[RRidxs, ZZidxs]
        # R_where_needed = self.eqR[RRidxs,ZZidxs]
        # this is psi refined in the cells
        bilinear_psi = np.sum(
            np.sum(
                psi_where_needed[:, :, np.newaxis, :]
                * self.yyy[np.newaxis, np.newaxis],
                -1,
            )[:, np.newaxis, :, :]
            * self.xxx[np.newaxis, :, :, np.newaxis],
            axis=-2,
        )
        # this is a vector of Rvalues in the refined cells
        # refined_R = np.sum(np.sum(R_where_needed[:,:,np.newaxis,:]*self.yyy[np.newaxis, np.newaxis], -1)[:,np.newaxis,:,:]*self.xxx[np.newaxis,:,:, np.newaxis], axis=-2)
        return bilinear_psi, refined_R

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
        # mask refined sub-cells outside the limiter
        masked_refined_jtor = (
            refined_jtor
            * self.full_masks[
                self.Ridx[self.mask_to_refine], self.Zidx[self.mask_to_refine], :, :
            ]
        )
        # average refined jtor on quarters
        refined_jtor_on_quarters = np.sum(
            masked_refined_jtor[:, :, :, np.newaxis]
            * self.quartermasks[np.newaxis, :, :, :],
            axis=(1, 2),
        )
        # explode the unrefined jtor on corresponding quarters
        jtor_on_quarters = np.tile(unrefined_jtor[:, :, np.newaxis], [1, 1, 4]) / 4
        # assign the refined averages to the corresponding grid cells
        jtor_on_quarters[
            self.Ridx[self.mask_to_refine], self.Zidx[self.mask_to_refine], 0
        ] = refined_jtor_on_quarters[:, 0]
        jtor_on_quarters[
            self.Ridx[self.mask_to_refine] + 1, self.Zidx[self.mask_to_refine], 1
        ] = refined_jtor_on_quarters[:, 1]
        jtor_on_quarters[
            self.Ridx[self.mask_to_refine] + 1, self.Zidx[self.mask_to_refine] + 1, 2
        ] = refined_jtor_on_quarters[:, 2]
        jtor_on_quarters[
            self.Ridx[self.mask_to_refine], self.Zidx[self.mask_to_refine] + 1, 3
        ] = refined_jtor_on_quarters[:, 3]
        self.jtor_on_quarters = 1.0 * jtor_on_quarters
        return np.sum(jtor_on_quarters, axis=-1)
