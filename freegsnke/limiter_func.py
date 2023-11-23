import numpy as np
from matplotlib.path import Path

class Limiter_handler:

    def __init__(self, eq, limiter):
        """Object to handle additional calculations due to the addition of a limiter
        with respect to a purely diverted plasma. This is primarily used by the profile functions.
        Each profile function has its own instance of a Limiter_handler.

        Parameters
        ----------
        eq : FreeGS equilibrium object
            Used as a source of info on the solver's grid.
        limiter : a tokamak.Wall object
            Contains a list of R and Z coordinates (the vertices) which define the region accessible to the plasma.
            The boundary itself is the limiter.
        """
        
        self.limiter = limiter
        self.eqR = eq.R
        self.eqZ = eq.Z
        self.dR = self.eqR[1, 0] - self.eqR[0, 0]
        self.dZ = self.eqZ[0, 1] - self.eqZ[0, 0]
        self.dRdZ = self.dR*self.dZ
        self.ker_signs = np.array([[1,-1],[-1,1]])[np.newaxis, :, :]
        self.eq_shape = np.shape(eq.R)
        self.build_mask_inside_limiter()
        self.limiter_points()



    def build_mask_inside_limiter(self, ):
        """Uses the coordinates of points along the edge of the limiter region
        to generate the mask of contained domain points.

        Parameters
        ----------
        eq : freeGS Equilibrium object
            Specifies the domain properties
        limiter : freeGS.machine.Wall object
            Specifies the limiter contour points
        Returns
        -------
        mask_inside_limiter : np.array
            Mask over the full domain of grid points inside the limiter region.
        """
        verts = np.concatenate((np.array(self.limiter.R)[:,np.newaxis],
                                np.array(self.limiter.Z)[:,np.newaxis]), axis=-1)
        path = Path(verts)

        points = np.concatenate((self.eqR[:,:,np.newaxis], self.eqZ[:,:,np.newaxis]), axis=-1)
        
        mask_inside_limiter = path.contains_points(points.reshape(-1,2))
        mask_inside_limiter = mask_inside_limiter.reshape(self.eq_shape)
        self.mask_inside_limiter = mask_inside_limiter
        

    def limiter_points(self, refine=6):
        """Based on the limiter vertices, it builds the refined list of points on the boundary
        of the region where the plasma is allowed. These points are those on which the flux
        function is interpolated to find the value of psi_boundary in the case of a limiter plasma.

        Parameters
        ----------
        refine : int, optional
            the upsampling ratio with respect to the solver's grid, by default 6.

        """
        verts = np.concatenate((np.array(self.limiter.R)[:,np.newaxis],
                            np.array(self.limiter.Z)[:,np.newaxis]), axis=-1)

        refined_ddiag = (self.dR**2 + self.dZ**2)**.5/refine

        fine_points = []
        for i in range(1,len(verts)):
            dv = verts[i:i+1] - verts[i-1:i]
            ndv = np.linalg.norm(dv)
            nn = np.round(ndv//refined_ddiag).astype(int)
            if nn:
                points = dv * np.arange(nn)[:,np.newaxis]/nn
                points += verts[i-1:i]
                fine_points.append(points)
        fine_points = np.concatenate(fine_points, axis=0)

        Rvals = self.eqR[:,0]
        Ridxs = np.sum(Rvals[np.newaxis,:] < fine_points[:,:1], axis=1)
        Zvals = self.eqZ[0,:]
        Zidxs = np.sum(Zvals[np.newaxis,:] < fine_points[:,1:2], axis=1)
        self.grid_per_limiter_fine_point = np.concatenate((Ridxs[:,np.newaxis], Zidxs[:,np.newaxis]), axis=-1)

        self.fine_point_per_cell = {}
        self.fine_point_per_cell_R = {}
        self.fine_point_per_cell_Z = {}
        for i in range(len(fine_points)):
            if (Ridxs[i], Zidxs[i]) not in self.fine_point_per_cell.keys():
                self.fine_point_per_cell[Ridxs[i], Zidxs[i]] = []
                self.fine_point_per_cell_R[Ridxs[i], Zidxs[i]] = []
                self.fine_point_per_cell_Z[Ridxs[i], Zidxs[i]] = []
            self.fine_point_per_cell[Ridxs[i], Zidxs[i]].append(i)
            self.fine_point_per_cell_R[Ridxs[i], Zidxs[i]].append([self.eqR[Ridxs[i]+1, Zidxs[i]] - fine_points[i,0],
                                                                   self.eqR[Ridxs[i], Zidxs[i]] - fine_points[i,0]])
            self.fine_point_per_cell_Z[Ridxs[i], Zidxs[i]].append([[self.eqZ[Ridxs[i], Zidxs[i]+1] - fine_points[i,1],
                                                                   self.eqZ[Ridxs[i], Zidxs[i]] - fine_points[i,1]]])
        for key in self.fine_point_per_cell.keys():
            self.fine_point_per_cell_R[key] = np.array(self.fine_point_per_cell_R[key])
            self.fine_point_per_cell_Z[key] = np.array(self.fine_point_per_cell_Z[key])
        self.fine_point = fine_points


    def interp_on_limiter_points_cell(self, id_R, id_Z, psi):
        """Calculates a bilinear interpolation of the flux function psi in the solver's grid
        cell [eq.R[id_R], eq.R[id_R + 1]] x [eq.Z[id_Z], eq.Z[id_Z + 1]]. The interpolation is returned directly for 
        the refined points on the limiter boundary that fall in that grid cell, as assigned
        through the self.fine_point_per_cell objects.


        Parameters
        ----------
        id_R : int
            index of the R coordinate for the relevant grid cell
        id_Z : int
            index of the Z coordinate for the relevant grid cell
        psi : np.array on the solver's grid
            Vaules of the total flux function ofn the solver's grid.

        Returns
        -------
        vals : np.array
            Collection of floating point interpolated values of the flux function
            at the self.fine_point_per_cell[id_R, id_Z] locations.
        """
        if (id_R, id_Z) in self.fine_point_per_cell_Z.keys():
            ker = psi[id_R:id_R+2, id_Z:id_Z+2][np.newaxis, :, :]
            ker *= self.ker_signs
            vals = np.sum(ker*self.fine_point_per_cell_Z[id_R, id_Z], axis=-1)
            vals = np.sum(vals*self.fine_point_per_cell_R[id_R, id_Z], axis=-1)
        else:
            vals = []
        return vals
    

    def interp_on_limiter_points(self, id_R, id_Z, psi):
        """Uses interp_on_limiter_points_cell to interpolate the flux function psi
        on the refined limiter boundary points relevant to the 9 cells
        {id_R-1, id_R, id_R+1} X {id_Z-1, id_Z, id_Z+1}. Interpolated values on the 
        boundary points relevant to the cells above are collated and returned. 
        This is called by self.core_mask_limiter with id_R, id_Z corresponding to the 
        grid cell outside the limiter (but in the diverted core) with the 
        highest psi value (referred to as id_psi_max_out in self.core_mask_limiter)

 
        Parameters
        ----------
        id_R : int
            index of the R coordinate for the relevant grid cell
        id_Z : _type_
            index of the Z coordinate for the relevant grid cell
        psi : _type_
            Vaules of the total flux function ofn the solver's grid.

        Returns
        -------
        vals : np.array
            Collection of floating point interpolated values of the flux function
            at the self.fine_point_per_cell locations relevant to all of the 9 cells
            {id_R-1, id_R, id_R+1} X {id_Z-1, id_Z, id_Z+1}
        """
        vals = []
        for i in np.arange(-1,2):
            for j in np.arange(-1,2):
                vals.append(self.interp_on_limiter_points_cell(id_R+i, id_Z+j, psi))
        vals = np.concatenate(vals)
        vals /= self.dRdZ
        return vals



    def core_mask_limiter(self, psi, psi_bndry, core_mask, 
                            limiter_mask_out,
                            #   limiter_mask_in,
                            #   linear_coeff=.5
                            ):
        """Checks if plasma is in a limiter configuration rather than a diverted configuration.
        This is obtained by checking whether the core mask deriving from the assumption of a diverted configuration
        implies an overlap with the limiter. If so, an interpolation of psi on the limiter boundary points
        is called to determine the value of psi_boundary and to recalculate the core_mask accordingly. 

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
        limiter_mask_out : np.array
            The mask identifying the border of the limiter, including only points 'outside it', not accessible to the plasma.
            Same size as psi. 
        


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
            # psi_max_out = np.amax(psi[offending_mask])
            # psi_max_in = np.amax(psi[(core_mask * limiter_mask_in).astype(bool)])
            # psi_bndry = linear_coeff*psi_max_out + (1-linear_coeff)*psi_max_in
            # core_mask = (psi > psi_bndry)*core_mask
                
            id_psi_max_out = np.unravel_index(np.argmax(psi-(10**6)*(1-offending_mask)), self.eq_shape)
            interpolated_on_limiter = self.interp_on_limiter_points(id_psi_max_out[0],
                                                                    id_psi_max_out[1],
                                                                    psi)
            psi_bndry = np.amax(interpolated_on_limiter)
            core_mask = (psi > psi_bndry)*core_mask  

        return psi_bndry, core_mask, flag_limiter




