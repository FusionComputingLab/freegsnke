import numpy as np
import freegs
from freegs.gradshafranov import Greens

class NewtonKrylov:
     
    def __init__(self, eq):
        #eq is an Equilibrium instance, it has to have the same domain and grid as 
        #the ones the solver will be called on
        
        R = eq.R
        Z = eq.Z
        self.R = R
        self.Z = Z
        
        #for reshaping
        nx,ny = np.shape(R)
        self.nx = nx
        self.ny = ny
        
        #for integration
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # List of indices on the boundary
        bndry_indices = np.concatenate(
            [
                [(x, 0) for x in range(nx)],
                [(x, ny - 1) for x in range(nx)],
                [(0, y) for y in range(ny)],
                [(nx - 1, y) for y in range(ny)],
            ]
        )
        self.bndry_indices = bndry_indices
        
        
        
        #linear solver for del*Psi=RHS with fixed RHS
        self.solver = freegs.multigrid.createVcycle(
            nx, ny, 
            freegs.gradshafranov.GSsparse4thOrder(eq.R[0,0], 
                                                  eq.R[-1,0], 
                                                  eq.Z[0,0], 
                                                  eq.Z[0,-1]), 
            nlevels=1, ncycle=1, niter=2, direct=True)
        
        
        #greenfunctions for boundary conditions
        greenfunc = np.zeros((len(bndry_indices),nx,ny))
        for i, [x, y] in enumerate(bndry_indices):
            # Calculate the response of the boundary point
            # to each cell in the plasma domain
            greenfunc[i] = Greens(R, Z, R[x, y], Z[x, y]) 
            # Prevent infinity/nan by removing (x,y) point
            greenfunc[i, x, y] = 0.0
        self.greenfunc = greenfunc*dR*dZ
        
        #mask of non-boundary domain
        self.boundary_mask = np.ones_like(R)
        self.boundary_mask[:,0] = 0
        self.boundary_mask[:,-1] = 0
        self.boundary_mask[0,:] = 0
        self.boundary_mask[-1,:] = 0
        
        #RHS/Jtor
        self.rhs_before_jtor = -freegs.gradshafranov.mu0*eq.R
                
        
            
    def freeboundary(self, plasma_psi, tokamak_psi, profiles):
        #tokamak_psi is psi from the currents assigned to the tokamak coils in eq, ie.
        #tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
        
        #jtor and RHS given tokamak_psi above and the input plasma_psi
        self.jtor = profiles.Jtor(self.R, self.Z, tokamak_psi+plasma_psi)
        self.rhs = self.rhs_before_jtor*self.jtor
        
        #calculates and assignes boundary conditions, NOT von Haugenow but 
        #exactly as in freegs
        self.psi_boundary = np.zeros_like(self.R)
        psi_bnd = np.sum(self.greenfunc*self.jtor[np.newaxis,:,:], axis=(-1,-2))
        for i, [x, y] in enumerate(self.bndry_indices):
            self.psi_boundary[x,y] = psi_bnd[i]    
        self.rhs[0, :] = self.psi_boundary[0, :]
        self.rhs[:, 0] = self.psi_boundary[:, 0]
        self.rhs[-1, :] = self.psi_boundary[-1, :]
        self.rhs[:, -1] = self.psi_boundary[:, -1]
        
    def F(self, plasma_psi, profiles, eq): #root problem on Psi
        self.freeboundary(plasma_psi, profiles, eq)
        return plasma_psi - self.solver(self.psi_boundary, self.rhs)
    
    def _F(self, plasma_psi): 
        #same as above, but uses private profiles and tokamak_psi
        self.freeboundary(plasma_psi, self.tokamak_psi, self.profiles)
        return plasma_psi - self.solver(self.psi_boundary, self.rhs)
    
    
    def Arnoldi_iteration(self, plasma_psi, #trial plasma_psi
                                starting_vec, #first vector for psi basis, both are in 2Dformat
                                Fplasma_psi=None, #residual of trial plasma_psi: F(plasma_psi)
                                n_k=10, #max number of basis vectors
                                conv_crit=1e-2, #add basis vector 
                                                #if orthogonal component is larger than
                                grad_eps=.1 #infinitesimal step
                         ):
        
        #basis in Psi space
        Q = np.zeros((self.nx*self.ny, n_k+1))
        #basis in grandient space
        G = np.zeros((self.nx*self.ny, n_k+1))
        
        #orthonormalize starting vec
        nr0 = np.linalg.norm(starting_vec)
        Q[:,0] = starting_vec.reshape(-1)/nr0
        
        
        if Fplasma_psi is None:
            Fplasma_psi = self._F(plasma_psi)
        
        
        n_it = 0
        #control on whether to add a new basis vector
        arnoldi_control = 1
        #use at least 3 orthogonal terms, but not more than n_k
        while ((n_it<3)+(arnoldi_control>0))*(n_it<n_k)>0:
            ri = self._F(plasma_psi + Q[:,n_it].reshape(self.nx,self.ny)*grad_eps) 
            ri -= Fplasma_psi
            ri /= grad_eps
            rin = ri.reshape(-1)
            #store the gradient
            G[:,n_it] = rin
            n_it += 1
            #orthonormalize the vector in Psi space
            for j in range(n_it):
                rin -= np.dot(Q[:,j].T, rin) * Q[:,j]
            nr0 = np.linalg.norm(rin)
            #store in Psi basis vectors
            Q[:,n_it] = rin/nr0
            arnoldi_control = (nr0>conv_crit)
        #make both basis available
        self.Q = Q[:,:n_it]
        self.G = G[:,:n_it]
            
    def dpsi(self, res0, clip=10):
        #solve the least sq problem in coeffs: min||G.coeffs+res0||^2
        self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.G.T, self.G)),
                                     self.G.T), -res0.reshape(-1))
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        #get the associated step in psi space
        dpsi = np.sum(self.Q*self.coeffs[np.newaxis,:], axis=1).reshape(self.nx,self.ny)
        #dpsi *= self.boundary_mask
        return dpsi
    
    #this is the solver itself
    #solves the forward GS problem: given 
    # - the set of active coil currents as in eq.tokamak,
    # - the plasma properties assigned by the object "profiles"
    # finds the equilibrium plasma_psi and assigns it to eq
    # The starting trial_plasma_psi is eq.plasma_psi
    def solve(self, eq, 
                    profiles,
                    rel_convergence=1e-6, 
                    n_k=8, #this is a good compromise between convergence and speed
                    conv_crit=1e-2, 
                    grad_eps=.01,
                    clip=10, #maximum absolute value of coefficients in psi space
                    verbose=False,
                    max_iter=30, #after these it just stops
                    conv_history=False #returns relative convergence
                    ):
        
        rel_c_history = []
        
        trial_plasma_psi = eq.plasma_psi
        self.profiles = profiles
        self.tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
        
        res0 = self._F(trial_plasma_psi)
        rel_change = np.amax(np.abs(res0))
        rel_change /= (np.amax(trial_plasma_psi)-np.amin(trial_plasma_psi))
        rel_c_history.append(rel_change)
        if verbose:
            print('rel_change_0', rel_change)
            
        it=0
        while rel_change>rel_convergence and it<max_iter:
            self.Arnoldi_iteration(trial_plasma_psi, 
                                   res0, #starting vector in psi space is the residual itself
                                   res0, #F(trial_plasma_psi) already calculated
                                   n_k, conv_crit, grad_eps)
            dpsi = self.dpsi(res0, clip)
            trial_plasma_psi += dpsi
            res0 = self._F(trial_plasma_psi)
            rel_change = np.amax(np.abs(res0))
            rel_change /= (np.amax(trial_plasma_psi)-np.amin(trial_plasma_psi))
            rel_c_history.append(rel_change)
            if verbose:
                print(rel_change, 'coeffs=', self.coeffs)
            it += 1
        
        #update eq with new solution
        eq.plasma_psi = trial_plasma_psi
        
        #if max_iter was hit, then message:
        if not it<max_iter:
            print('failed to converge with less than {} iterations'.format(max_iter))
            
        if conv_history:
            return np.array(rel_c_history)