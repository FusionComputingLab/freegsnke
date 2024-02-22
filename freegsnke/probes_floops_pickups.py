import os
this_dir , this_filename = os.path.split(__file__)

try:
    from freegs import critical
except:
    from freegs.freegs import critical
try:
    from freegs.gradshafranov import Greens
except:
    from freegs.freegs.gradshafranov import Greens
import numpy as np
import pickle

import MASTU_coils
from MASTU_coils import coils_dict

#pickles with the flux-loop and pickup-coil dictionaries
#these files are specific to MAST-U
floops_fh='floops.pk'
pickups_fh='pickups.pk'
floops_path=os.path.join(this_dir,floops_fh)
pickups_path=os.path.join(this_dir,pickups_fh)
tfile=open(floops_path,'rb')
floops=pickle.load(tfile)
tfile.close()
tfile=open(pickups_path,'rb')
pccoils=pickle.load(tfile)
tfile.close()

class quants_for_emulation:
    #needs coil_dict from MASTU_coils.py
    
    def __init__(self, eq , floops=floops, pccoils=pccoils):
        #pre-builds matrices for inductance calculations on new equilibria
        #eq need not be the actual one, only has to have same grid properties 
        #as those on which calculations will be made by calling other methods
        
        dR_dZ = (eq.R[0,0]-eq.R[1,0])*(eq.Z[0,0]-eq.Z[0,1])
        two_pi_dR_dZ = 2*np.pi*dR_dZ
        
        #reduced grid for plasma-related inductances
        red_idx = np.round(np.array(np.shape(eq.R))*
                           np.array([[12,100],[30,99]])/129).astype(int)
        red_R = eq.R[red_idx[0,0]:red_idx[0,1],
                     red_idx[1,0]:red_idx[1,1]]
        red_Z = eq.Z[red_idx[0,0]:red_idx[0,1],
                     red_idx[1,0]:red_idx[1,1]]
        self.red_idx = red_idx
        
        #for plasma-plasma flux
        greenmatrix = Greens(red_R[:,:,np.newaxis,np.newaxis], red_Z[:,:,np.newaxis,np.newaxis],
                             red_R[np.newaxis,np.newaxis,:,:], red_Z[np.newaxis,np.newaxis,:,:])
        greenmatrix *= two_pi_dR_dZ*dR_dZ
        self.greenmatrix = greenmatrix
                
        #for coil-plasma and plasma-coil fluxes 
        resgrid = np.shape(red_Z)
        coil_plasma_greens = np.zeros((len(coils_dict.keys()),resgrid[0],resgrid[1]))
        for i,labeli in enumerate(coils_dict.keys()):
            greenm = Greens(red_R[np.newaxis,:,:], red_Z[np.newaxis,:,:],
                            coils_dict[labeli]['coords'][0][:,np.newaxis,np.newaxis],
                            coils_dict[labeli]['coords'][1][:,np.newaxis,np.newaxis])
            greenm *= coils_dict[labeli]['polarity'][:,np.newaxis,np.newaxis]
            coil_plasma_greens[i] = two_pi_dR_dZ*np.sum(greenm, axis=0)
        self.coil_plasma_greens = coil_plasma_greens
        
        #for separatrix characterization
        self.Zvec = np.arange(len(red_Z[0]))[np.newaxis,:].astype('float64')
        self.Zvec -= np.mean(self.Zvec)
       
        self.dR_dZ = dR_dZ
        self.red_R = red_R
        
        self.fvac = eq.fvac()
        #
        ## the stuff below is needed for the magnetics
        #
        Rs=eq.R[:,0]
        Zs=eq.Z[0,:]
        # please keep both the private and self.-ed variables for ease with the typing in discpos below
        NX=len(Rs)
        NY=len(Zs)
        Rmin , Rmax = min(Rs) , max(Rs)
        Zmin , Zmax = min(Zs) , max(Zs)
        self.NX=NX
        self.NY=NY
        self.Rmin , self.Rmax = Rmin , Rmax
        self.Zmin , self.Zmax = Zmin , Zmax
        
        # fill in the dictionary of coordinates for each pickup coil 
        self.pccoils=pccoils
        self.floops=floops
        self.discpos={};
        for coil in self.pccoils:
            #discpos[coil['name']]
            tpos=coil['position']
            tpos=coil['position']
            tpR=tpos[0]
            tpZ=tpos[2]
            tRi = int((NX-1)*(tpR-Rmin)/(Rmax-Rmin))
            tZi = int((NY-1)*(tpZ-Zmin)/(Zmax-Zmin))
            tvec=coil['orientation_vector']
            #
            if (tRi<NX-1 and tRi>0):
                tRp=(tRi+1)*((NX-1)*(tpR-Rmin)/(Rmax-Rmin)>tRi+0.5)+tRi*((NX-1)*(tpR-Rmin)/(Rmax-Rmin)<=tRi+0.5);
                tRm=(tRi)*((NX-1)*(tpR-Rmin)/(Rmax-Rmin)>tRi+0.5)+(tRi-1)*((NX-1)*(tpR-Rmin)/(Rmax-Rmin)<=tRi+0.5);
            elif tRi<NX-1:
                tRp= tRi;
                tRm=0
            else:
                tRm= tRi-1;
                tRp=tRi
            dR=1.0#*(tRi<NX-1)+1.0*(tRi>0)
            dR=dR*(Rmax-Rmin)/NX;
            #
            if (tZi<NY-1 and tZi>0):
                tZp=(tZi+1)*((NY-1)*(tpZ-Zmin)/(Zmax-Zmin)>tZi+0.5)+tZi*((NY-1)*(tpZ-Zmin)/(Zmax-Zmin)<=tZi+0.5);
                tZm=(tZi)*((NY-1)*(tpZ-Zmin)/(Zmax-Zmin)>tZi+0.5)+(tZi-1)*((NY-1)*(tpZ-Zmin)/(Zmax-Zmin)<=tZi+0.5);
            elif tZi<NY-1:
                tZp= tZi;
                tZm=0
            else:
                tZm= tZi-1;
                tZp=tZi
            dZ=1.0#*(tRi<NX-1)+1.0*(tRi>0)
        #         if tZi<NY-1: tZp=tZi+1;
        #         else: tZp= tZi;
        #         if tZi>0: tZm=tZi-1;
        #         else: tZm= tZi;
        #         dZ=1.0*(tZi<NX-1)+1.0*(tZi>0)
            dZ=dZ*(Zmax-Zmin)/NY;
            #
            self.discpos[coil['name']]={'Ri':tRi, 'Zi':tZi ,'Rp':tRp,'Rm':tRm,'Zp':tZp,'Zm':tZm,'dR':dR,'dZ':dZ}
        
    def jtor_and_mask(self, eq, profiles):
        # eq is the actual one on which to calculate quantities
        # profiles is the associated ConstrainPaxisIp or ConstrainBetapIp obj
        # call this BEFORE calling method fluxes
        jtor = profiles.Jtor(eq.R, eq.Z, eq.psi())
        self.plasma_mask = jtor>0
        self.red_jtor = jtor[self.red_idx[0,0]:self.red_idx[0,1],
                             self.red_idx[1,0]:self.red_idx[1,1]]
        self.red_plasma_mask = self.red_jtor>0
        
        #separatrix characterization
        self.plasma_zloc = np.stack((np.mean(self.Zvec*self.red_plasma_mask, axis=1),
                                     np.sum(self.red_plasma_mask, axis=1)))
        
        self.tot_current = self.dR_dZ*np.sum(jtor)
        
        #to be devided by the conductivity \sigma
        plasma_resistance = np.sum(self.red_R*self.red_jtor)
        plasma_resistance *= 2*np.pi*self.dR_dZ/self.tot_current
        self.plasma_resistance = plasma_resistance
                
    def fluxes(self, eq):
        #needs jtor and plasma masks
        
        #plasma-plasma flux:
        plasma_self_flux = self.red_jtor[np.newaxis,np.newaxis,:,:]*self.greenmatrix
        plasma_self_flux *= self.red_plasma_mask[:,:,np.newaxis,np.newaxis]
        self.plasma_self_flux = np.sum(plasma_self_flux)
        
        #flux of plasma current on coils, 
        #the quantity of interest is the derivative wrp to: tot plasma current 
        #                                                   plasma parameters
        #                                                   coil currents
        coil_plasma_flux = self.red_jtor[np.newaxis,:,:]*self.coil_plasma_greens
        coil_plasma_flux = np.sum(coil_plasma_flux, axis=(1,2))
        self.coil_plasma_flux = coil_plasma_flux/MASTU_coils.nloops_per_coil
        
        #inductance of coils on plasma                                               
        plasma_coil_ind = (self.red_plasma_mask[np.newaxis,:,:])*self.coil_plasma_greens
        plasma_coil_ind = np.sum(plasma_coil_ind, axis=(1,2))
        self.plasma_coil_ind = plasma_coil_ind
        
        
    def quants_out(self, eq, profiles):
        #calls all that's needed on eq, in order, and returns results
        self.jtor_and_mask(eq, profiles)
        self.fluxes(eq)
        
        results = {}
        
        #total plasma current, plasma resistance term (to devide by conductivity)
        results['tot_Ip_Rp'] = [self.tot_current, self.plasma_resistance]
        
        #plasma-plasma flux:
        results['plasma_self_flux'] = self.plasma_self_flux
        
        #flux of plasma current on coils
        results['plasma_flux_on_coils'] = self.coil_plasma_flux
        
        #inductance of coils on plasma
        results['plasma_coil_ind'] = self.plasma_coil_ind
        
        #plasma centroid and width in Z of section inside separatrix
        #in grid units
        results['separatrix'] = [self.plasma_zloc, self.plasma_mask]
        
        return results
    
    def getpsis(self, eq,floops=floops):
        #this one is easy, just get psi(R,Z) at the flux-loop positions
        #you can also run it with a different dictionary of flux-loop positions
        Psilist=[];
        for floop in floops:
            tpos=floop['position']
            tpR=tpos[0]
            tpZ=tpos[1]
            tRi = int((self.NX-1)*(tpR-self.Rmin)/(self.Rmax-self.Rmin))
            tZi = int((self.NY-1)*(tpZ-self.Zmin)/(self.Zmax-self.Zmin))
            tpsiRZ=eq.psi()[tRi,tZi]
            Psilist.append(tpsiRZ)
        return Psilist
    
    def getBs_disc(self, eq):
        # NB: if you change eq object or the pickup-coils, you'll need to re-init the quants_for_emu so that the probe radii in self.discpos fall in the correct cells
        # the derivative estimation below uses D(sinh(arcsinh)) to that it's better behaved over a large value of psi's (e.g. close to the coils)
        fvac=self.fvac;
        Blist=[];
        for coil in self.pccoils:
            discs=self.discpos[coil['name']]
            tpos=coil['position']
            tpR=tpos[0]
            tpZ=tpos[2]
            tvec=coil['orientation_vector']
            #
            tRi=discs['Ri'] ; tZi = discs['Zi']
            tRp=discs['Rp'] ; tRm=discs['Rm'] ; tZp=discs['Zp'] ; tZm=discs['Zm'] ; dR=discs['dR'] ; dZ=discs['dZ']
            tB=0.0
            #
            if tvec[0]!=0.0:
                fm=eq.psi()[tRi,tZi] # 0.5*(eq.psi()[tRi,tZp]+eq.psi()[tRi,tZm])
                fm=np.sqrt(1.0+fm**2)
                df=fm*(np.arcsinh(eq.psi()[tRi,tZp])-np.arcsinh(eq.psi()[tRi,tZm]))/dZ
                ## or uncomment this and comment the above uncomment this if you prefer the old-school finite difference derivative, way faster but less accurate
                #df=(eq.psi()[tRi,tZp]-eq.psi()[tRi,tZm])/dZ
                tBr=-(1./tpR)*df
                tB+=tvec[0]*tBr; # eq.Br(tpR,tpZ);
            if tvec[1]!=0.0:
                tB+=tvec[1]*fvac/tpR; #eq.Btor(tpR,tpZ)
            if tvec[2]!=0.0:
                fm=eq.psi()[tRi,tZi] # 0.5*(eq.psi()[tRp,tZi]+eq.psi()[tRm,tZi])
                fm=np.sqrt(1.0+fm**2)
                df=fm*(np.arcsinh(eq.psi()[tRp,tZi])-np.arcsinh(eq.psi()[tRm,tZi]))/dR
                ## or uncomment this and comment the above if you prefer the old-school finite difference derivative, way faster but less accurate
                #df=(eq.psi()[tRp,tZi]-eq.psi()[tRm,tZi])/dR
                tBz=(1./tpR)*df
                tB+=tvec[2]*tBz; # eq.Bz(tpR,tpZ);
            ## the above replaces the slow spline-based sub below:
            #tB=eq.Br(tpR,tpZ)*tvec[0]+eq.Btor(tpR,tpZ)*tvec[1]+eq.Bz(tpR,tpZ)*tvec[2]
            Blist.append(tB)
        return Blist

        
 

        
        
        
    