#gfile file reader
#mfitz 2014
#requires https://pypi.python.org/pypi/fortranformat

#Will write and read files according to http://w3.pppl.gov/ntcc/TORAY/G_EQDSK.pdf
#Original EFIT code can be found here https://web.archive.org/web/19991009045705/http://lithos.gat.com:80/efit/weqdsku.f.txt

def COCOS(sign_Bp,sign_RφZ,sign_ρθφ):
#Sauter O and Medvedev S Y 2013 Tokamak coordinate conventions: COCOS Comput. Phys. Commun. 184 293302
#Table 1
# sign_RφZ   {1:1,-1:2},{1:3,-1:4},{1:5,-1:6},{1:7,-1:8}
# sign_Bp    1:{1:5,-1:6},{1:1,-1:2}  -1:{1:3,-1:4},{1:7,-1:8}
# sign_Bp,sign_ρθφ,sign_RφZ   {1:{-1:{1:5,-1:6},1:{1:1,-1:2}} , -1:{-1:{1:3,-1:4},1:{1:7,-1:8}}}
    c={1:{-1:{1:5,-1:6},1:{1:1,-1:2}} , -1:{-1:{1:3,-1:4},1:{1:7,-1:8}}}
    try:
        return  c[sign_Bp][sign_ρθφ][sign_RφZ]
    except:
        return 0

class Gfile:
    def __init__(self):
        self.A52=" "*48+"   0"
        self.nr=0
        self.nz=0
        self.rdim=0
        self.zdim=0
        self.rcentr=0
        self.rleft=0
        self.zmid=0
        self.rmaxis=0
        self.zmaxis=0
        self.bcentr=0
        self.current=0
        self.simag=0
        self.sibry=0
        self.f_dia=[]
        self.pressure=[]
        self.ffprime=[]
        self.pprime=[]
        self.psi_grid=[[]]
        self.q=[]
        self.n_bnd=0
        self.limitr=0
        self.R_bnd=[]
        self.Z_bnd=[]
        self.rlim=[]
        self.zlim=[]
        self.psin=[]
        self.xdum=0 #just a dummy to mimic eqdsktohelena behavior
        self.R=[]        # R coordinates
        self.Z=[]        # Z coordinates
        self.cocos=None
    def set(self,obj,value):  self.__dict__[obj]=value
    def get(self,obj): return self.__dict__[obj]
    def write(self,filename):
        import gfileReader
        gfileReader.writeGfile(self,filename)
    
def getGfile(filename):
    import numpy as np
    from io import StringIO
    gfile=Gfile() 
    import fortranformat as ff
    f2020=ff.FortranRecordReader('5e16.9')
    f2022=ff.FortranRecordReader('2i5')

    def read(handle,gfile,varList,File):
        line=handle.read(File.readline())
        for pel,el in enumerate(varList):
            gfile.set(el,line[pel])  
        return gfile

    def get1DArray(File,npsi):
        res=[]
        while(len(res)<npsi):
            line=f2020.read(File.readline())
            for l in line:
                if(l is not None): #don't append Nones
                    res.append(l)
        return np.array(res)
        
    File=open(filename,'r')
    gfile.A52=File.read(52)
    readStatement=lambda handle,varList:read(handle,gfile,varList,File)
    gfile=readStatement(ff.FortranRecordReader('2i4'),['nr','nz'])
    gfile=readStatement(f2020,['rdim','zdim','rcentr','rleft','zmid'])
    gfile=readStatement(f2020,['rmaxis','zmaxis','simag','sibry','bcentr'])
    gfile=readStatement(f2020,['current','simag','xdum','rmaxis','xdum']) 
    gfile=readStatement(f2020,['zmaxis','xdum','sibry','xdum','xdum'])  
    
    npsi=gfile.nr
    gfile.f_dia=get1DArray(File,npsi)
    gfile.pressure=get1DArray(File,npsi)
    gfile.ffprime=get1DArray(File,npsi)
    gfile.pprime=get1DArray(File,npsi)

    gfile.psi_grid=np.zeros((gfile.nr,gfile.nz))
    gfile.R = np.linspace(gfile.rleft,gfile.rleft+gfile.rdim,gfile.nr)
    gfile.Z = np.linspace(gfile.zmid-gfile.zdim/2.0,gfile.zmid+gfile.zdim/2.0 ,gfile.nz )

    stack=[]
    for j in range(gfile.nz):
        for i in range(gfile.nr):
            if(len(stack)==0):
                stack=f2020.read(File.readline())
            gfile.psi_grid[i][j]=stack.pop(0)

    gfile.q=get1DArray(File,npsi)
    gfile=readStatement(f2022,['n_bnd','limitr']) 
    stack=[]
    while(len(gfile.R_bnd)<gfile.n_bnd):
        if(len(stack)==0):
            stack=f2020.read(File.readline())
        gfile.R_bnd.append(stack.pop(0))
        if(len(stack)==0):
            stack=f2020.read(File.readline())
        gfile.Z_bnd.append(stack.pop(0))
 
    stack=[]
    while(len(gfile.rlim)<gfile.limitr):
        if(len(stack)==0):
            stack=f2020.read(File.readline())
        gfile.rlim.append(stack.pop(0))
        if(len(stack)==0):
            stack=f2020.read(File.readline())
        gfile.zlim.append(stack.pop(0))  

    #numpify
    gfile.R_bnd=np.array(gfile.R_bnd)
    gfile.Z_bnd=np.array(gfile.Z_bnd)
    gfile.rlim=np.array(gfile.rlim)
    gfile.zlim=np.array(gfile.zlim)

    for i in range(npsi):
        gfile.psin.append(float(i)/float(npsi-1))
 
    File.close()
    return gfile

def determineCOCOS(gfile,sign_RφZ=1):
    #cocos is defined by 3 signs - flux direction, toroidal direction, poloidal direction

    #toroidal direction is defined by eqdsk standard as (R φ Z) allowing only cocos 1 3 5 7
    # φ is counter-clockwise
    
    #sign of Bz is physically determined by direction of Ip
    #clockwise (negative) current gives positive outboard Bz
    import numpy as np
    sign_Bz=-int(sign_RφZ*np.sign(gfile.current))


    # But Bz also depends on flux
    # Bz=-sign_Bp*sign_RφZ*(1/R(dψ/dR))
    sign_Bp=sign_Bz*sign_RφZ*(-np.sign(gfile.sibry-gfile.simag))  

    #now some information on the direction of poloidal angle is required.
    #this is only contained in q(edge) and its relationship to B0 and Ip
    #see discussion in COCOS paper below fig 1. Sign of q, for Ip and B0 going
    #same direction, along with toroidal direction choice, determines q
    sign_ρθφ=sign_RφZ*np.sign(gfile.current)*np.sign(gfile.bcentr)*np.sign(gfile.q[-1])

    return COCOS(sign_Bp,sign_RφZ,sign_ρθφ)

def what_sign_Bp(cocos):
    for sign_Bp in [-1,1]:
        for sign_RφZ in [-1,1]:
            for sign_ρθφ in [-1,1]:
                if(COCOS(sign_Bp,sign_RφZ,sign_ρθφ)==cocos):
                    return sign_Bp
    print("invalid COCOS")
    raise ValueError

def what_sign_RPhiZ(cocos):
    for sign_Bp in [-1,1]:
        for sign_RφZ in [-1,1]:
            for sign_ρθφ in [-1,1]:
                if(COCOS(sign_Bp,sign_RφZ,sign_ρθφ)==cocos):
                    return sign_RφZ
    print("invalid COCOS")
    raise ValueError

def what_sign_rhoThetaPhi(cocos):
    for sign_Bp in [-1,1]:
        for sign_RφZ in [-1,1]:
            for sign_ρθφ in [-1,1]:
                if(COCOS(sign_Bp,sign_RφZ,sign_ρθφ)==cocos):
                    return sign_ρθφ 
    print("invalid COCOS")
    raise ValueError


def flip_sign_Bp(gfile):
    gfile.sibry=-gfile.sibry
    gfile.simag=-gfile.simag
    gfile.psi_grid=-gfile.psi_grid
    gfile.pprime=-gfile.pprime
    gfile.ffprime=-gfile.ffprime
    return gfile
    
def flip_sign_rhoThetaPhi(gfile):
    gfile.q=-gfile.q
    return gfile

def flip_sign_RPhiZ(gfile):
    gfile.sibry=-gfile.sibry
    gfile.simag=-gfile.simag
    gfile.psi_grid=-gfile.psi_grid
    gfile.pprime=-gfile.pprime
    gfile.ffprime=-gfile.ffprime
    gfile.q=-gfile.q
    gfile.current=-gfile.current
    gfile.bcentr=-gfile.bcentr
    gfile.f_dia=-gfile.f_dia
    return gfile

def convertCOCOS(gfile,target,origin_RφZ=1):
    origin=determineCOCOS(gfile,sign_RφZ=origin_RφZ)
    origin_Bp=what_sign_Bp(origin)
    target_Bp=what_sign_Bp(target)
    origin_ρθφ=what_sign_rhoThetaPhi(origin)
    target_ρθφ=what_sign_rhoThetaPhi(target)
    target_RφZ=what_sign_RPhiZ(target)
    
    if(origin_Bp!=target_Bp):
        gfile=flip_sign_Bp(gfile)
    if(origin_ρθφ!=target_ρθφ):
        gfile=flip_sign_rhoThetaPhi(gfile)
    if(origin_RφZ!=target_RφZ):
        gfile=flip_sign_RPhiZ(gfile)
    return gfile


def printCOCOStable():
    cs=[1,2,3,4,5,6,7,8]
    print("COCOS \t sign_Bp \t sign_RφZ \t sign_ρθφ")
    for cocos in cs:
        sign_Bp=what_sign_Bp(cocos)
        sign_RφZ=what_sign_RPhiZ(cocos)
        sign_ρθφ=what_sign_rhoThetaPhi(cocos)
        print("{} \t\t {} \t\t {} \t\t {}".format(cocos,sign_Bp,sign_RφZ,sign_ρθφ))

def writeGfile(gfile,filename):
        File=open(filename,'w')
        import numpy as np
        from io import StringIO
        import fortranformat as ff
        f2020=ff.FortranRecordWriter('5e16.9')
        f2022=ff.FortranRecordWriter('2i5')
        
        def writeStatement(handle,varList):
            lst=[]
            for pel,el in enumerate(varList):
                lst.append(gfile.get(el))
            File.write(handle.write(lst))
            File.write("\n")

        def writeArray(handle,variable):
            File.write(handle.write(gfile.get(variable)))
            File.write("\n")

        def writeOrderedPairs(handle,var1,var2):
            longArrayOfPairs=[]
            v1=gfile.get(var1)
            v2=gfile.get(var2)
            for pv,_ in enumerate(v1):
                longArrayOfPairs.append(v1[pv])
                longArrayOfPairs.append(v2[pv])
            #and pretend it's an array
            File.write(handle.write(longArrayOfPairs))
            File.write("\n")

        File.write(gfile.A52[0:52])
        writeStatement(ff.FortranRecordWriter('2i4'),['nr','nz'])
        writeStatement(f2020,['rdim','zdim','rcentr','rleft','zmid'])
        writeStatement(f2020,['rmaxis','zmaxis','simag','sibry','bcentr'])
        writeStatement(f2020,['current','simag','xdum','rmaxis','xdum']) 
        writeStatement(f2020,['zmaxis','xdum','sibry','xdum','xdum'])  

        writeArray(f2020,'f_dia')
        writeArray(f2020,'pressure')
        writeArray(f2020,'ffprime')
        writeArray(f2020,'pprime')

        ###2-D psi profile
        File.write(f2020.write(np.array(gfile.psi_grid).flatten(order='F')))
        File.write("\n")
        #####

        writeArray(f2020,'q')
        writeStatement(f2022,['n_bnd','limitr'])
        writeOrderedPairs(f2020,'R_bnd','Z_bnd')
        writeOrderedPairs(f2020,'rlim','zlim')

        File.close()


def getSoloviev(R0=3.0,a=1.0,alpha=1.0,B0=1.0,sigma=1.0,eps=0.4,tau=1.7,
                Idashsq=1.0,pdash=1.0):
    #GODEBLOED COORDINATES R,Z,φ , r,θ,φ
    #Bp=∇φx∇ψ
    #psi=-R A phi
    #BR=-1/R dpsi/dZ 
    #BZ= 1/R dpsi/dR
    #R Bphi= I(Ψ)

    sign_Bp=1
    sign_RφZ=-1 #will be flipped below
    sign_ρθφ=1

    import numpy as np
    gfile=Gfile()
    nr=50
    nz=100
    gfile.nr=nr
    gfile.nz=nz
    npsi=gfile.nr
    psin=np.linspace(0,1,npsi)

    def psin_xy(x,y):
        term1=(x-0.5*eps*(1.0-x**2))**2.0
        bracket1=(1-0.25*eps**2)
        bracket2=(1.0+eps*tau*x*(2.0+eps*x))
        bracket3=(y/sigma)**2
        term2=bracket1*bracket2*bracket3
        return term1+term2
    
    xaxis=np.linspace(-1,1,nr)
    yaxis=np.linspace(-1.5,1.5,nz)
    gfile.psi_grid=np.zeros((nr,nz))

    #Length of x must be number of columns in z. in contour
    #plt.contour(x,y,psi_grid.transpose())
    for i,x in enumerate(xaxis):
        for j,y in enumerate(yaxis):
            gfile.psi_grid[i][j]=psin_xy(x,y)

    psi_one=a**2.0*B0/alpha
    delta=1.0/eps*(np.sqrt(1+eps**2.0)-1.0)**2.0

    gfile.simag=0.0
    gfile.sibry=psi_one  
    gfile.R=a*xaxis+R0
    gfile.Z=a*yaxis
    gfile.rleft=gfile.R[0]
    gfile.rdim=gfile.R[-1]-gfile.R[0]
    gfile.rcentr=(gfile.R[0]+gfile.R[-1])/2.0
    gfile.zmid=(gfile.Z[0]+gfile.Z[-1])/2.0
    gfile.zdim=gfile.Z[-1]-gfile.Z[0]
    gfile.rmaxis=a*delta+R0
    gfile.zmaxis=0.0
    gfile.bcentr=B0
    gfile.n_bnd=gfile.nr
    gfile.limitr=gfile.nr

    Isq=Idashsq*(1.0-psin)+R0**2.0*B0**2
    
    # eqdsk is in cylindircal r phi z so sign of Bphi must change
    gfile.f_dia=-1.0*np.sign(B0)*np.sqrt(Isq)
    gfile.ffprime=0.5*Idashsq*np.ones(npsi)
    sign_RφZ=1
    gfile.cocos=COCOS(sign_Bp,sign_RφZ,sign_ρθZ)
    gfile.pressure=pdash*(1.0-psin)
    gfile.pprime=-pdash*np.ones(npsi)
    gfile.q=np.zeros(npsi)
    
    gfile.R_bnd=np.zeros(npsi)
    gfile.Z_bnd=np.zeros(npsi)
    gfile.rlim=np.zeros(npsi)
    gfile.zlim=np.zeros(npsi)

    gfile.psi_grid=gfile.psi_grid*psi_one

    return gfile


def makeSplinesFromGfile(gfile):
    global __psi_spline
    global __F_spline
    global __psi_edge
    global __psi_axis
    global __q_spline
    global __p_spline
    global __Zmag
    global __path
    global __B_R_spline
    global __B_Z_spline
    global __B_Phi_spline
    global __b_R_spline
    global __b_Z_spline
    global __b_Phi_spline
    global __RB_Phi_spline
    global __mu0
    global __sign_Bp
    global __sign_RφZ
    global __sign_ρθφ
    global __modB_spline

    import numpy as np

    __mu0=4*np.pi*1e-7

    if(not gfile.cocos):
        gfile.cocos=determineCOCOS(gfile)

    if(not gfile.cocos):
        print("WARNING - cannot determine COCOS, signs will be wrong")
        gfile.cocos=3

    __sign_Bp=what_sign_Bp(gfile.cocos)
    __sign_RφZ=what_sign_RPhiZ(gfile.cocos)
    __sign_ρθφ=what_sign_rhoThetaPhi(gfile.cocos)

    from scipy import interpolate
    __psi_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,gfile.psi_grid,s=0)
    __F_spline=interpolate.UnivariateSpline(gfile.psin,gfile.f_dia,s=0,ext=3) 
    __q_spline=interpolate.UnivariateSpline(gfile.psin,gfile.q,s=0,ext=3) 
    __p_spline=interpolate.UnivariateSpline(gfile.psin,gfile.pressure,s=0,ext=1) 

    __psi_axis=gfile.simag
    __psi_edge=gfile.sibry
    __Zmag=gfile.zmaxis
    
    #inside outside detection
    import matplotlib.path as mpath
    bound=np.column_stack((gfile.R_bnd,gfile.Z_bnd))
    __path=mpath.Path(bound)

    
    def twogrid(x,y):
        nx,ny=len(x),len(y)
        px,py=np.mgrid[0:nx,0:ny]
        xx,yy=np.zeros(px.shape),np.zeros(py.shape)
        xx.flat,yy.flat=x[px.flatten()],y[py.flatten()]
        return xx,yy
    
    RR,ZZ=twogrid(gfile.R,gfile.Z)
    B_R=np.zeros(RR.shape)
    B_Z=np.zeros(RR.shape)
    B_Phi=np.zeros(RR.shape)

    B_R.flat=B_R_RZ(RR.flatten(),ZZ.flatten())
    B_Z.flat=B_Z_RZ(RR.flatten(),ZZ.flatten())
    B_Phi.flat=B_Phi_RZ(RR.flatten(),ZZ.flatten())

    #B component splines for derivatives
    __B_R_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,B_R,s=0)
    __B_Z_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,B_Z,s=0)
    __B_Phi_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,B_Phi,s=0)
    __RB_Phi_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,RR*B_Phi,s=0)

    #drift relevant splines
    modB=np.zeros(RR.shape)
    modB.flat=modB_RZ(RR.flatten(),ZZ.flatten())
    __modB_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,modB,s=0)
    
    b_R=np.zeros(RR.shape)
    b_Z=np.zeros(RR.shape)
    b_Phi=np.zeros(RR.shape)
    b_R.flat=b_R_RZ(RR.flatten(),ZZ.flatten())
    b_Z.flat=b_Z_RZ(RR.flatten(),ZZ.flatten())
    b_Phi.flat=b_Phi_RZ(RR.flatten(),ZZ.flatten())

    __b_R_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,b_R,s=0)
    __b_Z_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,b_Z,s=0)
    __b_Phi_spline=interpolate.RectBivariateSpline(gfile.R,gfile.Z,b_Phi,s=0)
    

    
def F_psin(psin):
    return __F_spline(psin)

def P_psin(psin):
    return __p_spline(psin)

def P_RZ(R,Z):
    return P_psin(psin_RZ(R,Z))

def psi_RZ(R,Z):
    return __psi_spline(R,Z,grid=False)

def psin_RZ(R,Z):
    return (__psi_spline(R,Z,grid=False)-__psi_axis)/(__psi_edge-__psi_axis)

def gradPsi_R_RZ(R,Z):
    return __psi_spline(R,Z,dx=1,dy=0,grid=False)

def gradPsi_Z_RZ(R,Z):
    return __psi_spline(R,Z,dx=0,dy=1,grid=False)

def gradB_R_RZ(R,Z):
    return __modB_spline(R,Z,dx=1,dy=0,grid=False)

def gradB_Z_RZ(R,Z):
    return __modB_spline(R,Z,dx=0,dy=1,grid=False)

def gradPsi_Z_RZ(R,Z):
    return __psi_spline(R,Z,dx=0,dy=1,grid=False)

def B_R_RZ(R,Z):
    return __sign_Bp*__sign_RφZ*(1.0/R)*gradPsi_Z_RZ(R,Z)

def B_Z_RZ(R,Z):
    return -__sign_Bp*__sign_RφZ*(1.0/R)*gradPsi_R_RZ(R,Z)

def B_Phi_RZ(R,Z):
    return F_psin(psin_RZ(R,Z))/R

def modB_RZ(R,Z):
    return (B_R_RZ(R,Z)**2.0+B_Z_RZ(R,Z)**2.0+B_Phi_RZ(R,Z)**2.0)**0.5

def q_R(R):
    return __q_spline(psin_RZ(R,__Zmag))

def isInside(R,Z):
    import numpy as np
    points=np.column_stack((R,Z))
    return __path.contains_points(points)

def j_R_RZ(R,Z):
    #(curl B)_R
    return -__B_Phi_spline(R,Z,dx=0,dy=1,grid=False)/__mu0

def j_Phi_RZ(R,Z):
    #(curl B)_Phi
    return (__B_R_spline(R,Z,dx=0,dy=1,grid=False)-__B_Z_spline(R,Z,dx=1,dy=0,grid=False))/__mu0

def j_Z_RZ(R,Z):
    #(curl B)_Z
    return (1/R)*__RB_Phi_spline(R,Z,dx=1,dy=0,grid=False)/__mu0

def vgradB_R_RZ(R,Z):
    return 0.5*B_Phi_RZ(R,Z)*gradB_Z_RZ(R,Z)/modB_RZ(R,Z)**2

def vgradB_Z_RZ(R,Z):
    return -0.5*B_Phi_RZ(R,Z)*gradB_R_RZ(R,Z)/modB_RZ(R,Z)**2

def vgradB_Phi_RZ(R,Z):
    return 0.5*(B_Z_RZ(R,Z)*gradB_R_RZ(R,Z)-B_R_RZ(R,Z)*gradB_Z_RZ(R,Z))/modB_RZ(R,Z)**2

def jxB_R_RZ(R,Z):
    return j_Phi_RZ(R,Z)*B_Z_RZ(R,Z)-j_Z_RZ(R,Z)*B_Phi_RZ(R,Z)

def jxB_Z_RZ(R,Z):
    return j_R_RZ(R,Z)*B_Phi_RZ(R,Z)-j_Phi_RZ(R,Z)*B_R_RZ(R,Z)

def jxB_Phi_RZ(R,Z):
    return j_Z_RZ(R,Z)*B_R_RZ(R,Z)-j_R_RZ(R,Z)*B_Z_RZ(R,Z)

def b_R_RZ(R,Z):
    return B_R_RZ(R,Z)/modB_RZ(R,Z)

def b_Z_RZ(R,Z):
    return B_Z_RZ(R,Z)/modB_RZ(R,Z)

def b_Phi_RZ(R,Z):
    return B_Phi_RZ(R,Z)/modB_RZ(R,Z)

def kappa_R_RZ(R,Z):
    #last terms are Christoffels!
    return b_R_RZ(R,Z)*__b_R_spline(R,Z,dx=1,dy=0,grid=False)+b_Z_RZ(R,Z)*__b_R_spline(R,Z,dx=0,dy=1,grid=False)-b_Phi_RZ(R,Z)**2/R

def kappa_Phi_RZ(R,Z):
    #last terms are Christoffels!
    return b_R_RZ(R,Z)*__b_Phi_spline(R,Z,dx=1,dy=0,grid=False)+b_Z_RZ(R,Z)*__b_Phi_spline(R,Z,dx=0,dy=1,grid=False)+b_Phi_RZ(R,Z)*b_R_RZ(R,Z)/R

def kappa_Z_RZ(R,Z):
    return b_R_RZ(R,Z)*__b_Z_spline(R,Z,dx=1,dy=0,grid=False)+b_Z_RZ(R,Z)*__b_Z_spline(R,Z,dx=0,dy=1,grid=False)

def vkappa_R_RZ(R,Z):
    return (B_Phi_RZ(R,Z)*kappa_Z_RZ(R,Z)-B_Z_RZ(R,Z)*kappa_Phi_RZ(R,Z))/modB_RZ(R,Z)

def vkappa_Phi_RZ(R,Z):
    return (B_Z_RZ(R,Z)*kappa_R_RZ(R,Z)-B_R_RZ(R,Z)*kappa_Z_RZ(R,Z))/modB_RZ(R,Z)

def vkappa_Z_RZ(R,Z):
    return (B_R_RZ(R,Z)*kappa_Phi_RZ(R,Z)-B_Phi_RZ(R,Z)*kappa_R_RZ(R,Z))/modB_RZ(R,Z)

def writeVtk(gfile,filename,nPhi=6,gridType="unstructured",nx=None,ny=None,nz=None):
    import numpy as np

    def threegrid(x,y,z):
        nx,ny,nz=len(x),len(y),len(z)
        px,py,pz=np.mgrid[0:nx,0:ny,0:nz]
        xx,yy,zz=np.zeros(px.shape),np.zeros(py.shape),np.zeros(pz.shape)
        xx.flat,yy.flat,zz.flat=x[px.flatten()],y[py.flatten()],z[pz.flatten()]
        return xx,yy,zz

    RR,ZZ,PP=threegrid(gfile.R,gfile.Z,np.linspace(0,np.pi*2,nPhi))
    
    if(gridType=="unstructured"):
        RR,ZZ,PP=threegrid(gfile.R,gfile.Z,np.linspace(0,np.pi*2,nPhi))
        nPoints=len(RR.flatten())
        XX=RR*np.cos(PP)
        YY=RR*np.sin(PP)
        pointHeader="UNSTRUCTURED_GRID\nPOINTS {} double\n".format(nPoints)

    elif(gridType=="structured"):
        if(not nx):
            nx=2*len(gfile.R)
        if(not ny):
            ny=2*len(gfile.R)
        if(not nz):
            nz=len(gfile.Z) 

        X=np.linspace(-gfile.R[-1],gfile.R[-1],nx)
        Y=np.linspace(-gfile.R[-1],gfile.R[-1],ny)
        Z=np.linspace(gfile.Z[0],gfile.Z[-1],nz)

        XX,YY,ZZ=threegrid(X,Y,Z)
        RR=np.sqrt(XX**2+YY**2)
        PP=np.arctan2(YY,XX)
        branchCut=np.where(PP.flatten()<0)
        PP.flat[branchCut]=PP.flat[branchCut]+2*np.pi
        nPoints=len(RR.flatten())
        pointHeader="STRUCTURED_GRID\nDIMENSIONS {} {} {}\nPOINTS {} double\n".format(nx,ny,nz,nPoints)

    else:
        raise ValueError("unsupported grid type: {}".format(gridType))

    scalars=["Pressure","PsiPoloidal"]
    vectors=["B","B_T","B_P","j","j_T","j_P","vgradB","jxB","kappa","vkappa"]
    items={}
    

    def componentX(component_Phi,component_R):
        return -component_Phi*np.sin(PP.flatten())+component_R*np.cos(PP.flatten())
    
    def componentY(component_Phi,component_R):
        return component_Phi*np.cos(PP.flatten())+component_R*np.sin(PP.flatten())
    
    items['PsiPoloidal']=psi_RZ(RR.flatten(),ZZ.flatten())
    items['Pressure']=P_RZ(RR.flatten(),ZZ.flatten())

    items['B_R']=B_R_RZ(RR.flatten(),ZZ.flatten())
    items['B_Z']=B_Z_RZ(RR.flatten(),ZZ.flatten())
    items['B_Phi']=B_Phi_RZ(RR.flatten(),ZZ.flatten())

    items['j_R']=j_R_RZ(RR.flatten(),ZZ.flatten())
    items['j_Z']=j_Z_RZ(RR.flatten(),ZZ.flatten())
    items['j_Phi']=j_Phi_RZ(RR.flatten(),ZZ.flatten())
    
    items['vgradB_R']=vgradB_R_RZ(RR.flatten(),ZZ.flatten())
    items['vgradB_Z']=vgradB_Z_RZ(RR.flatten(),ZZ.flatten())
    items['vgradB_Phi']=vgradB_Phi_RZ(RR.flatten(),ZZ.flatten())

    items['jxB_R']=jxB_R_RZ(RR.flatten(),ZZ.flatten())
    items['jxB_Z']=jxB_Z_RZ(RR.flatten(),ZZ.flatten())
    items['jxB_Phi']=jxB_Phi_RZ(RR.flatten(),ZZ.flatten())

    items['kappa_R']=kappa_R_RZ(RR.flatten(),ZZ.flatten())
    items['kappa_Z']=kappa_Z_RZ(RR.flatten(),ZZ.flatten())
    items['kappa_Phi']=kappa_Phi_RZ(RR.flatten(),ZZ.flatten())

    items['vkappa_R']=vkappa_R_RZ(RR.flatten(),ZZ.flatten())
    items['vkappa_Z']=vkappa_Z_RZ(RR.flatten(),ZZ.flatten())
    items['vkappa_Phi']=vkappa_Phi_RZ(RR.flatten(),ZZ.flatten())

    for vector in ["B","j","vgradB","jxB","kappa","vkappa"]:
        for version in ["","_T","_P"]:
            for component in ["_X","_Y","_Z"]:
                name=vector+version+component
                poloidal=1.0
                toroidal=1.0
                if "_T"==version:
                    poloidal=0.0
                if "_P"==version:
                    toroidal=0.0
                if "_X"==component:
                    items[name]=componentX(toroidal*items[vector+"_Phi"],poloidal*items[vector+"_R"])
                if "_Y"==component:
                    items[name]=componentY(toroidal*items[vector+"_Phi"],poloidal*items[vector+"_R"])
                if "_Z"==component:
                    items[name]=poloidal*items[vector+"_Z"]
                    
                    
    def sanitize(data):
        data.flat[RR.flatten()<gfile.R[0]]=0
        data.flat[RR.flatten()>gfile.R[-1]]=0

    for scalar in scalars:
        item=items[scalar]
        sanitize(item)

    for vector in vectors:
        for component in ["X","Y","Z"]:
            name=vector+"_"+component
            item=items[name]
            sanitize(item)

    #include pressure only within LCFS
    items['Pressure'][np.logical_not(isInside(RR.flatten(),ZZ.flatten()))]=0
    
    with open(filename,"w") as File:
        File.write("# vtk DataFile Version 2.0\n")
        File.write(gfile.A52+"\n")
        File.write("ASCII\n")
        File.write("DATASET "+pointHeader)
        x=XX.flatten()
        y=YY.flatten()
        z=ZZ.flatten()
        for p,_ in enumerate(x):            
            File.write("{} {} {} \n".format(x[p],y[p],z[p]))
        File.write("POINT_DATA {} \n".format(nPoints))
        
        for scalar in scalars:
            item=items[scalar]
            File.write("SCALARS {} double 1\n".format(scalar))
            File.write("LOOKUP_TABLE default\n")
            for p,_ in enumerate(x): 
                File.write("{} \n".format(item[p]))

        for vector in vectors:
            File.write("VECTORS {} double\n".format(vector))
            for p,_ in enumerate(x): 
                for component in ["X","Y","Z"]:
                    name=vector+"_"+component
                    item=items[name]
                    File.write("{} ".format(item[p]))
                File.write("\n")


        def replaceLastComma():
            File.seek(File.tell()-1)
            File.write("\n")
            
        scalars=["PointID","x","y","z","Pressure","PsiPoloidal"]
        items["PointID"]=np.arange(len(x))
        items["x"]=x
        items["y"]=y
        items["z"]=z

        with open(filename+".csv","w") as File:
              
            #titles
            for scalar in scalars:
                File.write(scalar+",")
            for vector in vectors:
                for component in ["X","Y","Z"]:
                    name=vector+"_"+component
                    File.write(name+",")
   
            replaceLastComma()

            for p,_ in enumerate(x): #for each point
                #go through each scalar name in the local namespace
                for scalar in scalars:
                    item=items[scalar]
                    File.write(str(item[p]))
                    File.write(",")
                #repeat for each components of the vectors
                for vector in vectors:
                    for component in ["X","Y","Z"]:
                        name=vector+"_"+component
                        item=items[name]
                        File.write(str(item[p]))
                        File.write(",")
                #end of line
                replaceLastComma()
