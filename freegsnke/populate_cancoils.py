import numpy as np

import os
this_dir , this_filename = os.path.split(__file__)

def cancoils(name):
    t_can_slabs=[]
    t_can_tab=np.genfromtxt(name,delimiter=',',names=True)#skip_header=1)
    for line in t_can_tab:
        t_can_slabs=t_can_slabs+[ [ {'R':line['R'] ,'Z':line['Z'] ,'dR':line['dR'] ,'dZ':line['dZ'] } ] ]
    # now start splitting the coils and replacing them in the list
    t_can_coils=[]
    for slab in t_can_slabs:
        t_can_coils=t_can_coils+splitcoil(slab)
    return t_can_coils

def splitcoil(coil, dRsplit=100.0 , dZsplit=100.0):
    #coil must be a list of {'R':... , 'Z':... , 'dR':... , 'dZ':... } dicts
    tcoil=coil.copy()
    cont=False
    for incoil in tcoil:
        if incoil['dR']>dRsplit:
            newcoil1= {'R':(incoil['R']+0.25*incoil['dR']),'Z':incoil['Z'],'dR':0.5*incoil['dR'],'dZ':incoil['dZ']}
            newcoil2= {'R':(incoil['R']-0.25*incoil['dR']),'Z':incoil['Z'],'dR':0.5*incoil['dR'],'dZ':incoil['dZ']}
            tcoil=tcoil+[newcoil1,newcoil2]
            tcoil.remove(incoil)
            cont=True
        elif incoil['dZ']>dZsplit:
            newcoil1= {'R':(incoil['R']),'Z':(incoil['Z']+0.25*incoil['dZ']),'dR':incoil['dR'],'dZ':0.5*incoil['dZ']}
            newcoil2= {'R':(incoil['R']),'Z':(incoil['Z']-0.25*incoil['dZ']),'dR':incoil['dR'],'dZ':0.5*incoil['dZ']}
            tcoil=tcoil+[newcoil1,newcoil2]
            tcoil.remove(incoil)
            cont=True
        else:
            pass
    if cont: tcoil=splitcoil(tcoil)
    return tcoil

can_names=['D1','D2','D3','D5','Dp','D5','D6','D7','P4','P5']
coilcans={}
for name in can_names:
    tcancoils=cancoils(this_dir+'/MASTU_pass/'+name+'Can.csv')
    # print(name)
    # print(tcancoils)
    for (i,coil) in enumerate(tcancoils):
        coilcans['can_'+name+'upper_'+str(i)]= {'R':coil['R']/1000.0,'Z':coil['Z']/1000.0,'dR':coil['dR']/1000.0,'dZ':coil['dZ']/1000.0}
        coilcans['can_'+name+'lower_'+str(i)]= {'R':coil['R']/1000.0,'Z':-coil['Z']/1000.0,'dR':coil['dR']/1000.0,'dZ':coil['dZ']/1000.0}
# print(coilcans)
# print(len(coilcans))