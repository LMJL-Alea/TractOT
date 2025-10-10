import numpy as np
from new_grid import newgrid_streamline,newgrid_MAS
from utils import compute_ksize

def trk_to_numpy_tract(tract,nb_streamline=1,streamlineId=None,d=3):
    #Transform a .trk file of tract into a list of numpy array 
    nb_pts=[]
    m=[]
    for i in range(0,nb_streamline):
        if streamlineId is None:
            streamline = tract.streamlines[i]
        else:
            streamline = tract.streamlines[streamlineId[i]]
        
        nb_pts+=[streamline.shape[0]]    
        m+=[streamline]
    if nb_streamline==1:
        return m[0],nb_pts[0]
    else:
        return m,nb_pts
            
def trk_to_numpy_tract_interpolate(tract,nb_streamline=1,nb_pts=50,streamlineId=None,d=3):
    #Transform a .trk file of tract into numpy array
    #Every streamline are re-sample on a uniform grid with nb_pts elements 
    m=np.zeros((nb_streamline,nb_pts,d))
    m_raw,_=trk_to_numpy_tract(tract,nb_streamline,streamlineId,d)
    
    if nb_streamline ==1:
        m[0]=newgrid_streamline(m_raw,nb_pts=nb_pts)
        
    else:
        for i in range(0,nb_streamline):
            m[i]=newgrid_streamline(m_raw[i],nb_pts=nb_pts)
    
    if nb_streamline==1:
        return m[0],nb_pts
    else:
        return m,nb_pts    
    

