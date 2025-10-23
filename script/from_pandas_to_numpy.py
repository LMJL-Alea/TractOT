import numpy as np
from new_grid import newgrid_streamline,newgrid_MAS
from utils import compute_ksize

def pandas_to_numpy_tract(tract,nb_streamline=1,streamlineId=None,d=3):
    #Transform a pandas.csv file of tract into a list of numpy array 
    nb_pts=[]
    m=[]
    for i in range(0,nb_streamline):
        if streamlineId is None:
            streamline = tract[tract["StreamlineId"]==i+1]
        else:
            streamline = tract[tract["StreamlineId"]==streamlineId[i]]
        
        nb_pts+=[streamline.shape[0]]    
        m+=[streamline.iloc[:,:3].to_numpy()]
    return m,nb_pts
            
def pandas_to_numpy_tract_interpolate(tract,nb_streamline=1,nb_pts=50,streamlineId=None,d=3):
    #Transform a .csv file of tract into numpy array
    #Every streamline are re-sample on a uniform grid with nb_pts elements 
    m=np.zeros((nb_streamline,nb_pts,d))
    m_raw,_=pandas_to_numpy_tract(tract,nb_streamline,streamlineId,d)
    
    if nb_streamline ==1:
        m[0]=newgrid_streamline(m_raw,nb_pts=nb_pts)
        
    else:
        for i in range(0,nb_streamline):
            m[i]=newgrid_streamline(m_raw[i],nb_pts=nb_pts)
    
    return m,nb_pts    
    
def pandas_to_numpy_MAT(tract_MCM,nb_MAS=1,streamlineId=None,d=3):
    #Transform a .csv file of MAT into a list of numpy array
    ksize=compute_ksize(tract_MCM)
    
    m=[]
    wI=[]
    I=[]
    w=[]
    S=[]
    nb_MAP=[]
    for i in range(0,nb_MAS):
        if streamlineId is None:
            streamline = tract_MCM[tract_MCM["StreamlineId"]==i+1]
        else:
            streamline = tract_MCM[tract_MCM["StreamlineId"]==streamlineId[i]]
        
        m+=[streamline.iloc[:,:3].to_numpy()]
        
        wI+=[np.ascontiguousarray(streamline.iloc[:,6].to_numpy())]
        I+=[streamline.iloc[:,7+ksize].to_numpy()]
        
        w+=[np.ascontiguousarray(streamline.iloc[:,7:7+ksize].to_numpy())]
        S+=[array_to_cov(streamline.iloc[:,8+ksize:].to_numpy())]
        
        nb_MAP+=[streamline.shape[0]]  
        
    return m,wI,I,w,S,nb_MAP
    
    
def pandas_to_numpy_MAT_interpolate(tract_MCM,nb_MAS=1,nb_MAP=50,streamlineId=None,d=3):
    #Transform a .csv file of MAT into numpy array
    #Every MAS are re-sample on a uniform grid with nb_MAPs elements 
    ksize=compute_ksize(tract_MCM)
    
    m=np.zeros((nb_MAS,nb_MAP,d))
    wI=np.zeros((nb_MAS,nb_MAP))
    I=np.zeros((nb_MAS,nb_MAP))
    w=np.zeros((nb_MAS,nb_MAP,ksize))
    S=np.zeros((nb_MAS,nb_MAP,ksize,d,d))
    
    
    m_raw,wI_raw,I_raw,w_raw,S_raw,_=pandas_to_numpy_MAT(tract_MCM,nb_MAS,streamlineId,d)
    
    if nb_MAS ==1:
        m[0],wI[0],I[0],w[0],S[0]=newgrid_MAS(m_raw,wI_raw,I_raw,w_raw,S_raw,nb_MAP=nb_MAP)
        
    else:
        for i in range(0,nb_MAS):
            m[i],wI[i],I[i],w[i],S[i]=newgrid_MAS(m_raw[i],wI_raw[i],I_raw[i],w_raw[i],S_raw[i],nb_MAP=nb_MAP)
    
    return m,wI,I,w,S,nb_MAP
    
def array_to_cov(l,d=3):
    #Convert an array of 6 elements into the corresponding covariance matrix
    ksize=int((l.shape[1])/6)
    nb_pts=l.shape[0]
    
    lr=l.reshape((nb_pts,ksize,6))
    S=np.zeros((nb_pts,ksize,d,d)) 

    S[:,:,0,0]=lr[:,:,0]
    S[:,:,:2,1]=lr[:,:,1:3]
    S[:,:,:,2]=lr[:,:,3:]
    S[:,:,1,0]=lr[:,:,1]
    S[:,:,2,:2]=lr[:,:,3:5]
    return S
    
def array_to_cov_neighbor(l,d=3):
    #Convert an array of 6 elements into the corresponding covariance matrix
    #For multiple neighbors 
    #l in (nb_pts,3,3,3,6)
    
    nb_pts=l.shape[0]
    d=l.shape[1]
    S=np.zeros((nb_pts,3,3,3,d,d))

    S[:,:,:,:,0,0]=l[:,:,:,:,0]
    S[:,:,:,:,:2,1]=l[:,:,:,:,1:3]
    S[:,:,:,:,:,2]=l[:,:,:,:,3:]
    S[:,:,:,:,1,0]=l[:,:,:,:,1]
    S[:,:,:,:,2,:2]=l[:,:,:,:,3:5]
    return S

def cov_to_array_MAS(S,d=3):
    #Convert covariance matrix into array of 6 elements
    nb_pts,ksize=S.shape[0],S.shape[1]
    l=np.zeros((nb_pts,ksize,6))
    l[:,:,0]=S[:,:,0,0]
    l[:,:,1:3]=S[:,:,:2,1]
    l[:,:,3:]=S[:,:,:,-1]
    return l.reshape((nb_pts,-1))

def cov_to_array_MAT(S,d=3):
    #Convert covariance matrix into array of 6 elements
    #At the scale of MAT
    nb_MAS,nb_pts,ksize=S.shape[0],S.shape[1],S.shape[2]
    l=np.zeros((nb_MAS,nb_pts,6*ksize))
    for i in range(nb_MAS):
        l[i]=cov_to_array_MAS(S[i])
    return l.reshape((nb_MAS*nb_pts,-1))
    

