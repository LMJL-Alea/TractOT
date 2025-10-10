# Interpolation of a streamline with/without GMM
# Build n uniform element support on an input streamline 

import numpy as np
from utils import sqrtm
from projection_GMM import proj_GMM_MAS

def newgrid_streamline(m, nb_pts): #Without GMM
    #Compute streamline on a uniform grid with nb_pts 
    
    segments = m[1:] - m[:-1]
    lens_segments = np.sqrt(np.sum(segments**2, axis=1))
    dists = np.zeros(m.shape[0])
    dists[1:] = np.cumsum(lens_segments)
    len_segments = np.sum(lens_segments)
    
    # Compute new grid
    grid = np.linspace(start=0, stop=len_segments, num=nb_pts-1,endpoint=False)
    
    #check wich segment goes grid
    idxs_segments=(dists[:,None]<=grid[None]).sum(0)-1
    segments_starts,segments_ends=dists[idxs_segments],dists[idxs_segments+1]
    
    #Compute weights of interpolante
    ts=(grid-segments_starts)/(segments_ends-segments_starts)
    
    #Linear interpolation
    return np.concatenate((m[idxs_segments] + ts[:,None] * (m[idxs_segments + 1] - m[idxs_segments]),m[-1][None]))
    
def average_GMM(w1,S1,w2,S2,ts):
    #Compute average of nb_MAP*2 mixture
    ksize=S1.shape[1]
    k1,k2=np.count_nonzero(w1,axis=-1),np.count_nonzero(w2,axis=-1)
    kmax=np.max(np.concatenate((k1[:,None],k2[:,None]),axis=-1),axis=-1)
    w3,S3=np.concatenate(((1-ts[:,None])*w1,ts[:,None]*w2),axis=1),np.concatenate((S1,S2),axis=1)
    w,S=proj_GMM_MAS(kmax,ksize,w3,S3)
    return w,S
    
def newgrid_MAS(m,wI,I,w,S,nb_MAP,d=3): #With GMM
    #Compute MAS on a uniform grid with nb_MAPs 
    k=S.shape[1]
    segments = m[1:] - m[:-1]
    lens_segments = np.sqrt(np.sum(segments**2, axis=1))
    dists = np.zeros(m.shape[0])
    dists[1:] = np.cumsum(lens_segments)
    len_segments = np.sum(lens_segments)
    
    # Compute new grid
    grid = np.linspace(start=0, stop=len_segments, num=nb_MAP-1,endpoint=False)
    
    #check wich segment goes grid
    idxs_segments=(dists[:,None]<=grid[None]).sum(0)-1
    segments_starts,segments_ends=dists[idxs_segments],dists[idxs_segments+1]
    
    #Compute weights of interpolante
    ts=(grid-segments_starts)/(segments_ends-segments_starts)
    
    #Linear interpolation
    m1=np.concatenate((m[idxs_segments] + ts[:,None] * (m[idxs_segments + 1] - m[idxs_segments]),m[-1][None]))
    wI1=np.concatenate((wI[idxs_segments] + ts * (wI[idxs_segments + 1] - wI[idxs_segments]),wI[-1][None]))
    I1=np.concatenate((I[idxs_segments] + ts * (I[idxs_segments + 1] - I[idxs_segments]),I[-1][None]))
    
    #Average of GMM 
    w1=np.zeros((nb_MAP,k))
    w1[0],w1[-1]=w[0],w[-1]
    S1=np.zeros((nb_MAP,k,d,d))
    S1[0],S1[-1]=S[0],S[-1]
    
    w1[1:-1],S1[1:-1]=average_GMM(w[idxs_segments[1:]],S[idxs_segments[1:]],w[idxs_segments[1:]+1],S[idxs_segments[1:]+1],ts[1:])
        
    return m1,wI1,I1,w1,S1
