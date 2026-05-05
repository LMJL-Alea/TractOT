import numpy as np
import scipy as sc
    
def sqrtms(M):
    #square root of symmetric matrix (much faster than sqrtm)
    Val, Vec = np.linalg.eigh(M)
    Val=np.maximum(Val,0)
    Val = np.sqrt(Val)
    Q = np.einsum("...jk,...k->...jk", Vec, Val)
    return np.einsum("...jk,...kl->...jl", Q, np.swapaxes(Vec, -1, -2))
    
def sqrtm(M,sym=False):
    #square root of matrix
    Val, Vec = np.linalg.eig(M)
    Val=np.maximum(Val,0)
    Val = np.sqrt(Val)
    Q = np.einsum("...jk,...k->...jk", Vec, Val)
    return np.einsum("...jk,...kl->...jl", Q, np.swapaxes(Vec, -1, -2))

  
def random_MASs(nb_MAS,nb_MAP,k,d=3):
    #Sample random MASs
    m=np.random.normal(size=(nb_MAS,nb_MAP,d))
    
    w=abs(np.random.standard_normal((nb_MAS,nb_MAP,k+1)))
    w/=np.sum(w,-1)[:,:,None]
    wI=w[:,:,0]
    w=w[:,:,1:]

    I=abs(np.random.standard_normal((nb_MAS,nb_MAP)))
    S=np.zeros((nb_MAS,nb_MAP,k,d,d))
    for i in range(nb_MAS):
        for j in range(nb_MAP):
            S[i,j]=random_covs(k,d)

    return m,wI,I,w,S
    
def random_mixture(k,d):
    return random_covs(k,d),random_weight(k)
    
def random_covs(k,d):
    S=np.zeros((k,d,d))
    for i in range(k):
        U=np.diag(np.random.uniform(0,10,d))
        P=sc.stats.ortho_group.rvs(d)
        S[i,:,:]=P@U@(P.T)
    return S

def random_weight(k):
    w=abs(np.random.standard_normal(k))
    return w/sum(w)   


def concat_w_MAP(wI,w):
    w_stack=np.zeros(w.shape[0]+1)
    w_stack[0]=wI
    w_stack[1:]=w
    return w_stack
    
def concat_w_MAS(wI,w):
    w_stack=np.zeros((w.shape[0],w.shape[1]+1))
    w_stack[:,0]=wI
    w_stack[:,1:]=w
    return w_stack
    
def compute_ksize(MAT):
    return int((MAT.shape[1]-8)/7)
    
    
####################################
# Covariances matrices to 1D array #
# 1D array to covariances matrices #
####################################

def cov_to_array_MAS(S,d=3):
    #Convert covariance matrix into array of 6 elements
    #At the scale of MAS
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
    
def array_to_cov_MAS(l,d=3):
    #Convert an array of kx6 elements into the corresponding covariances matrices
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
    #Convert an array of kx6 elements into the corresponding covariances matrices
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

def most_colinear_compartment(m,S):
    #Compute the most colinear compartment of S
    #m in R(nb_pts,d)
    #S in R(nb_pts,k,d,d)
    #return nb_pts x idx
    
    dir_m=np.zeros(m.shape)
    dir_m[1:-1]=m[2:]-m[:-2]
    dir_m[0]=m[1]-m[0]
    dir_m[-1]=m[-1]-m[-2]
    dir_m/=np.linalg.norm(dir_m,axis=1)[:,None]

    val,vec=np.linalg.eigh(S)
    largest_vec=np.multiply(vec[:,:,:,-1],0,where=(val[:,:,-1]==0)[:,:,None],out=vec[:,:,:,-1]) #avoid eigenvector of null matrix
    
    return np.argmin(np.arccos(abs(np.clip(np.sum(dir_m[:,None]*largest_vec,-1),-1.0,1.0))),1)

