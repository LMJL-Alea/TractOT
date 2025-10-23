import numpy as np
from distances import distances_aniso_along
from utils import sqrtms

#In parallel wrt to nb of points

def update_P(w1,S1,S2):
    k2=S2.shape[1]
    C=distances_aniso_along(S1,S2)
    idxs=np.argmin(C,axis=-1)
    non_zero=idxs[:,:,None]==np.arange(k2)
    return w1[:,:,None]*non_zero
    
def update_w(P):
    return np.sum(P,1)

def update_S(P,w1,S1,S2,eps):
    k2=P.shape[-1]
    return gaussians_barycenter_fixpoint(np.repeat(S1[:,:,None,:,:],k2,axis=2), P, max_itr=50, eps=eps,Sigma_init=S2)

def proj_GMM_MAS(k2,ksize,w1,S1,max_itr=10,eps=1e-7):
    #k2 output nb of compartment
    #ksize size of the output array
    nb_pts=w1.shape[0]
    d=S1.shape[-1]

    #Flexible nb of compartments
    if np.array(k2).shape == (): #Should find a better test
        k2=np.int32(k2*np.ones(nb_pts))
        
    #Initialisation
    u=np.argsort(w1)[:,::-1]
    w1=np.take_along_axis(w1,u,1)
    S1=np.take_along_axis(S1,u[:,:,None,None],1)
    S2=np.zeros((nb_pts,ksize,d,d))
    for i in range(nb_pts):
        S2[i,:k2[i]]=np.copy(S1[i,:k2[i]])
        
    
    for i in range(max_itr):
        P=update_P(w1,S1,S2)
        
        S2_new=update_S(P,w1,S1,S2,eps)
        
        #Check convergence
        diff=np.linalg.norm(S2_new-S2)
        if diff <eps:
            S2 = S2_new
            break
        S2 = S2_new
    
    w2=update_w(P)      
    return w2,S2
    
def gaussians_barycenter_fixpoint(S, P, max_itr=50, eps=1e-7,Sigma_init=None):
    #Compute barycenters of gausians by fix point algo
    Pweight=np.divide(P,np.sum(P,1)[:,None,:],where=(np.sum(P,1)!=0)[:,None,:],out=np.zeros(P.shape))
    
    if Sigma_init is None:
        Sigma=np.sum(sqrtms(S)*Pweight[:,:,:,None,None],1)
    else:
        Sigma=Sigma_init
        
    for i in range(max_itr): #scale with d not with K
        Sigma_sq=sqrtms(Sigma)
        Sigma_new = np.sum(sqrtms(np.einsum('m...ij,m...jk,m...kl->m...il',Sigma_sq,S,Sigma_sq))*Pweight[:,:,:,None,None],1)
        
        # check convergence
        diff = np.linalg.norm(Sigma - Sigma_new)
        if diff <= eps:
            Sigma = Sigma_new
            break
            
        Sigma = Sigma_new
    return Sigma

