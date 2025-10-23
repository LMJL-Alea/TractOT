import numpy as np
from utils import sqrtms,concat_w_MAP,concat_w_MAS
import ot
from ICP import ICP_tracts

def distance_aniso(S1,S2):
    #S1 in R(d,d) and S2 in R(d,d)
    #return scalar
    sq1=sqrtms(S1)
    C=np.einsum("...kl,...lm,...mn->...kn",sq1,S2,sq1)
    sq_C=sqrtms(C)
    tr1 = np.trace(S1)
    tr2 = np.trace(S2) 
    return np.maximum(tr1 + tr2 -2*np.trace(sq_C),0)
 
def distances_aniso(S1,S2):
    #S1 in R(k1,d,d) and S2 in R(k2,d,d)
    #return C in R(k1,k2)
    sq1=sqrtms(S1)
    C=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_C=sqrtms(C)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)[:,None]
    tr2 = np.trace(S2,axis1=-2,axis2=-1)[None,:]  # (1, k_t)
    return np.maximum(tr1 + tr2 -2*np.trace(sq_C,axis1=-2,axis2=-1),0)

def distances_aniso_along(S1,S2): 
    #S1 in R(nb_pts,k1,d,d) and S2 in R(nb_pts,k2,d,d)
    #return C in R(nb_pts,k1,k2)
    sq1=sqrtms(S1)
    C=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_C=sqrtms(C)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)[:,:,None]
    tr2 = np.trace(S2,axis1=-2,axis2=-1)[:,None,:]  # (1, k_t)
    return np.maximum(tr1 + tr2 -2*np.trace(sq_C,axis1=-2,axis2=-1),0)
    
def distances_iso_aniso(I1,I2,S1,S2):
    #I1,I2 in R
    #S1 in R(k1,d,d) and S2 in R(k2,d,d)
    #return C in R(k1,k2)
    k1,d=S1.shape[0],S1.shape[-1]
    k2=S2.shape[0]
    C=np.zeros((k1+1,k2+1))
    
    sq1=sqrtms(S1)
    sq2=sqrtms(S2)
    A=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_A=sqrtms(A)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)
    tr2 = np.trace(S2,axis1=-2,axis2=-1) 

    C[1:,1:]=tr1[:,None] + tr2[None,:] -2*np.trace(sq_A,axis1=-2,axis2=-1)

    C[0,0]=d*(I1+I2-2*np.sqrt(I1*I2))
    C[0,1:]=d*I1+tr2-2*np.sqrt(I1)*np.trace(sq2,axis1=-2,axis2=-1)
    C[1:,0]=d*I2+tr1-2*np.sqrt(I2)*np.trace(sq1,axis1=-2,axis2=-1)
    return np.maximum(C,0)
    
def distances_iso_aniso_along(I1,I2,S1,S2):
    #I1,I2 in R(nb_pts)
    #S1 in R(nb_pts,k1,d,d) and S2 in R(nb_pts,k2,d,d)
    #return C in R(nb_pts,k1,k2)

    nb_pts,k1,d=S1.shape[0],S1.shape[1],S1.shape[-1]
    k2=S2.shape[1]
    C=np.zeros((nb_pts,k1+1,k2+1))
    
    sq1=sqrtms(S1)
    sq2=sqrtms(S2)
    A=np.einsum("...ikl,...jlm,...imn->...ijkn",sq1,S2,sq1)
    sq_A=sqrtms(A)
    tr1 = np.trace(S1,axis1=-2,axis2=-1)
    tr2 = np.trace(S2,axis1=-2,axis2=-1)
    C[:,1:,1:]=np.maximum(tr1[:,:,None] + tr2[:,None,:] -2*np.trace(sq_A,axis1=-2,axis2=-1),0)

    C[:,0,0]=d*(I1+I2-2*np.sqrt(I1*I2))
    C[:,0,1:]=d*I1[:,None]+tr2-2*np.sqrt(I1)[:,None]*np.trace(sq2,axis1=-2,axis2=-1)
    C[:,1:,0]=d*I2[:,None]+tr1-2*np.sqrt(I2)[:,None]*np.trace(sq1,axis1=-2,axis2=-1)
    return np.maximum(C,0)
    
def distances_spatial_iso_aniso(m1,m2,I1,I2,S1,S2,alpha=.5):
    #m1,m2 in R(d)
    #I1,I2 in R
    #S1 in R(k1,d,d) and S2 in R(k2,d,d)
    #return C in R(k1,k2)
    return alpha*np.sum((m1-m2)**2)+(1-alpha)*distances_iso_aniso(I1,I2,S1,S2)
    
def distances_spatial_iso_aniso_along(m1,m2,I1,I2,S1,S2,alpha=.5):
    #m1,m2 in R(nb_pts,d)
    #I1,I2 in R(nb_pts)
    #S1 in R(nb_pts,k1,d,d) and S2 in R(nb_pts,k2,d,d)
    #return C in R(nb_pts,k1,k2)
    return alpha*np.sum((m1-m2)**2,1)[:,None,None]+(1-alpha)*distances_iso_aniso_along(I1,I2,S1,S2)
    
def distance_MAP(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5):
    #m1,m2 in R(d)
    #I1,I2 in R
    #S1 in R(k1,d,d) and S2 in R(k2,d,d)
    #w1 in R(k1) and w2 in R(k2)
    #return scalar
    
    C=distances_spatial_iso_aniso(m1,m2,I1,I2,S1,S2,alpha)
    
    w1_stack=concat_w_MAP(wI1,w1)
    w2_stack=concat_w_MAP(wI2,w2) 
    
    return ot.emd2(w1_stack,w2_stack,C)
    
def distances_MAP(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5):
    #m1,m2 in R(nb_MAP1,d),R(nb_MAP2,d)
    #I1,I2 in R(nb_MAP1),R(nb_MAP2,d)
    #S1 in R(nb_MAP1,k1,d,d) and S2 in R(nb_MAP2,k2,d,d)
    #w1 in R(nb_MAP1,k1) and w2 in R(nb_MAP2,k2)
    #return C in (nb_MAP1,nb_MAP2)
    
    nb_MAPs1,nb_MAPs2=m1.shape[0],m2.shape[0]
    C=np.zeros((nb_MAPs1,nb_MAPs2))
    for i in range(nb_MAPs1):
        for j in range(nb_MAPs2):
            C[i,j]=distance_MAP(m1[i],m2[j],wI1[i],wI2[j],I1[i],I2[j],w1[i],w2[j],S1[i],S2[j],alpha)
    return C
    
def distance_MAS(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,wMAS1=None,wMAS2=None,alpha=.5):
    #m1,m2 in R(nb_MAP1,d),R(nb_MAP2,d)
    #I1,I2 in R(nb_MAP1),R(nb_MAP2,d)
    #S1 in R(nb_MAP1,k1,d,d) and S2 in R(nb_MAP2,k2,d,d)
    #w1 in R(nb_MAP1,k1) and w2 in R(nb_MAP2,k2)
    #return scalar
    if wMAS1 is None:
        wMAS1=np.ones(m1.shape[0])/m1.shape[0]
        
    if wMAS2 is None:
        wMAS2=np.ones(m2.shape[0])/m2.shape[0]
        
    C=distances_MAP(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha)
    return ot.emd2(wMAS1,wMAS2,C)
    
def distance_MAS_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5): 
    #m1,m2 in R(nb_MAP,d),R(nb_MAP,d)
    #I1,I2 in R(nb_MAP),R(nb_MAP,d)
    #S1 in R(nb_MAP,k1,d,d) and S2 in R(nb_MAP,k2,d,d)
    #w1 in R(nb_MAP,k1) and w2 in R(nb_MAP,k2)
    #return scalar
    C=distances_spatial_iso_aniso_along(m1,m2,I1,I2,S1,S2,alpha)
    nb_MAP=C.shape[0]

    D=0
    w1_stack=concat_w_MAS(wI1,w1)
    w2_stack=concat_w_MAS(wI2,w2)    
    for i in range(nb_MAP):
        #if np.sum(w1_stack) != 0 and np.sum(w2_stack)!=0: #for voxel out of image
        D+=ot.emd2(w1_stack[i],w2_stack[i],C[i])
    return D/nb_MAP
    
def distances_MAS_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=0.5):
    #m1,m2 in R(nb_MAS1,nb_MAP,d),R(nb_MAS2,nb_MAP,d)
    #I1,I2 in R(nb_MAS1,nb_MAP),R(nb_MAS2,nb_MAP,d)
    #S1 in R(nb_MAS1,nb_MAP,k1,d,d) and S2 in R(nb_MAS2,nb_MAP,k2,d,d)
    #w1 in R(nb_MAS1,nb_MAP,k1) and w2 in R(nb_MAS2,nb_MAP,k2)
    #return C in R(nb_MAS1,nb_MAS2)
    nb_MAS1=m1.shape[0]
    nb_MAS2=m2.shape[0]
    
    C=np.zeros((nb_MAS1,nb_MAS2))
    for i in range(nb_MAS1):
        for j in range(nb_MAS2):
            C[i,j]=distance_MAS_idx(m1[i],m2[j],wI1[i],wI2[j],I1[i],I2[j],w1[i],w2[j],S1[i],S2[j],alpha)

    return C
    
    
def distance_MAT_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha=.5,reflexion=False,lam1=0,lam2=0):
    nb_MAS=m1.shape[0]
    
    m1_new,m2_new,R,t,sigma=ICP_tracts(m1,m2,reflexion,lam1,lam2,itr=3)
    
    if reflexion:
        Ox=np.eye(3)
        Ox[0,0]=-1    
        S1_new=Ox@S1@Ox
    else:
        S1_new=S1
    
    w1_new=w1
    S1_new=R.T@S1_new@R
    wI1_new=wI1
    I1_new=I1 #iso invariant by rotation
    
    wI2_new=wI2[sigma]
    I2_new=I2[sigma]
    w2_new=w2[sigma]
    S2_new=S2[sigma]

    d=0
    for i in range(nb_MAS):
        d+=distance_MAS_idx(m1_new[i],m2_new[i],wI1_new[i],wI2_new[i],I1_new[i],I2_new[i],w1_new[i],w2_new[i],S1_new[i],S2_new[i],alpha=.5)
        d+=lam1*np.sum((np.eye(3)-R)**2)+lam2*np.sum(t**2)
    return d
