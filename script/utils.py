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
    
    
def build_MAT_from_tract(t,comp):
    MAT=t.copy(deep=True)
    d={}
    d['MostColinearIndex']=np.zeros(MAT.shape[0])
    d['FreeWaterWeight']=np.zeros(MAT.shape[0])
    for i in range(1,comp+1):
        d['Tensor'+str(i)+'Weight']=np.zeros(MAT.shape[0])
    d['FreeWaterParameter1']=np.zeros(MAT.shape[0])
    for i in range(1,comp+1):
        for j in range(1,7):
            d['Tensor'+str(i)+'Parameter'+str(j)]=np.zeros(MAT.shape[0])
    return MAT.assign(**d)

    

