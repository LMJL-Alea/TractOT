import numpy as np
from distances import distances_MAS_idx
from projection_GMM import proj_GMM_MAS

########################
## Clustering of MASs ##
########################

def compute_variance(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha):
    #Compute distances between centroid and input MASs in the cluster
    nb_MAS1=m1.shape[0]
    nb_MAS2=m2.shape[0]

    C=distances_MAS_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha)
      
    idxs=np.argmin(C,axis=-1)
    non_zero=idxs[:,None]==np.arange(nb_MAS2)
    P=np.ones(nb_MAS1)[:,None]*non_zero #TODO:Maybe a better formulation?
    
    return np.sum(P*C,0)

def compute_size_cluster(P):
    #Compute the number of elements in the cluster
    return P.sum(0)


def update_P(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha):
    #Compute closest centroid for each MASs
    #the distance between two MASs is the sum of MW between MAPs with same index 
    nb_MAS1=m1.shape[0]
    nb_MAS2=m2.shape[0]
    
    C=distances_MAS_idx(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha)
        
    idxs=np.argmin(C,axis=-1)
    non_zero=idxs[:,None]==np.arange(nb_MAS2)
    return np.ones(nb_MAS1)[:,None]*non_zero # Maybe a better formulation?

def update_m(Pweight,m1):
    #Interpolation of means
    return (Pweight[:,:,None,None]*m1[:,None]).sum(0)

def update_wI_I(Pweight,wI1,I1):
    #Interpolation of weights and values of isotropic compartments
    wI2=(Pweight[:,:,None]*wI1[:,None]).sum(0)
    I2=(Pweight[:,:,None]*I1[:,None]).sum(0)
    return wI2,I2

def update_w_S(P,w1,S1):
    #Quantization of weights and covariance of anisotropic compartments
    ksize=S1.shape[2]
    nb_centroid=P.shape[1]
    nb_MAP=S1.shape[1]
    w2,S2=np.zeros((nb_centroid,nb_MAP,ksize)),np.zeros((nb_centroid,nb_MAP,ksize,3,3))
    
    for i in range(nb_centroid):
        print(i,end=' ',flush=True)
        idx_temp=np.where(P[:,i]==1)[0]
        w1_temp=w1[idx_temp,:].transpose(1,0,2)
        S1_temp=S1[idx_temp,:].transpose(1,0,2,3,4)

        kmax=np.max(np.count_nonzero(w1_temp,axis=-1))        
        ksize=S1_temp.shape[2]
        
        w1_temp=w1_temp.reshape(nb_MAP,-1)
        w1_temp/=idx_temp.shape
        S1_temp=S1_temp.reshape(nb_MAP,-1,3,3)

        w2[i,:,:kmax],S2[i,:,:kmax]=proj_GMM_MAS(kmax,ksize,w1_temp,S1_temp,max_itr=10,eps=1e-5)
    print()
    return w2,S2

def clustering_MASs(nb_centroid,m1,wI1,I1,w1,S1,max_itr=20,eps=1e-7,alpha=0.5):
    #Perform clustering via LLoyd algorithm
    #Alternate between update of cluster and assignment on clusters
    
    #TODO initialization via odering of weights (along compartments)
    m2,wI2,I2,w2,S2=m1[:nb_centroid],wI1[:nb_centroid],I1[:nb_centroid],w1[:nb_centroid],S1[:nb_centroid]
    
    for i in range(max_itr):
        print(i,'th iteration',flush=True)
        
        print("Update P",flush=True)
        P=update_P(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha)
        Pweight=np.divide(P,np.sum(P,0)[None],where=(np.sum(P,0)!=0)[None,:],out=np.zeros(P.shape))
        print("Update spatial",flush=True)
        m2_new=update_m(Pweight,m1)
        print("Update iso",flush=True)
        wI2_new,I2_new=update_wI_I(Pweight,wI1,I1)
        print("Update aniso",flush=True)
        w2_new,S2_new=update_w_S(P,w1,S1)
        
        diff=np.linalg.norm(m2_new-m2)+np.linalg.norm(wI2_new-wI2)+np.linalg.norm(I2_new-I2)+np.linalg.norm(w2_new-w2)+np.linalg.norm(S2_new-S2)
        print('diff',diff,flush=True)
        if diff <eps:
            m2=m2_new
            wI2=wI2_new
            I2=I2_new
            w2=w2_new
            S2 = S2_new
            break
        m2=m2_new
        wI2=wI2_new
        I2=I2_new
        w2=w2_new
        S2 = S2_new
    

    V=compute_variance(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,alpha)
    size_clusters=compute_size_cluster(P)
    return m2,wI2,I2,w2,S2,size_clusters,V



#################################
### Clustering of Streamlines ###
#################################
def compute_size_cluster_tract(P):
    #compute size of the clusters
    return P.sum(0)
    
def compute_variance_tract(m1,m2,alpha):
    #Compute distances between centroid and input MASs in the cluster
    nb_tract1=m1.shape[0]
    nb_tract2=m2.shape[0]
    nb_pts=m1.shape[1]
    C=np.sum((m1[:,None]-m2[None,:])**2,(2,3))
      
    idxs=np.argmin(C,axis=-1)
    non_zero=idxs[:,None]==np.arange(nb_tract2)
    P=np.ones(nb_tract1)[:,None]*non_zero
    
    return np.sum(P*C,0)
    
def update_P_tract(m1,m2,alpha):
    #Compute assignment to the closest centroid
    #the distance between two tracts is the sum of the spatial distance between points with same index  
    nb_tract1=m1.shape[0]
    nb_tract2=m2.shape[0]
    nb_pts=m1.shape[1]
    
    #C=np.sum((m1[:,None]-m2[None,:])**2,(2,3)) #all point of the tract
    C=np.sum((m1[:,None,:,-1]-m2[None,:,:,-1])**2,2) #Upper point of CST
    
    idxs=np.argmin(C,axis=-1)
    non_zero=idxs[:,None]==np.arange(nb_tract2)
    return np.ones(nb_tract1)[:,None]*non_zero # Maybe a better formulation?
    
def clustering_tract(nb_centroid,m1,max_itr=20,eps=1e-7,alpha=0.5):
    #Perform clustering with LLoyd algorithm 
    m2=m1[:nb_centroid]
    for i in range(max_itr):
        print(i,end=' ',flush=True)
        P=update_P_tract(m1,m2,alpha)
        Pweight=np.divide(P,np.sum(P,0)[None],where=(np.sum(P,0)!=0)[None,:],out=np.zeros(P.shape))
        m2_new=update_m(Pweight,m1)

        diff=np.linalg.norm(m2_new-m2)
        #print(i,':',diff)
        if diff <eps:
            m2=m2_new
            break
            
        m2=m2_new
    
    V=compute_variance_tract(m1,m2,alpha)
    size_cluster=compute_size_cluster_tract(P)
    return m2,size_cluster,V

