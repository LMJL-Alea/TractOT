import numpy as np

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

