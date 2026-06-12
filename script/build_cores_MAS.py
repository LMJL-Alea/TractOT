import numpy as np
from sklearn.cluster import KMeans
import pickle

from projection_GMM import proj_GMM_MAS
from utils import cov_to_array_MAS


def average_MAS_per_labels(n_clusters,m,wI,I,w,S,conf,labels_svm,labels_clus):       
    """
    Compute cores MAS 

    This function compute cores MAS by averaging of MAS, given clusters  

    Parameters
    ----------
    n_clusters : positive integer
        Number of target cores MAS
    m : numpy array 
        Euclidean coordinate of the MAS. 
        Is shape is ,nb_MASxnb_MAPx3
    wI : numpy array
        An array of positive weights for the isotropic comparment. 
        Is shape is nb_MASxnb_MAP
    I: numpy array
        Array of the eigenvalues for the isotropic covariate
        Is shape is nb_MASxnb_MAP
    w: numpy array
        An array of positive weights for the anisotropic comparment.
        Is shape is nb_MASxnb_MAPxnb_compartement
    S: positive scalar
        An array of the covariates matrices for the anisotropic part
        Is shape is nb_MASxnb_MAPxnb_compartmentxdxd
    conf: numpy array
        Confidence score for the points of the MAS
        Is shape is nb_MASxnb_MAP
    labels_svm: numpy array
        Label of each MAP given by an SVM 
        Is shape is nb_MASxnb_MAP
    labels_clus: numpy array
        Label of each MAS
        Is shape is nb_MAS
        
    Returns
    Note returns are list since cores MAS might have different number of MAP
    -------
    cores_m : list of numpy array
        A list of array for the Euclidean coordinate of the cores MAS
        List has length n_clusters
    cores_wI: list of numpy array
        A list of array for weights of the isotropic compartment of the cores MAS
        List has length n_clusters
    cores_I: list of numpy array
        A list of array for eigenvalue of the isotropic compartment of the cores MAS
        List has length n_clusters
    cores_w: list of numpy array
        A list of array for weights of the anisotropic compartment of the cores MAS
        List has length n_clusters
    cores_S: list of numpy array
        A list of array for the anisotropic covariates the cores MAS
        List has length n_clusters
    idx_svm_l: list of numpy array
        A list with svm labels of each point of the cores MAS 
    idx_cluster: list of numpy array
        A list with cluster labels of each point of the cores MAS
        Note that each MAS as the label along its MAP
    size_cluster: list of numpy array
        A list containg the number of MAP that have been averaging to obtain each MAP of the cores MAS
    """
    nb_pts=m.shape[1]
    K_size=w.shape[-1]

    cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm_l,idx_cluster,size_cluster=[],[],[],[],[],[],[],[]
    
    for i in range(n_clusters):
        print('l = ',i,flush=True)
        
        nb_pts_i=np.unique(labels_svm[labels_clus==i]).shape[0]
        idx_cluster_i=i*np.ones(nb_pts_i)
        
        cores_m_i=np.zeros((nb_pts_i,3))
        cores_wI_i=np.zeros(nb_pts_i)
        cores_I_i=np.zeros(nb_pts_i)
        cores_w_i=np.zeros((nb_pts_i,K_size))
        cores_S_i=np.zeros((nb_pts_i,K_size,3,3))
        idx_svm_i=np.zeros(nb_pts_i)
        size_cluster_i=np.zeros(nb_pts_i)
        
        for j,idx_svm in enumerate(np.unique(labels_svm[labels_clus==i])):
            print(idx_svm,end=' ',flush=True)
            idx_svm_i[j]=idx_svm
            #print(size_cluster,end=' ')
            
            size_cluster_i[j]=(w[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]).shape[0]
            
            k2=np.max(np.count_nonzero(w[labels_clus==i][labels_svm[labels_clus==i]==idx_svm],axis=1))
            #print('k2 ',k2)
            
            #Normalization of the score
            conf_idx_svm_clus=conf[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]
            
            #spatial
            cores_m_i[j]=np.mean(m[labels_clus==i][labels_svm[labels_clus==i]==idx_svm],axis=0)
            
            #iso
            cores_wI_i[j]=np.mean(wI[labels_clus==i][labels_svm[labels_clus==i]==idx_svm])
            cores_I_i[j]=np.mean(I[labels_clus==i][labels_svm[labels_clus==i]==idx_svm])
    
            #Aniso #TODO add scores
            cores_w_i[j][:k2],cores_S_i[j][:k2]=proj_GMM_MAS(k2,k2,
                                                   (w[labels_clus==i][labels_svm[labels_clus==i]==idx_svm]*conf_idx_svm_clus[:,None]).reshape(1,-1),
                                                   S[labels_clus==i][labels_svm[labels_clus==i]==idx_svm].reshape(1,-1,3,3),
                                                   max_itr=30)                           
            
            cores_w_i[j]=(cores_w_i[j]/np.sum(cores_w_i[j]))-cores_wI_i[j]/K_size
            
        print(flush=True)    
        cores_m+=[cores_m_i]
        cores_wI+=[cores_wI_i]
        cores_I+=[cores_I_i]
        cores_w+=[cores_w_i]
        cores_S+=[cov_to_array_MAS(cores_S_i)]
        idx_cluster+=[idx_cluster_i]
        idx_svm_l+=[idx_svm_i]
        size_cluster+=[size_cluster_i]
        
    return cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm_l,idx_cluster,size_cluster



def infer_labels_svm_endpoint(path_svm,n_clusters,m,random_state=0):
    
    nb_streamline,nb_pts=m.shape[0],m.shape[1]
    
    with open(path_svm, 'rb') as file:
        svc = pickle.load(file)
    labels_svm=svc.predict(X=m.reshape(-1,3)).reshape(nb_streamline,nb_pts)
    
    m_endpoint=m[labels_svm==nb_pts]
    m_endpoint=m[:,-1][labels_svm[:,-1]==nb_pts]
    #print(m_endpoint.shape)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(m_endpoint)

    m_centroid_endpoint=kmeans.cluster_centers_
    labels_clus=kmeans.predict(m[:,-1,:])
    return labels_svm,labels_clus
