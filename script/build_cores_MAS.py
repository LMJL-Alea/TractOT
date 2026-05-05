import numpy as np
from sklearn.cluster import KMeans
import pickle

from projection_GMM import proj_GMM_MAS
from utils import cov_to_array_MAS

def average_MAS_per_labels(n_clusters,m,wI,I,w,S,conf,labels_svm,labels_clus):
    
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
