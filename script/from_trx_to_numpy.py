import numpy as np
from utils import array_to_cov_MAS

def trx_to_numpy_tract_array(tract,nb_streamline=1,streamlineId=None,d=3):
    #Transform a .trx file of augmented tract into numpy arrays
    #Every streamline must have the same nb of points
    
    nb_pts=tract.streamlines[0].shape[0]
    k=np.int32(tract.data_per_point["aniso"][0].shape[1]/6)

    m=np.zeros((nb_streamline,nb_pts,d))
    wI=np.zeros((nb_streamline,nb_pts))
    I=np.zeros((nb_streamline,nb_pts))
    w=np.zeros((nb_streamline,nb_pts,k))
    S=np.zeros((nb_streamline,nb_pts,k,d,d))
    conf=np.zeros((nb_streamline,nb_pts))
    
    for i in range(0,nb_streamline):
        if streamlineId is None:
            m[i] = tract.streamlines[i]
            wI[i] = tract.data_per_point["weight iso"][i].reshape(-1)
            I[i] = tract.data_per_point["iso"][i].reshape(-1)
            w[i] = tract.data_per_point["weights aniso"][i]
            S[i] = array_to_cov_MAS(tract.data_per_point["aniso"][i])
            conf[i] = tract.data_per_point["confidence"][i].reshape(-1)
        else:
            m[i] = tract.streamlines[streamlineId[i]]            
            wI[i] = tract.data_per_point["weight iso"][streamlineId[i]].reshape(-1)
            I[i] = tract.data_per_point["iso"][streamlineId[i]].reshape(-1)
            w[i] = tract.data_per_point["weights aniso"][streamlineId[i]]
            S[i] = array_to_cov_MAS(tract.data_per_point["aniso"][streamlineId[i]])
            conf[i] = tract.data_per_point["confidence"][streamlineId[i]].reshape(-1)
    return m,wI,I,w,S,conf,nb_pts 


def trx_to_numpy_tract_list(tract,nb_streamline=1,d=3):
    #Transform a .trx file of augmented tract into list of numpy arrays
    #Every streamline can have different number of point
    
    nb_MAS=len(tract.streamlines)
    
    m,wI,I,w,S,labels_svm,nb_pts=[],[],[],[],[],[],[]
    
    for i in range(nb_MAS):
        m+=[tract.streamlines[i]]
        wI+=[tract.data_per_point["weight iso"][i].reshape(-1)]
        I+=[tract.data_per_point["iso"][i].reshape(-1)]
        w+=[tract.data_per_point["weights aniso"][i]]
        S+=[array_to_cov_MAS(tract.data_per_point["aniso"][i])]
        labels_svm+=[tract.data_per_point["svm"][i].reshape(-1)]
        nb_pts+=[tract.streamlines[i].shape[0]]
        
    return m,wI,I,w,S,labels_svm,nb_pts 

