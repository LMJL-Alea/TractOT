import numpy as np
import pandas

def numpy_to_pandas_streamlines(m,idx_streamline=1):
    streamline=pandas.DataFrame({'X': m[:, 0], 
                  'Y': m[:, 1],
                  'Z': m[:,2],
                  'PointId':np.arange(1,len(m)+1),
                  "StreamlineId":idx_streamline*np.ones(len(m),dtype=np.int8)})   
    return streamline
                 
def numpy_to_pandas_tract(list_m):
    tract=numpy_to_pandas_streamlines(list_m[0])
    for i in range(1,len(list_m)):
        tract=pandas.concat([tract,numpy_to_pandas_streamlines(list_m[i],idx_streamline=i+1)],ignore_index=True)
    return tract
    
def numpy_to_pandas_tract_interpolate(m,nb_streamline,nb_pts):
    tract=pandas.DataFrame({'X': m[:, 0], 
                  'Y': m[:, 1],
                  'Z':m[:,2],
                  'PointId':np.repeat(np.arange(1,nb_pts+1)[None],nb_streamline,0).reshape(-1),
                  "StreamlineId":np.repeat(np.arange(1,nb_streamline+1),nb_pts)})
    return tract

def numpy_to_pandas_MAS(m,wI,I,w,S,idx_MAS=1):
    ksize=w.shape[-1]
    MAS=pandas.DataFrame({'X': m[:, 0], 
                  'Y': m[:, 1],
                  'Z': m[:,2],
                  'PointId':np.arange(1,len(m)+1),
                  "StreamlineId":idx_MAS*np.ones(len(m),dtype=np.int8),
                  "MostColinearIndex":np.zeros(len(m)),
                  "FreeWaterWeight":wI})

    dico={}
    for i in range(ksize):
        dico['Tensor'+str(i+1)+'Weight']=w[:,i]
    dico['FreeWaterParameter1']=I

    l=cov_to_array_MAS(S,d=3)
    for i in range(ksize):
        for j in range(0,6):
            dico['Tensor'+str(i+1)+'Parameter'+str(j+1)]=l[:,6*i+j]
    return MAS.assign(**dico)



def numpy_to_pandas_MAT(list_m,list_wI,list_I,list_w,list_S):
    MAT=numpy_to_pandas_MAS(list_m[0],list_wI[0],list_I[0],list_w[0],list_S[0])
                  
    for i in range(1,len(list_m)):
        MAT=pandas.concat([MAT,numpy_to_pandas_MAS(list_m[i],list_wI[i],list_I[i],list_w[i],list_S[i],idx_MAS=i+1)],ignore_index=True)
    return MAT    
    
    
def numpy_to_pandas_MAT_interpolate(m,wI,I,w,S,nb_MAS,nb_MAP):
    ksize=w1.shape[-1]
    MAT=pandas.DataFrame({'X': m[:,:, 0].reshape(-1), 
                  'Y': m[:,:, 1].reshape(-1),
                  'Z':m[:,:,2].reshape(-1),
                  'PointId':np.repeat(np.arange(1,nb_MAP+1)[None],nb_MAS,0).reshape(-1),
                  "StreamlineId":np.repeat(np.arange(1,nb_MAS+1),nb_MAP),
                  "MostColinearIndex":np.zeros(nb_MAS*nb_MAP),
                  "FreeWaterWeight":wI.reshape(-1)})
    dico={}
    for i in range(ksize):
        dico['Tensor'+str(i+1)+'Weight']=w[:,:,i].reshape(-1)
    dico['FreeWaterParameter1']=I.reshape(-1)

    l=cov_to_array_MAT(S)
    for i in range(ksize):
        for j in range(0,6):
            dico['Tensor'+str(i+1)+'Parameter'+str(j+1)]=l[:,6*i+j]
    return MAT.assign(**dico)


def cov_to_array_MAS(S,d=3):
    #Convert covariance matrix into array of 6 elements
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
    
    
    

