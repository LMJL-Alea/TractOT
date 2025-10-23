import numpy as np

#Given a tract and grid
#Reduce the number of streamline by averaging the streamlines which ended in the same cube 
# Particularly efficient for the CST tract as it ensures to capture the variability of streamlines 
#Note that we can't control the exact size of m_new

def down_sampling_tract(nb_step,m1,upper=-1):
    #nb cube is nb_step^3
    
    miin=np.amin(m1[:,upper],0)
    maax=np.amax(m1[:,upper],0)
    idx_along_dim=np.int32(np.floor((m1[:,upper]-miin[None])/(maax[None]-miin[None])*nb_step))
    unique,idx=np.unique(idx_along_dim,axis=0,return_inverse=True)
    m2=np.zeros((unique.shape[0],m1.shape[1],m1.shape[2]))
    for i in range(unique.shape[0]):
        idx_assignment=np.where(idx==i)[0]
        m2[i]=np.mean(m1[idx_assignment],0)
    return m2
