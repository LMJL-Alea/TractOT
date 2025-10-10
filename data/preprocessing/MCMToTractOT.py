import numpy as np
import pandas
import nrrd
import argparse
from dipy.io.streamline import load_tractogram

parser = argparse.ArgumentParser()
parser.add_argument('-t','--tract', type=str,required=True,help="selected tract")
parser.add_argument('-s','--subject', type=str,required=True,help="selected subject")
parser.add_argument('-p','--path', type=str,required=True,help="selected path")
args = parser.parse_args()
tract_name=args.tract
subject=args.subject
path=args.path

#tract_name='CST_left'
#subject=599469
#path='/home/gui/Documents/Post-doc/data/HCP_105/'

import sys
path2='/home/gui/Documents/Post-doc/code/TractOT/'
sys.path.append(path2+'script/')
from from_pandas_to_numpy import array_to_cov_neighbor
from from_numpy_to_pandas import numpy_to_pandas_MAT
from projection_GMM import proj_GMM_MAS

d=3

#Compute neigbor kernel
step=np.array([0,1,-1])
step_i, step_j, step_k = np.meshgrid(step, step, step, indexing='ij')
kernel = np.stack([step_i, step_j, step_k], axis=-1)[None]


#Load coordinate of mcm image
mcm_coordinate,header= nrrd.read(path+str(subject)+'/04-AlignedMCM/spatial.nrrd')
origin_img=header['space origin']
step_img=np.diag(header['space directions'][1:])

#Load weights of compartements and value of iso compartment
mcm_w,_=nrrd.read(path+str(subject)+'/04-AlignedMCM/MCM_avg_aligned/MCM_avg_aligned_weights.nrrd')    
mcm_I,_=nrrd.read(path+str(subject)+'/04-AlignedMCM/MCM_avg_aligned/MCM_avg_aligned_0.nrrd')
ksize=mcm_w.shape[0]-1 #shape of the array aka maximum number of compartment
    
#Load aniso compartment
mcm_cov=[]
for i in range(ksize):
    mcm_cov+=[nrrd.read(path+str(subject)+'/04-AlignedMCM/MCM_avg_aligned/MCM_avg_aligned_'+str(i+1)+'.nrrd')[0]]

    
#Load tract
tract = load_tractogram(
            path+str(subject)+'/06-AlignedTracts/'+str(tract_name)+'_clustered_newgrid_ordered_aligned.vtk', 
            reference=path+'000000/02-TransfoToMNISpace/averageDTI.nii.gz',
            bbox_valid_check=False,
            trk_header_check=False)
origin_tract=tract.space_attributes[0][:3,-1]
step_tract=np.diag(tract.space_attributes[0])[:3]

nb_streamline=len(tract.streamlines)
            
list_m,list_wI,list_I,list_w,list_S=[],[],[],[],[]
            
print(nb_streamline,end='->',flush=True)
for si in range(0,nb_streamline): #nb of streamlines
#for si in range(1,3):
    #if si%500==0:
    print(" ",si,end=' ',flush=True)
    streamline_coordinate=tract.streamlines[si] #Get coordinate of the streamline
    nb_pts=streamline_coordinate.shape[0]
    
    #Compute closest voxel from tract coordinate then compute its neigbord
    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
    if np.all((streamline_coordinate[0]*step_tract-mcm_coordinate[:,idx[0,0],idx[0,1],idx[0,2]]*step_img)>1.25):
        print("mauvais voxel",Flush=True)
    idx_neig=idx[:,None,None,None,:]+kernel
        
    #Compute distance to each neigbor
    voxel=mcm_coordinate[:,idx_neig[:,:,:,:,0],idx_neig[:,:,:,:,1],idx_neig[:,:,:,:,2]].transpose(1,0,2,3,4)
    D=np.sum((voxel-streamline_coordinate[:,:,None,None,None])**2,1)
    
    #Get weights of compartements and value of iso compartment 
    I=mcm_I[idx_neig[:,:,:,:,0],idx_neig[:,:,:,:,1],idx_neig[:,:,:,:,2]]
    wI=mcm_w[0,idx_neig[:,:,:,:,0],idx_neig[:,:,:,:,1],idx_neig[:,:,:,:,2]]
    w=mcm_w[1:,idx_neig[:,:,:,:,0],idx_neig[:,:,:,:,1],idx_neig[:,:,:,:,2]].transpose(1,2,3,4,0)
    kmax=np.max(np.count_nonzero(w,axis=-1),axis=(1,2,3)) #Number of non null compartment  
    
    #Compute weights of neigbor wrt to distance
    w_dist=1/D
    mask=np.ones((nb_pts,3,3,3))*(np.sum(w,-1)+wI>(1-1e-7)) #Mask for neighbor voxels out of the image    
    w_dist=w_dist*mask
    w_dist/=w_dist.sum((1,2,3))[:,None,None,None]
            
    # Interpolation of iso
    If=np.sum(I*w_dist,(1,2,3))
    wIf=np.sum(wI*w_dist,(1,2,3))
    
    #Get aniso compartment
    S=np.zeros((nb_pts,3,3,3,ksize,d,d))
    for i in range(0,ksize):
        S[:,:,:,:,i]=array_to_cov_neighbor(mcm_cov[i][:,idx_neig[:,:,:,:,0],idx_neig[:,:,:,:,1],idx_neig[:,:,:,:,2]].transpose(1,2,3,4,0))
    
    #Interpolation of aniso
    w=w*w_dist[:,:,:,:,None]
    wf,Sf=proj_GMM_MAS(kmax,ksize,w.reshape(nb_pts,-1),S.reshape(nb_pts,-1,d,d),itr=50,eps=1e-7)
                
    
    list_m+=[streamline_coordinate]
    list_wI+=[wIf]
    list_I+=[If]
    list_w+=[wf]
    list_S+=[Sf]
                
MAT=numpy_to_pandas_MAT(list_m,list_wI,list_I,list_w,list_S)            
MAT.to_csv(path+str(subject)+'/07-AugmentedTractsOT/'+str(tract_name)+'_avg_clustered_newgrid_ordered_aligned.csv',index=False)   
print('',flush=True)

