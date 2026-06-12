#Add MCM with an average over neigbhor voxels
# average in the sense of optimal tranpost metric
import numpy as np
import nrrd
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.io.utils import Space

parser = argparse.ArgumentParser()
parser.add_argument('-t','--tractogram_path', type=str,required=True,help="path of input tractogram")
parser.add_argument('-o','--output_path_tractogram',type=str,required=True,help="output path of obtained augmented tractogram in .trx")
parser.add_argument('-c','--mcm_path',type=str,required=True,help="path of the mcm images")
parser.add_argument('-w','--weights_path',type=str,required=True,help="weights image")
parser.add_argument('-s','--coordinate_path',type=str,required=True,help="image of coordinate")
parser.add_argument('-r','--reference_path',type=str,required=False,help="reference image or .trk")

args = parser.parse_args()
tractogram_path=args.tractogram_path
output_path_tractogram=args.output_path_tractogram
mcm_path=args.mcm_path
weights_path=args.weights_path
coordinate_path=args.coordinate_path

if args.reference_path:
    reference_path=args.reference_path
else:
    reference='same'


import sys
sys.path.append('../script/')
from utils import array_to_cov_neighbor, cov_to_array_MAS,most_colinear_compartment
from projection_GMM import proj_GMM_MAS
from distances import distance_confidence_along

d=3

#Compute neigbor kernel
step=np.array([0,1,-1])
step_i, step_j, step_k = np.meshgrid(step, step, step, indexing='ij')
kernel_neighbor = np.stack([step_i, step_j, step_k], axis=-1)[None]


#Load coordinate of mcm image
mcm_coordinate,header= nrrd.read(coordinate_path)
origin_img=header['space origin']
step_img=np.diag(header['space directions'][1:])

#Load weights of compartements and value of iso compartment
mcm_w,_=nrrd.read(weights_path)    
mcm_I,_=nrrd.read(mcm_path+'0.nrrd')
ksize=mcm_w.shape[0]-1 #shape of the array aka maximum number of compartment
    
#Load aniso compartment
mcm_cov=[]
for i in range(ksize):
    mcm_cov+=[nrrd.read(mcm_path+str(i+1)+'.nrrd')[0]]

    
#Load tract
tract = load_tractogram(
            tractogram_path, 
            reference=reference_path,#to_space=Space.LPSMM)
            bbox_valid_check=False,
            trk_header_check=False)
origin_tract=tract.space_attributes[0][:3,-1]
step_tract=np.diag(tract.space_attributes[0])[:3]

nb_streamline=len(tract.streamlines)
            
list_m,list_wI,list_I,list_w,list_S,list_conf,list_col=[],[],[],[],[],[],[]
            
print(nb_streamline,end='->',flush=True)
for si in range(0,nb_streamline): #nb of streamlines
    if si%10==0:
        print(" ",si,end=' ',flush=True)
    streamline_coordinate=tract.streamlines[si] #Get coordinate of the streamline
    nb_pts=streamline_coordinate.shape[0]
    
    #Compute closest voxel from tract coordinate then compute its neigbord
    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
    #if np.all((streamline_coordinate[0]*step_tract-mcm_coordinate[:,idx[0,0],idx[0,1],idx[0,2]]*step_img)>1.25):
    #    print("mauvais voxel",Flush=True)
    idx_neighbor=idx[:,None,None,None,:]+kernel_neighbor
        
    #Compute distance to each neigbor
    voxels_neighbor=mcm_coordinate[:,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]].transpose(1,0,2,3,4)
    D=np.sum((voxels_neighbor-streamline_coordinate[:,:,None,None,None])**2,1)
    
    #Get weights of compartements and value of iso compartment 
    I=mcm_I[idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]]
    wI=mcm_w[0,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]]
    w=mcm_w[1:,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]].transpose(1,2,3,4,0)
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
        S[:,:,:,:,i]=array_to_cov_neighbor(mcm_cov[i][:,idx_neighbor[:,:,:,:,0],idx_neighbor[:,:,:,:,1],idx_neighbor[:,:,:,:,2]].transpose(1,2,3,4,0))
    
    #Interpolation of aniso
    w=w*w_dist[:,:,:,:,None]
    wf,Sf=proj_GMM_MAS(kmax,ksize,w.reshape(nb_pts,-1),S.reshape(nb_pts,-1,d,d),max_itr=50,eps=1e-7)
    
    #Compute distance target source
    D=distance_confidence_along(w.reshape(nb_pts,-1),wf,S.reshape(nb_pts,-1,d,d),Sf)
    conf=1/(D+1e-8)

    #Compute most colinear compartment
    idx_col=most_colinear_compartment(streamline_coordinate,Sf)         
    
    list_m+=[streamline_coordinate]
    list_wI+=[wIf]
    list_I+=[If]
    list_w+=[wf]
    list_S+=[cov_to_array_MAS(Sf)]
    list_conf+=[conf]
    list_col+=[idx_col]
 
#Normalize confidence 
max_confidence=0
for i in range(len(list_conf)):
    if max(list_conf[i])>max_confidence:
        max_confidence=max(list_conf[i])
for i in range(len(list_conf)):
    list_conf[i]/=max_confidence 
 
tract.streamlines=list_m
dico={}
dico['weight iso']=list_wI
dico['iso']=list_I
dico['weights aniso']=list_w
dico['aniso']=list_S
dico['confidence']=list_conf
dico['colinear']=list_col
tract.data_per_point=dico
save_tractogram(tract,output_path_tractogram)
print('',flush=True)
