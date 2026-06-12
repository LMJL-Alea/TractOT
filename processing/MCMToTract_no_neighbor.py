#Add MCM without lifting of the neighbor voxels
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
from utils import array_to_cov_neighbor,cov_to_array_MAS,most_colinear_compartment

d=3


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
            
list_m,list_wI,list_I,list_w,list_S,list_col=[],[],[],[],[],[]
            
print(nb_streamline,end='->',flush=True)
for si in range(0,nb_streamline): #nb of streamlines
#for si in range(1,3):
    #if si%500==0:
    print(" ",si,end=' ',flush=True)
    streamline_coordinate=tract.streamlines[si] #Get coordinate of the streamline
    nb_pts=streamline_coordinate.shape[0]
    
    #Compute closest voxel from tract coordinate then compute its neigbord
    idx=np.int32(np.round((streamline_coordinate-origin_tract)/step_tract))
    #if np.all((streamline_coordinate[0]*step_tract-mcm_coordinate[:,idx[0,0],idx[0,1],idx[0,2]]*step_img)>1.25):
    #    print("mauvais voxel",Flush=True)
        
    
    #Get weights of compartements and value of iso compartment 
    If=mcm_I[idx[:,0],idx[:,1],idx[:,2]]
    wIf=mcm_w[0,idx[:,0],idx[:,1],idx[:,2]]
    wf=mcm_w[1:,idx[:,0],idx[:,1],idx[:,2]].transpose(1,0)
    

    
    #Get aniso compartment
    Sf=np.zeros((nb_pts,ksize,d,d))
    for i in range(0,ksize):
        Sf[:,i]=array_to_cov(mcm_cov[i][:,idx[:,0],idx[:,1],idx[:,2]].transpose(1,0))[:,0]

    #Compute most colinear compartment
    idx_col=most_colinear_compartment(streamline_coordinate,Sf)
    
    list_m+=[streamline_coordinate]
    list_wI+=[wIf]
    list_I+=[If]
    list_w+=[wf]
    list_S+=[Sf]
    list_col+=[idx_col]
                
tract.streamlines=list_m
dico={}
dico['weight iso']=list_wI
dico['iso']=list_I
dico['weights aniso']=list_w
dico['aniso']=list_S
dico['colinear']=list_col
tract.data_per_point=dico
save_tractogram(tract,output_path_tractogram)
print('',flush=True)
