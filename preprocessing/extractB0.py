#Concatenate DWI

import argparse
import os
import numpy as np
from dipy.io.image import load_nifti,save_nifti
from dipy.io.gradients import read_bvals_bvecs

parser = argparse.ArgumentParser()
parser.add_argument('-i','--path', type=str,required=True,help="path")
parser.add_argument('-m','--image', type=str,required=True,help="image name")

args = parser.parse_args()
path=args.path
image=args.image

dwi,header= load_nifti(path+'1eddy_'+image+'.nii.gz')
header[-1,-1]=4.3
b0=dwi[:,:,:,0:2]
b0[:,:,:,-1]=dwi[:,:,:,-1] #b0 is first and last 
save_nifti(path+'B0_eddy_'+image+'.nii.gz',b0,header)
