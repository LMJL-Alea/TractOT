#Concatenate DWI

import argparse
import os
import numpy as np
from dipy.io.image import load_nifti,save_nifti
from dipy.io.gradients import read_bvals_bvecs

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_paths_images', nargs='+', type=str,required=True,help='input path for the images')
parser.add_argument('-b','--input_paths_bvals', nargs='+', type=str,required=True,help='input path for the bvals')
parser.add_argument('-g','--input_paths_bvecs', nargs='+', type=str,required=True,help='input path for the bvecs')

parser.add_argument('-o','--output_paths_images', type=str,required=True,help='output path for the images')
parser.add_argument('-c','--output_paths_bvals', type=str,required=True,help='output path for the bvals')
parser.add_argument('-f','--output_paths_bvecs', type=str,required=True,help='output path for the bvecs')

args = parser.parse_args()
imgs=args.input_paths_images
bvals=args.input_paths_bvals
bvecs=args.input_paths_bvecs

out_imgs=args.output_paths_images
out_bvals=args.output_paths_bvals
out_bvecs=args.output_paths_bvecs


#print(imgs,flush=True)
#img_dwi=['DTI_B700_64dir_PA_','DTI_B1000_64dir_PA_','DTI_B2000_64dir_PA_']

for i in range(len(imgs)):
    if i==0:
        dwi,header= load_nifti(imgs[i])
        bval,bvec=bval,bvec=read_bvals_bvecs(bvals[i],bvecs[i])
    else:
        dwi_i,header_i=load_nifti(imgs[i])
        bval_i,bvec_i=read_bvals_bvecs(bvals[i],bvecs[i])

        dwi=np.concatenate((dwi,dwi_i),axis=-1)
        bval=np.concatenate((bval,bval_i),axis=-1)
        bvec=np.concatenate((bvec,bvec_i),axis=0)

save_nifti(out_imgs,dwi,header)
np.savetxt(out_bvals, bval.astype(int),fmt='%i', newline=" ")
np.savetxt(out_bvecs, bvec.T, newline="\n", delimiter="  ")
