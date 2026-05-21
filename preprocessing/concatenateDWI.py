#Concatenate DWI

import argparse
import os
import numpy as np
from dipy.io.image import load_nifti,save_nifti
from dipy.io.gradients import read_bvals_bvecs

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-s','--subject', type=str,required=True,help="subject")

args = parser.parse_args()
path=args.input_path
subject=args.subject


img_dwi=['DTI_B700_64dir_PA_','DTI_B1000_64dir_PA_','DTI_B2000_64dir_PA_']

for i,img in enumerate(img_dwi):
    if i==0:
        dwi,header= load_nifti(path+'2EPI_'+img+subject+'.nii.gz')
        bval,bvec=bval,bvec=read_bvals_bvecs(path+img+subject+'.bval',path+img+subject+'.bvec')
    else:
        dwi_i,header_i= load_nifti(path+'2EPI_'+img+subject+'.nii.gz')
        bval_i,bvec_i=read_bvals_bvecs(path+img+subject+'.bval',path+img+subject+'.bvec')

        dwi=np.concatenate((dwi,dwi_i),axis=-1)
        bval=np.concatenate((bval,bval_i),axis=-1)
        bvec=np.concatenate((bvec,bvec_i),axis=0)

save_nifti(path+'3concatenate_DTI_PA_'+subject+'.nii.gz',dwi,header)
np.savetxt(path+'3concatenate_DTI_PA_'+subject+".bval", bval.astype(int),fmt='%i', newline=" ")
np.savetxt(path+'3concatenate_DTI_PA_'+subject+".bvec", bvec.T, newline="\n", delimiter="  ")
