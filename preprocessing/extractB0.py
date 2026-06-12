#Extract B0

import argparse
import os
import numpy as np
from dipy.io.image import load_nifti,save_nifti
from dipy.io.gradients import read_bvals_bvecs

parser = argparse.ArgumentParser()
parser.add_argument('-i','--image_input', type=str,required=True,help="input dwi image")
parser.add_argument('-o','--b0_output', type=str,required=True,help="output B0, first and last slice")

args = parser.parse_args()
image_input=args.image_input
b0_output=args.b0_output

dwi,header= load_nifti(image_input)
#header[-1,-1]=4.3
b0=dwi[:,:,:,0:2]
b0[:,:,:,-1]=dwi[:,:,:,-1] #b0 is first and last 
save_nifti(b0_output,b0,header)
