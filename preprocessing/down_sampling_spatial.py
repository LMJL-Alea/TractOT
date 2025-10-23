# Reduce the number of streamline
# By downsampling

import numpy as np
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram

import sys
sys.path.append('../script/')
from down_sampling import down_sampling_tract
from from_vtk_to_numpy import vtk_to_numpy_tract_array

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-r','--reference_path',type=str,required=True,help="reference image or .trk")
parser.add_argument('-c','--nb_step', type=int,required=True,help="number of steps of the grid")

args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
nb_step=args.nb_step

tract = load_tractogram(
            input_path,
            reference=reference)

nb_streamline=len(tract.streamlines)
m1,_=vtk_to_numpy_tract_array(tract,nb_streamline=nb_streamline)

m2=down_sampling_tract(nb_step,m1,upper=-1)
tract.streamlines=m2
print("Obtained tracto has ",m2.shape[0],' streamlines',flush=True)
save_tractogram(tract,output_path) 

