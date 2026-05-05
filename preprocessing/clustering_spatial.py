# Reduce the number of streamline
# By clustering

import numpy as np
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram

import sys
sys.path.append('../script/')
from clustering import clustering_tract
from from_vtk_to_numpy import vtk_to_numpy_tract_array

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-s','--nb_centroid', type=int,required=True,help="number of centroid")
parser.add_argument('-r','--reference_path',type=str,required=False,help="reference image or .trk")

args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
nb_centroid=args.nb_centroid

tract = load_tractogram(
            input_path,
            reference=reference)

nb_streamline=len(tract.streamlines)
m1,_=vtk_to_numpy_tract_array(tract,nb_streamline=nb_streamline)

m2,_,_=clustering_tract(nb_centroid,m1,max_itr=1000,eps=1e-7,alpha=0.5)
tract.streamlines=m2

save_tractogram(tract,output_path) 

