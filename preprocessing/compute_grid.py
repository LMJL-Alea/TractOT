#Uniformise the number of points on every streamlines

import numpy as np
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram

import sys
sys.path.append('../script/')
from new_grid import newgrid_streamline

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-r','--reference_path',type=str,required=True,help="reference image or .trk")
parser.add_argument('-n','--nb_pts', type=int,required=True,help="number of points on each streamlines")

args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
nb_pts=args.nb_pts

tract = load_tractogram(
            input_path,
            reference=reference)

nb_streamline=len(tract.streamlines)

m=[]
for i in range(nb_streamline):
    m+=[newgrid_streamline(tract.streamlines[i],nb_pts=nb_pts)]

tract.streamlines=m

save_tractogram(tract,output_path) 

