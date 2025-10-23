# Reorder streamlines given an reference streamline
# Indice are set to bet 0-> bottom of the brain, -1 -> top of the brain

import numpy as np
from dipy.io.streamline import load_tractogram,save_tractogram
import argparse

import sys
sys.path.append('../script/')
from new_grid import newgrid_streamline

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-r1','--reference_path',type=str,required=True,help="reference image or .trk")
parser.add_argument('-r2','--reference_path_streamline',type=str,required=True,help="reference streamline")


args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
reference_path_streamline=args.reference_path_streamline

nb_pts=10

ref_tract=load_tractogram(reference_path_streamline,
                         reference=reference)
                         
ref_streamline=ref_tract.streamlines[0]
m1 = newgrid_streamline(ref_streamline, nb_pts=nb_pts)

tract = load_tractogram(
            input_path,
            reference=reference)

streamlines=tract.streamlines
for i in range(len(streamlines)):
    mi=newgrid_streamline(streamlines[i],nb_pts=nb_pts)

    d1=np.linalg.norm(m1-mi)
    d2=np.linalg.norm(m1-mi[::-1])
    if d2<d1:
        tract.streamlines[i]=streamlines[i][::-1]

save_tractogram(tract,output_path) 
print('',flush=True)
