# Load vtk aligned file in numpy
# Reoder them 
# Save them in .csv with pandas

import numpy as np
import pandas
from dipy.io.streamline import load_tractogram,save_tractogram
import argparse

import sys
sys.path.append('../../script/')
from new_grid import newgrid_streamline
from from_numpy_to_pandas import numpy_to_pandas_tract
parser = argparse.ArgumentParser()
parser.add_argument('-t','--tract', type=str,required=True,help="selected tract")
parser.add_argument('-s','--subject', type=str,required=True,help="selected subject")
parser.add_argument('-p','--path', type=str,required=True,help="selected path")
args = parser.parse_args()
tract_name=args.tract
subject=args.subject
path=args.path
#tract_name='CST_left'
#subject=599469




nb_pts=10

ref_tract=load_tractogram(path+'000000/06-AlignedTracts/'+str(tract_name)+'_ref_aligned.vtk',
                         reference=path+'000000/02-TransfoToMNISpace/averageDTI.nii.gz')
ref_streamline=ref_tract.streamlines[0]
m1 = newgrid_streamline(ref_streamline, nb_pts=nb_pts)

tract = load_tractogram(
            path+str(subject)+'/06-AlignedTracts/'+str(tract_name)+'_aligned.vtk',
            reference=path+'000000/02-TransfoToMNISpace/averageDTI.nii.gz')

streamlines=tract.streamlines
for i in range(len(streamlines)):
    mi=newgrid_streamline(streamlines[i],nb_pts=nb_pts)

    d1=np.linalg.norm(m1-mi)
    d2=np.linalg.norm(m1-mi[::-1])
    if d2<d1:
        tract.streamlines[i]=streamlines[i][::-1]

save_tractogram(tract,path+str(subject)+'/06-AlignedTracts/'+str(tract_name)+'_ordered_aligned.vtk',bbox_valid_check=False) 
