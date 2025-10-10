import numpy as np
import pandas
import argparse

import sys
sys.path.append('../../script/')
from clustering import clustering_tract
from from_trk_to_numpy import trk_to_numpy_tract_interpolate
from dipy.io.streamline import load_tractogram,save_tractogram

parser = argparse.ArgumentParser()
parser.add_argument('-t','--tract', type=str,required=True,help="selected tract")
parser.add_argument('-s','--subject', type=str,required=True,help="selected subject")
parser.add_argument('-p','--path', type=str,required=True,help="selected path")
parser.add_argument('-n','--nb_centroid', type=int,required=True,help="number of centroid")
args = parser.parse_args()
tract_name=args.tract
subject=args.subject
path=args.path
nb_centroid=args.nb_centroid

#tract='CST_left'
#subject=599469
#path=/home/gui/Documents/Post-doc/data/HCP_105/
#nb_centroid=100

nb_pts=40

tract = load_tractogram(
            path+str(subject)+'/06-AlignedTracts/'+str(tract_name)+'_ordered_aligned.vtk',
            reference=path+'000000/02-TransfoToMNISpace/averageDTI.nii.gz')

nb_streamline=len(tract.streamlines)
m1,nb_pts=trk_to_numpy_tract_interpolate(tract,nb_streamline=nb_streamline,nb_pts=nb_pts)

m2,_,_=clustering_tract(nb_centroid,m1,itr=1000,eps=1e-7,alpha=0.5)
tract.streamlines=m2

save_tractogram(tract,path+str(subject)+'/06-AlignedTracts/'+str(tract_name)+'_clustered_newgrid_ordered_aligned2.vtk',bbox_valid_check=False) 
