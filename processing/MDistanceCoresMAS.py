import numpy as np
from dipy.io.streamline import load_tractogram
import argparse

import sys
sys.path.append('../script/')
from distances import distance_cores_svm
from from_trx_to_numpy import trx_to_numpy_tract_list

parser = argparse.ArgumentParser()
parser.add_argument('-i','--path', type=str,required=True,help="path of the dataset")
parser.add_argument('-o','--output_path', type=str,required=True,help="output path")
parser.add_argument('-L','--numbers_cores',type=int,required=True,help="number of cores MAS")
parser.add_argument('-T','--name_tract',type=str,required=True,help="name of the tract")
parser.add_argument('-d','--dataset', type=str,required=True,help='data set of the subject')
parser.add_argument('-s','--subjects', nargs='+', type=str,required=True,help='name of the subject')
parser.add_argument('-a','--alpha', type=float,required=True,help='alpha parameter, trade off spatial/diffusion')


args = parser.parse_args()
path=args.path
output_path=args.output_path
nb_MAS=args.numbers_cores
TRACT=args.name_tract
dataset=args.dataset
SUBJECTS=args.subjects
alpha=args.alpha

M=np.zeros((len(SUBJECTS),len(SUBJECTS)))

for i,s1 in enumerate(SUBJECTS):
    print(s1,end=': ',flush=True)
    cores1 = load_tractogram(path+"/"+s1+'/06-AugmentedTractsOT/7cores_MAS_'+TRACT+'_L'+str(nb_MAS)+'.trx', reference='same')
    m1,wI1,I1,w1,S1,labels_svm1,_=trx_to_numpy_tract_list(cores1,nb_MAS)
    for j,s2 in enumerate(SUBJECTS):
        if j>i:
            cores2 = load_tractogram(path+"/"+s2+'/06-AugmentedTractsOT/7cores_MAS_'+TRACT+'_L'+str(nb_MAS)+'.trx', reference='same')
            m2,wI2,I2,w2,S2,labels_svm2,_=trx_to_numpy_tract_list(cores2,nb_MAS)  
            cost,_,_,C_nb_pts=distance_cores_svm(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,labels_svm1,labels_svm2,alpha=alpha)
            M[i,j]=cost
            #print(C_nb_pts.min())
    print()
M+=M.T

np.savetxt(output_path+'/'+dataset+"_"+TRACT+'_alpha'+str(alpha)+'_L'+str(nb_MAS)+'.txt', M)

