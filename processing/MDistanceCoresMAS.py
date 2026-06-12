import numpy as np
from dipy.io.streamline import load_tractogram
import argparse
import pandas as pd
import os.path

import sys
sys.path.append('../script/')
from distances import distance_cores_svm
from from_trx_to_numpy import trx_to_numpy_tract_list


parser = argparse.ArgumentParser()
parser.add_argument('-p','--path', type=str,required=True,help="path to the dataset")
parser.add_argument('-d1','--dataset_1', type=str,required=True,help="first dataset")
parser.add_argument('-d2','--dataset_2', type=str,required=True,help="second dataset")
parser.add_argument('-s1','--subjects1', nargs='+', type=str,required=True,help='name of the subject for first data set')
parser.add_argument('-s2','--subjects2', nargs='+', type=str,required=True,help='name of the subject for second data set')
parser.add_argument('-T','--name_tract',type=str,required=True,help="name of the tract")
parser.add_argument('-L','--numbers_cores',type=int,required=True,help="number of cores MAS")
parser.add_argument('-a','--alpha', type=float,required=True,help='alpha parameter, trade off spatial/diffusion')


args = parser.parse_args()
path=args.path
d1=args.dataset_1
d2=args.dataset_2
subjects1=args.subjects1
subjects2=args.subjects2
TRACT=args.name_tract
nb_MAS=args.numbers_cores
alpha=args.alpha



if d1>d2: #'HCP_105'<'PPMI/Control','PPMI/Control'<'PPMI/PD','PPMI/PD'<'PPMI/Prodromal'
    d1,d2=d2,d1

path1=path+str(d1)
path2=path+str(d2)

if "/" in d1:
    d1=d1.replace("/","_")
if "/" in d2:
    d2=d2.replace("/","_")

#Load all subject of the data
all_s1=pd.read_csv(path1+'/000000/all_subject.csv')["Subject ID"].to_numpy()
all_s2=pd.read_csv(path2+'/000000/all_subject.csv')["Subject ID"].to_numpy()

M = pd.DataFrame(np.zeros((len(all_s1),len(all_s2))), index=all_s1, columns=all_s2)

#If it does not exist yet save it, otherwise load it
if os.path.isfile(path+'matrix_distance/'+d1+'_'+d2+'_'+TRACT+'_L'+str(nb_MAS)+'_alpha_'+str(alpha)+'.csv'):
    print('file already exist, loading it')
    M = pd.read_csv(path+'matrix_distance/'+d1+'_'+d2+'_'+TRACT+'_L'+str(nb_MAS)+'_alpha_'+str(alpha)+'.csv',index_col=0)
else:  
    print('file does not exist writing it')
    M.to_csv(path+'matrix_distance/'+d1+'_'+d2+'_'+TRACT+'_L'+str(nb_MAS)+'_alpha_'+str(alpha)+'.csv')
    M = pd.read_csv(path+'matrix_distance/'+d1+'_'+d2+'_'+TRACT+'_L'+str(nb_MAS)+'_alpha_'+str(alpha)+'.csv',index_col=0)
M.index = M.index.astype(str)

#Compute distance patients
for i,s1 in enumerate(subjects1):
    print(s1,end=': ',flush=True)
    cores1 = load_tractogram(path1+"/"+s1+'/06-AugmentedTractsOT/7cores_MAS_'+TRACT+'_L'+str(nb_MAS)+'.trx', reference='same')
    m1,wI1,I1,w1,S1,labels_svm1,_=trx_to_numpy_tract_list(cores1,nb_MAS)
    for j,s2 in enumerate(subjects2):
        if int(s2)>int(s1):
            if M.at[s1, s2] != 0:
                print('Overwritting a distance',end=' ',flush=True)
            cores2 = load_tractogram(path2+"/"+s2+'/06-AugmentedTractsOT/7cores_MAS_'+TRACT+'_L'+str(nb_MAS)+'.trx', reference='same')
            m2,wI2,I2,w2,S2,labels_svm2,_=trx_to_numpy_tract_list(cores2,nb_MAS)  
            cost,_,_,C_nb_pts=distance_cores_svm(m1,m2,wI1,wI2,I1,I2,w1,w2,S1,S2,labels_svm1,labels_svm2,alpha=alpha)
            M.at[s1, s2] = cost
            M.at[s2, s1] = cost
            #print(C_nb_pts.min())
    print()
    
#M.to_csv(path+'matrix_distance/'+d1+'_'+d2+'_'+TRACT+'_L'+str(nb_MAS)+'_alpha_'+str(alpha)+'.csv')
