#Compute Cores MAS
# point of the core MAS are average of GM after clustering of MAS and svm parcellation

import argparse
import numpy as np
from dipy.io.streamline import load_tractogram,save_tractogram
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('-t','--augmented_tractogram_path', type=str,required=True,help="path of input augmented tractogram")
parser.add_argument('-o','--output_path_cores_MAS',type=str,required=True,help="output path of obtained cores MAS in .trx")
parser.add_argument('-s','--svm_path',type=str,required=True,help="path of the learned svm")
parser.add_argument('-L','--number_of_cores_MAS',type=int,required=True,help="number of cores MAS")

args = parser.parse_args()
augmented_tractogram_path=args.augmented_tractogram_path
output_path_cores_MAS=args.output_path_cores_MAS
svm_path=args.svm_path
number_of_cores_MAS=args.number_of_cores_MAS

import sys
sys.path.append('../script/')
from from_trx_to_numpy import trx_to_numpy_tract_array
from build_cores_MAS import average_MAS_per_labels,infer_labels_svm_endpoint

MAS = load_tractogram(augmented_tractogram_path, reference='same')

nb_pts=40
nb_streamline=len(MAS.streamlines)

m,wI,I,w,S,conf,_=trx_to_numpy_tract_array(MAS,nb_streamline)
#print(m.shape,wI.shape,I.shape,w.shape,S.shape,conf.shape)

print("Predict parcellation",flush=True)
labels_svm,labels_clus=infer_labels_svm_endpoint(svm_path,number_of_cores_MAS,m,random_state=0)
#print(labels_svm.shape,labels_clus.shape)

cores_m,cores_wI,cores_I,cores_w,cores_S,idx_svm,idx_cluster,size_cluster=average_MAS_per_labels(number_of_cores_MAS,m,wI,I,w,S,conf,labels_svm,labels_clus)

#print(len(cores_m),len(cores_wI),len(cores_I),len(cores_w),len(cores_S),len(idx_svm),len(idx_cluster),len(size_cluster))
#print(cores_m[0].shape,cores_wI[0].shape,cores_I[0].shape,cores_w[0].shape,cores_S[0].shape,idx_svm[0].shape,idx_cluster[0].shape,size_cluster[0].shape)

print(cores_wI[0][0],cores_m[0][0],flush=True)

cores=MAS.from_sft(cores_m,MAS)
dico={}
dico['weight iso']=cores_wI
dico['iso']=cores_I
dico['weights aniso']=cores_w
dico['aniso']=cores_S
dico['svm']=idx_svm
dico['cluster']=idx_cluster
dico['size cluster']=size_cluster
cores.data_per_point=dico
save_tractogram(cores,output_path_cores_MAS)
print('',flush=True)
