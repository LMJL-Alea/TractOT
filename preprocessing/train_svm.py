#Train svm to perform parcellation

import numpy as np
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram
from sklearn.svm import SVC
import pickle

import sys
sys.path.append('../script/')
from from_vtk_to_numpy import vtk_to_numpy_tract_array

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
parser.add_argument('-r','--reference_path',type=str,required=True,help="reference image or .trk")
parser.add_argument('-t','--nb_streamliness_train', type=int,required=True,help="number of streamlines to train the SVM")

args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path
reference=args.reference_path
n_train=args.nb_streamliness_train

tract_train = load_tractogram(
            input_path,
            reference=reference)

n_train=min(n_train,len(tract_train.streamlines))
m_train,_=vtk_to_numpy_tract_array(tract_train,n_train)
nb_pts=m_train.shape[1]

samples=m_train.reshape(-1,3).tolist()
classes=np.repeat(np.arange(1,nb_pts+1)[None,:],n_train,0).reshape(-1).tolist()

print()
print("train SVM",flush=True)
svc = SVC(C=.5, kernel='poly')
svc.fit(X=samples, y=classes)

print("save SVM",flush=True)
with open(output_path, 'wb') as file:
    pickle.dump(svc, file)
