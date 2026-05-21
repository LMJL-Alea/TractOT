# select closest revB0 for DWI
# and rename

import os
import json
import datetime
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--path', type=str,required=True,help="path")
parser.add_argument('-s','--subject',type=str,required=True,help="subject")

args = parser.parse_args()
path=args.path
SUBJECT=args.subject

DWI=['DTI_B700_64dir_PA_','DTI_B1000_64dir_PA_','DTI_B2000_64dir_PA_']
revB0=['DTI_revB0_AP_'+SUBJECT,'DTI_revB0_AP_'+SUBJECT+'a','DTI_revB0_AP_'+SUBJECT+'b','DTI_revB0_AP_'+SUBJECT+'c']

acquisition_time_dwi=[]
acquisition_time_revB0=[]

for i,img in enumerate(DWI):
    with open(path+SUBJECT+'/00-Preprocessing/'+img+SUBJECT+'.json') as f:
        t=json.load(f)["AcquisitionTime"]
        acquisition_time_dwi+=[datetime.datetime(year=2000,month=1,day=1,hour=int(t[0:2]),minute=int(t[3:5]))]

for i,img in enumerate(revB0):
    with open(path+SUBJECT+'/00-Preprocessing/'+img+'.json') as f:
        t=json.load(f)["AcquisitionTime"]
        acquisition_time_revB0+=[datetime.datetime(year=2000,month=1,day=1,hour=int(t[0:2]),minute=int(t[3:5]))]

D=np.zeros((len(acquisition_time_dwi),len(acquisition_time_revB0)))

for i in range(len(acquisition_time_dwi)):
    for j in range(len(acquisition_time_revB0)):
        D[i,j]=abs((acquisition_time_dwi[i]-acquisition_time_revB0[j]).total_seconds())

rename=['rev_B0_DTI_B700_64dir_PA','rev_B0_DTI_B1000_64dir_PA','rev_B0_DTI_B2000_64dir_PA']
for i in range(len(acquisition_time_dwi)):
    idx=np.argmin(D[i,:])
    os.rename(path+SUBJECT+'/00-Preprocessing/'+revB0[idx]+'.json', path+SUBJECT+'/00-Preprocessing/'+rename[i]+'_'+SUBJECT+'.json')
    os.rename(path+SUBJECT+'/00-Preprocessing/'+revB0[idx]+'.nii.gz', path+SUBJECT+'/00-Preprocessing/'+rename[i]+'_'+SUBJECT+'.nii.gz')
