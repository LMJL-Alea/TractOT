#Convert trk to vtk
import numpy as np
import argparse
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.io.utils import Space

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input_path', type=str,required=True,help="input path")
parser.add_argument('-o','--output_path',type=str,required=True,help="output path")
args = parser.parse_args()
input_path=args.input_path
output_path=args.output_path

tract = load_tractogram(
            input_path,
            reference='same')#,to_space=Space.LPSMM)

save_tractogram(tract,output_path,to_space=Space.LPSMM) 



