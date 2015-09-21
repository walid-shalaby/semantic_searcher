# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(walid-shalaby)s
"""

#python3 predict_corr.py /media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/EN-RG-65.txt /home/wshalaby/Desktop/wikipedia/EN-wform.w.5.cbow.neg10.400.subsmpl.txt /media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/results/predict/EN-RG-65.txt

import sys
import os
from scipy.spatial.distance import cosine

def main(args):
    if len(args)==4:        
        fin = open(args[1], 'r')
        fvec = open(args[2], 'r')        
        fout = open(args[3], 'w')
        
        # load vectors in dictionary keyed by term
        vectors = {}
        while True:
            line = fvec.readline().strip()
            if line=="":
                break            
            line = line.split("\t")            
            vectors[line[0]] = [float(x) for x in line[1:]]
      
        fvec.close()
    
        # load samples and calculate similarities
        for sample in fin.readlines():
            sample = sample.strip().lower().split(",")
            if len(sample)==2:
                if sample[0] in vectors.keys():
                    if sample[1] in vectors.keys():
                        fout.write(sample[0]+','+sample[1]+','+str(1-cosine(vectors[sample[0]],vectors[sample[1]]))+os.linesep)
                    else:
                        print(sample[1]+" is OOV! "+str(sample))
                        fout.write(sample[0]+','+sample[1]+',0.0'+os.linesep)
                else:
                    print(sample[0]+" is OOV! "+str(sample))
                    fout.write(sample[0]+','+sample[1]+',0.0'+os.linesep)
            else:
                print("invalid sample format: "+str(sample))
        fin.close()
        fout.close()
    
    else:
        print("invalid arguments!")
    

main(sys.argv)