# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(walid-shalaby)s
"""

import os
import sys

def formatter(path,out):
    for f in os.listdir(path):
        if(os.path.isdir(path+'/'+f)):
            formatter(path+'/'+f,out)
        else:
            print(path+'/'+f)
            inp = open(path+'/'+f)
            for line in inp:
                line = line.replace(os.linesep,'')
                if(line[0]=='('):
                    out.write(path[path.rfind('/')+1:]+'\t'+line.replace('r:','').replace('a:','').replace('p:','').replace('m:','').replace(',','\t').replace('pvalue=','').replace('SpearmanrResult(correlation=','').replace('(','').replace(')','').replace(' ','').replace('\'','')+os.linesep) 
            inp.close()

out = open(sys.argv[2],"w")
out.write('data set'+'\t'+'best_r'+'\t'+'pv'+'\t'+'best_p'+'\t'+'pv'+'\t'+'best_mean'+'\t'+'r'+'\t'+'pv'+'\t'+'p'+'\t'+'pv'+'\t'+'mean'+'\t'+'file'+'\t'+'lambda@best_r'+'\t'+'r@best_r'+'\t'+'pv'+'\t'+'p@best_r'+'\t'+'pv'+'\t'+'mean@best_r'+'\t'+'lambda@best_p'+'\t'+'r@best_p'+'\t'+'pv'+'\t'+'p@best_p'+'\t'+'pv'+'\t'+'mean@best_p'+'\t'+'lambda@best_mean'+'\t'+'r@best_mean'+'\t'+'pv'+'\t'+'p@best_mean'+'\t'+'pv'+'\t'+'mean@best_mean'+'\t'+'lambda@1'+'\t'+'r@1'+'\t'+'pv'+'\t'+'p@1'+'\t'+'pv'+'\t'+'mean@1'+os.linesep)
formatter(sys.argv[1],out)
out.close()
