# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import os
import gensim
import csv
import multiprocessing 
from multiprocessing import Pool

model_path = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki-doc2vec.model"
keep_path = "/home/wshalaby/work/github/WikiToolbox/20newsgroups.simple.esa.concepts.500+tree.lst"
mappings_path = "/home/wshalaby/work/github/WikiToolbox/wiki_ids_titles.csv"
outpath = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
outfile = open(outpath,"w")
model = None

def get_sims(id):
    global model
    sims = model.docvecs.most_similar(id,topn=50)
    return {id:sims}
    
def main():
    global model

    print("loading model")
    model = gensim.models.Doc2Vec.load(model_path)
    model.init_sims(replace=True)
    
    ids=['21233','545687','323557']
    cnt = 1
    srclis = []
    for id in ids:
        cnt = cnt + 1
        srclis.append([id])
        if cnt % 8==0:
            p = Pool(8)
            res = p.map(get_sims, srclis)
            p.close()
            p.join()
            print(res)
            srclis = []
            res.clear()

    p = Pool(8)
    p.map(get_sims, srclis)
    p.close()
    p.join()
    print(res)
    
    outfile.close()        
    
main()

