## -*- coding: utf-8 -*-
#"""
#Created on Mon Sep 6 22:33:40 2016
#
#@author: wshalaby
#"""
#
#import os
#import gensim
#import csv
#import multiprocessing 
#from multiprocessing import Pool
#
##model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/models/wiki-doc2vec.model"
#model_path = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki-doc2vec.model"
##keep_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups.simple.esa.concepts.500+tree.lst"
#keep_path = "/home/wshalaby/work/github/WikiToolbox/20newsgroups.simple.esa.concepts.500+tree.lst"
##mappings_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/root/Desktop/data-and-indices/patents-data/CLEF-IP/experiments/wiki_ids_titles.csv"
#mappings_path = "/home/wshalaby/work/github/WikiToolbox/wiki_ids_titles.csv"
##outpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
#outpath = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
#outfile = open(outpath,"w")
#mappings = {}
#model = None
#
#def get_sims(src):
#    global res, mappings, model
#    cnt = src[0]
#    id = src[1]
#    title = src[2]
#    print(cnt+" "+title)
#    sims = []
#    sims = model.docvecs.most_similar(id,topn=50)
#    ext_sims = []
#    for sim in sims:
#        #if sim[0] in mappings.keys():
#        try:
#            ext_sims.append((mappings[sim[0]],sim[1]))
#        except:
#            continue
#    
#    return {title:ext_sims}
#    
#def write_sims(res):
#    global outfile
#    for item in res:
#        for title, sims in item.items():
#            outfile.write(title+"/\\/\\1000")    
#            for sim in sims:
#                outfile.write("#\\$#"+sim[0]+"/\\/\\"+str(int(1000*round(sim[1],3))))
#                #outfile.flush()
#            outfile.write(os.linesep)
#        
#def main():
#    global mappings, model
#    to_keep = set()
#    print("loading keey only")
#    with open(keep_path) as titles:
#        for title in titles:
#            to_keep.add(title.replace(os.linesep,""))
#        
#    print("loaded "+str(len(to_keep))+" keep only")
#    
#    print("loading mappings")
#    records = csv.DictReader(open(mappings_path))
#    for pair in records:
#        if len(to_keep)==0 or pair['title'] in to_keep:
#            mappings[pair['id']] = pair['title']
#        
#    print("loaded "+str(len(mappings))+" mappings")
#    
#    print("loading model")
#    model = gensim.models.Doc2Vec.load(model_path)
#    #wiki_ids = list(model.docvecs.doctags)
#    #print("loaded "+str(len(wiki_ids))+" ids")
#    
#    cnt = 1
#    srclis = []
#    for id,title in mappings.items():
#        cnt = cnt + 1
#        srclis.append([str(cnt),id,title])
#        if cnt % 8==0:
#            p = Pool(8)
#            res = p.map(get_sims, srclis)
#            p.close()
#            p.join()
#            write_sims(res)
#            srclis = []
#            res.clear()
#
#    p = Pool(8)
#    p.map(get_sims, srclis)
#    p.close()
#    p.join()
#    write_sims(res)
#    
#    outfile.close()        
#    
#main()
#

## -*- coding: utf-8 -*-
#"""
#Created on Mon Sep 6 22:33:40 2016
#
#@author: wshalaby
#"""
#
#import os
#import gensim
#import csv
#
#model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/models/wiki-doc2vec.model"
##model_path = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki-doc2vec.model"
#keep_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups.simple.esa.concepts.500+tree.lst"
##keep_path = "/home/wshalaby/work/github/WikiToolbox/20newsgroups.simple.esa.concepts.500+tree.lst"
#mappings_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/root/Desktop/data-and-indices/patents-data/CLEF-IP/experiments/wiki_ids_titles.csv"
##mappings_path = "/home/wshalaby/work/github/WikiToolbox/wiki_ids_titles.csv"
#outpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
##outpath = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
#
#to_keep = set()
#if keep_path!="":
#    print("loading keey only")
#    with open(keep_path) as titles:
#        for title in titles:
#            to_keep.add(title.replace(os.linesep,""))
#    
#print("loaded "+str(len(to_keep))+" keep only")
#
#print("loading mappings")
#mappings = {}
#records = csv.DictReader(open(mappings_path))
#for pair in records:
#    if len(to_keep)==0 or pair['title'] in to_keep:
#        mappings[pair['id']] = pair['title']
#    
#print("loaded "+str(len(mappings))+" mappings")
#
#print("loading model")
#model = gensim.models.Doc2Vec.load(model_path)
#model.init_sims(replace=True)
##wiki_ids = list(model.docvecs.doctags)
##print("loaded "+str(len(wiki_ids))+" ids")
#
#outfile = open(outpath,"w")
#
#cnt = 1
#for id,title in mappings.items():
#    print(str(cnt)+" "+title)
#    cnt = cnt + 1
#    outfile.write(title+"/\\/\\1000")
#    sims = model.docvecs.most_similar(id,topn=50)
#    for sim in sims:
#        try:
#            outfile.write("#\\$#"+mappings[sim[0]]+"/\\/\\"+str(int(1000*round(sim[1],3))))
#        except:
#            continue
#        #outfile.flush()
#    outfile.write(os.linesep)
#    
#outfile.close()

## -*- coding: utf-8 -*-
#"""
#Created on Mon Sep 6 22:33:40 2016
#
#@author: wshalaby
#"""
#
#import os
#import gensim
#import csv
#import multiprocessing 
#from joblib import Parallel, delayed
#from multiprocessing import Pool
#
#model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/models/wiki-doc2vec.model"
##model_path = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki-doc2vec.model"
#keep_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups.simple.esa.concepts.500+tree.lst"
##keep_path = "/home/wshalaby/work/github/WikiToolbox/20newsgroups.simple.esa.concepts.500+tree.lst"
#mappings_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/root/Desktop/data-and-indices/patents-data/CLEF-IP/experiments/wiki_ids_titles.csv"
##mappings_path = "/home/wshalaby/work/github/WikiToolbox/wiki_ids_titles.csv"
#outpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
##outpath = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
#outfile = open(outpath,"w")
#mappings = {}
#model = None
#
#def get_sims(src):
#    global mappings, model
#    cnt = src[0]
#    id = src[1]
#    title = src[2]
#    print(cnt+" "+title)
#    sims = []
#    sims = model.docvecs.most_similar(id,topn=50)
#    ext_sims = []
#    for sim in sims:
#        #if sim[0] in mappings.keys():
#        try:
#            ext_sims.append((mappings[sim[0]],sim[1]))
#        except:
#            continue
#    
#    return {title:ext_sims}
#    
#def write_sims(res):
#    global outfile
#    for item in res:
#        for title, sims in item.items():
#            outfile.write(title+"/\\/\\1000")    
#            for sim in sims:
#                outfile.write("#\\$#"+sim[0]+"/\\/\\"+str(int(1000*round(sim[1],3))))
#                #outfile.flush()
#            outfile.write(os.linesep)
#        
#def main():
#    global mappings, model
#    to_keep = set()
#    print("loading keey only")
#    with open(keep_path) as titles:
#        for title in titles:
#            to_keep.add(title.replace(os.linesep,""))
#        
#    print("loaded "+str(len(to_keep))+" keep only")
#    
#    print("loading mappings")
#    records = csv.DictReader(open(mappings_path))
#    for pair in records:
#        if len(to_keep)==0 or pair['title'] in to_keep:
#            mappings[pair['id']] = pair['title']
#        
#    print("loaded "+str(len(mappings))+" mappings")
#    
#    print("loading model")
#    model = gensim.models.Doc2Vec.load(model_path)
#    model.init_sims(replace=True)
#    #wiki_ids = list(model.docvecs.doctags)
#    #print("loaded "+str(len(wiki_ids))+" ids")
#    
#    cnt = 1
#    srclis = []
#    for id,title in mappings.items():
#        cnt = cnt + 1
#        srclis.append([str(cnt),id,title])
#        if cnt % 8==0:#((int)(multiprocessing.cpu_count()/2)) == 0:
#            res = Parallel(n_jobs=8)(delayed(get_sims)(i) for i in srclis)
#            write_sims(res)
#            srclis = []
#            res.clear()
#
#    res = Parallel(n_jobs=8)(delayed(get_sims)(i) for i in srclis)
#    write_sims(res)
#            
#    outfile.close()        
#    
#main()
#


# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import os
import gensim
import csv
import multiprocessing 
from joblib import Parallel, delayed
from multiprocessing import Pool

import _compat_pickle
_compat_pickle.IMPORT_MAPPING.update({
    'UserDict': 'collections',
    'UserList': 'collections',
    'UserString': 'collections',
    'whichdb': 'dbm',
    'StringIO':  'io',
    'cStringIO': 'io',
})

#model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/models/wiki-doc2vec.model"
model_path = "/scratch/wshalaby/doc2vec/wiki-doc2vec.model"
#model_path = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki-doc2vec.model"
#keep_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups.simple.esa.concepts.500+tree.lst"
keep_path = "/scratch/wshalaby/doc2vec/20newsgroups.simple.esa.concepts.500+tree.lst"
#keep_path = "/home/wshalaby/work/github/WikiToolbox/20newsgroups.simple.esa.concepts.500+tree.lst"
#mappings_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/root/Desktop/data-and-indices/patents-data/CLEF-IP/experiments/wiki_ids_titles.csv"
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
#mappings_path = "/home/wshalaby/work/github/WikiToolbox/wiki_ids_titles.csv"
#outpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt11"
outpath = "/scratch/wshalaby/doc2vec/wiki_sims.txt"
#outpath = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/wiki_sims.txt"
outfile = open(outpath,"w")
mappings = {}
models = [None,None,None,None,None,None,None,None,None,None]

def get_sims(srcs):
    global mappings, models
    #print("loading model")
    #model = gensim.models.Doc2Vec.load(model_path)
    #model.init_sims(replace=True)
    res = {}
    for src in list(srcs.values())[0]:
        cnt = src[0]
        id = src[1]
        title = src[2]
        print(cnt+" "+title)
        sims = []
        sims = models[list(srcs.keys())[0]].docvecs.most_similar(id,topn=50)
        ext_sims = []
        for sim in sims:
            #if sim[0] in mappings.keys():
            try:
                ext_sims.append((mappings[sim[0]],sim[1]))
            except:
                continue
        res[title] = ext_sims
    return res
    
def write_sims(res):
    global outfile
    for item in res:
        for title, sims in item.items():
            outfile.write(title+"/\\/\\1000")    
            for sim in sims:
                outfile.write("#\\$#"+sim[0]+"/\\/\\"+str(int(1000*round(sim[1],3))))
                #outfile.flush()
            outfile.write(os.linesep)
    
def main():
    global mappings, models
    
    for i in range(4):
        print("loading model")
        models[i] = gensim.models.Doc2Vec.load(model_path)
        models[i].init_sims(replace=True)
    #wiki_ids = list(model.docvecs.doctags)
    #print("loaded "+str(len(wiki_ids))+" ids")
    
    to_keep = set()
    print("loading keey only")
    with open(keep_path) as titles:
        for title in titles:
            to_keep.add(title.replace(os.linesep,""))
        
    print("loaded "+str(len(to_keep))+" keep only")
    
    print("loading mappings")
    records = csv.DictReader(open(mappings_path))
    for pair in records:
        if len(to_keep)==0 or pair['title'] in to_keep:
            mappings[pair['id']] = pair['title']
        
    print("loaded "+str(len(mappings))+" mappings")
    
    cnt = 1
    srclis = []
    for id,title in mappings.items():
        cnt = cnt + 1
        srclis.append([str(cnt),id,title])
        if cnt % 400==0:
            res = Parallel(n_jobs=4)(delayed(get_sims)({i%100:srclis[i:i+99]}) for i in [0,100,200,300])
            write_sims(res)
            srclis = []
            res.clear()

    res = Parallel(n_jobs=4)(delayed(get_sims)({i%100:srclis[i:i+99]}) for i in [0,100,200,300])
    write_sims(res)
            
    outfile.close()        
    
main()

