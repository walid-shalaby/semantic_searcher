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

model_path = "/scratch/wshalaby/doc2vec/models/wikipedia-2016.500.30out.model"
outpath = "/scratch/wshalaby/doc2vec/models/wiki_sims.500.30out.txt"
wiki_esa_mapping_path = "/scratch/wshalaby/doc2vec/wiki_esa_ids.tsv"
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = "/scratch/wshalaby/doc2vec/wikipedia-2016.500.30out.lst"

model_path = "/scratch/wshalaby/doc2vec/models/wikipedia-2016.500.30out.model"
outpath = "/scratch/wshalaby/doc2vec/models/wiki_sims.500.30out.txt"
wiki_esa_mapping_path = "/scratch/wshalaby/doc2vec/wiki_esa_ids.tsv"
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = "/scratch/wshalaby/doc2vec/wikipedia-2016.500.30out.lst"

model_path = '/scratch/wshalaby/doc2vec/models/word2vec/concepts.500.30out.model'
outpath =   "/scratch/wshalaby/doc2vec/models/word2vec/concepts-sims-wvec.500.30out.txt"
wiki_esa_mapping_path = "/scratch/wshalaby/doc2vec/wiki_esa_ids.tsv"
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = "/scratch/wshalaby/doc2vec/wikipedia-2016.500.30out.lst"

model_path = '/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter.500.30out.model'
outpath =   "/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter-sims-wvec.500.30out.txt"
wiki_esa_mapping_path = "/scratch/wshalaby/doc2vec/wiki_esa_ids.tsv"
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = "/scratch/wshalaby/doc2vec/wikipedia-2016.500.30out.lst"

model_path = '/scratch/wshalaby/doc2vec/models/word2vec/concepts.model'
outpath =   "/scratch/wshalaby/doc2vec/models/word2vec/concepts.txt"
wiki_esa_mapping_path = ""
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = ""

model_path = '/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter.model'
outpath =   "/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter.vecs.txt"
wiki_esa_mapping_path = ""
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = ""

model_path = '/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter.model'
outpath =   "/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter.vecs.txt"
wiki_esa_mapping_path = ""
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = ""

#model_path = '/home/wshalaby/work/phd-courses/ML/project/model/jobs-vecs-dm.model'
#outpath =   "/home/wshalaby/work/phd-courses/ML/project/model/jobs-vecs-dm.vecs"
#wiki_esa_mapping_path = ""
#mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
#keep_path = ""

model_path = "/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.model"
outpath = "/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.vecs.txt"
mappings_path = "/home/walid/work/github/semantic_searcher/titles_redirects_ids_anno.csv"
keep_path = ""

model_path = "/home/walid/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
outpath = "/home/walid/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.vecs.txt"
wiki_esa_mapping_path = ""
mappings_path = "/home/walid/work/github/semantic_searcher/titles_redirects_ids.csv"
keep_path = ""

outfile = None
w2v_model_type = 'title'
mappings = {}
models = [None,None,None,None,None,None,None,None,None,None]

import sys
def get_vecs_w2vec(srcs):
    global mappings, models, w2v_model_type
    res = {}
    for src in list(srcs.values())[0]:
        cnt = src[0]
        id = src[1]
        title = src[2]
        print(cnt+" ("+id+") ("+title+")")
        try:
            if w2v_model_type=='title':
                res[title] = list(models[list(srcs.keys())[0]][title])
            elif w2v_model_type=='id':
                res[id] = list(models[list(srcs.keys())[0]]['id'+title+'di'])
        except:
            print("Oops..."+cnt+" ("+id+") ("+title+"), doc doesn't exist!")
            continue
    return res
    
def get_vecs_doc2vec(srcs):
    global mappings, models
    res = {}
    for src in list(srcs.values())[0]:
        cnt = src[0]
        id = src[1]
        title = src[2]
        if id in models[list(srcs.keys())[0]].docvecs.doctags:
            print(cnt+" ("+id+") ("+title+")")
            res[title] = list(models[list(srcs.keys())[0]][id])
    return res
    
def write_vecs(res, format):
    import pickle
    import os
    global outfile    
    for item in res:
        for title, vec in item.items():
            if format=='raw':
                pickle.dump((title,vec),outfile)
            elif format=='txt':
                outfile.write(title+'\t'+str(vec).replace(' ','').replace('[','').replace(']','')+os.linesep)

def load_vecs(filename):
    import pickle
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

#items = load(myfilename)

def main():
    global mappings, models, outfile, w2v_model_type
    
    workers = int(sys.argv[2])
    format = sys.argv[3]
    w2v_model_type = sys.argv[4]

    if format=='raw':
        outfile = open(outpath,"wb")
    elif format=='txt':
        outfile = open(outpath,"w")

    for i in range(workers):
        print("loading model")
        if len(sys.argv)>1 and sys.argv[1]=='w2v':
            models[i] = gensim.models.Word2Vec.load(model_path)
        else:
            models[i] = gensim.models.Doc2Vec.load(model_path)
    
    to_keep = set()
    if keep_path!="":
        print("loading keey only")
        with open(keep_path) as titles:
            for title in titles:
                to_keep.add(title.replace(os.linesep,""))
    
    wiki_esa_mapping = {}
    if wiki_esa_mapping_path!="":
        print("loading wiki esa mappings")
        with open(wiki_esa_mapping_path) as wiki_esas:
            for wiki_esa in wiki_esas:
                tokens = wiki_esa.replace(os.linesep,"").split("\t")
                wiki_esa_mapping[tokens[1]] = tokens[0]            
    
    print("loaded "+str(len(wiki_esa_mapping))+" wiki esa mappings")
    
    print("loading mappings")
    records = csv.DictReader(open(mappings_path))
    for pair in records:
        if len(to_keep)==0 or pair['title'] in to_keep:
            if len(wiki_esa_mapping)>0:
                if len(sys.argv)>1 and sys.argv[1]=='w2v':
                    mappings[pair['title']] = wiki_esa_mapping[pair['id']]
                else:
                    mappings[wiki_esa_mapping[pair['id']]] = pair['title']
            else:
                mappings[pair['title']] = pair['id']
        
    print("loaded "+str(len(mappings))+" mappings")
    
    cnt = 1
    srclis = []
    for k,v in mappings.items():
        cnt = cnt + 1
        if len(sys.argv)>1 and sys.argv[1]=='w2v':
            if w2v_model_type=='title':
                srclis.append([str(cnt),v,k])
            elif w2v_model_type=='id':
                srclis.append([str(cnt),k,v])
        else:
            srclis.append([str(cnt),k,v])
        if cnt % workers*100==0:
            if len(sys.argv)>1 and sys.argv[1]=='w2v':
                res = Parallel(n_jobs=workers)(delayed(get_vecs_w2vec)({i%100:srclis[i:i+99]}) for i in range(0,100*workers,100))
            else:
                res = Parallel(n_jobs=workers)(delayed()({i%100:srclis[i:i+99]}) for i in range(0,100*workers,100))
            write_vecs(res, format)
            srclis = []
            res.clear()

    if len(sys.argv)>1 and sys.argv[1]=='w2v':
        res = Parallel(n_jobs=workers)(delayed(get_vecs_w2vec)({i%100:srclis[i:i+99]}) for i in range(0,100*workers,100))
    else:
        res = Parallel(n_jobs=workers)(delayed(get_vecs_doc2vec)({i%100:srclis[i:i+99]}) for i in range(0,100*workers,100))
    write_vecs(res, format)
            
    outfile.close()        
    
# gen_doc2vec_wiki_vecs.py w2v 1 raw|txt title|id
print('use title in case of w2v concepts model and id in case of w2v annotated titles')
main()


