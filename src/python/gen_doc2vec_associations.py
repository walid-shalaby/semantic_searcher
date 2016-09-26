# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import os
import collections
import gensim

model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/clef-2010/samples/models/wiki-doc2vec.model"
topics_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/root/Desktop/data-and-indices/patents-data/CLEF-IP/experiments/topics.txt"
#topics_path = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/experiments/topics.txt"
prefix = "topic_"
outpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/root/Desktop/data-and-indices/patents-data/CLEF-IP/experiments/relevancy/qrel-doc2v-abstract.txt"

print("loading topics")
dic = {}
with open(topics_path) as topics:
    for topic in topics:
        tokens = topic.replace(os.linesep,'').split('_')
        dic[tokens[0]] = tokens[1]
od = collections.OrderedDict(sorted(dic.items()))
print("loading model")
model = gensim.models.Doc2Vec.load(model_path)

outfile = open(outpath,"w")
cnt = 1
for topic,id in od.items():
    if prefix+id in model.docvecs.doctags:
        rel = model.docvecs.most_similar(prefix+id,topn=1100)
        print(str(cnt)+": "+prefix+id)
        cnt = cnt + 1
#        if cnt>10:
#            break
        i = 1
        for (id,_) in rel:
            if i>1000:
                break
            if id.find(prefix)!=-1:
                continue
            outfile.write(topic+" 0 "+id+" "+str(i)+" 0 0"+os.linesep)
            i = i + 1
    else:
        print(id+" is missing")
    
outfile.close()
