# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import sys
import gensim
import csv
import os
from time import gmtime, strftime

ind_patern = '<doc><field name=\"id\">"%s</field><field name=\"title\"><![CDATA[%s]]></field><field name=\"vector\"><![CDATA[%s]]></field></doc>'
titles = {}
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles.csv"

#outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-wvecs-20160820.txt'

modelpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter.500.30out.model'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.10iter.500.30out.ind'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.10iter.500.30out.txt'
keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"

modelpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts.500.30out.model'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.500.30out.ind'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.500.30out.txt'
keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"

modelpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts.model'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.ind'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.txt'
keeponly_path = ""

modelpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter.model'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.10iter.ind'
outpath =   '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-wvec.10iter.txt'
keeponly_path = ""

print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))

to_keep = set()
if keeponly_path!="":
    print("loading keey only")
    with open(keeponly_path) as keeponly_titles:
        for title in keeponly_titles:
            to_keep.add(title.replace(os.linesep,""))

    print("loaded "+str(len(to_keep))+" keeponly titles")

print("loading titles")
records = csv.DictReader(open(titles_path))
count = 0
for pair in records:
    count = count + 1
    if len(to_keep)==0 or pair['title'] in to_keep:
        titles[pair['title']] = pair['id']
    
    
print("loaded "+str(count)+" titles and keeping "+str(len(titles))+" only")

print("loading model "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))    
model = gensim.models.Word2Vec.load(modelpath)
print("flushing "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
with open(outpath,"w") as output:
    for term in model.vocab.keys():
        try:
            id = titles[term]
            if len(sys.argv)>1 and sys.argv[1]=='ind':
                output.write((ind_patern)%(id,term,' '.join(map(str,model[term])))+os.linesep)
            else:
                output.write(('%s\t%s\t%s')%(id,term,' '.join(map(str,model[term])))+os.linesep)
        except:
            print("Oops!...("+term+") not found...")
            continue
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print('Done!')
