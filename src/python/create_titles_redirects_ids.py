# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import os
import gensim

model_path = "/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.model"
outpath = "/home/walid/work/github/semantic_searcher/titles_redirects_ids_anno.csv"
mappings_path = "/home/walid/work/github/semantic_searcher/titles_redirects_ids.csv"

print("loading model")
model = gensim.models.Word2Vec.load(model_path)
titles_dic = set()
for title in model.vocab.keys():
    if title.find('id')==0 and title.find('di')==len(title)-2:
        titles_dic.add(title[2:len(title)-2])

print("found ("+str(len(titles_dic))+") titles")

count = 0
with open(mappings_path,'r') as titles_csv:
	outfile = open(outpath,"w")
	outfile.write('title,redirect,id'+os.linesep)
	for line in titles_csv:
		x = 10
		line = line.strip('\n')
		id = line[line.rfind(',')+1:]
		if id in titles_dic:
			count = count + 1
			outfile.write(line+os.linesep)

print("found ("+str(count)+")")