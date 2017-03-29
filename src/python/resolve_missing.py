# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""
#python3 resolve_missing.py > still_missing.txt

#cat samples-out-concepts-auto.motor-conc-allnorm-0.85.txt.out|grep '...not found'|sed 's/...not found//g'|sed 's/[[:digit:]]\+\t//g' | sort | uniq -c | sort -nr > notfound.txt
#cat samples-out-concepts-conc-allnorm-0.0-xaa-xab.txt.out.backup | grep _ | grep '...not found'|sed 's/...not found//g'|sed 's/.*\t//g' | sort | uniq -c | sort -nr > notfound.txt
#cat samples-out-concepts-conc-allnorm-0.0-xaa-xab.txt.out.backup | grep _ | grep '...not found'|sed 's/...not found//g'|sed 's/[[:digit:]]\+_//g' | sort | uniq -c | sort -nr > notfound-class.txt

import os
import gensim
import re

'''
import _compat_pickle
_compat_pickle.IMPORT_MAPPING.update({
    'UserDict': 'collections',
    'UserList': 'collections',
    'UserString': 'collections',
    'whichdb': 'dbm',
    'StringIO':  'io',
    'cStringIO': 'io',
})
'''

missing_path = "/home/wshalaby/work/github/semantic_searcher/src/python/missing1.txt"
missing_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/missing.txt"
missing_path = "/home/walid/work/github/semantic_searcher/python/missing.txt"
missing_path = "/home/walid/work/github/semantic_searcher/python/missing_all.txt"
missing_path = "/home/walid/work/github/semantic_searcher/python/notfound-nocount.txt"

out_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/concepts-10iter-dim500-wind5-skipgram1.vocab"
out_path = "/home/wshalaby/work/github/semantic_searcher/src/python/resolved_missing1.txt"
out_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/resolved_missing.txt"
out_path = "/home/walid/work/github/semantic_searcher/python/concepts-10iter-dim500-wind5-skipgram1.vocab"
out_path = "/home/walid/work/github/semantic_searcher/python/resolved_missing.txt"
out_path = "/home/walid/work/github/semantic_searcher/python/resolved_missing_all.txt"
out_path = "/home/walid/work/github/semantic_searcher/python/resolved_missing_all_still.txt"

still_path = "/home/wshalaby/work/github/semantic_searcher/src/python/still_missing1.txt"
still_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/still_missing.txt"
still_path = "/home/walid/work/github/semantic_searcher/python/still_missing.txt"
still_path = "/home/walid/work/github/semantic_searcher/python/still_missing_all.txt"
still_path = "/home/walid/work/github/semantic_searcher/python/still_missing_all_still.txt"

model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/home/wshalaby/work/github/semantic_searcher/src/python/concepts-10iter-dim500-wind5-skipgram1.vocab"
model_path = "/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"
model_path = "/home/walid/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/home/walid/work/github/semantic_searcher/python/concepts-10iter-dim500-wind5-skipgram1.vocab"

titles_path = "/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles_redirects_seealso.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects_seealso.csv"
titles_path = "/home/walid/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/titles_redirects_ids.csv"

from commons import load_titles

titles = {}
redirects = {}
seealso = {}

mode = 'resolve'
if mode=='flush':
    model = gensim.models.Word2Vec.load(model_path)
    with open(out_path,'w') as f:
        for i in model.vocab:
            f.write(i+os.linesep)
elif mode=='resolve':
    to_keep = set()
    titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
    print('done loading')

    vocab = {}
    fout = open(out_path,'w')
    fstill = open(still_path,'w')  
    #model = gensim.models.Word2Vec.load(model_path)
    for redirect,title in redirects.items(): # add redirects
        if 1==1:# title in vocab.values():
            surface = redirect.replace(',',' ').replace('+',' ').replace('!',' ').replace('-',' ').replace('.',' ').replace('/',' ').replace('\'s',' s').replace('\'n',' n').replace('U.S.','U S ').replace('2.0','2 0').replace('&amp;',' ').replace('\'',' ').replace(';',' ').replace('&',' ').replace('/',' ').replace('?',' ').replace('&quot;',' ')
            surface = re.sub('\\s+', ' ', surface).strip()
            #print('redirect:'+surface+'\t'+title)
            vocab[surface] = title
    fin = open(model_path,'r')
    for line in fin: # add model vocabulary
        surface = line.strip('\n').replace(',',' ').replace('+',' ').replace('!',' ').replace('-',' ').replace('.',' ').replace('/',' ').replace('\'s',' s').replace('\'n',' n').replace('U.S.','U S ').replace('2.0','2 0').replace('&amp;',' ').replace('\'',' ').replace(';',' ').replace('&',' ').replace('/',' ').replace('?',' ').replace('&quot;',' ')
        surface = re.sub('\\s+', ' ', surface).strip()
        vocab[surface] = line.strip('\n')
    
    print("processing")
    with open(missing_path,'r') as f:
        for line in f:
            line = line.strip('\n')
            #tokens = line.split()
            printed = False
            #for i in model.vocab:
            #for i in vocab:
            found = 0
            if line in vocab:
                fout.write('\''+line+'\': '+'\''+vocab[line].replace('\'','\\\'')+'\','+os.linesep)
                fout.flush()
                printed = True        
                '''
                elif 1!=1:
                    for t in tokens:
                        if i.find(t)!=-1:
                            found += 1
                    if found>1 and found>=len(tokens)-1:
                        fout.write('\''+line+'\': '+'\''+i+'\','+os.linesep)
                        fout.flush()
                        printed = True
                '''
            if printed==False:
                fstill.write(line+os.linesep)
                fstill.flush()

