# -*- coding: utf-8 -*-
"""
Created on Tues Jan 3 22:33:40 2017

@author: wshalaby
"""

import gensim
import os
from time import gmtime, strftime
import multiprocessing

in_path = '/scratch/wshalaby/doc2vec/test.svm'
in_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.svm'
in_path = '/home/walid/work/github/semantic_searcher/test.svm'
in_path2 = '/home/wshalaby/work/github/dexter-relatedness/data/learning/trec/trec-ent/training.svm'
in_path = '/home/wshalaby/work/github/dexter-relatedness/data/learning/trec/trec-ent/test.svm'

outpath1 = '/scratch/wshalaby/doc2vec/test.trec'
outpath2 = '/scratch/wshalaby/doc2vec/test.concepts-10iter-dim500-wind5-skipgram1.trec'
outpath2 = '/scratch/wshalaby/doc2vec/test.w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.trec'

outpath1 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.trec'
outpath2 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.concepts-10iter-dim500-wind5-skipgram1.trec'
outpath2 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.trec'

outpath1 = '/home/walid/work/github/semantic_searcher/test.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/test.concepts-10iter-dim500-wind5-skipgram1.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/test.w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.trec'

outpath1 = '/home/wshalaby/work/github/dexter-relatedness/data/learning/trec/trec-ent/test.trec'
outpath2 = '/home/wshalaby/work/github/dexter-relatedness/data/learning/trec/trec-ent/test.concepts-10iter-dim500-wind5-skipgram1.trec'
outpath2 = '/home/wshalaby/work/github/dexter-relatedness/data/learning/trec/trec-ent/test.w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.trec'

model_path = "/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"

model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"
model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"

model_path = "/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"
model_path = "/home/walid/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"

with open(outpath1,'w') as outp1:
    qrels = {}
    entities = {}
    conll_recs = set()
    entities_conll_recs = set()
    entities_qrels = set()
    with open(in_path,'r') as inp:
        for line in inp:
            line = line.strip(' \n')
            tokens = line.split(' ')
            conll_rec = tokens[len(tokens)-1]
            conll_recs.add(conll_rec)
            ent_pair = tokens[len(tokens)-2].split('-')
            label = tokens[0]
            qrel = tokens[1]
            #qrels.add(qrel)
            if len(ent_pair)!=2:
                print('Oops...unexpected entity pair')
                break
            entities_conll_recs.add(ent_pair[0]+'_'+conll_rec)
            entities_qrels.add(ent_pair[0]+'_'+qrel)
            if ent_pair[0] < ent_pair[1]:
                normalized_ent_pair = ent_pair[0]+'-'+ent_pair[1]
            else:
                normalized_ent_pair = ent_pair[1]+'-'+ent_pair[0]
            if True==True and label=='1':
                if ent_pair[0] in entities:
                    entities[ent_pair[0]].add(ent_pair[1])                    
                else:
                    entities[ent_pair[0]] = set(ent_pair[1])

            if qrel in qrels:
                qrels[qrel].add(tokens[len(tokens)-2])                    
            else:
                qrels[qrel] = set(tokens[len(tokens)-2])
            #if conll_rec=='1':
            #    outp1.write(line+os.linesep)
            '''
            if ent_pair[1] in entities:
                entities[ent_pair[1]].add(ent_pair[0])
            else:
                entities[ent_pair[1]] = set(ent_pair[0])
            '''
        print(len(entities))
        total = 0
        for ent in entities:
            total += len(entities[ent])
        print(1.0*total/len(entities))
        print(len(conll_recs))
        print(len(entities_conll_recs))
        print(len(qrels))
        for qrel in qrels:
            total += len(qrels[qrel])
        print(1.0*total/len(qrels))        
        print(len(entities_qrels))
        #print(entities)
        #print(entities['11867'])
        #print(len(entities['11867']))
    '''
    with open(in_path2,'r') as inp:
#        qrels = []
#        entities = {}
#        conll_recs = set()
#        entities_conll_recs = set()
        for line in inp:
            line = line.strip(' \n')
            tokens = line.split(' ')
            conll_rec = tokens[len(tokens)-1]
            conll_recs.add(conll_rec)
            ent_pair = tokens[len(tokens)-2].split('-')
            label = tokens[0]
            if len(ent_pair)!=2:
                print('Oops...unexpected entity pair')
                break
            entities_conll_recs.add(ent_pair[0]+'_'+conll_rec)
            if ent_pair[0] < ent_pair[1]:
                normalized_ent_pair = ent_pair[0]+'-'+ent_pair[1]
            else:
                normalized_ent_pair = ent_pair[1]+'-'+ent_pair[0]
            if True==True and label=='1':
                if ent_pair[0] in entities:
                    entities[ent_pair[0]].add(ent_pair[1])
                else:
                    entities[ent_pair[0]] = set(ent_pair[1])
            #if conll_rec=='1':
            #    outp1.write(line+os.linesep)
        print(len(entities))
        total = 0
        for ent in entities:
            total += len(entities[ent])
        print(1.0*total/len(entities))
        print(len(conll_recs))
        print(len(entities_conll_recs))
        #print(entities)
        #print(entities['11867'])
        #print(len(entities['11867']))
    '''

'''
print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model = gensim.models.Word2Vec.load(model_path)
print("model loaded "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
'''

print('Done!')
