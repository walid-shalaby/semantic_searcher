# -*- coding: utf-8 -*-
"""
Created on Tues Jan 3 22:33:40 2017

@author: wshalaby
"""

"""
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles_redirects_seealso.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles_redirects.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/titles_redirects_ids.csv"
titles_path = "/home/walid/work/github/semantic_searcher/titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/titles_redirects_ids.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids_anno.csv"

in_path = '/scratch/wshalaby/doc2vec/test.svm'
in_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.svm'
in_path = '/home/walid/work/github/semantic_searcher/test.svm'
in_path = '/home/wshalaby/work/github/dexter-relatedness/data/learning/trec/trec-ent/training.svm'
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
"""

import sys
import os
import gensim
import numpy as np

def getSimilarity(item): # sort out by similarity
    return item[2]


def getQid(item):
    return int(item[0].split(':')[1])


def main(in_path, model, outpath1, outpath2, mode, titles=None):    
    qrels = {}
    ent_pairs = set()
    missing_sources = set()
    missing_candidates = []
    ids = {}
    for k,v in titles.items():
        ids[v] = k
    with open(in_path,'r') as inp:
        for line in inp:
            line = line.strip(os.linesep)
            tokens = line.split(' ')
            qrel = tokens[1]
            pair = tokens[len(tokens)-2]
            pair_tokens = pair.split('-')
            source_ent = pair_tokens[0]
            candidate_ent = pair_tokens[1]
            if source_ent not in ids:
                missing_sources.add(source_ent)
                continue
            if candidate_ent not in ids:
                missing_candidates.append(candidate_ent)
                continue
            ent_pairs.add(pair)
            label = int(tokens[0])
            if label==1:
                sim = 1.0
                rank = 1
            else:
                sim = 0.0
                rank = 0
            if qrel not in qrels: # first time this qrel appears, create a new set to store its source-target pairs
                qrels[qrel] = [(pair,rank,sim)]
            else:
                qrels[qrel].append((pair,rank,sim)) # add source-target entities
        print('total entities:',len(qrels))
        total = 0
        for qrel in qrels:
            total += len(qrels[qrel])
        print('average candidates:',1.0*total/len(qrels))
        print('total unique entity pairs:',len(ent_pairs))
        print('total missing sources:',len(missing_sources))
        print('missing sources:',missing_sources)
        print('total missing candidates:',len(missing_candidates))
        # sort out candidates by similarity (ones with label 1 on top of those with label 0)
        qrels_list = [(qrel,sorted(list(qrels[qrel]), key=getSimilarity, reverse=True)) for qrel in qrels]
        # sort out by qid
        qrels_list = sorted(qrels_list, key=getQid)
        # write gold standard qrel file
        with open(outpath1, 'w') as outp1:
            for qrel_tuple in qrels_list:
                qid = qrel_tuple[0]
                for qrel in qrel_tuple[1]:
                    # each line formatted as: qid iter(0)   docno (source-candidate)      rank (1|0 depending on label)
                    outp1.write(qid+'\t0\t'+qrel[0]+'\t'+str(qrel[1])+os.linesep)
        # calculate ent-ent similarities
        ent_pairs_sims = {}
        for pair in ent_pairs:
            pair_tokens = pair.split('-')
            if mode=='w2v':
                source = 'id'+pair_tokens[0]+'di'
                candidate = 'id'+pair_tokens[1]+'di'
            elif mode=='conc':
                source = ids[pair_tokens[0]]
                candidate = ids[pair_tokens[1]]
            if source in model and candidate in model:
                ent_pairs_sims[pair] = model.similarity(source,candidate)#1-np.random.rand()
            #else:
            #    print('Opps...either source or candidate entities are missing', pair)
        # write gold standard qrel file
        with open(outpath2, 'w') as outp2:
            for qrel_tuple in qrels_list:
                qid = qrel_tuple[0]
                qrels = []
                # skip qrels with no relevant entities, i.e. all those that have only irrelevant entities (0 labels only)
                count = 0
                for gold_qrel in qrel_tuple[1]:
                    gold_rank = gold_qrel[1]
                    if gold_rank==1:
                        count += 1
                if count==0:
                    print('skipping source entity:',qid)
                    continue
                #elif qid=='qid:2121':
                #    print(qrel_tuple[1])
                for gold_qrel in qrel_tuple[1]: # list of candidates (source-candidate, gold_rank, gold_sim)
                    pair = gold_qrel[0]
                    if pair in ent_pairs_sims:
                        qrels.append((pair,0,ent_pairs_sims[pair]))
                qrels = sorted(qrels, key=getSimilarity, reverse=True) # sort out by distance asc (less distance higher rank).
                for i in range(len(qrels)):
                    qrels[i] = (qrels[i][0],i+1,qrels[i][2]) # update ranks according to similarities
                for qrel in qrels:
                    # each line formatted as: qid iter(0)   docno (source_candidate)      rank  sim   run_id (0)
                    outp2.write(qid+'\t0\t'+qrel[0]+'\t'+str(qrel[1])+'\t'+str(qrel[2])+'\t0'+os.linesep)


main(in_path, model, outpath1, outpath2, mode, titles)
in_path = sys.argv[1]
model_path = sys.argv[2]
outpath1 = sys.argv[3]
outpath2 = sys.argv[4]

#model = gensim.models.Word2Vec.load(model_path)
model = 10

mode = 'w2v'
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/titles_redirects_ids_anno.csv"
from commons import load_titles
to_keep = set()
titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
in_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.svm'
outpath1 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.trec'
outpath2 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.trec'
#total entities: 3308
#average candidates: 89.29897218863361
#total unique entity pairs: 217305
#total missing sources: 8
#missing sources: {'368881', '18642444', '66513', '842061', '1157753', '1724515', '353781', '21758160'}
#total missing candidates: 6392
#skipping source entity: qid:5165
#skipping source entity: qid:11109
#skipping source entity: qid:11111
#skipping source entity: qid:11112
#trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test-w2v.trec test.w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.trec 
#map                     all 0.5408
#ndcg_cut_1              all 0.6286
#ndcg_cut_5              all 0.5802
#ndcg_cut_10             all 0.6059

'''
=========================================
raw mincnt 5
model_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim500-wind5-cnt5-skipgram1.model'
in_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/test.svm'
outpath1 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim500-wind5-cnt5-skipgram1.out1.trec'
outpath2 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim500-wind5-cnt5-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'18642444', '353781', '1724515', '842061', '1157753', '66513', '368881'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
map                     all     0.5322
ndcg_cut_1              all     0.6203
ndcg_cut_5              all     0.5698
ndcg_cut_10             all     0.5959
../trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 ../w2v-plain-anno-titles-10iter-dim500-wind5-cnt5-skipgram1.out1.trec ../w2v-plain-anno-titles-10iter-dim500-wind5-cnt5-skipgram1.out1.trec
===========================================
4.4m wind7
model_path = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind7-cnt1-skipgram1.model'
outpath1 = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind7-cnt1-skipgram1.out1.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind7-cnt1-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1724515', '66513', '353781', '368881', '18642444', '842061', '1157753'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 ../w2v-plain-anno-titles-4.4-10iter-dim500-wind7-cnt1-skipgram1.out1.trec ../w2v-plain-anno-titles-4.4-10iter-dim500-wind7-cnt1-skipgram1.out2.trec
map                     all     0.5460
ndcg_cut_1              all     0.6280
ndcg_cut_5              all     0.5837
ndcg_cut_10             all     0.6095
===========================================
4.4m wind9
model_path = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.model'
outpath1 = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.out1.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1724515', '842061', '18642444', '1157753', '353781', '66513', '368881'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 ../w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.out1.trec ../w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.out2.trec
map                     all     0.5497 (BEST)
ndcg_cut_1              all     0.6280 (BEST)
ndcg_cut_5              all     0.5873 (BEST)
ndcg_cut_10             all     0.6147 (BEST)
===========================================
4.4m + concepts wind5
model_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-4.4-concepts-20160820-no-redirects-dic-10iter-dim500-wind7-cnt1-skipgram1.model'
outpath1 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-4.4-concepts-20160820-no-redirects-dic-10iter-dim500-wind7-cnt1-skipgram1.out1.trec'
outpath2 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-4.4-concepts-20160820-no-redirects-dic-10iter-dim500-wind7-cnt1-skipgram1.out2.trec'
total entities: 3308
average candidates: 89.29897218863361
total unique entity pairs: 217305
total missing sources: 8
missing sources: {'21758160', '842061', '353781', '66513', '18642444', '1724515', '1157753', '368881'}
total missing candidates: 6392
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 ../w2v-plain-anno-titles-raw-10iter-dim500-wind5-cnt1-skipgram1.out1.trec ../w2v-plain-anno-titles-raw-10iter-dim500-wind5-cnt1-skipgram1.out2.trec
map                     all     0.5397
ndcg_cut_1              all     0.6281
ndcg_cut_5              all     0.5775
ndcg_cut_10             all     0.6032
===========================================

raw wind5
model_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-raw-10iter-dim500-wind5-cnt1-skipgram1.model'
outpath1 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-raw-10iter-dim500-wind5-cnt1-skipgram1.out1.trec'
outpath2 = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-raw-10iter-dim500-wind5-cnt1-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'18642444', '353781', '1724515', '842061', '1157753', '66513', '368881'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 ../w2v-plain-anno-titles-raw-10iter-dim500-wind5-cnt1-skipgram1.out1.trec ../w2v-plain-anno-titles-raw-10iter-dim500-wind5-cnt1-skipgram1.out2.trec
map                     all     0.5397
ndcg_cut_1              all     0.6281
ndcg_cut_5              all     0.5775
ndcg_cut_10             all     0.6032
===========================================
4.4m iter1 wind5
model_path = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-1iter-dim500-wind5-skipgram1.model'
outpath1 = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-1iter-dim500-wind5-skipgram1.out1.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-1iter-dim500-wind5-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1157753', '842061', '353781', '368881', '66513', '1724515', '18642444'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 ../w2v-plain-anno-titles-1iter-dim500-wind5-skipgram1.out1.trec ../w2v-plain-anno-titles-1iter-dim500-wind5-skipgram1.out2.trec
map                     all     0.4689
ndcg_cut_1              all     0.5738
ndcg_cut_5              all     0.5101
ndcg_cut_10             all     0.5264
=========================================
'''
mode = 'conc'
titles_path = "/home/walid/work/github/semantic_searcher/titles_redirects_ids.csv"
from commons import load_titles
to_keep = set()
titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
in_path = '/home/walid/work/github/semantic_searcher/test.svm'
outpath1 = '/home/walid/work/github/semantic_searcher/test.trec'
outpath1 = '/home/walid/work/github/semantic_searcher/test.trec.self'
outpath1 = '/home/walid/work/github/semantic_searcher/test.10iter.new.trec'
outpath1 = '/home/walid/work/github/semantic_searcher/test.1iter.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/test.concepts-10iter-dim500-wind5-skipgram1.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/test.w2v-plain-anno-titles-1iter-dim200-wind5-skipgram1.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/test.w2v-plain-anno-titles-10iter-new-dim500-wind5-skipgram1.trec'
outpath2 = '/home/walid/work/github/semantic_searcher/test.w2v-plain-anno-titles-10iter-new-dim500-wind5-skipgram1.trec.self'
#trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test-conc.trec test.concepts-10iter-dim500-wind5-skipgram1.trec
#map                     all 0.4465
#ndcg_cut_1              all 0.5032
#ndcg_cut_5              all 0.4772
#ndcg_cut_10             all 0.5027

'''
=========================================
self-dic-sentence-per-article
/home/walid/work/github/semantic_searcher/concepts-self-dic-10iter-dim500-wind3-skipgram1.model
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1724515', '1157753', '66513', '368881', '18642444', '842061', '353781'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval  -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind3-skipgram1.out1.trec test.concepts-self-dic-10iter-dim500-wind3-skipgram1.out2.trec
map                     all     0.4309
ndcg_cut_1              all     0.4799
ndcg_cut_5              all     0.4568
ndcg_cut_10             all     0.4849

/home/walid/work/github/semantic_searcher/concepts-self-dic-10iter-dim500-wind4-skipgram1.model
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1724515', '1157753', '66513', '368881', '18642444', '842061', '353781'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval  -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind4-skipgram1.out1.trec test.concepts-self-dic-10iter-dim500-wind4-skipgram1.out2.trec
map                     all     0.4386
ndcg_cut_1              all     0.4929
ndcg_cut_5              all     0.4679
ndcg_cut_10             all     0.4934

/home/walid/work/github/semantic_searcher/concepts-self-dic-10iter-dim500-wind6-skipgram1.model
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1724515', '1157753', '66513', '368881', '18642444', '842061', '353781'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
../trec_eval.9.0/trec_eval  -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind6-skipgram1.out1.trec test.concepts-self-dic-10iter-dim500-wind6-skipgram1.out2.trec                                                                                                                                     
map                     all     0.4534
ndcg_cut_1              all     0.5210
ndcg_cut_5              all     0.4848
ndcg_cut_10             all     0.5094

/home/walid/work/github/semantic_searcher/concepts-self-dic-10iter-dim500-wind7-skipgram1.model
/scratch/wshalaby/doc2vec/concepts-self-dic-sentence-per-article/concepts-self-dic-10iter-dim500-wind7-skipgram1.model
outpath1 = 'concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind7-skipgram1.out1.trec'
outpath2 = 'concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind7-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1724515', '1157753', '66513', '368881', '18642444', '842061', '353781'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind7-skipgram1.out1.trec concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind7-skipgram1.out2.trec
map                     all     0.4604
ndcg_cut_1              all     0.5256
ndcg_cut_5              all     0.4938
ndcg_cut_10             all     0.5187

/scratch/wshalaby/doc2vec/concepts-self-dic-sentence-per-article/concepts-self-dic-10iter-dim500-wind9-skipgram1.model
outpath1 = 'concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind9-skipgram1.out1.trec'
outpath2 = 'concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind9-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'66513', '842061', '1724515', '368881', '18642444', '353781', '1157753'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind9-skipgram1.out1.trec concepts-self-dic-sentence-per-article/test.concepts-self-dic-10iter-dim500-wind9-skipgram1.out2.trec
map                     all 0.4635 (BEST)
ndcg_cut_1              all 0.5349 (BEST)
ndcg_cut_5              all 0.4956 (BEST)
ndcg_cut_10             all 0.5217 (BEST)

/scratch/wshalaby/doc2vec/models/word2vec/w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.model
outpath1 = 'w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.out1.trec'
outpath2 = 'w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.out2.trec'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'368881', '1157753', '842061', '353781', '1724515', '18642444', '66513'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.out1.trec w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.out2.trec
map                     all 0.4629 (BEST)
ndcg_cut_1              all 0.5339 (BEST)
ndcg_cut_5              all 0.4959 (BEST)
ndcg_cut_10             all 0.5205 (BEST)

self-dic-sentence-per-line

model_path = '/scratch/wshalaby/doc2vec/concepts-self-dic-10iter-dim500-wind4-skipgram1.model'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'353781', '1724515', '66513', '368881', '842061', '18642444', '1157753'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind4-skipgram1.sentence-per-line.out1.trec test.concepts-self-dic-10iter-dim500-wind4-skipgram1.sentence-per-line.out2.trec
map                     all 0.4383
ndcg_cut_1              all 0.4893
ndcg_cut_5              all 0.4653
ndcg_cut_10             all 0.4923

model_path = '/scratch/wshalaby/doc2vec/concepts-self-dic-10iter-dim500-wind5-skipgram1.model'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'842061', '66513', '1157753', '18642444', '353781', '1724515', '368881'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind5-skipgram1.sentence-per-line.out1.trec test.concepts-self-dic-10iter-dim500-wind5-skipgram1.sentence-per-line.out2.trec
map                     all 0.4398
ndcg_cut_1              all 0.4874
ndcg_cut_5              all 0.4674
ndcg_cut_10             all 0.4943

model_path = '/scratch/wshalaby/doc2vec/concepts-self-dic-10iter-dim500-wind6-skipgram1.model'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'353781', '1724515', '66513', '368881', '842061', '18642444', '1157753'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind6-skipgram1.sentence-per-line.out1.trec test.concepts-self-dic-10iter-dim500-wind6-skipgram1.sentence-per-line.out2.trec
map                     all 0.4426
ndcg_cut_1              all 0.4950
ndcg_cut_5              all 0.4722
ndcg_cut_10             all 0.4970

model_path = '/scratch/wshalaby/doc2vec/concepts-self-dic-10iter-dim500-wind7-skipgram1.model'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'353781', '842061', '368881', '1724515', '18642444', '66513', '1157753'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind7-skipgram1.sentence-per-line.out1.trec test.concepts-self-dic-10iter-dim500-wind7-skipgram1.sentence-per-line.out2.trec
map                     all 0.4425
ndcg_cut_1              all 0.4893
ndcg_cut_5              all 0.4705
ndcg_cut_10             all 0.4969

model_path = '/scratch/wshalaby/doc2vec/concepts-self-dic-10iter-dim500-wind8-skipgram1.model'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1157753', '842061', '353781', '18642444', '1724515', '368881', '66513'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind8-skipgram1.sentence-per-line.out1.trec test.concepts-self-dic-10iter-dim500-wind8-skipgram1.sentence-per-line.out2.trec
map                     all 0.4448
ndcg_cut_1              all 0.4980
ndcg_cut_5              all 0.4735
ndcg_cut_10             all 0.4993

model_path = '/scratch/wshalaby/doc2vec/concepts-self-dic-10iter-dim500-wind9-skipgram1.model'
total entities: 3309
average candidates: 89.78210939860985
total unique entity pairs: 218607
total missing sources: 7
missing sources: {'1157753', '842061', '353781', '18642444', '1724515', '368881', '66513'}
total missing candidates: 4777
skipping source entity: qid:5165
skipping source entity: qid:11109
skipping source entity: qid:11111
skipping source entity: qid:11112
trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test.concepts-self-dic-10iter-dim500-wind9-skipgram1.sentence-per-line.out1.trec test.concepts-self-dic-10iter-dim500-wind9-skipgram1.sentence-per-line.out2.trec
map                     all 0.4451
ndcg_cut_1              all 0.4923
ndcg_cut_5              all 0.4731
ndcg_cut_10             all 0.5008
=========================================
'''
#trec_eval.9.0/trec_eval -m map -m ndcg_cut.1,5,10 test.10iter.new.trec /home/walid/work/github/semantic_searcher/test.w2v-plain-anno-titles-10iter-new-dim500-wind5-skipgram1.trec
#total entities: 3309
#average candidates: 89.78210939860985
#total unique entity pairs: 218607
#total missing sources: 7
#missing sources: {'1157753', '368881', '66513', '842061', '1724515', '353781', '18642444'}
#total missing candidates: 4777
#skipping source entity: qid:5165
#skipping source entity: qid:11109
#skipping source entity: qid:11111
#skipping source entity: qid:11112
#map                     all 0.5384
#ndcg_cut_1              all 0.6278
#ndcg_cut_5              all 0.5757
#ndcg_cut_10             all 0.6025

main(in_path, model, outpath1, outpath2, mode, titles)

titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"
mode = 'conc'
from commons import load_titles
to_keep = set()
titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
model_path = 'concepts-self-dic-10iter-dim500-wind7-skipgram1.model'
model = gensim.models.Word2Vec.load(model_path)
model.init_sims(replace=True)
outpath1 = 'test.concepts-self-dic-10iter-dim500-wind8-skipgram1.sentence-per-line.out1.trec'
outpath2 = 'test.concepts-self-dic-10iter-dim500-wind8-skipgram1.sentence-per-line.out2.trec'



#=========================================================================
# convert ids into titles
import os
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"
from commons import load_titles
to_keep = set()
titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
inpath = 'w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.out1.trec'
outpath = 'w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.out1.titles.trec'
ids = {}
for k,v in titles.items():
    ids[v] = k
with open(outpath, 'w') as output:
    with open(inpath,'r') as input:
        for line in input:
            tokens = line.strip('\n').split('\t')
            id12 = tokens[2].split('-')
            tokens[2] = ids[id12[0]].replace(' ','_')+'||'+ids[id12[1]].replace(' ','_')
            output.write('\t'.join(tokens)+os.linesep)

#=========================================================================