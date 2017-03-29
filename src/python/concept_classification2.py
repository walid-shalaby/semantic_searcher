# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

#python3 concept_classification.py >& samples-out-concepts-auto.motor-anno-allnorm-0.0.txt.out
#python3 concept_classification3.py >& samples-out-concepts-auto.motor-conc-matchnorm-0.0.txt.out
#python3 concept_classification2.py >& samples-out-concepts-auto.motor-anno-matchnorm-0.0.txt.out
#python3 concept_classification1.py >& samples-out-concepts-auto.motor-conc-allnorm-0.0.txt.out

#python3 concept_classification.py >& samples-out-concepts-auto.elec-anno-allnorm-0.0.txt.out
#python3 concept_classification3.py >& samples-out-concepts-auto.elec-conc-matchnorm-0.0.txt.out
#python3 concept_classification2.py >& samples-out-concepts-auto.elec-anno-matchnorm-0.0.txt.out
#python3 concept_classification1.py >& samples-out-concepts-auto.elec-conc-allnorm-0.0.txt.out

#pyspark --driver-memory 13g --executor-memory 13g --conf "spark.kryoserializer.buffer.max = 512m"

#spark-submit concept_classification.py --driver-memory 32g --executor-memory 32g --conf "spark.kryoserializer.buffer.max = 1024m"

mode = 'spark_concepts'
titles_to_loadBC = None
loaded_vecsBC = None
topics_dicBC = None
topicsBC = None
redirectsBC = None
titlesBC = None

if mode!='spark_concepts' and mode!='spark_anno' :
    import gensim
    from joblib import Parallel, delayed
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    import _compat_pickle
    _compat_pickle.IMPORT_MAPPING.update({
        'UserDict': 'collections',
        'UserList': 'collections',
        'UserString': 'collections',
        'whichdb': 'dbm',
        'StringIO':  'io',
        'cStringIO': 'io',
    })

else:
    from functools import partial

import os
from sys import getsizeof
import csv
from time import gmtime, strftime
import numpy as np
from scipy import spatial
from commons import load_titles
import multiprocessing
from multiprocessing import Pool
from skipgram_esa_mappings import skipgram_esa_mappings_dic
#print(skipgram_esa_mappings_dic)


titles = {}
redirects = {}
seealso = {}

titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles_redirects_seealso.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles_redirects.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects.csv"
titles_path = "/home/walid/work/github/semantic_searcher/titles_redirects_ids.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"

keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"
keeponly_path = ""

model_path = "/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/home/wshalaby/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"
model_path = "/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"
model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/home/walid/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"

topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups/representation/tree.20newsgroups.simple.esa.concepts.newrefine.500"
topics_inpath = "/home/wshalaby/work/github/semantic_searcher/src/python/tree.20newsgroups.simple.esa.concepts.newrefine.500"
topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.500.leaves"
topics_inpath = "/home/wshalaby/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.500.autos.elctronics.leaves"
topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.500.autos.elctronics.leaves"
topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.500.autos.elctronics.leaves"
topics_inpath = "/home/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.500.autos.elctronics.leaves"
topics_inpath = "/home/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.500.autos.motors.leaves"
topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.500.autos.motors.leaves"
topics_inpath = "/home/wshalaby/work/github/semantic_searcher/src/python/tree.20newsgroups.simple.esa.concepts.newrefine.500.autos.motors.leaves"
topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.500.autos.motors.leaves"
topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"
topics_inpath = "/home/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"
topics_inpath = "/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"
topics_inpath = "/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.2recs"
topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"
topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.2recs"

samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples.txt"
samples_inpath = "/home/wshalaby/work/github/semantic_searcher/src/python/samples.txt"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups/representation/20newsgroups.simple.esa.concepts.500.auto.elec.8"
samples_inpath = "/home/wshalaby/work/github/semantic_searcher/src/python/samples.motor.autos.txt"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups/representation/20newsgroups.simple.esa.concepts.500.auto.elec.103037"
samples_inpath = "/home/wshalaby/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.auto.elec"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups/representation/20newsgroups.simple.esa.concepts.500.auto.elec"
samples_inpath = "/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.auto.elec"
samples_inpath = "/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.auto.motor"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups/representation/20newsgroups.simple.esa.concepts.500.auto.motor"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups/representation/20newsgroups.simple.esa.concepts.500"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/xab"
samples_inpath = "/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500"
samples_inpath = "/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.10recs"
samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500"
samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.10recs"

output_path = "/home/walid/work/github/semantic_searcher/python/missing_all.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/20newsgroups/representation/20newsgroups.simple.esa.concepts.500-out"
output_path = "/home/wshalaby/work/github/semantic_searcher/src/python/samples.txt"
output_path = "/home/wshalaby/work/github/semantic_searcher/src/python/samples-out.txt"
output_path = "/home/wshalaby/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-conc.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/missing.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-anno.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-cosine-auto.elec.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-conc.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-anno-allnorm-0.85.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-conc-matchnorm-0.85.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-anno-matchnorm-0.85.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-conc-allnorm-0.85.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-anno-allnorm-0.85.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-conc-matchnorm-0.85.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-anno-matchnorm-0.85.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-conc-allnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-cosine-auto.motor.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-anno-allnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-conc-matchnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-anno-matchnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.elec-conc-allnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-anno-allnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-conc-matchnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-anno-matchnorm-0.85.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-auto.motor-conc-allnorm-0.85.txt"

output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-conc-allnorm-0.0.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-conc-allnorm-0.0-xab.txt"

output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-cosine.txt"

vecs_path = "/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter.vecs.txt"
vecs_path = "/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.vecs.txt"
vecs_path = "/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter-dim500-wind5-skipgram1.vecs.txt"
vecs_path = "concepts-10iter-dim500-wind5-skipgram1.vecs.txt"

wiki_esa_mapping_path = "/scratch/wshalaby/doc2vec/wiki_esa_ids.tsv"
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = "/scratch/wshalaby/doc2vec/wikipedia-2016.500.30out.lst"

output_path_hdfs = 'samples-out-concepts-conc-allnorm-0.0.txt'

topics_dic = None
model = None
fout = None
y = []
y_pred = []
y_top = []
y_top_pred = []
topics = {}

def load_vecs(filename, format):
    import pickle
    if format=='raw':
        f = open(filename, "rb")
    elif format=='txt':
        f = open(filename, "r")
    while True:
        try:
            if format=='raw':
                yield pickle.load(f)
            elif format=='txt':
                tokens = f.readline().split('\t')
                if len(tokens)==2:
                    yield ((tokens[0],[float(x) for x in tokens[1].split(',')]))
                else:
                    print('Oops! Malformed line: ' + str(tokens))
                    break
        except EOFError:
            break

def check_vector(line):
    global titles_to_loadBC
    tokens = line.split('\t')
    if len(tokens)==2 and tokens[0] in titles_to_loadBC.value.keys():
        return [(tokens[0],tokens[1])]
    else:
        return [None]

def resolve_esa_concepts(line):
    #global titlesBC, redirectsBC, skipgram_esa_mappings_dicBC
    topic_name = line[0:line.find('\t')]
    topic_vector = line[line.rfind('\t')+1:]        
    if topic_vector.find(';')==-1: # none line e.g., sample 53718
        return [None]
    titles_to_load = []
    for concept in topic_vector.split(';'):
        if len(concept)>0:
            org_concept_name = concept[0:concept.find(',')]
            try:
                concept_name = skipgram_esa_mappings_dicBC.value[concept_name]                
            except:
                concept_name = org_concept_name                
            
            # look up concept id
            try:
                if mode=='spark_anno': # anno titles
                    concept_id = titlesBC.value[concept_name]
                else:
                    if concept_name in titlesBC.value.keys():
                        concept_id = concept_name
                    else: # may be a redirect
                        concept_id = redirectsBC.value[concept_name]
                titles_to_load.append((org_concept_name,concept_id))
            except:
                #print(concept_name+'...not found')
                titles_to_load.append((org_concept_name,'NOT_-_FOUND'))
                continue
    return titles_to_load

def get_max_similarity1(line):
    global skipgram_esa_mappings_dic
    print(len(skipgram_esa_mappings_dic))
    sample_info = SampleInfo()
    sample_info.sample_name = line
    print(line)
    return sample_info

def get_max_similarity2(line):
    global model, titles, redirects, topics_dic, topics, skipgram_esa_mappings_dic
    sample_info = SampleInfo()

    if line.find('#')==0: # none line e.g., sample 53718 if commented it out
        return sample_info
    
    sample = line[0:line.find('\t')].split('\\')
    sample_info.sample_name = sample[len(sample)-1]
    #if sample_name=='53718':
    #    process = True
    #    continue
    #if process==False:
    #    continue
    sample_info.sample_class = sample[len(sample)-2]
    sample_vector = line[line.rfind('\t')+1:]
    accum_vector = []
    if sample_vector.find(';')==-1: # none line e.g., sample 53718
        print(line)
        return sample_info
    #max_concept_weight = 0.0
    #print(line)
    for concept in sample_vector.split(';'):
        if len(concept)>0:
            concept_name = concept[0:concept.find(',')]
            concept_weight = float(concept[concept.rfind(',')+1:])            
   
def get_max_similarity_spark(line):
    #global titles_to_loadBC, loaded_vecsBC, topics_dicBC, topicsBC
    result = ''

    if line.find('#')==0: # none line e.g., sample 53718 if commented it out
        return [None]

    sample_info = SampleInfo()

    sample = line[0:line.find('\t')].split('\\')
    sample_info.sample_name = sample[len(sample)-1]
    sample_info.sample_class = sample[len(sample)-2]
    sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
    sample_vector = line[line.rfind('\t')+1:]
    accum_vector = []
    if sample_vector.find(';')==-1: # none line e.g., sample 53718
        return [None]
    #max_concept_weight = 0.0
    #print(line)
    for concept in sample_vector.split(';'):
        if len(concept)>0:
            concept_name = concept[0:concept.find(',')]
            concept_weight = float(concept[concept.rfind(',')+1:])
            #if concept_weight>max_concept_weight:
            #    max_concept_weight = concept_weight
            # look up concept id
            
            if 1!=1:#model.vector_size==200: # anno titles
                concept_id = titles_to_loadBC.value[concept_name]
            else:
                concept_id = titles_to_loadBC.value[concept_name]
            if concept_id != 'NOT_-_FOUND':
                # accumulate weighted concept vector
                vect = np.array(loaded_vecsBC.value[concept_id])
                accum_vector.append((vect,concept_weight,concept_name))#np.array(model['12806'])
            else:
                result += sample_info.sample_name+'\t'+concept_name+'...not found'+os.linesep
                accum_vector.append((None,concept_weight,concept_name))
    
    #for (concept_vector,weight,concept_name) in accum_vector:
    #    print(concept_name,concept_vector==None)
    #print(len(accum_vector))
    # normalize weights
    #accum_vector = [(concept_vector,weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]

    # calculate sample norm
    sample_norm = np.linalg.norm([weight for (concept_vector,weight,concept_name) in accum_vector])
    
    result += 'SAMPLE_NORM\t'+sample_info.sample_name+'\t'+str(sample_norm)+os.linesep

    sample_info.max_all_sim = 0.0
    sample_info.max_topic_name = ''
    sample_info.max_topic_id = None
        
    for k,v in topicsBC.value.items(): # loop on topics
        #print("testing topic: ("+k+")")
        cur_sim = 0.0
        topic_norm = []
        all_topic_norm = v[1]
        topic_accum_vector = v[0]
        for (sample_concept_vector,sample_weight,concept_name) in accum_vector: # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            max_sim = 0.0
            max_sim_weight = 0.0
            max_topic_name = ''
            for (topic_concept_vector,topic_weight,topic_name) in topic_accum_vector: # v is topic accum_vector in the format (vector,weight)
                #if concept_name=="Pimp My Ride" and topic_name=="Pimp My Ride":
                #    print("testing topic concept: ("+topic_name+")")
                if topic_name==concept_name:
                    max_sim = 1.0
                    max_sim_weight = topic_weight
                    result += sample_info.sample_name+'\t'+concept_name+'...exact match'+os.linesep
                    max_topic_name = topic_name
                    break
                elif sample_concept_vector is not None and topic_concept_vector is not None:
                    sim = 1 - spatial.distance.cosine(topic_concept_vector,sample_concept_vector)
                else:
                    sim = -10.0
                if sim>max_sim:
                    max_sim = sim
                    max_sim_weight = topic_weight
                    max_topic_name = topic_name
            #print(concept_name, max_sim_weight, max_sim)
            if max_sim>=-100.0:#0.85:#-100.0:#
                topic_norm.append(max_sim_weight)
                cur_sim += max_sim_weight*sample_weight*max_sim
                result += 'MAXIMA\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+max_topic_name+'\t'+str(max_sim_weight)+'\t'+str(max_sim)+os.linesep

        if len(topic_norm)>0:
            #cur_sim = cur_sim/(sample_norm*np.linalg.norm(topic_norm))
            cur_sim = cur_sim/(sample_norm*all_topic_norm)

        result += 'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(cur_sim)+os.linesep
        if cur_sim>sample_info.max_all_sim:
            sample_info.max_all_sim = cur_sim
            sample_info.max_topic_name = k
            sample_info.max_topic_id = topics_dicBC.value[k]

    result += sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
    return result

def get_max_similarity(line):
    global model, titles, redirects, topics_dic, topics, skipgram_esa_mappings_dic
    sample_info = SampleInfo()

    #if line.find('\\103227')==-1:
    #    return sample_info
    #if line.find('\\103116')==-1:
    #    return sample_info
    #if line.find('\\103124')==-1:
    #    return sample_info
    #if line.find('\\103047')==-1 and line.find('\\103046')==-1 and line.find('\\103049')==-1 and line.find('\\103048')==-1 and line.find('\\103040')==-1:
    #    return sample_info

    if line.find('#')==0: # none line e.g., sample 53718 if commented it out
        return sample_info
    
    sample = line[0:line.find('\t')].split('\\')
    sample_info.sample_name = sample[len(sample)-1]
    #if sample_name=='53718':
    #    process = True
    #    continue
    #if process==False:
    #    continue
    sample_info.sample_class = sample[len(sample)-2]
    sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
    sample_vector = line[line.rfind('\t')+1:]
    accum_vector = []
    if sample_vector.find(';')==-1: # none line e.g., sample 53718
        return sample_info
    #max_concept_weight = 0.0
    #print(line)
    for concept in sample_vector.split(';'):
        if len(concept)>0:
            concept_name = concept[0:concept.find(',')]
            if concept_name in skipgram_esa_mappings_dic.keys():
                concept_name = skipgram_esa_mappings_dic[concept_name]

            concept_weight = float(concept[concept.rfind(',')+1:])
            #if concept_weight>max_concept_weight:
            #    max_concept_weight = concept_weight
            # look up concept id
            try:
                if model.vector_size==200: # anno titles
                    concept_id = titles[concept_name]
                else:
                    if concept_name in model.vocab:
                        concept_id = concept_name
                    else: # may be a redirect
                        concept_id = redirects[concept_name]
                # accumulate weighted concept vector
                vect = np.array(model[concept_id])
                accum_vector.append((vect,concept_weight,concept_name))#np.array(model['12806'])
            except:
                print(sample_info.sample_name+'\t'+concept_name+'...not found')
                accum_vector.append((None,concept_weight,concept_name))

    #for (concept_vector,weight,concept_name) in accum_vector:
    #    print(concept_name,concept_vector==None)
    #print(len(accum_vector))
    # normalize weights
    #accum_vector = [(concept_vector,weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]

    # calculate sample norm
    sample_norm = np.linalg.norm([weight for (concept_vector,weight,concept_name) in accum_vector])
    
    print(os.linesep+'SAMPLE_NORM\t'+sample_info.sample_name+'\t'+str(sample_norm)+os.linesep)

    sample_info.max_all_sim = 0.0
    sample_info.max_topic_name = ''
    sample_info.max_topic_id = None
        
    for k,v in topics.items(): # loop on topics
        #print("testing topic: ("+k+")")
        cur_sim = 0.0
        topic_norm = []
        all_topic_norm = v[1]
        topic_accum_vector = v[0]
        for (sample_concept_vector,sample_weight,concept_name) in accum_vector: # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            max_sim = 0.0
            max_sim_weight = 0.0
            max_topic_name = ''
            for (topic_concept_vector,topic_weight,topic_name) in topic_accum_vector: # v is topic accum_vector in the format (vector,weight)
                #if concept_name=="Pimp My Ride" and topic_name=="Pimp My Ride":
                #    print("testing topic concept: ("+topic_name+")")
                if topic_name==concept_name:
                    max_sim = 1.0
                    max_sim_weight = topic_weight
                    print(sample_info.sample_name+'\t'+concept_name+'...exact match')
                    max_topic_name = topic_name
                    break
                elif sample_concept_vector is not None and topic_concept_vector is not None:
                    sim = 1 - spatial.distance.cosine(topic_concept_vector,sample_concept_vector)
                else:
                    sim = -10.0
                if sim>max_sim:
                    max_sim = sim
                    max_sim_weight = topic_weight
                    max_topic_name = topic_name
            #print(concept_name, max_sim_weight, max_sim)
            if max_sim>=-100.0:#0.85:#-100.0:#
                topic_norm.append(max_sim_weight)
                cur_sim += max_sim_weight*sample_weight*max_sim
                print(os.linesep+'MAXIMA\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+max_topic_name+'\t'+str(max_sim_weight)+'\t'+str(max_sim)+os.linesep)

        if len(topic_norm)>0:
            #cur_sim = cur_sim/(sample_norm*np.linalg.norm(topic_norm))
            cur_sim = cur_sim/(sample_norm*all_topic_norm)

        print(os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(cur_sim)+os.linesep)
        if cur_sim>sample_info.max_all_sim:
            sample_info.max_all_sim = cur_sim
            sample_info.max_topic_name = k
            sample_info.max_topic_id = topics_dic[k]

    return sample_info

class SampleInfo:
    def __init__(self):
        self.max_topic_id = None
        self.max_topic_name = None
        self.sample_class = None
        self.sample_name = None
        self.max_all_sim = 0.0

def write_sims(samples_info):
    global y, y_pred, y_top, y_top_pred, topics_dic, top_topics, fout

    for sample_info in samples_info:
        if sample_info.max_topic_id==None:
            continue
        y.append(topics_dic[sample_info.sample_class])
        y_pred.append(sample_info.max_topic_id)            
        if sample_info.sample_class.find('atheism')!=-1 or sample_info.sample_class.find('religion')!=-1:
            y_top.append(top_topics['religion'])
        else:
            y_top.append(top_topics[sample_info.sample_class.split('.')[0]])
        if sample_info.max_topic_name.find('atheism')!=-1 or sample_info.max_topic_name.find('religion')!=-1:
            y_top_pred.append(top_topics['religion'])
        else:                
            y_top_pred.append(top_topics[sample_info.max_topic_name.split('.')[0]])
        
        fout.write(sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep)
        fout.flush()
        print('processed ('+sample_info.sample_name+')')

def main():
    global titles, redirects, topics_dic, model, fout, y, y_pred, y_top, y_top_pred, topics, top_topics, mode, titles_to_loadBC, loaded_vecsBC, topics_dicBC, topicsBC#, redirectsBC, titlesBC, skipgram_esa_mappings_dicBC

    print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    #mode = 'max_multi_thread'
    if mode!='cosine' and mode!='test' and mode!='check_file':
        to_keep = set()
        if keeponly_path!="":
            print("loading keey only")
            with open(keeponly_path) as keeponly_titles:
                for title in keeponly_titles:
                    to_keep.add(title.replace(os.linesep,""))
        print("loaded "+str(len(to_keep))+" keeponly titles")

        titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
        print("keeping "+str(len(titles))+" titles only")
        print("loaded "+str(len(redirects))+" redirects")

        if mode!='spark_concepts' and mode!='spark_anno' :
            print("loading model")
            model = gensim.models.Word2Vec.load(model_path)
            print("model loaded")

    if mode=='cosine':
        print("loading topics")
        topics = {}
        topics_dic = {}
        top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
        cur_topic = 1
        with open(topics_inpath,'r') as topics_file:
            for line in topics_file:
                line = line.strip('\n')
                if len(line)>0:
                    topic_name = line[0:line.find('\t')]
                    if topic_name in topics.keys():
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = {}
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            accum_vector[concept_name] = concept_weight
                    
                    topics[topic_name] = (accum_vector, np.linalg.norm(list(accum_vector.values())))
                    topics_dic[topic_name] = cur_topic
                    cur_topic = cur_topic + 1

        print("processing samples")
        fout = open(output_path,'a')
        y = []
        y_pred = []
        y_top = []
        y_top_pred = []
        with open(samples_inpath,'r') as samples_file:
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    sample = line[0:line.find('\t')].split('\\')
                    sample_name = sample[len(sample)-1]
                    sample_class = sample[len(sample)-2]
                    sample_name = sample_name+'_'+sample_class
                    print('processing ('+sample_name+')')
                    sample_vector = line[line.rfind('\t')+1:]
                    accum_vector = {}
                    for concept in sample_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            accum_vector[concept_name] = concept_weight
                    
                    max_sim = 0.0
                    max_topic_name = 'unknown.unknown'
                    max_topic_id = 0
                    for topic_name,topic_concepts_weights in topics.items():
                        topic_weights = topic_concepts_weights[0]
                        topic_norm = topic_concepts_weights[1]
                        sim = 0.0
                        for sample_concept,sample_concept_weight in accum_vector.items():
                            if sample_concept in topic_weights.keys():                                
                                #print(topic_name,topic_weights[sample_concept],sample_concept,sample_concept_weight)
                                sim += sample_concept_weight*topic_weights[sample_concept]

                        #print(topic_norm,np.linalg.norm(list(accum_vector.values())))
                        sim = sim / (topic_norm*np.linalg.norm(list(accum_vector.values())))

                        if sim>max_sim:
                            max_sim = sim
                            max_topic_name = topic_name
                            max_topic_id = topics_dic[topic_name]
                    
                    fout.write(sample_name+'\t'+sample_class+'\t'+str(topics_dic[sample_class])+'\t'+max_topic_name+'\t'+str(max_topic_id)+'\t'+str(max_sim)+os.linesep)

        fout.close()

    elif mode=='missing':
        print("loading topics")
        topics = {}
        topics_dic = {}
        missing = set()
        top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
        cur_topic = 1
        with open(topics_inpath,'r') as topics_file:
            for line in topics_file:
                line = line.strip('\n')
                if len(line)>0:
                    topic_name = line[0:line.find('\t')]
                    if topic_name in topics.keys():
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = {}
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    if concept_name not in titles.keys(): # may be a redirect
                                        concept_name = redirects[concept_name]
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]                            
                                # accumulate weighted concept vector
                            except:
                                if concept_name not in skipgram_esa_mappings_dic.keys():
                                    print(concept_name+'...not found')
                                    missing.add(topic_name+'\t'+concept_name)
                                continue

        print("processing samples")
        fout = open(output_path,'a')
        with open(samples_inpath,'r') as samples_file:
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    sample = line[0:line.find('\t')].split('\\')
                    sample_name = sample[len(sample)-1]
                    sample_class = sample[len(sample)-2]
                    sample_vector = line[line.rfind('\t')+1:]
                    for concept in sample_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    if concept_name not in titles.keys(): # may be a redirect
                                        concept_name = redirects[concept_name]
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]                            
                                # accumulate weighted concept vector
                            except:
                                if concept_name not in skipgram_esa_mappings_dic.keys():
                                    print(concept_name+'...not found')
                                    missing.add(concept_name)
                                continue

        for m in missing:
            fout.write(m+os.linesep)

        fout.close()

    elif mode=='mean':
        print("loading topics")
        topics = {}
        topics_dic = {}
        top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
        cur_topic = 1
        with open(topics_inpath,'r') as topics_file:
            for line in topics_file:
                line = line.strip('\n')
                if len(line)>0:
                    topic_name = line[0:line.find('\t')]
                    if topic_name in topics.keys():
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = np.array([0.0]*model.vector_size)
                    accum_weight = 0.0
                    count = 0
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    if concept_name not in titles.keys(): # may be a redirect
                                        concept_name = redirects[concept_name]
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]                            
                                # accumulate weighted concept vector
                                accum_vector += concept_weight*np.array(model[concept_id])#np.array(model['12806'])
                                accum_weight += concept_weight
                                count = count + 1                    
                            except:
                                print(concept_name+'...not found')
                                continue
                    if count>0:
                        accum_vector =  accum_vector/accum_weight # weighted average all concepts
                    topics[topic_name] = accum_vector
                    topics_dic[topic_name] = cur_topic
                    cur_topic = cur_topic + 1

        print("processing samples")
        fout = open(output_path,'a')
        y = []
        y_pred = []
        y_top = []
        y_top_pred = []
        with open(samples_inpath,'r') as samples_file:
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    sample = line[0:line.find('\t')].split('\\')
                    sample_name = sample[len(sample)-1]
                    sample_class = sample[len(sample)-2]
                    sample_vector = line[line.rfind('\t')+1:]
                    accum_vector = np.array([0.0]*model.vector_size)
                    accum_weight = 0.0
                    count = 0
                    for concept in sample_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    if concept_name not in titles.keys(): # may be a redirect
                                        concept_name = redirects[concept_name]
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]
                                # accumulate weighted concept vector
                                accum_vector += concept_weight*np.array(model[concept_id])#np.array(model['12806'])
                                accum_weight += concept_weight
                                count = count + 1                    
                            except:
                                #print(concept_name+'...not found')
                                continue
                    if count>0:
                        accum_vector =  accum_vector/accum_weight # average all concepts
                    max_sim = 0.0
                    max_topic_name = ''
                    max_topic_id = 0
                    for k,v in topics.items():
                        sim = 1 - spatial.distance.cosine(accum_vector,v)
                        if sim>max_sim:
                            max_sim = sim
                            max_topic_name = k
                            max_topic_id = topics_dic[k]
                    y.append(topics_dic[sample_class])
                    y_pred.append(max_topic_id)            
                    if sample_class.find('atheism')!=-1 or sample_class.find('religion')!=-1:
                        y_top.append(top_topics['religion'])
                    else:
                        y_top.append(top_topics[sample_class.split('.')[0]])
                    if max_topic_name.find('atheism')!=-1 or max_topic_name.find('religion')!=-1:
                        y_top_pred.append(top_topics['religion'])
                    else:                
                        y_top_pred.append(top_topics[max_topic_name.split('.')[0]])
                    
                    fout.write(sample_name+'\t'+sample_class+'\t'+str(topics_dic[sample_class])+'\t'+max_topic_name+'\t'+str(max_topic_id)+'\t'+str(max_sim)+os.linesep)

        fout.close()
        print(f1_score(y, y_pred, average='micro'))
        print(accuracy_score(y, y_pred))
        #print(f1_score(y_top, y_top_pred, average='micro'))

    elif mode=='max_single_thread':
        print("loading topics")
        topics = {}
        topics_dic = {}
        top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
        cur_topic = 1
        with open(topics_inpath,'r') as topics_file:
            for line in topics_file:
                line = line.strip('\n')
                if len(line)>0:
                    topic_name = line[0:line.find('\t')]
                    if topic_name in topics.keys():
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]
                                # accumulate weighted concept vector
                                accum_vector.append((np.array(model[concept_id]),concept_weight,concept_name))#np.array(model['12806'])
                            except:
                                print(concept_name+'...not found')
                                continue
                    # calculate topic norm
                    topic_norm =  np.linalg.norm([weight for (concept_vector,weight,concept_name) in accum_vector])
                    topics[topic_name] = (accum_vector, topic_norm)
                    topics_dic[topic_name] = cur_topic
                    cur_topic = cur_topic + 1

        print("processing samples")
        fout = open(output_path,'a')
        y = []
        y_pred = []
        y_top = []
        y_top_pred = []
        with open(samples_inpath,'r') as samples_file:
            #process = False
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    sample = line[0:line.find('\t')].split('\\')
                    sample_name = sample[len(sample)-1]
                    if sample_name=='53718':
                    #    process = True
                        continue
                    #if process==False:
                    #    continue
                    sample_class = sample[len(sample)-2]
                    sample_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    if sample_vector.find(';')==-1: # none line e.g., sample 53718
                        continue
                    for concept in sample_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]
                                # accumulate weighted concept vector
                                accum_vector.append((np.array(model[concept_id]),concept_weight))#np.array(model['12806'])
                            except:
                                #print(concept_name+'...not found')
                                continue
                    # calculate sample norm
                    sample_norm = np.linalg.norm([weight for (concept_vector,weight) in accum_vector])
                    
                    max_all_sim = 0.0
                    max_topic_name = ''
                    max_topic_id = 0
                        
                    for k,v in topics.items(): # loop on topics
                        cur_sim = 0.0
                        topic_norm = []#v[1]
                        topic_accum_vector = v[0]
                        for (sample_concept_vector,sample_weight) in accum_vector: # loop on each concept in sample
                            max_sim = 0.0
                            max_sim_weight = 0.0
                            for (topic_concept_vector,topic_weight) in topic_accum_vector: # v is topic accum_vector in the format (vector,weight)
                                sim = 1 - spatial.distance.cosine(topic_concept_vector,sample_concept_vector)
                                if sim>max_sim:
                                    max_sim = sim
                                    max_sim_weight = topic_weight                                
                            topic_norm.append(max_sim_weight)
                            cur_sim += max_sim_weight*sample_weight*max_sim

                        cur_sim = cur_sim/(sample_norm*np.linalg.norm(topic_norm))

                        if cur_sim>max_all_sim:
                            max_all_sim = cur_sim
                            max_topic_name = k
                            max_topic_id = topics_dic[k]

                    y.append(topics_dic[sample_class])
                    y_pred.append(max_topic_id)            
                    if sample_class.find('atheism')!=-1 or sample_class.find('religion')!=-1:
                        y_top.append(top_topics['religion'])
                    else:
                        y_top.append(top_topics[sample_class.split('.')[0]])
                    if max_topic_name.find('atheism')!=-1 or max_topic_name.find('religion')!=-1:
                        y_top_pred.append(top_topics['religion'])
                    else:                
                        y_top_pred.append(top_topics[max_topic_name.split('.')[0]])
                    
                    fout.write(sample_name+'\t'+sample_class+'\t'+str(topics_dic[sample_class])+'\t'+max_topic_name+'\t'+str(max_topic_id)+'\t'+str(max_all_sim)+os.linesep)
                    fout.flush()
                    print('processed ('+sample_name+')')

        fout.close()

        print(f1_score(y, y_pred, average='micro'))
        print(f1_score(y_top, y_top_pred, average='micro'))

    elif mode=='max_multi_thread':
        print("loading topics")
        topics = {}
        topics_dic = {}
        top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
        cur_topic = 1
        with open(topics_inpath,'r') as topics_file:
            for line in topics_file:
                line = line.strip('\n')
                if len(line)>0:
                    topic_name = line[0:line.find('\t')]
                    if topic_name in topics.keys():
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    max_concept_weight = 0.0
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            if concept_weight>max_concept_weight:
                                max_concept_weight = concept_weight
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]
                                # accumulate weighted concept vector
                                accum_vector.append((np.array(model[concept_id]),concept_weight,concept_name))#np.array(model['12806'])
                            except:
                                print(concept_name+'...not found')
                                accum_vector.append((None,concept_weight,concept_name))#np.array(model['12806'])
                                continue
                    
                    # normalize weights
                    #print('max',max_concept_weight)
                    #accum_vector = [(concept_vector,weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]

                    # calculate topic norm
                    topic_norm =  np.linalg.norm([weight for (concept_vector,weight,concept_name) in accum_vector])
                    print(os.linesep+'TOPIC_NORM\t'+topic_name+'\t'+str(topic_norm)+os.linesep)
                    topics[topic_name] = (accum_vector, topic_norm)
                    topics_dic[topic_name] = cur_topic
                    cur_topic = cur_topic + 1

        print("processing samples")
        fout = open(output_path,'a')
        y = []
        y_pred = []
        y_top = []
        y_top_pred = []
        with open(samples_inpath,'r') as samples_file:
            in_records = []
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    in_records.append(line)

                    if len(in_records)%(multiprocessing.cpu_count()-1)==0:
                        p = Pool(multiprocessing.cpu_count()-1)
                        samples_info = p.map(get_max_similarity, in_records)                            
                        p.close()
                        p.join()
                        write_sims(samples_info)
                        in_records = []

        if len(in_records)>0:
            p = Pool(multiprocessing.cpu_count()-1)
            samples_info = p.map(get_max_similarity, in_records)                
            p.close()
            p.join()
            write_sims(samples_info)

        fout.close()

        print(f1_score(y, y_pred, average='micro'))
        print(f1_score(y_top, y_top_pred, average='micro'))

    elif mode=='spark_concepts' or mode=='spark_anno':
        from pyspark import SparkContext
        sc = SparkContext(appName="concept classification")
        
        titles_to_load = {}
        loaded_vecs = {}

        print('broadcasting titles')
        titlesBC = sc.broadcast(titles)
        print('titles broadcasted')
        
        print('broadcasting redirects')
        redirectsBC = sc.broadcast(redirects)
        print('redirects broadcasted')

        print('broadcasting skipgram_esa_mappings_dic')
        skipgram_esa_mappings_dicBC = sc.broadcast(skipgram_esa_mappings_dic)
        print('skipgram_esa_mappings_dic broadcasted')
        
        print('resolving topic titles')
        topics = sc.textFile(topics_inpath_hdfs)
        topics_titles = topics.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        #for pair in topics_titles.collect():
        #    titles_to_load[pair[0]] = pair[1]

        #print('topic titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')
        print('resolving sample titles')
        samples = sc.textFile(samples_inpath_hdfs)
        samples_titles = samples.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        #samples_titles.union(topics_titles).distinct().saveAsTextFile('tmpooos1')
        for pair in samples_titles.union(topics_titles).distinct().collect():
            titles_to_load[pair[1]] = pair[0]
        print('sample titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')

        print('broadcasting titles_to_load')
        titles_to_loadBC = sc.broadcast(titles_to_load)
        print('titles_to_load broadcasted')
        
        print('loading vectors')
        vecs_data = sc.textFile(vecs_path)
        vecs = vecs_data.filter(lambda line: len(line)>1).flatMap(lambda line: check_vector(line)).filter(lambda line: line is not None).map(lambda pair: pair[0]+'\t'+pair[1])
        #for pair in vecs.collect():
        #    loaded_vecs[pair[0]] = pair[1]
        #print('vectors loaded')
        vecs.saveAsTextFile('concept_classification_tmp_vecs1.vecs.txt')
        res = os.system('hdfs dfs -text concept_classification_tmp_vecs1.vecs.txt/*>concept_classification_tmp_vecs.vecs.txt')
        if res!=0:
            print('ERROR getting vectors from HDFS')
        else:
            print('loading needed vectors')
            with open('concept_classification_tmp_vecs.vecs.txt') as vecs_file:
                for line in vecs_file:
                    line = line.strip('\n')
                    if len(line)>0:
                        tokens = line.strip('\t')
                        if len(tokens)!=2:
                            print('invalid line: '+line)
                        else:
                            loaded_vecs[tokens[0]] = np.array([float(x) for x in tokens[1].split(',')])

                    print('we need ('+str(len(loaded_vecs.keys()))+') vectors of size ('+str(getsizeof(loaded_vecs))+')')

                print('broadcasting vectors')
                loaded_vecsBC = sc.broadcast(loaded_vecs)
                print('vectors broadcasted')  

            print("loading topics")
            topics = {}
            topics_dic = {}
            top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
            cur_topic = 1
            with open(topics_inpath,'r') as topics_file:
                for line in topics_file:
                    line = line.strip('\n')
                    if len(line)>0:
                        topic_name = line[0:line.find('\t')]
                        if topic_name in topics.keys():
                            line = line[line.find('\t')+1:]
                            topic_name = line[0:line.find('\t')]
                        topic_vector = line[line.rfind('\t')+1:]
                        accum_vector = []
                        max_concept_weight = 0.0
                        for concept in topic_vector.split(';'):
                            if len(concept)>0:
                                concept_name = concept[0:concept.find(',')]
                                concept_weight = float(concept[concept.rfind(',')+1:])
                                if concept_weight>max_concept_weight:
                                    max_concept_weight = concept_weight
                                # look up concept id
                                
                                concept_id = titles_to_load[concept_name]
                                if concept_id != 'NOT_-_FOUND':
                                    # accumulate weighted concept vector
                                    accum_vector.append((np.array(vecs[concept_id]),concept_weight,concept_name))#np.array(model['12806'])
                                else:
                                    print(concept_name+'...not found')
                                    accum_vector.append((None,concept_weight,concept_name))#np.array(model['12806'])                                                    
                        
                        # normalize weights
                        #print('max',max_concept_weight)
                        #accum_vector = [(concept_vector,weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]

                        # calculate topic norm
                        topic_norm =  np.linalg.norm([weight for (concept_vector,weight,concept_name) in accum_vector])
                        print(os.linesep+'TOPIC_NORM\t'+topic_name+'\t'+str(topic_norm)+os.linesep)
                        topics[topic_name] = (accum_vector, topic_norm)
                        topics_dic[topic_name] = cur_topic
                        cur_topic = cur_topic + 1

            print('broadcasting topics_dic')
            topics_dicBC = sc.broadcast(topics_dic)
            print('topics_dic broadcasted')

            print('broadcasting topics')
            topicsBC = sc.broadcast(topics)
            print('topics broadcasted')      

            samples = sc.textFile(samples_inpath_hdfs)
            results = samples.flatMap(lambda x : get_max_similarity_spark(x)).filter(lambda line: line is not None)
            results.saveAsTextFile(output_path_hdfs)

        sc.stop()
        
        '''
        vecs_data = load_vecs(vecs_path, 'txt')
        for i,pair in enumerate(vecs_data):
        #if i>10:
        #    break
            if(pair[0] in titles_to_load.keys()):
                vecs[pair[0]] = pair[1]
        print('vectors loaded')
        
        with open(topics_inpath,'r') as topics_file:
            for line in topics_file:
                line = line.strip('\n')
                if len(line)>0:
                    topic_name = line[0:line.find('\t')]
                    if topic_name in topics.keys():
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            # look up concept id
                            try:
                                if mode=='spark_anno': # anno titles
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in titles.keys():
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]
                                titles_to_load.add(concept_id)
                            except:
                                print(concept_name+'...not found')
                                continue                    
        print('topic titles resolved')
        print('resolving sample titles')
        with open(samples_inpath,'r') as samples_file:
            in_records = []
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    sample_vector = line[line.rfind('\t')+1:]
                    sample_name = sample[len(sample)-1]
                    sample_class = sample[len(sample)-2]
                    sample_name = sample_name+'_'+sample_class
                    
                    if sample_vector.find(';')==-1: # none line e.g., sample 53718
                        continue
                    for concept in sample_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic.keys():
                                concept_name = skipgram_esa_mappings_dic[concept_name]

                            # look up concept id
                            try:
                                if mode=='spark_anno': # anno titles
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in titles.keys():
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]
                                titles_to_load.add(concept_id)
                            except:
                                print(sample_name+'\t'+concept_name+'...not found')
                    
        print('sample titles resolved')
        
        print('broadcasting vectors')
        vecsBC = sc.broadcast(vecs)
        print('vectors broadcasted')

        
        print("processing samples")
        fout = open(output_path,'a')
        y = []
        y_pred = []
        y_top = []
        y_top_pred = []
        with open(samples_inpath,'r') as samples_file:
            in_records = []
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    in_records.append(line)

                    if len(in_records)%(multiprocessing.cpu_count()-1)==0:
                        p = Pool(multiprocessing.cpu_count()-1)
                        samples_info = p.map(get_max_similarity, in_records)                            
                        p.close()
                        p.join()
                        write_sims(samples_info)
                        in_records = []

        if len(in_records)>0:
            p = Pool(multiprocessing.cpu_count()-1)
            samples_info = p.map(get_max_similarity, in_records)                
            p.close()
            p.join()
            write_sims(samples_info)

        fout.close()
    '''                
    elif mode=='test':
        p = Pool(12)#multiprocessing.cpu_count()-1)
        samples_info = p.map(get_max_similarity1, [str(i) for i in range(0,10)])
        for s in samples_info:
            print('-'+s.sample_name)
        p.close()
        p.join()
    elif mode=='check_file':
        print("processing samples")
        with open(samples_inpath,'r') as samples_file:
            in_records = []
            count = 0
            for line in samples_file:
                line = line.strip('\n')
                count = count+1
                if len(line)>0:
                    print(count)                
                    in_records.append(line)

                    if len(in_records)%50==0:
                        p = Pool(multiprocessing.cpu_count()-2)
                        p.map(get_max_similarity2, in_records)                            
                        p.close()
                        p.join()
                        in_records = []

        if len(in_records)>0:
            p = Pool(multiprocessing.cpu_count()-2)
            p.map(get_max_similarity2, in_records)                
            p.close()
            p.join()

main()
#model.similar_by_vector(topics['computer'])
#1 - spatial.distance.cosine(topics['computer'], model['i8086'])
#concepts
#0.871595330739
#0.8845
#titles
#0.887452880991
#0.8955
