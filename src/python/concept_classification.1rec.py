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

#pyspark --driver-memory 13g --executor-memory 13g --conf "spark.kryoserializer.buffer.max = 1024m"

#spark-submit --driver-memory 13g --executor-memory 13g --num-executors 50 --conf spark.kryoserializer.buffer.max=1024m concept_classification.py

#spark_concepts
#spark-submit --driver-memory 13g --executor-memory 13g --conf spark.driver.maxResultSize=4g --conf spark.kryoserializer.buffer.max=1024m concept_classification.py

#spark_resolve_concepts
#spark-submit --driver-memory 13g --executor-memory 13g concept_classification.py

#hungarian_spark_resolved_cosines
#spark-submit --driver-memory 13g --executor-memory 13g --num-executors 200 --py-files commons.py,munkres.py concept_classification.py
#nohup spark-submit --driver-memory 13g --executor-memory 13g --py-files commons.py,munkres.py concept_classification.py >& hungarian_spark_resolved_cosines.out &
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.elecs.txt|grep -P 'rec.autos\t2\trec.autos\t2'|wc -l
#903
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.elecs.txt|grep -P 'rec.autos\t2\tsci.electronics\t1'|wc -l
#85
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.elecs.txt|grep -P 'sci.electronics\t1\tsci.electronics\t1'|wc -l
#958
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.elecs.txt|grep -P 'sci.electronics\t1\trec.autos\t2'|wc -l
#36
#(903+958)/((903+85)+(958+36))
#0.9389505549949546
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.motors.txt|grep -P 'rec.autos\t2\trec.autos\t2'|wc -l
#975
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.motors.txt|grep -P 'rec.autos\t2\trec.motorcycles\t1'|wc -l
#9
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.motors.txt|grep -P 'rec.motorcycles\t1\trec.motorcycles\t1'|wc -l
#462
#cat samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.motors.txt|grep -P 'rec.motorcycles\t1\trec.autos\t2'|wc -l
#509
#(975+462)/((975+9)+(462+509))
#0.7350383631713555

mode = 'spark_concepts'
mode = 'max_multi_thread_resolved_anno'
mode = 'max_multi_thread_resolved'
mode = 'spark_concept_vectors'
mode = 'spark_resolve_concepts'
mode = 'spark_concept_vectors_cosines'
mode = 'max_multi_thread_resolved_cosines'
mode = 'hungarian_multi_thread_resolved_cosines'
mode = 'hungarian_spark_resolved_cosines'

topic_vecs_matBC = None
topic_titlesBC = None
titles_to_loadBC = None
topic_titles_to_loadBC = None
sample_titles_to_loadBC = None
loaded_vecsBC = None
topics_dicBC = None
topicsBC = None
redirectsBC = None
titlesBC = None
skipgram_esa_mappings_dicBC = None
topic_titlesBC = None
sample_titlesBC = None

import io

if mode not in ('spark_concepts','spark_anno','spark_resolve_concepts','spark_concept_vectors','spark_anno_vectors','spark_concept_vectors_cosines','max_multi_thread_resolved_cosines','hungarian_multi_thread_resolved_cosines','hungarian_spark_resolved_cosines'):
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
titles_cosines = {}
topic_titles = {}
sample_titles = {}

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

keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"
keeponly_path = ""

model_path = "/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/home/wshalaby/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"
model_path = "/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim200-wind5-skipgram1.model"
model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/home/walid/work/github/semantic_searcher/concepts-10iter-dim500-wind5-skipgram1.model"
model_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concept_classification_anno_tmp_vecs.vecs.txt"
model_path = "/home/walid/work/github/semantic_searcher/concept_classification_tmp_vecs.vecs.txt"

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
topics_inpath = "/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.2recs"
topics_inpath = "/home/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"
topics_inpath = "/home/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved"
topics_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"
topics_inpath = "/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"

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
samples_inpath = "/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.10recs"
samples_inpath = "/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500"
samples_inpath = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.resolved"
samples_inpath = "/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.resolved"

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

output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-concepts-conc-allnorm-0.0-xab.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-conc-allnorm-0.0.txt"
output_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/samples-out-anno-conc-allnorm-0.0.resolved.txt"
output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-conc-allnorm-0.0.resolved.txt"

#output_path = "/home/walid/work/github/semantic_searcher/python/samples-out-concepts-cosine.txt"

output_path_hdfs = 'samples-out-concepts-conc-allnorm-0.0.txt'

if mode in ('max_multi_thread_resolved_cosines','hungarian_multi_thread_resolved_cosines','hungarian_spark_resolved_cosines'):
    from munkres import Munkres, print_matrix, make_cost_matrix

    #concepts
    #175
    cosines_inpath = '/home/walid/work/github/semantic_searcher/concept_classification_cosines.txt'
    topics_inpath = '/home/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved'
    samples_inpath = '/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.resolved'
    output_path = '/home/walid/work/github/semantic_searcher/python/samples-out-concepts-conc-allnorm-0.0.resolved.cosines.txt'   

    #79
    cosines_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concept_classification_cosines.txt'
    topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved'
    samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.resolved'
    output_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/samples-out-concepts-conc-allnorm-0.0.resolved.cosines.txt'

    #urc
    #cosines_inpath = '/scratch/wshalaby/doc2vec/concept_classification_cosines.txt.2recs'
    cosines_inpath = '/scratch/wshalaby/doc2vec/concept_classification_cosines.txt'
    topics_inpath = '/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved'
    samples_inpath = '/scratch/wshalaby/doc2vec/20newsgroups.simple.esa.concepts.500.resolved'
    
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.resolved.3recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.resolved"
    
    output_path = '/scratch/wshalaby/doc2vec/samples-out-concepts-conc-allnorm-0.0.resolved.cosines.txt'
    output_path_hdfs = 'samples-out-concepts-conc-allnorm-0.85.hung.resolved.txt'
    
    #anno
    #79
    cosines_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concept_classification_anno_cosines.txt'
    topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved'    
    samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.anno.resolved'    
    output_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/samples-out-anno-conc-allnorm-0.0.resolved.cosines.txt'
    
    #175
    cosines_inpath = '/home/walid/work/github/semantic_searcher/concept_classification_anno_cosines.txt'    
    topics_inpath = '/home/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved'
    samples_inpath = '/home/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.anno.resolved'
    output_path = '/home/walid/work/github/semantic_searcher/python/samples-out-anno-conc-allnorm-0.0.resolved.cosines.txt'
    
    #urc
    #cosines_inpath = '/scratch/wshalaby/doc2vec/concept_classification_anno_cosines.txt.2recs'
    cosines_inpath = '/scratch/wshalaby/doc2vec/concept_classification_anno_cosines.txt'
    topics_inpath = '/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.autos.motors'
    topics_inpath = '/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.autos.elecs'
    topics_inpath = '/scratch/wshalaby/doc2vec/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved'
    samples_inpath = '/scratch/wshalaby/doc2vec/20newsgroups.simple.esa.concepts.500.anno.resolved'
    
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved.3recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved.autos.elecs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved.autos.motors"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved.autos.elecs.1rec"
    
    output_path = '/scratch/wshalaby/doc2vec/samples-out-anno-conc-allnorm-0.0.resolved.cosines.txt'
    output_path_hdfs = 'samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.elecs.txt'
    output_path_hdfs = 'samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.motors.txt'
    output_path_hdfs = 'samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.autos.elecs.1rec.txt'
    output_path_hdfs = 'samples-out-concepts-conc-allnorm-0.85.hung.anno.resolved.1rec.txt'
    
    if mode=='max_multi_thread_resolved_cosines':
        from commons import load_cosines
    else:
        from commons import load_cosines_spark
        from commons import get_topic_titles
        #concepts
        cosines_inpath = 'concept_classification_cosines.txt'
        #anno
        cosines_inpath = 'concept_classification_anno_cosines.txt.3recs'
        cosines_inpath = 'concept_classification_anno_cosines.txt'        
        
if mode=='spark_resolve_concepts':
    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.2recs"
    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500"
    
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.10recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500"
    
    #anno
    topics_outpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.3recs"
    topics_outpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved"
    topics_notfound_outpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.notfound"
    
    titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids_anno.csv"
        
    samples_outpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved"
    samples_notfound_outpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.notfound"
    
    #concepts
    topics_outpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved.3recs"
    topics_outpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved"
    topics_notfound_outpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.notfound"
    
    titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"

    samples_outpath_hdfs = "20newsgroups.simple.esa.concepts.500.resolved"
    samples_notfound_outpath_hdfs = "20newsgroups.simple.esa.concepts.500.notfound"
    
if mode in ('spark_concept_vectors','spark_anno_vectors','spark_concept_vectors_cosines'):
    '''
    vecs_path = "/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter.vecs.txt"
    vecs_path = "/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.vecs.txt"
    vecs_path = "/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter-dim500-wind5-skipgram1.vecs.txt"
    '''
    
    #anno        
    titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids_anno.csv"
    
    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.3recs"
    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved"
    
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved.3recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved"
    
    vecs_path = "w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.vecs.txt"

    cosines_outpath = 'concept_classification_anno_cosines.txt.3recs'
    cosines_outpath = 'concept_classification_anno_cosines.txt'
    
    #concepts
    titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"

    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved"
    
    #samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.10recs"
    #samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500"

    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.resolved.3recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.resolved"
    
    vecs_path = "concepts-10iter-dim500-wind5-skipgram1.vecs.txt"

    cosines_outpath = 'concept_classification_cosines.txt'
    
if mode in ('spark_concepts','spark_anno'):
    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved"
    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved.3recs"

    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.3recs"
    topics_inpath_hdfs = "tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved"
    
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.10recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500"

    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.resolved.3recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.resolved"
    
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved.3recs"
    samples_inpath_hdfs = "20newsgroups.simple.esa.concepts.500.anno.resolved"


vecs_outpath = 'concept_classification_tmp_vecs.vecs.txt'
vecs_outpath = 'concept_classification_anno_tmp_vecs.vecs.txt'

wiki_esa_mapping_path = "/scratch/wshalaby/doc2vec/wiki_esa_ids.tsv"
mappings_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
keep_path = "/scratch/wshalaby/doc2vec/wikipedia-2016.500.30out.lst"

topics_dic = None
model = None
fout = None
y = []
y_pred = []
y_top = []
y_top_pred = []
topics = {}

def load_vecs(filename, format='txt'):
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

def cosines(pair):
    global topic_vecs_matBC, topic_titlesBC
    import numpy as np
    result = []
    dots = list(np.array([pair[1]]).dot(topic_vecs_matBC.value)[0])
    for title,dot in zip(topic_titlesBC.value,dots): # topic_titlesBC is a list of (sample title,norm pairs)
        sample_title = pair[0]
        sample_title_norm = np.linalg.norm(pair[1])
        topic_title = title[0]
        topic_title_norm = title[1]
        cos = dot/(sample_title_norm*topic_title_norm)
        result.append((sample_title,topic_title+'\t'+str(cos)))
    return result
    #for k,v in topic_vecsBC.value.items(): # return list of pairs (sample title, topic_title+tab+cosine)
    #    result.append((pair[0],k+str(pair[1].dot(v))))
    #return result

def get_vector(line, vec_mode='all', vec_format='txt'):
    global titles_to_loadBC, topic_titles_to_loadBC, sample_titles_to_loadBC
    import numpy as np
    tokens = line.split('\t')
    if len(tokens)==2: 
        if vec_format=='txt':
            vec = tokens[1]
        elif vec_format=='lst':
            vec = np.array([float(x) for x in tokens[1].split(',')])
        if vec_mode=='all' and tokens[0] in titles_to_loadBC.value:
            return [(tokens[0],vec)]
        elif vec_mode=='topics' and tokens[0] in topic_titles_to_loadBC.value:
            return [(tokens[0],vec)]
        elif vec_mode=='samples' and tokens[0] in sample_titles_to_loadBC.value:
            return [(tokens[0],vec)]
        else:
            return [None]
    else:
        return [None]

def check_vector(line):
    global titles_to_loadBC
    tokens = line.split('\t')
    if len(tokens)==2 and tokens[0] in titles_to_loadBC.value.values():
        return [(tokens[0],tokens[1])]
    else:
        return [None]

def resolve_esa_concepts(line):
    global titlesBC, redirectsBC, skipgram_esa_mappings_dicBC
    topic_name = line[0:line.find('\t')]
    topic_vector = line[line.rfind('\t')+1:]
    if topic_vector.find(';')==-1: # none line e.g., sample 53718
        return [None]
    titles_mappings = []
    for concept in topic_vector.split(';'):
        if len(concept)>0:
            org_concept_name = concept[0:concept.find(',')]
            try:
                concept_name = skipgram_esa_mappings_dicBC.value[org_concept_name]                
            except:
                concept_name = org_concept_name                
            
            # look up concept id
            try:
                if mode=='spark_anno': # anno titles
                    concept_id = titlesBC.value[concept_name]
                else:
                    if concept_name in titlesBC.value:
                        concept_id = concept_name
                    else: # may be a redirect
                        concept_id = redirectsBC.value[concept_name]
                titles_mappings.append((org_concept_name,concept_id))
            except:
                #print(concept_name+'...not found')
                titles_mappings.append((org_concept_name,u'NOT_-_FOUND'))                
    return titles_mappings

#st = sc.parallelize(['misc.forsale  sale offer shipping  forsale sell price brand obo       The Apprentice (UK series four),4.119606971740723;UniSquare,3.959080219268799;Volkswagen,3.9620440006256104;Vodafone,3.292428731918335;Privatization in Iran,3.443326473236084;DeLorean Motor Company,3.706087112426758;Tesco,5.543150901794434;Invincible class aircraft carrier,4.362002849578857;Citigroup,2.9731836318969727;Foreclosure,3.991910457611084;Big C,3.3920843601226807;Kogan Technologies,3.638162136077881;Store brand,3.559232473373413;AT T Mobility,4.15373420715332;No frills,3.305943250656128'])
#str = st.flatMap(lambda line: resolve_esa_concepts(line))
#str.collect()

def resolve_esa_concepts_line(line):
    global titlesBC, redirectsBC, skipgram_esa_mappings_dicBC
    topic_name = line[0:line.rfind('\t')]
    topic_vector = line[line.rfind('\t')+1:]
    if topic_vector.find(';')==-1: # none line e.g., sample 53718
        return [None]
    notfound_titles = []
    new_topic_vector = topic_name+'\t'
    for concept in topic_vector.split(';'):
        if len(concept)>0:
            tokens = concept.split(',')
            org_concept_name = tokens[0]
            concept_weight = tokens[1]
            try:
                concept_name = skipgram_esa_mappings_dicBC.value[org_concept_name]
            except:
                concept_name = org_concept_name                
            
            # look up concept id
            try:
                if mode=='spark_anno': # anno titles
                    concept_id = titlesBC.value[concept_name]
                else:
                    if concept_name in titlesBC.value:
                        concept_id = concept_name
                    else: # may be a redirect
                        concept_id = redirectsBC.value[concept_name]
                new_topic_vector += concept_id+',,,,'+concept_weight+';;;;'
            except:
                #print(concept_name+'...not found')
                notfound_titles.append(('NOT_-_FOUND',org_concept_name))                
    return notfound_titles+[(new_topic_vector)]

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
    global titles_to_loadBC, loaded_vecsBC, topics_dicBC, topicsBC
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
            if concept_id != u'NOT_-_FOUND':
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
    return [result]

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
            if concept_name in skipgram_esa_mappings_dic:
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

def get_max_similarity_resolved(line):
    global model, titles, redirects, topics_dic, topics, skipgram_esa_mappings_dic, mode
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
    if sample_vector.find(';;;;')==-1: # none line e.g., sample 53718
        return sample_info
    #max_concept_weight = 0.0
    #print(line)
    for concept in sample_vector.split(';;;;'):
        if len(concept)>0:
            concept_name = concept[0:concept.find(',,,,')]
            concept_weight = float(concept[concept.rfind(',,,,')+4:])
            #if concept_weight>max_concept_weight:
            #    max_concept_weight = concept_weight
            # look up concept id
            if mode=='max_multi_thread_resolved_anno': # anno titles
                if concept_name in model:
                    concept_id = concept_name
                    # accumulate weighted concept vector
                    vect = np.array(model[concept_id])
                    accum_vector.append((vect,concept_weight,concept_name))#np.array(model['12806'])
                else:
                    print(sample_info.sample_name+'\t'+concept_name+'...not found')
                    #accum_vector.append((None,concept_weight,concept_name))
            else:
                if concept_name in titles:
                    concept_id = concept_name
                    # accumulate weighted concept vector
                    vect = np.array(model[concept_id])
                    accum_vector.append((vect,concept_weight,concept_name))#np.array(model['12806'])
                else:
                    print(sample_info.sample_name+'\t'+concept_name+'...not found')
                    #accum_vector.append((None,concept_weight,concept_name))

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
                    #print(os.linesep+'CONSIM\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+topic_name+'\t'+str(topic_weight)+'\t'+str(max_sim)+os.linesep)
                    break
                elif sample_concept_vector is not None and topic_concept_vector is not None:
                    sim = 1 - spatial.distance.cosine(topic_concept_vector,sample_concept_vector)
                    #print(os.linesep+'CONSIM\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+topic_name+'\t'+str(topic_weight)+'\t'+str(sim)+os.linesep)
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
def get_max_similarity_resolved_cosines(line):
    global topics_dic, topics, skipgram_esa_mappings_dic, topic_titles, titles_cosines
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
    if sample_vector.find(';;;;')==-1: # none line e.g., sample 53718
        return sample_info
    #max_concept_weight = 0.0
    #print(line)
    for concept in sample_vector.split(';;;;'):
        if len(concept)>0:
            concept_name = concept[0:concept.find(',,,,')]
            concept_weight = float(concept[concept.rfind(',,,,')+4:])
            accum_vector.append((concept_weight,concept_name))
            
    # normalize weights
    #accum_vector = [(weight/max_concept_weight,concept_name) for (weight,concept_name) in accum_vector]

    # calculate sample norm
    sample_norm = np.linalg.norm([weight for (weight,concept_name) in accum_vector])
    
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
        for (sample_weight,concept_name) in accum_vector: # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            max_sim = 0.0
            max_sim_weight = 0.0
            max_topic_name = ''
            for (topic_weight,topic_name) in topic_accum_vector: # v is topic accum_vector in the format (vector,weight)
                #if concept_name=="Pimp My Ride" and topic_name=="Pimp My Ride":
                #    print("testing topic concept: ("+topic_name+")")
                if topic_name==concept_name:
                    max_sim = 1.0
                    max_sim_weight = topic_weight
                    print(sample_info.sample_name+'\t'+concept_name+'...exact match')
                    max_topic_name = topic_name
                    #print(os.linesep+'CONSIM\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+topic_name+'\t'+str(topic_weight)+'\t'+str(max_sim)+os.linesep)
                    break
                elif topic_name in topic_titles and sample_info.sample_name in sample_titles:
                    sim = titles_cosines[sample_titles[sample_info.sample_name]][topic_titles[topic_name]]
                    #print(os.linesep+'CONSIM\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+topic_name+'\t'+str(topic_weight)+'\t'+str(sim)+os.linesep)
                else:
                    print('Opps...unexpected error one of ('+topic_name+'\t'+sample_info.sample_name+') not found')
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

def get_hungarian_similarity_resolved_cosines(line):
    global topics_dic, topics, topic_titles, titles_cosines
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
    if sample_vector.find(';;;;')==-1: # none line e.g., sample 53718
        return sample_info
    #max_concept_weight = 0.0
    #print(line)
    for concept in sample_vector.split(';;;;'):
        if len(concept)>0:
            concept_name = concept[0:concept.find(',,,,')]
            concept_weight = float(concept[concept.rfind(',,,,')+4:])
            accum_vector.append((concept_weight,concept_name))
            
    # normalize weights
    #accum_vector = [(weight/max_concept_weight,concept_name) for (weight,concept_name) in accum_vector]

    # calculate sample norm
    sample_norm = np.linalg.norm([weight for (weight,concept_name) in accum_vector])
    
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
        if len(topic_accum_vector)<len(accum_vector):
            print('Warning...',sample_info.sample_name,len(accum_vector),k,len(topic_accum_vector))

        cost_matrix = [[0.0 for i in range(len(topic_accum_vector))] for j in range(len(accum_vector))]
        for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            max_sim = 0.0
            max_sim_weight = 0.0
            max_topic_name = ''
            for j,(topic_weight,topic_name) in enumerate(topic_accum_vector): # v is topic accum_vector in the format (vector,weight)
                #if concept_name=="Pimp My Ride" and topic_name=="Pimp My Ride":
                #    print("testing topic concept: ("+topic_name+")")
                if topic_name==concept_name:
                    cost_matrix[i][j] = 1.0-1.0 #least cost
                    print(sample_info.sample_name+'\t'+concept_name+'...exact match')
                    #print(os.linesep+'CONSIM\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+topic_name+'\t'+str(topic_weight)+'\t'+str(max_sim)+os.linesep)
                elif topic_name in topic_titles and concept_name in titles_cosines and topic_titles[topic_name] in titles_cosines[concept_name]:
                    cost_matrix[i][j] = 1.0 - titles_cosines[concept_name][topic_titles[topic_name]]
                else:
                    cost_matrix[i][j] = 1.0
        
        m = Munkres()
        indexes = m.compute(cost_matrix)
        for row, col in indexes:
            sample_weight = accum_vector[row][0]
            concept_name = accum_vector[row][1]
            max_sim_weight = topic_accum_vector[col][0]
            max_topic_name = topic_accum_vector[col][1]
            max_sim = 1.0-cost_matrix[row][col]
            print(os.linesep+'MAXIMA\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+max_topic_name+'\t'+str(max_sim_weight)+'\t'+str(max_sim)+os.linesep)
            topic_norm.append(max_sim_weight) # topic weight
            cur_sim += max_sim_weight*sample_weight*max_sim

        if len(topic_norm)>0:
            #cur_sim = cur_sim/(sample_norm*np.linalg.norm(topic_norm))
            cur_sim = cur_sim/(sample_norm*all_topic_norm)

        print(os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(cur_sim)+os.linesep)
        if cur_sim>sample_info.max_all_sim:
            sample_info.max_all_sim = cur_sim
            sample_info.max_topic_name = k
            sample_info.max_topic_id = topics_dic[k]

    print(os.linesep+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep)
    return sample_info

def get_hungarian_similarity_resolved_cosines_spark(line):
    global topics_dicBC, topicsBC, topic_titlesBC, titles_cosinesBC
    
    if line.find('#')==0: # none line e.g., sample 53718 if commented it out
        return [None]
    
    result = u''
    #result = []
    sample_info = SampleInfo()
    sample = line[0:line.find('\t')].split('\\')
    sample_info.sample_name = sample[len(sample)-1]
    sample_info.sample_class = sample[len(sample)-2]
    sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
    sample_vector = line[line.rfind('\t')+1:]
    accum_vector = []
    if sample_vector.find(';;;;')==-1: # none line e.g., sample 53718
        return [None]
    #max_concept_weight = 0.0
    #print(line)
    for concept in sample_vector.split(';;;;'):
        if len(concept)>0:
            concept_name = concept[0:concept.find(',,,,')]
            concept_weight = float(concept[concept.rfind(',,,,')+4:])
            accum_vector.append((concept_weight,concept_name))
            
    # normalize weights
    #accum_vector = [(weight/max_concept_weight,concept_name) for (weight,concept_name) in accum_vector]

    # calculate sample norm
    sample_norm = np.linalg.norm([weight for (weight,concept_name) in accum_vector])
    
    result += 'SAMPLE_NORM\t'+sample_info.sample_name+'\t'+str(sample_norm)+os.linesep
    #result.append(['SAMPLE_NORM\t',sample_info.sample_name,'\t',str(sample_norm),os.linesep])

    sample_info.max_all_sim = 0.0
    sample_info.max_topic_name = ''
    sample_info.max_topic_id = None
        
    for k,v in topicsBC.value.items(): # loop on topics
        #print("testing topic: ("+k+")")
        cur_sim = 0.0
        topic_norm = []
        all_topic_norm = v[1]
        topic_accum_vector = v[0]
        if len(topic_accum_vector)<len(accum_vector):
            result += 'WARNING...'+sample_info.sample_name+str(len(accum_vector))+k+str(len(topic_accum_vector))+os.linesep
            #result.append(['WARNING...',sample_info.sample_name,str(len(accum_vector)),k,str(len(topic_accum_vector)),os.linesep])

        cost_matrix = [[0.0 for i in range(len(topic_accum_vector))] for j in range(len(accum_vector))]
        result += sample_info.sample_name+'\t'+k+'\t'+'MAT_DIM\t('+str(len(cost_matrix))+','+str(len(cost_matrix[0]))+')'+os.linesep
        # we need to discard all rows in the cost matrix whose elements are all 1s. 
        # this means these the similarities between these sample titles and all topic titles are zeros
        # so no need to include them in the cost matrix for efficiency
        # we also don't need to include in the cost matrix a sample title which has an exact match with a topic title
        valid_titles = []
        for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
            valid_row = False
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            max_sim = 0.0
            max_sim_weight = 0.0
            max_topic_name = ''
            # search for exact match
            exact_found = False
            for pair in topic_accum_vector:
                if concept_name==pair[1]:
                    exact_found = True                    
                    result += sample_info.sample_name+'\t'+concept_name+'...exact match'+os.linesep                    
                    max_sim_weight = pair[0]
                    max_topic_name = pair[1]
                    max_sim = 1.0
                    result += 'MAXIMA\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+max_topic_name+'\t'+str(max_sim_weight)+'\t'+str(max_sim)+os.linesep
                    #result.append(['MAXIMA\t',sample_info.sample_name,'\t',k,'\t',concept_name,'\t',str(sample_weight),'\t',max_topic_name,'\t',str(max_sim_weight),'\t',str(max_sim),os.linesep])
                    topic_norm.append(max_sim_weight) # topic weight
                    cur_sim += max_sim_weight*sample_weight*max_sim
                    break
            if exact_found==False:
                for j,(topic_weight,topic_name) in enumerate(topic_accum_vector): # v is topic accum_vector in the format (vector,weight)
                    '''
                    if topic_name==concept_name:
                        cost_matrix[i][j] = 1.0-1.0 #least cost
                        result += sample_info.sample_name+'\t'+concept_name+'...exact match'+os.linesep
                        valid_row = True
                    el
                    '''
                    if topic_name in topic_titlesBC.value and concept_name in titles_cosinesBC.value and topic_titlesBC.value[topic_name] in titles_cosinesBC.value[concept_name]:
                        cost_matrix[i][j] = 1.0 - titles_cosinesBC.value[concept_name][topic_titlesBC.value[topic_name]]
                        valid_row = True
                    else:
                        cost_matrix[i][j] = 1.0

            if valid_row==True: # if false then all similarities are 0s or below threshold for this sample title with topic titles
                valid_titles.append(i)                
            #else:
            #    result += 'ZERO similarities...'+os.linesep
            #    #result.append(['ZERO similarities...',os.linesep])
                
        if len(valid_titles)>0:
            valid_cost_matrix = [cost_matrix[i] for i in valid_titles]
            result += sample_info.sample_name+'\t'+k+'\t'+'MAT_NEW_DIM\t('+str(len(valid_cost_matrix))+','+str(len(valid_cost_matrix[0]))+')'+os.linesep
            valid_accum_vector = [accum_vector[i] for i in valid_titles]
            m = Munkres()
            indexes = m.compute(valid_cost_matrix)
            for row, col in indexes:
                sample_weight = valid_accum_vector[row][0]
                concept_name = valid_accum_vector[row][1]
                max_sim_weight = topic_accum_vector[col][0]
                max_topic_name = topic_accum_vector[col][1]
                max_sim = 1.0-valid_cost_matrix[row][col]
                result += 'MAXIMA\t'+sample_info.sample_name+'\t'+k+'\t'+concept_name+'\t'+str(sample_weight)+'\t'+max_topic_name+'\t'+str(max_sim_weight)+'\t'+str(max_sim)+os.linesep
                #result.append(['MAXIMA\t',sample_info.sample_name,'\t',k,'\t',concept_name,'\t',str(sample_weight),'\t',max_topic_name,'\t',str(max_sim_weight),'\t',str(max_sim),os.linesep])
                topic_norm.append(max_sim_weight) # topic weight
                cur_sim += max_sim_weight*sample_weight*max_sim

        if len(topic_norm)>0:
            #cur_sim = cur_sim/(sample_norm*np.linalg.norm(topic_norm))
            cur_sim = cur_sim/(sample_norm*all_topic_norm)

        result += 'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(cur_sim)+os.linesep
        #result.append(['SIMILA\t',sample_info.sample_name,'\t',k,'\t',str(cur_sim),os.linesep])
        if cur_sim>sample_info.max_all_sim:
            sample_info.max_all_sim = cur_sim
            sample_info.max_topic_name = k
            sample_info.max_topic_id = topics_dicBC.value[k]
    
    result += 'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dicBC.value[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
    #result.append([sample_info.sample_name,'\t',sample_info.sample_class,'\t',str(topics_dicBC.value[sample_info.sample_class]),'\t',sample_info.max_topic_name,'\t',str(sample_info.max_topic_id),'\t',str(sample_info.max_all_sim),os.linesep])
    return [result]
    #return [''.join(result)]

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
    global titles, redirects, topics_dic, topic_vecs_matBC, topic_titlesBC, model, fout, y, y_pred, y_top, y_top_pred, topics, top_topics, mode, titles_to_loadBC, topic_titles_to_loadBC, sample_titles_to_loadBC, loaded_vecsBC, topics_dicBC, topicsBC, redirectsBC, titlesBC, skipgram_esa_mappings_dicBC, titles_cosines, topic_titles, titles_cosinesBC, titles_cosinesBC

    print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    if mode!='cosine' and mode!='test' and mode!='check_file':
        to_keep = set()
        if keeponly_path!="":
            print("loading keey only")
            with open(keeponly_path) as keeponly_titles:
                for title in keeponly_titles:
                    to_keep.add(title.replace(os.linesep,""))
        print("loaded "+str(len(to_keep))+" keeponly titles")

        if mode=='max_multi_thread_resolved_cosines':
            titles_cosines, topic_titles, sample_titles = load_cosines_2d(cosines_inpath)            
            print("loaded "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            #titles_cosines, topic_titles = load_cosines(cosines_inpath)
        elif mode in ('hungarian_multi_thread_resolved_cosines'):
            titles_cosines, topic_titles = load_cosines(cosines_inpath, min_cosine=0.85)
            print("loaded "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        elif mode!='hungarian_spark_resolved_cosines':
            titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
        print("keeping "+str(len(titles))+" titles only")
        print("loaded "+str(len(redirects))+" redirects")

        if mode not in ('spark_concepts','spark_anno','spark_resolve_concepts','spark_concept_vectors','spark_anno_vectors','spark_concept_vectors_cosines','max_multi_thread_resolved_cosines','hungarian_multi_thread_resolved_cosines','hungarian_spark_resolved_cosines'):
            print("loading model")
            if mode in ('max_multi_thread_resolved','max_multi_thread_resolved_anno'):
                model = {}
                for pair in load_vecs(model_path):
                    model[pair[0]] = pair[1]
            else:
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = {}
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic:
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
                            if concept_name in skipgram_esa_mappings_dic:
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
                            if sample_concept in topic_weights:                                
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
                    if topic_name in topics:
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
                                    if concept_name not in titles: # may be a redirect
                                        concept_name = redirects[concept_name]
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]                            
                                # accumulate weighted concept vector
                            except:
                                if concept_name not in skipgram_esa_mappings_dic:
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
                                    if concept_name not in titles: # may be a redirect
                                        concept_name = redirects[concept_name]
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in model.vocab:
                                        concept_id = concept_name
                                    else: # may be a redirect
                                        concept_id = redirects[concept_name]                            
                                # accumulate weighted concept vector
                            except:
                                if concept_name not in skipgram_esa_mappings_dic:
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = np.array([0.0]*model.vector_size)
                    accum_weight = 0.0
                    count = 0
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic:
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    if concept_name not in titles: # may be a redirect
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
                            if concept_name in skipgram_esa_mappings_dic:
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            concept_weight = float(concept[concept.rfind(',')+1:])
                            # look up concept id
                            try:
                                if model.vector_size==200: # anno titles
                                    if concept_name not in titles: # may be a redirect
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic:
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
                            if concept_name in skipgram_esa_mappings_dic:
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    max_concept_weight = 0.0
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic:
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

    elif mode=='max_multi_thread_resolved' or mode=='max_multi_thread_resolved_anno':
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    max_concept_weight = 0.0
                    for concept in topic_vector.split(';;;;'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',,,,')]
                            concept_weight = float(concept[concept.rfind(',,,,')+4:])
                            if concept_weight>max_concept_weight:
                                max_concept_weight = concept_weight
                            # look up concept id
                            if  mode=='max_multi_thread_resolved_anno': # anno titles
                                if concept_name in model:
                                    # accumulate weighted concept vector
                                    accum_vector.append((np.array(model[concept_name]),concept_weight,concept_name))#np.array(model['12806'])
                                else:
                                    print(concept_name+'...not found')
                            else:
                                if concept_name in titles:
                                    # accumulate weighted concept vector
                                    accum_vector.append((np.array(model[concept_name]),concept_weight,concept_name))#np.array(model['12806'])
                                else: # may be a redirect
                                    print(concept_name+'...not found')
                                    #accum_vector.append((None,concept_weight,concept_name))#np.array(model['12806'])
                                                
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
                        samples_info = p.map(get_max_similarity_resolved, in_records)                            
                        p.close()
                        p.join()
                        write_sims(samples_info)
                        in_records = []

        if len(in_records)>0:
            p = Pool(multiprocessing.cpu_count()-1)
            samples_info = p.map(get_max_similarity_resolved, in_records)                
            p.close()
            p.join()
            write_sims(samples_info)

        fout.close()

        print(f1_score(y, y_pred, average='micro'))
        print(f1_score(y_top, y_top_pred, average='micro'))

    elif mode=='max_multi_thread_resolved_cosines':
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    max_concept_weight = 0.0
                    for concept in topic_vector.split(';;;;'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',,,,')]
                            concept_weight = float(concept[concept.rfind(',,,,')+4:])
                            if concept_weight>max_concept_weight:
                                max_concept_weight = concept_weight
                            
                            accum_vector.append((concept_weight,concept_name))
                                                
                    # normalize weights
                    #print('max',max_concept_weight)
                    #accum_vector = [(weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]

                    # calculate topic norm
                    topic_norm =  np.linalg.norm([weight for (weight,concept_name) in accum_vector])
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
                        samples_info = p.map(get_max_similarity_resolved_cosines, in_records)
                        p.close()
                        p.join()
                        write_sims(samples_info)
                        in_records = []

        if len(in_records)>0:
            p = Pool(multiprocessing.cpu_count()-1)
            samples_info = p.map(get_max_similarity_resolved_cosines, in_records)                
            p.close()
            p.join()
            write_sims(samples_info)

        fout.close()

        print(f1_score(y, y_pred, average='micro'))
        print(f1_score(y_top, y_top_pred, average='micro'))

    elif mode=='hungarian_multi_thread_resolved_cosines':
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    max_concept_weight = 0.0
                    for concept in topic_vector.split(';;;;'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',,,,')]
                            concept_weight = float(concept[concept.rfind(',,,,')+4:])
                            if concept_weight>max_concept_weight:
                                max_concept_weight = concept_weight
                            
                            accum_vector.append((concept_weight,concept_name))
                                                
                    # normalize weights
                    #print('max',max_concept_weight)
                    #accum_vector = [(weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]

                    # calculate topic norm
                    topic_norm =  np.linalg.norm([weight for (weight,concept_name) in accum_vector])
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
                        samples_info = p.map(get_hungarian_similarity_resolved_cosines, in_records)
                        p.close()
                        p.join()
                        write_sims(samples_info)
                        in_records = []

        if len(in_records)>0:
            p = Pool(multiprocessing.cpu_count()-1)
            samples_info = p.map(get_hungarian_similarity_resolved_cosines, in_records)                
            p.close()
            p.join()
            write_sims(samples_info)

        fout.close()

        print(f1_score(y, y_pred, average='micro'))
        print(f1_score(y_top, y_top_pred, average='micro'))

    elif mode=='hungarian_spark_resolved_cosines':
        from pyspark import SparkContext
        sc = SparkContext(appName="concept classification- hungarian resolved cosines")

        print("loading cosines")
        all_cosines = sc.textFile(cosines_inpath)
        needed_cosines = all_cosines.filter(lambda line: len(line)>1).flatMap(lambda line: load_cosines_spark(line,min_cosine=0.85)).filter(lambda line: line is not None)
        titles_cosines = {}
        for cosine in needed_cosines.collect():
            for k,v in cosine.items():
                titles_cosines[k] = v
        topic_titles = get_topic_titles(all_cosines.first())
        
        print('broadcasting topic_titles ('+str(len(topic_titles.keys()))+')')
        topic_titlesBC = sc.broadcast(topic_titles)
        print('topic_titles broadcasted')

        print('broadcasting titles_cosines ('+str(len(titles_cosines.keys()))+')')
        titles_cosinesBC = sc.broadcast(titles_cosines)
        print('titles_cosines broadcasted')
                
        print("loading topics")
        topics = {}
        topics_dic = {}
        top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
        cur_topic = 1
        with io.open(topics_inpath,'r',encoding='utf-8') as topics_file:
            for line in topics_file:
                line = line.strip('\n')
                if len(line)>0:
                    topic_name = line[0:line.find('\t')]
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    accum_vector = []
                    max_concept_weight = 0.0
                    for concept in topic_vector.split(';;;;'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',,,,')]
                            concept_weight = float(concept[concept.rfind(',,,,')+4:])
                            if concept_weight>max_concept_weight:
                                max_concept_weight = concept_weight
                            
                            accum_vector.append((concept_weight,concept_name))
                                                
                    # normalize weights
                    #print('max',max_concept_weight)
                    #accum_vector = [(weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]

                    # calculate topic norm
                    topic_norm =  np.linalg.norm([weight for (weight,concept_name) in accum_vector])
                    print(os.linesep+'TOPIC_NORM\t'+topic_name+'\t'+str(topic_norm)+os.linesep)
                    topics[topic_name] = (accum_vector, topic_norm)
                    topics_dic[topic_name] = cur_topic
                    cur_topic = cur_topic + 1
            
            print('broadcasting topics_dic ('+str(len(topics_dic.keys()))+')')
            topics_dicBC = sc.broadcast(topics_dic)
            print('topics_dic broadcasted')

            print('broadcasting topics ('+str(len(topics.keys()))+')')
            topicsBC = sc.broadcast(topics)
            print('topics broadcasted')      
        
        print("processing samples")
        samples = sc.textFile(samples_inpath_hdfs)
        results = samples.flatMap(lambda x : get_hungarian_similarity_resolved_cosines_spark(x)).filter(lambda line: line is not None)
        os.system('hdfs dfs -rm -r '+output_path_hdfs)
        results.saveAsTextFile(output_path_hdfs)
            
        sc.stop()
    
    elif mode=='spark_resolve_concepts':
        from pyspark import SparkContext
        sc = SparkContext(appName="concept resolver")
        
        print('broadcasting titles ('+str(len(titles.keys()))+')')
        titlesBC = sc.broadcast(titles)
        print('titles broadcasted')
        
        print('broadcasting redirects ('+str(len(redirects.keys()))+')')
        redirectsBC = sc.broadcast(redirects)
        print('redirects broadcasted')

        print('broadcasting skipgram_esa_mappings_dic ('+str(len(skipgram_esa_mappings_dic.keys()))+')')
        skipgram_esa_mappings_dicBC = sc.broadcast(skipgram_esa_mappings_dic)
        print('skipgram_esa_mappings_dic broadcasted')
        
        if topics_inpath_hdfs!="":
            print('resolving topic titles')
            topics_records = sc.textFile(topics_inpath_hdfs)
            #topics = sc.parallelize(['sales    Inventory,0.12;Stream,0.1'])
            topics_titles = topics_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts_line(line)).filter(lambda line: line is not None)
            os.system('hdfs dfs -rm -r '+topics_outpath_hdfs)
            topics_titles.filter(lambda line: line[0]!='NOT_-_FOUND').saveAsTextFile(topics_outpath_hdfs)
            os.system('hdfs dfs -rm -r '+topics_notfound_outpath_hdfs)
            topics_titles.filter(lambda line: line[0]=='NOT_-_FOUND').distinct().map(lambda pair: pair[1]).saveAsTextFile(topics_notfound_outpath_hdfs)            

        #for pair in topics_titles.collect():
        #    titles_to_load[pair[0]] = pair[1]

        #print('topic titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')
        if samples_inpath_hdfs!="":
            print('resolving sample titles')
            samples_records = sc.textFile(samples_inpath_hdfs)
            #samples = sc.parallelize(['sample1    Inventory,0.12;Water,0.1','sample2    Stream,0.12;Basketball,0.1'])
            samples_titles = samples_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts_line(line)).filter(lambda line: line is not None)
            os.system('hdfs dfs -rm -r '+samples_outpath_hdfs)
            samples_titles.filter(lambda line: line[0]!='NOT_-_FOUND').saveAsTextFile(samples_outpath_hdfs)
            os.system('hdfs dfs -rm -r '+samples_notfound_outpath_hdfs)
            samples_titles.filter(lambda line: line[0]=='NOT_-_FOUND').distinct().map(lambda pair: pair[1]).saveAsTextFile(samples_notfound_outpath_hdfs)
            
        sc.stop()
    
    elif mode in ('spark_concept_vectors','spark_anno_vectors'):
        from pyspark import SparkContext
        sc = SparkContext(appName="concept classification")
        
        titles_to_load = {}
        loaded_vecs = {}

        print('broadcasting titles ('+str(len(titles.keys()))+')')
        titlesBC = sc.broadcast(titles)
        print('titles broadcasted')
        
        print('broadcasting redirects ('+str(len(redirects.keys()))+')')
        redirectsBC = sc.broadcast(redirects)
        print('redirects broadcasted')

        print('broadcasting skipgram_esa_mappings_dic ('+str(len(skipgram_esa_mappings_dic.keys()))+')')
        skipgram_esa_mappings_dicBC = sc.broadcast(skipgram_esa_mappings_dic)
        print('skipgram_esa_mappings_dic broadcasted')
        
        print('resolving topic titles')
        topics_records = sc.textFile(topics_inpath_hdfs)
        #topics = sc.parallelize(['sales    Inventory,0.12;Stream,0.1'])
        topics_titles = topics_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        #for pair in topics_titles.collect():
        #    titles_to_load[pair[0]] = pair[1]

        #print('topic titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')
        print('resolving sample titles')
        samples_records = sc.textFile(samples_inpath_hdfs)
        #samples = sc.parallelize(['sample1    Inventory,0.12;Water,0.1','sample2    Stream,0.12;Basketball,0.1'])
        samples_titles = samples_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        all_titles_to_load = samples_titles.union(topics_titles).distinct()
        os.system('hdfs dfs -rm -r concept_classification_tmp_titles.txt')
        all_titles_to_load.saveAsTextFile('concept_classification_tmp_titles.txt')
        #res = os.system('hdfs dfs -cat concept_classification_tmp_titles.txt/*>concept_classification_tmp_titles.txt')
        res = 0
        if res!=0:
            print('ERROR getting titles from HDFS')
        for pair in all_titles_to_load.collect():
            titles_to_load[pair[1]] = pair[0]
        print('sample titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')

        print('broadcasting titles_to_load ('+str(len(titles_to_load.keys()))+')')
        titles_to_loadBC = sc.broadcast(titles_to_load)
        print('titles_to_load broadcasted')
                
        print('loading vectors')
        vecs_data = sc.textFile(vecs_path)
        vecs = vecs_data.filter(lambda line: len(line)>1).flatMap(lambda line: get_vector(line)).filter(lambda line: line is not None).map(lambda pair: pair[0]+'\t'+pair[1])
        os.system('hdfs dfs -rm -r '+vecs_outpath)
        vecs.saveAsTextFile(vecs_outpath)
        res = os.system('hdfs dfs -cat '+vecs_outpath+'/*>'+vecs_outpath)
        sc.stop()        
    
    elif mode in ('spark_concept_vectors_cosines'):
        '''
        from pyspark import SparkContext
        sc = SparkContext(appName="concept classification - vectors cosines")
        
        topic_titles_to_load = {}
        sample_titles_to_load = {}
        titles_to_load = {}
        loaded_vecs = {}

        print('broadcasting titles ('+str(len(titles.keys()))+')')
        titlesBC = sc.broadcast(titles)
        print('titles broadcasted')
        
        print('broadcasting redirects ('+str(len(redirects.keys()))+')')
        redirectsBC = sc.broadcast(redirects)
        print('redirects broadcasted')

        print('broadcasting skipgram_esa_mappings_dic ('+str(len(skipgram_esa_mappings_dic.keys()))+')')
        skipgram_esa_mappings_dicBC = sc.broadcast(skipgram_esa_mappings_dic)
        print('skipgram_esa_mappings_dic broadcasted')
        
        print('resolving topic titles')
        topics_records = sc.textFile(topics_inpath_hdfs)
        #topics = sc.parallelize(['sales    Inventory,0.12;Stream,0.1'])
        topics_titles = topics_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        for pair in topics_titles.collect():
            topic_titles_to_load[pair[1]] = pair[0]
            titles_to_load[pair[1]] = pair[0]
        print('topic titles resolved, total ('+str(len(topic_titles_to_load.keys()))+') titles')

        print('resolving sample titles')
        samples_records = sc.textFile(samples_inpath_hdfs)
        samples_titles = samples_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        #os.system('hdfs dfs -rm -r concept_classification_tmp_titles.txt')
        #all_titles_to_load.saveAsTextFile('concept_classification_tmp_titles.txt')
        for pair in samples_titles.collect():
            sample_titles_to_load[pair[1]] = pair[0]
            titles_to_load[pair[1]] = pair[0]
        print('sample titles resolved, total ('+str(len(sample_titles_to_load.keys()))+') titles')

        print('all titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')

        print('broadcasting topic_titles_to_load ('+str(len(topic_titles_to_load.keys()))+')')
        topic_titles_to_loadBC = sc.broadcast(topic_titles_to_load)
        print('topic_titles_to_load broadcasted')
        
        print('broadcasting sample_titles_to_load ('+str(len(sample_titles_to_load.keys()))+')')
        sample_titles_to_loadBC = sc.broadcast(sample_titles_to_load)
        print('sample_titles_to_load broadcasted')

        print('broadcasting titles_to_load ('+str(len(titles_to_load.keys()))+')')
        titles_to_loadBC = sc.broadcast(titles_to_load)
        print('titles_to_load broadcasted')

        print('loading vectors')
        vecs_data = sc.textFile(vecs_path)
        vecs = vecs_data.filter(lambda line: len(line)>1).flatMap(lambda line: get_vector(line)).filter(lambda line: line is not None).map(lambda pair: pair[0]+'\t'+pair[1])
        os.system('hdfs dfs -rm -r '+vecs_outpath)
        vecs.saveAsTextFile(vecs_outpath)
        res = os.system('hdfs dfs -cat '+vecs_outpath+'/*>'+vecs_outpath)
        
        topic_vecs = vecs.flatMap(lambda line: get_vector(line,vec_mode='topics',vec_format='lst')).filter(lambda line: line is not None)
        count = topic_vecs.count()
        topic_vecs_mat = None
        topic_vecs_titles = []
        for i,pair in enumerate(topic_vecs.collect()):
            if topic_vecs_mat is None:
                topic_vecs_mat = np.zeros((count,len(pair[1])))
            topic_vecs_mat[i,:] = np.array([pair[1]])
            topic_vecs_titles.append((pair[0],np.linalg.norm(pair[1])))

        print('broadcasting topic_vecs_mat ('+str(count)+')')
        topic_vecs_matBC = sc.broadcast(topic_vecs_mat.transpose())
        print('topic_vecs_mat broadcasted')

        print('broadcasting topic_vecs_titles ('+str(len(topic_vecs_titles))+')')
        topic_titlesBC = sc.broadcast(topic_vecs_titles)
        print('topic_vecs_titles broadcasted')

        sample_vecs = vecs.flatMap(lambda line: get_vector(line,vec_mode='samples',vec_format='lst')).filter(lambda line: line is not None)
        all_cosines = sample_vecs.flatMap(lambda line: cosines(line)).reduceByKey(lambda a,b:a+'\t'+b).map(lambda (a,b):a+'\t'+b)
        os.system('hdfs dfs -rm -r '+cosines_outpath)
        all_cosines.saveAsTextFile(cosines_outpath)
        sc.stop()
        '''
    elif mode in ('spark_concepts','spark_anno'):
        from pyspark import SparkContext
        sc = SparkContext(appName="concept classification")
        
        titles_to_load = {}
        loaded_vecs = {}

        print('broadcasting titles ('+str(len(titles.keys()))+')')
        titlesBC = sc.broadcast(titles)
        print('titles broadcasted')
        
        print('broadcasting redirects ('+str(len(redirects.keys()))+')')
        redirectsBC = sc.broadcast(redirects)
        print('redirects broadcasted')

        print('broadcasting skipgram_esa_mappings_dic ('+str(len(skipgram_esa_mappings_dic.keys()))+')')
        skipgram_esa_mappings_dicBC = sc.broadcast(skipgram_esa_mappings_dic)
        print('skipgram_esa_mappings_dic broadcasted')
        
        print('resolving topic titles')
        topics_records = sc.textFile(topics_inpath_hdfs)
        #topics = sc.parallelize(['sales    Inventory,0.12;Stream,0.1'])
        topics_titles = topics_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        #for pair in topics_titles.collect():
        #    titles_to_load[pair[0]] = pair[1]

        #print('topic titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')
        print('resolving sample titles')
        samples_records = sc.textFile(samples_inpath_hdfs)
        #samples = sc.parallelize(['sample1    Inventory,0.12;Water,0.1','sample2    Stream,0.12;Basketball,0.1'])
        samples_titles = samples_records.filter(lambda line: len(line)>1).flatMap(lambda line: resolve_esa_concepts(line)).filter(lambda line: line is not None).distinct()
        all_titles_to_load = samples_titles.union(topics_titles).distinct()
        os.system('hdfs dfs -rm -r concept_classification_tmp_titles.txt')
        all_titles_to_load.saveAsTextFile('concept_classification_tmp_titles.txt')
        #res = os.system('hdfs dfs -cat concept_classification_tmp_titles.txt/*>concept_classification_tmp_titles.txt')
        res = 0
        if res!=0:
            print('ERROR getting titles from HDFS')
        for pair in all_titles_to_load.collect():
            titles_to_load[pair[0]] = pair[1]
        print('sample titles resolved, total ('+str(len(titles_to_load.keys()))+') titles')

        print('broadcasting titles_to_load ('+str(len(titles_to_load.keys()))+')')
        titles_to_loadBC = sc.broadcast(titles_to_load)
        print('titles_to_load broadcasted')
                
        print('loading vectors')
        ##vecs_data = sc.textFile(vecs_path)
        ##vecs = vecs_data.filter(lambda line: len(line)>1).flatMap(lambda line: check_vector(line)).filter(lambda line: line is not None).map(lambda pair: pair[0]+'\t'+pair[1])
        #for pair in vecs.collect():
        #    loaded_vecs[pair[0]] = pair[1]
        #print('vectors loaded')
        ##os.system('hdfs dfs -rm -r concept_classification_tmp_vecs.vecs.txt')
        ##vecs.saveAsTextFile('concept_classification_tmp_vecs.vecs.txt')
        vecs = sc.textFile('concept_classification_tmp_vecs.vecs.txt')
        #res = os.system('hdfs dfs -cat concept_classification_tmp_vecs.vecs.txt/*>concept_classification_tmp_vecs.vecs.txt')
        res = 0
        if res!=0:
            print('ERROR getting vectors from HDFS')
        else:            
            print('loading needed vectors')
            with io.open('concept_classification_tmp_vecs.vecs.txt','r',encoding='utf-8') as vecs_file:            
                for line in vecs_file:##vecs.collect():#vecs_file:
                    line = line.strip('\n')
                    if len(line)>0:
                        tokens = line.split('\t')
                        if len(tokens)!=2:
                            print('invalid line: '+line)
                        else:
                            loaded_vecs[tokens[0]] = np.array([float(x) for x in tokens[1].split(',')])
            
            print('we need ('+str(len(loaded_vecs.keys()))+') vectors of size ('+str(getsizeof(loaded_vecs))+')')

            print('broadcasting vectors ('+str(len(loaded_vecs.keys()))+')')
            loaded_vecsBC = sc.broadcast(loaded_vecs)
            print('vectors broadcasted')  

            print("loading topics")
            topics = {}
            topics_dic = {}
            top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
            cur_topic = 1
            with io.open(topics_inpath,'r',encoding='utf-8') as topics_file:
                for i,line in enumerate(topics_file):
                    line = line.strip('\n')
                    #print('line: '+str(i))
                    if len(line)>0:
                        topic_name = line[0:line.find('\t')]
                        if topic_name in topics:
                            line = line[line.find('\t')+1:]
                            topic_name = line[0:line.find('\t')]
                        topic_vector = line[line.rfind('\t')+1:]
                        accum_vector = []
                        max_concept_weight = 0.0
                        for j,concept in enumerate(topic_vector.split(';')):
                            #print('concept: '+str(i)+'_'+str(j))
                            if len(concept)>0:
                                concept_name = concept[0:concept.find(',')]
                                concept_weight = float(concept[concept.rfind(',')+1:])
                                if concept_weight>max_concept_weight:
                                    max_concept_weight = concept_weight
                                # look up concept id
                                #print(type(concept_name))
                                #print(type(titles_to_load.keys()[0]))
                                concept_id = titles_to_load[concept_name]
                                if concept_id != u'NOT_-_FOUND':
                                    # accumulate weighted concept vector
                                    accum_vector.append((np.array(loaded_vecs[concept_id]),concept_weight,concept_name))#np.array(model['12806'])
                                else:
                                    print(concept_name.encode('utf-8')+'...not found')
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
        
            print('broadcasting topics_dic ('+str(len(topics_dic.keys()))+')')
            topics_dicBC = sc.broadcast(topics_dic)
            print('topics_dic broadcasted')

            print('broadcasting topics ('+str(len(topics.keys()))+')')
            topicsBC = sc.broadcast(topics)
            print('topics broadcasted')      
            
            print("processing samples")
            samples = sc.textFile(samples_inpath_hdfs)
            results = samples.flatMap(lambda x : get_max_similarity_spark(x)).filter(lambda line: line is not None)
            os.system('hdfs dfs -rm -r '+output_path_hdfs)
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
                    if topic_name in topics:
                        line = line[line.find('\t')+1:]
                        topic_name = line[0:line.find('\t')]
                    topic_vector = line[line.rfind('\t')+1:]
                    for concept in topic_vector.split(';'):
                        if len(concept)>0:
                            concept_name = concept[0:concept.find(',')]
                            if concept_name in skipgram_esa_mappings_dic:
                                concept_name = skipgram_esa_mappings_dic[concept_name]
                            # look up concept id
                            try:
                                if mode=='spark_anno': # anno titles
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in titles:
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
                            if concept_name in skipgram_esa_mappings_dic:
                                concept_name = skipgram_esa_mappings_dic[concept_name]

                            # look up concept id
                            try:
                                if mode=='spark_anno': # anno titles
                                    concept_id = titles[concept_name]
                                else:
                                    if concept_name in titles:
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
