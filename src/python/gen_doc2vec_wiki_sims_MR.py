# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

#hdfs dfs -rm -r concepts-10iter.vecs.sims
#spark-submit --executor-memory 14G --total-executor-cores 100 gen_doc2vec_wiki_sims_MR.py concepts-10iter.vecs.txt concepts-10iter.vecs.sims
#hdfs dfs -text concepts-10iter.vecs.sims/* > /scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter.vecs.sims
#hdfs dfs -rm -r hdfs://taipan/user/wshalaby/concepts-10iter-sample.vecs.sims
#spark-submit gen_doc2vec_wiki_sims_MR.py concepts-10iter-sample.vecs.txt concepts-10iter-sample.vecs.sims
#hdfs dfs -text concepts-10iter-sample.vecs.sims/* > /scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter-sample.vecs.sims

from pyspark import SparkContext

'''
def load_vecs(filename):
    import pickle
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
'''

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

def tokenize_line(line):
    tokens = line.split('\t')
    if len(tokens)==2:
        return (tokens[0],[float(x) for x in tokens[1].split(',')])

def cosine(t): # t = (t1,t2), t1 = ('title1',vec1), t2 = ('title2',vec2)
    from scipy import spatial
    label1 = t[0][0]
    vec1 = t[0][1]
    label2 = t[1][0]
    vec2 = t[1][1]
    result = 1 - spatial.distance.cosine(vec1, vec2)
    return (label1, (label2, result))

def get_top_similarities(tup, topn): # tup is (label,array of (label,sim))
    import numpy as np
    label = tup[0]
    labels = []
    sims = []
    result = []
    for sim in tup[1]:
        labels.append(sim[0])
        sims.append(sim[1])
    indexes = np.argsort(sims)
    cnt = 0
    for i in range(0,len(indexes)):
        indx = indexes[len(indexes)-i-1]
        result.append(labels[indx]+':'+str(sims[indx]))
        cnt = cnt + 1
        if cnt==topn:
            break
    return label+'\t\t'+'\t'.join(result)

def main():
    import sys

    #format = sys.argv[1]
    #print('running in '+format+' format')

    inpath = sys.argv[1]
    outpath = sys.argv[2]

    sc = SparkContext(appName="CosineSimilarity")

    #vecs_data = load_vecs('/scratch/wshalaby/doc2vec/models/word2vec/concepts-10iter.vecs.txt',format)
    #vecs = sc.parallelize(vecs_data)
    vecs = sc.textFile(inpath).map(lambda line: tokenize_line(line))
    cross_vecs = vecs.cartesian(vecs).filter(lambda ((a,b),(c,d)): a!=c)
    similarities = cross_vecs.map(lambda record: cosine(record)).filter(lambda (a,(b,sim)): sim>=0.6)
    top_similarities = similarities.groupByKey().map(lambda x: get_top_similarities(x,50))
    top_similarities.saveAsTextFile(outpath)

main()
