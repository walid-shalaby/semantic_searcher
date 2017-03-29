# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

#spark-submit --driver-memory 13g --executor-memory 13g --py-files commons.py cosines_extractor.py ./concept_classification_cosines.txt ./concept_classification_cosines.0.85.txt 0.85
#nohup nohup python3 cosines_extractor.py concept_classification_cosines.txt ./concept_classification_cosines_sparse.txt 0.00001 local >& concept_classification_cosines_sparse.txt.out &
#spark-submit --driver-memory 13g --executor-memory 13g --py-files commons.py cosines_extractor.py concept_classification_anno_cosines.txt ./concept_classification_anno_cosines.0.85.txt 0.85
#nohup python3 cosines_extractor.py concept_classification_anno_cosines.txt ./concept_classification_anno_cosines_sparse.txt 0.00001 local >& concept_classification_anno_cosines_sparse.txt.out &

#spark-submit --driver-memory 4g --executor-memory 4g --conf spark.driver.maxResultSize=4g --conf spark.kryoserializer.buffer.max=1024m --py-files commons.py cosines_extractor.py concept_classification_anno_cosines.txt ./concept_classification_anno_cosines_sparse.txt 0.1

import sys
import io
import os
import multiprocessing
from multiprocessing import Pool
from itertools import repeat

def write_header(output, title, dic):
    line = title
    for title,id in dic.items():
        line += '\t'+str(id)+':'+title

    output.write(line+os.linesep)

def write_line(output, title, dic):
    line = title
    for id,cosine in dic.items():
        line += '\t'+str(id)+':'+str(cosine)

    output.write(line+os.linesep)

mode = 'spark'
cosines_inpath = sys.argv[1]#'concept_classification_anno_cosines.txt'
outpath = sys.argv[2]#'concept_classification_anno_cosines.0.85.txt'
threshold = float(sys.argv[3])
if len(sys.argv)>4:
    mode = sys.argv[4]
    
if mode == 'spark':
    from commons import load_cosines_spark
    from pyspark import SparkContext
    sc = SparkContext(appName="concept classification- extract cosines")
elif mode=='local':
    from commons import load_cosines_line

from commons import get_topic_titles

with io.open(outpath,'w',encoding='utf-8') as output:
    if mode=='spark':
        all_cosines = sc.textFile(cosines_inpath)
        topic_titles = get_topic_titles(all_cosines.first())

        write_header(output, 'sample_title', topic_titles)
        
        print("loading cosines")
        needed_cosines = all_cosines.filter(lambda line: len(line)>1).flatMap(lambda line: load_cosines_spark(line,min_cosine=threshold)).filter(lambda line: line is not None)
        titles_cosines = {}
        for cosine in needed_cosines.collect():
            for sample_title,cosines_dic in cosine.items():
                write_line(output, sample_title, cosines_dic)
                
    elif mode=='local':
        with io.open(cosines_inpath,'r',encoding='utf-8') as inp:
            print("loading cosines")
            in_records = []
            for i,line in enumerate(inp):
                if (i+1)%10000==0:
                    print('processed (',i+1,') samples')
                    sys.stdout.flush()
                if i==0:
                    topic_titles = get_topic_titles(line)
                    write_header(output, 'sample_title', topic_titles)        
                
                in_records.append(line)
                if len(in_records)%(multiprocessing.cpu_count()-1)==0:
                    p = Pool(multiprocessing.cpu_count()-1)
                    cosines = p.map(load_cosines_line, zip(in_records,repeat(threshold)))                            
                    p.close()
                    p.join()
                    for cosine in cosines:
                        for sample_title,cosines_dic in cosine.items():
                            write_line(output, sample_title, cosines_dic)
                    in_records = []

            if len(in_records)>0:
                p = Pool(multiprocessing.cpu_count()-1)
                cosines = p.map(load_cosines_line, zip(in_records,repeat(threshold)))                            
                p.close()
                p.join()
                for cosine in cosines:
                    for sample_title,cosines_dic in cosine.items():
                        write_line(output, sample_title, cosines_dic)