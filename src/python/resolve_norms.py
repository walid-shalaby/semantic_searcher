# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import os
import numpy as np

with open('sample_topic_sqr.txt','r') as topics:
    with open('SAMPLE_TOPIC_NORM.txt','w') as new_topics:
        for line in topics:
            try:
                tokens = line.strip('\n').split('\t')
                new_topics.write(tokens[0]+'\t'+tokens[1]+'\t'+str(np.sqrt(float(tokens[2])))+os.linesep)            
            except:
                print(line)
'''
with open('TOPIC_NORM.txt','r') as topics:
    with open('TOPIC_NORM_SQRT.txt','w') as new_topics:
        for line in topics:
            tokens = line.strip('\n').split('\t')
            new_topics.write(tokens[0]+'\t'+tokens[1]+'\t'+str(np.sqrt(float(tokens[1])))+os.linesep)

with open('SAMPLE_NORM.txt','r') as topics:
    with open('SAMPLE_NORM_SQRT.txt','w') as new_topics:
        for line in topics:
            tokens = line.strip('\n').split('\t')
            new_topics.write(tokens[0]+'\t'+tokens[1]+'\t'+str(np.sqrt(float(tokens[1])))+os.linesep)            
'''            