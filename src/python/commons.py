# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(walid-shalaby)s
"""

import csv
import sys
import numpy as np
import os
from scipy import spatial
import multiprocessing
from multiprocessing import Pool

ng_topic_mappings = {'rec.motorcycles':'autos.sports', 'comp.windows.x':'computer', 'talk.politics.guns':'politics', 'talk.politics.mideast':'politics', 'talk.religion.misc':'religion', 'rec.autos':'autos.sports', 'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports', 'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer', 'sci.space':'science', 'talk.politics.misc':'politics', 'sci.electronics':'science', 'comp.graphics':'computer', 'comp.os.ms.windows.misc':'computer', 'sci.med':'science', 'sci.crypt':'science', 'soc.religion.christian':'religion', 'alt.atheism':'religion', 'misc.forsale':'sale'}

class SampleInfo:
     def __init__(self):
         self.max_topic_id = None
         self.max_topic_name = None
         self.sample_class = None
         self.sample_name = None
         self.max_all_sim = 0.0

from collections import namedtuple
Sample = namedtuple('Sample',['sample_info','sample_vec'])
MaxInstance = namedtuple('MaxInstance',['sample_info','score','diff'])

def getWeight(item):
    return item[0]


def load_topics(topics_inpath, conc_sep=';;;;', weight_sep=',,,,',multi=False):
    # multi indicates that the topic might have mutiple representations (vectors) in 
    # the given file. Useful when creating topic vectors using few labeled samples
    cur_topic = 1
    topics = {}
    topics_dic = {}
    with open(topics_inpath,'r') as topics_file:
        for line in topics_file:
            line = line.strip('\n')
            if len(line)>0:
                topic_name = line[0:line.find('\t')]
                if topic_name in topics and multi==False:
                    line = line[line.find('\t')+1:]
                    topic_name = line[0:line.find('\t')]
                topic_vector = line[line.rfind('\t')+1:]
                accum_vector = []
                max_concept_weight = 0.0
                for concept in topic_vector.split(conc_sep):
                    if len(concept)>0:
                        concept_name = concept[0:concept.find(weight_sep)]
                        concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                        if concept_weight>max_concept_weight:
                            max_concept_weight = concept_weight                            
                        accum_vector.append((concept_weight,concept_name))                                                
                # normalize weights
                #print('max',max_concept_weight)
                #accum_vector = [(weight/max_concept_weight,concept_name) for (concept_vector,weight,concept_name) in accum_vector]
                # calculate topic norm
                topic_norm =  np.linalg.norm([weight for (weight,concept_name) in accum_vector])
                print(os.linesep+'TOPIC_NORM\t'+topic_name+'\t'+str(topic_norm)+os.linesep)
                if topic_name not in topics:
                    topics[topic_name] = (accum_vector, topic_norm)
                    topics_dic[topic_name] = cur_topic
                    cur_topic = cur_topic + 1
                elif multi==True:
                    cur_accum_vector, cur_topic_norm = topics[topic_name]
                    if isinstance(cur_accum_vector[0],list)==True:
                        cur_accum_vector.append(accum_vector)
                        cur_topic_norm.append(topic_norm)
                    else:
                        cur_accum_vector = [cur_accum_vector,accum_vector]
                        cur_topic_norm = [cur_topic_norm,topic_norm]
                    topics[topic_name] = (cur_accum_vector,cur_topic_norm)
                else:
                    print('error, duplicate topics unexpected')
                    return
    return topics, topics_dic


def flush_samples_text(samples_inpath, output_path):
    import os
    samples = []
    with open(samples_inpath,'r') as samples_file:
        with open(output_path,'w') as output:
            for line in samples_file:
                line = line.strip('\n')
                if len(line)>0:
                    sample = line[0:line.find('\t')].split('\\')
                    sample_name = sample[len(sample)-1]
                    sample_class = sample[len(sample)-2]
                    sample_name = sample_name+'_'+sample_class
                    sample_text = line[line.find('\t')+1:line.rfind('\t')]
                    output.write('\t'.join([sample_class,sample_name,sample_text])+os.linesep)


def load_samples(samples_inpath):
    samples = []
    with open(samples_inpath,'r') as samples_file:
        for line in samples_file:
            line = line.strip('\n')
            if len(line)>0:
                samples.append(line)
    return samples


def classify_samples_bottomup(samples=None, topics=None, topics_dic=None, dims=[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, conc_sep=';;;;', weight_sep=',,,,', outpath=None):
    dims.reverse()
    for dim in dims:
        # Initialize  correct counts
        sub_confusion_dic = {}
        super_confusion_dic = {}        
        super_total_correct = {}
        sup_total_correct = {}
        sub_cnt = 0
        super_cnt = 0
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0
            sub_confusion_dic[k] = sub_cnt
            sub_cnt += 1
            if v not in super_confusion_dic:
                super_confusion_dic[v] = super_cnt
                super_cnt += 1            
        # allocate confusion matrices
        super_confusion_mat = np.zeros((super_cnt,super_cnt))
        sub_confusion_mat = np.zeros((sub_cnt,sub_cnt))        
        # final result
        result = ''
        total_correct_sup = 0
        total_correct_super = 0
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            # calculate sample norm
            sample_norm = np.linalg.norm([weight for (weight,concept_name) in accum_vector])    
            for k,v in topics.items(): # loop on topics
                if k not in topic_mappings:
                    continue
                cur_sim = 0.0
                topic_norm = []
                all_topic_norm = v[1]
                topic_accum_vector = v[0]
                topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
                # calculate topic norm
                topic_norm = np.linalg.norm([weight for (weight,concept_name) in topic_accum_vector])                
                for sample_weight,sample_concept_name in accum_vector: # loop on each concept in sample
                    max_sim = 0.0
                    max_sim_weight = 0.0
                    for topic_weight,topic_concept_name in topic_accum_vector: # loop on each concept in topic                    
                        if topic_concept_name==sample_concept_name:
                            max_sim = 1.0
                            max_sim_weight = topic_weight
                            break                    
                    cur_sim += max_sim_weight*sample_weight*max_sim
                cur_sim = cur_sim/(sample_norm*topic_norm)
                result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(cur_sim)+os.linesep
                if cur_sim>sample_info.max_all_sim:
                    sample_info.max_all_sim = cur_sim
                    sample_info.max_topic_name = k
                    sample_info.max_topic_id = topics_dic[k]
            if sample_info.max_topic_name is not None:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
            else:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\tNone\tNone\t'+str(sample_info.max_all_sim)+os.linesep
            if sample_info.sample_class==sample_info.max_topic_name:
                total_correct_sup += 1
                sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
            if sample_info.max_topic_name is not None and topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
                total_correct_super += 1
                super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1
            if sample_info.max_topic_name is not None: # update confusion matrix
                sub_confusion_mat[sub_confusion_dic[sample_info.sample_class]][sub_confusion_dic[sample_info.max_topic_name]] += 1
                super_confusion_mat[super_confusion_dic[topic_mappings[sample_info.sample_class]]][super_confusion_dic[topic_mappings[sample_info.max_topic_name]]] += 1
            if total%100==0:
                print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        #print confusion matrix
        ks = ['']*len(sub_confusion_dic)
        for k,v in sub_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in sub_confusion_mat[i]])+os.linesep
        ks = ['']*len(super_confusion_dic)
        for k,v in super_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in super_confusion_mat[i]])+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_topdown(samples=None, super_topics=None, topics=None, topics_dic=None, dims=[1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, conc_sep=';;;;', weight_sep=',,,,', outpath=None):
    dims.reverse()
    for dim in dims:
        # Initialize  correct counts
        sub_confusion_dic = {}
        super_confusion_dic = {}        
        super_total_correct = {}
        sup_total_correct = {}
        sub_cnt = 0
        super_cnt = 0
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0
            sub_confusion_dic[k] = sub_cnt
            sub_cnt += 1
            if v not in super_confusion_dic:
                super_confusion_dic[v] = super_cnt
                super_cnt += 1            
        # allocate confusion matrices
        super_confusion_mat = np.zeros((super_cnt,super_cnt))
        sub_confusion_mat = np.zeros((sub_cnt,sub_cnt))        
        # final result
        result = ''
        total_correct_sup = 0
        total_correct_super = 0
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            # calculate sample norm
            sample_norm = np.linalg.norm([weight for (weight,concept_name) in accum_vector])    
            max_super_sim = 0.0
            max_super_name = None
            for ksuper,vsuper in super_topics.items():
                if ksuper not in topic_mappings.values():
                    continue
                cur_sim = 0.0
                topic_norm = []
                all_topic_norm = vsuper[1]
                topic_accum_vector = vsuper[0]
                topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
                # calculate topic norm
                topic_norm = np.linalg.norm([weight for (weight,concept_name) in topic_accum_vector])                
                for sample_weight,sample_concept_name in accum_vector: # loop on each concept in sample
                    max_sim = 0.0
                    max_sim_weight = 0.0
                    for topic_weight,topic_concept_name in topic_accum_vector: # loop on each concept in topic                    
                        if topic_concept_name==sample_concept_name:
                            max_sim = 1.0
                            max_sim_weight = topic_weight
                            break                    
                    cur_sim += max_sim_weight*sample_weight*max_sim
                cur_sim = cur_sim/(sample_norm*topic_norm)
                result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+ksuper+'\t'+str(cur_sim)+os.linesep
                if cur_sim>max_super_sim:
                    max_super_sim = cur_sim
                    max_super_name = ksuper
            if max_super_name is not None:
                for k,v in topics.items(): # loop on topics
                    if k not in topic_mappings or topic_mappings[k]!=max_super_name:
                        continue
                    cur_sim = 0.0
                    topic_norm = []
                    all_topic_norm = v[1]
                    topic_accum_vector = v[0]
                    topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
                    # calculate topic norm
                    topic_norm = np.linalg.norm([weight for (weight,concept_name) in topic_accum_vector])                
                    for sample_weight,sample_concept_name in accum_vector: # loop on each concept in sample
                        max_sim = 0.0
                        max_sim_weight = 0.0
                        for topic_weight,topic_concept_name in topic_accum_vector: # loop on each concept in topic                    
                            if topic_concept_name==sample_concept_name:
                                max_sim = 1.0
                                max_sim_weight = topic_weight
                                break                    
                        cur_sim += max_sim_weight*sample_weight*max_sim
                    cur_sim = cur_sim/(sample_norm*topic_norm)
                    result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(cur_sim)+os.linesep
                    if cur_sim>sample_info.max_all_sim:
                        sample_info.max_all_sim = cur_sim
                        sample_info.max_topic_name = k
                        sample_info.max_topic_id = topics_dic[k]
                if sample_info.max_topic_name is not None:
                    result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
                else:
                    result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\tNone\tNone\t'+str(sample_info.max_all_sim)+os.linesep
                if sample_info.sample_class==sample_info.max_topic_name:
                    total_correct_sup += 1
                    sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
                if sample_info.max_topic_name is not None and topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
                    total_correct_super += 1
                    super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1
                if sample_info.max_topic_name is not None: # update confusion matrix
                    sub_confusion_mat[sub_confusion_dic[sample_info.sample_class]][sub_confusion_dic[sample_info.max_topic_name]] += 1
                    super_confusion_mat[super_confusion_dic[topic_mappings[sample_info.sample_class]]][super_confusion_dic[topic_mappings[sample_info.max_topic_name]]] += 1
            if total%100==0:
                print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        #print confusion matrix
        ks = ['']*len(sub_confusion_dic)
        for k,v in sub_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in sub_confusion_mat[i]])+os.linesep
        ks = ['']*len(super_confusion_dic)
        for k,v in super_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in super_confusion_mat[i]])+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_bottomup(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted=True):
    for dim in dims:
        # final result
        result = ''
        # Initialize  correct counts
        sub_confusion_dic = {}
        super_confusion_dic = {}
        super_total_correct = {}
        sup_total_correct = {}
        sub_cnt = 0
        super_cnt = 0
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0
            sub_confusion_dic[k] = sub_cnt
            sub_cnt += 1
            if v not in super_confusion_dic:
                super_confusion_dic[v] = super_cnt
                super_cnt += 1            
        # allocate confusion matrices
        super_confusion_mat = np.zeros((super_cnt,super_cnt))
        sub_confusion_mat = np.zeros((sub_cnt,sub_cnt))
        #calcuate aggregated vector for all the topics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = float(not weighted)
            topic_accum_vectors = v[0]
            # see if the topic have mutiple representations (vectors) 
            # useful when creating topic vectors using few labeled samples
            if not isinstance(topic_accum_vectors[0],list):
                topic_accum_vectors = [v[0]]
            sub_topics_vecs[k] = []
            for topic_accum_vector in topic_accum_vectors:
                topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
                vec =  np.zeros(model.vector_size)
                for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                    try:
                        if mode=='conc':
                            if weighted==True:
                                vec += sample_weight*model[concept_name]
                            else:
                                vec += model[concept_name]
                            accum_weight += sample_weight
                        elif mode=='w2v':
                            if weighted==True:
                                vec += sample_weight*model['id'+titles[concept_name]+'di']
                                accum_weight += sample_weight
                            else:
                                vec += model['id'+titles[concept_name]+'di']                        
                    except:
                        print(os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep)
                        result+=os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep
                        continue
                if (vec==np.zeros(model.vector_size)).all()==True:
                    print(os.linesep+'FATAL\t'+k+'...no concepts found'+os.linesep)
                    return
                vec /= accum_weight
                sub_topics_vecs[k].append(vec)
        total_correct_sup = 0
        total_correct_super = 0
        discount = 0
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = float(not weighted)
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            sample_vec += sample_weight*model[concept_name]
                            accum_weight += sample_weight
                        else:
                            sample_vec += model[concept_name]                        
                    elif mode=='w2v':
                        if weighted==True:
                            sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                        else:
                            sample_vec += model['id'+titles[concept_name]+'di']                            
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            if (sample_vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+sample_info.sample_name+'...no concepts found'+os.linesep)
                discount += 1
                continue
            sample_vec /= accum_weight
            max_sim = 0
            max_topic = ''
            for k,v in topics.items(): # loop on topics
                if k not in topic_mappings:
                    continue
                for vec in sub_topics_vecs[k]:
                    sim = 1 - spatial.distance.cosine(vec,sample_vec)
                    result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(sim)+os.linesep
                    if sim>sample_info.max_all_sim:
                        sample_info.max_all_sim = sim
                        sample_info.max_topic_name = k
                        sample_info.max_topic_id = topics_dic[k]
            if sample_info.max_topic_name is not None:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
            else:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\tNone\tNone\t'+str(sample_info.max_all_sim)+os.linesep
            if sample_info.sample_class==sample_info.max_topic_name:
                total_correct_sup += 1
                sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
            if sample_info.max_topic_name is not None and topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
                total_correct_super += 1
                super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1
            if sample_info.max_topic_name is not None: # update confusion matrix
                sub_confusion_mat[sub_confusion_dic[sample_info.sample_class]][sub_confusion_dic[sample_info.max_topic_name]] += 1
                super_confusion_mat[super_confusion_dic[topic_mappings[sample_info.sample_class]]][super_confusion_dic[topic_mappings[sample_info.max_topic_name]]] += 1
            total -= discount
            if total%100==0:
                print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        #print confusion matrix
        ks = ['']*len(sub_confusion_dic)
        for k,v in sub_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in sub_confusion_mat[i]])+os.linesep
        ks = ['']*len(super_confusion_dic)
        for k,v in super_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in super_confusion_mat[i]])+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)

def classify_samples_with_densification_bottomup_before_topics_from_samples(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted=True):
    for dim in dims:
        # final result
        result = ''
        # Initialize  correct counts
        sub_confusion_dic = {}
        super_confusion_dic = {}
        super_total_correct = {}
        sup_total_correct = {}
        sub_cnt = 0
        super_cnt = 0
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0
            sub_confusion_dic[k] = sub_cnt
            sub_cnt += 1
            if v not in super_confusion_dic:
                super_confusion_dic[v] = super_cnt
                super_cnt += 1            
        # allocate confusion matrices
        super_confusion_mat = np.zeros((super_cnt,super_cnt))
        sub_confusion_mat = np.zeros((sub_cnt,sub_cnt))
        #calcuate aggregated vector for all the topics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = float(not weighted)
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            vec += sample_weight*model[concept_name]
                        else:
                            vec += model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        if weighted==True:
                            vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                        else:
                            vec += model['id'+titles[concept_name]+'di']                        
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep
                    continue
            if (vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+k+'...no concepts found'+os.linesep)
                return
            vec /= accum_weight
            sub_topics_vecs[k] = vec            
        total_correct_sup = 0
        total_correct_super = 0
        discount = 0
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = float(not weighted)
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            sample_vec += sample_weight*model[concept_name]
                            accum_weight += sample_weight
                        else:
                            sample_vec += model[concept_name]                        
                    elif mode=='w2v':
                        if weighted==True:
                            sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                        else:
                            sample_vec += model['id'+titles[concept_name]+'di']                            
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            if (sample_vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+sample_info.sample_name+'...no concepts found'+os.linesep)
                discount += 1
                continue
            sample_vec /= accum_weight
            max_sim = 0
            max_topic = ''
            for k,v in topics.items(): # loop on topics
                if k not in topic_mappings:
                    continue
                sim = 1 - spatial.distance.cosine(sub_topics_vecs[k],sample_vec)
                result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(sim)+os.linesep
                if sim>sample_info.max_all_sim:
                    sample_info.max_all_sim = sim
                    sample_info.max_topic_name = k
                    sample_info.max_topic_id = topics_dic[k]
            if sample_info.max_topic_name is not None:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
            else:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\tNone\tNone\t'+str(sample_info.max_all_sim)+os.linesep
            if sample_info.sample_class==sample_info.max_topic_name:
                total_correct_sup += 1
                sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
            if sample_info.max_topic_name is not None and topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
                total_correct_super += 1
                super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1
            if sample_info.max_topic_name is not None: # update confusion matrix
                sub_confusion_mat[sub_confusion_dic[sample_info.sample_class]][sub_confusion_dic[sample_info.max_topic_name]] += 1
                super_confusion_mat[super_confusion_dic[topic_mappings[sample_info.sample_class]]][super_confusion_dic[topic_mappings[sample_info.max_topic_name]]] += 1
            total -= discount
            if total%100==0:
                print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        #print confusion matrix
        ks = ['']*len(sub_confusion_dic)
        for k,v in sub_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in sub_confusion_mat[i]])+os.linesep
        ks = ['']*len(super_confusion_dic)
        for k,v in super_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in super_confusion_mat[i]])+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)

def classify_samples_with_densification_bottomup_bootstrap_multithread(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted=True):
    params = []
    for dim in dims:
        params.append({'dims':[dim],'samples':samples, 'topics':topics, 'topics_dic':topics_dic, 'topic_mappings':topic_mappings, 'model':model, 'conc_sep':conc_sep, 'weight_sep':weight_sep, 'mode':mode, 'titles':titles, 'outpath':outpath, 'weighted':weighted})
    p = Pool(multiprocessing.cpu_count()-1)
    p.map(classify_samples_with_densification_bottomup_bootstrap_thread,params)


def classify_samples_with_densification_bottomup_bootstrap_thread(params):
    classify_samples_with_densification_bottomup_bootstrap(samples=params['samples'], topics=params['topics'], topics_dic=params['topics_dic'], dims=params['dims'], topic_mappings=params['topic_mappings'], model=params['model'], conc_sep=params['conc_sep'], weight_sep=params['weight_sep'], mode=params['mode'], titles=params['titles'], outpath=params['outpath'], weighted=params['weighted'])


def classify_samples_with_densification_bottomup_bootstrap(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted=True, weighted_bs=True, min_bs_score=0.0, min_bs_diff=0.0):
    default_max_instance = MaxInstance(None,-1.0, 100)
    for dim in dims:
        # final result
        result = ''
        # Initialize  correct counts
        sub_confusion_dic = {}
        super_confusion_dic = {}
        super_total_correct = {}
        sup_total_correct = {}
        sub_cnt = 0
        super_cnt = 0
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0
            sub_confusion_dic[k] = sub_cnt
            sub_cnt += 1
            if v not in super_confusion_dic:
                super_confusion_dic[v] = super_cnt
                super_cnt += 1            
        # allocate confusion matrices
        super_confusion_mat = np.zeros((super_cnt,super_cnt))
        sub_confusion_mat = np.zeros((sub_cnt,sub_cnt))
        #calcuate aggregated vector for all the topics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = float(not weighted)
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            vec += sample_weight*model[concept_name]
                        else:
                            vec += model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        if weighted==True:
                            vec += sample_weight*model['id'+titles[concept_name]+'di']
                        else:
                            vec += model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep
                    continue
            if (vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+k+'...no concepts found'+os.linesep)
                return
            vec /= accum_weight
            sub_topics_vecs[k] = vec            
        total_correct_sup = 0
        total_correct_super = 0
        # process samples
        # compute dense representation
        samples_dic = {}
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = float(not weighted)
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            sample_vec += sample_weight*model[concept_name]
                        else:
                            sample_vec += model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        if weighted==True:
                            sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                        else:
                            sample_vec += model['id'+titles[concept_name]+'di']                            
                except:
                    #print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    #result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            if (sample_vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+sample_info.sample_name+'...no concepts found'+os.linesep)
                continue
            sample_vec /= accum_weight
            samples_dic[sample_info.sample_name] = Sample(sample_info, sample_vec)
        all_samples = set(samples_dic.keys())
        label_max_instance = dict(zip(topic_mappings.keys(),[default_max_instance for _ in topic_mappings.keys()]))
        label_weighted_scores = dict(zip(topic_mappings.keys(),len(topic_mappings)*[1]))
        total = 0
        Topic = namedtuple('Topic',['name','score'])
        while len(all_samples)>0:
            for sample_name,sample in samples_dic.items():
                if sample_name in all_samples: # not assigned yet
                    min_diff = 100.0
                    best_topic = Topic('',-1.0)
                    for topic in topic_mappings: # loop on topics
                        sim = abs(1 - spatial.distance.cosine(sub_topics_vecs[topic],sample.sample_vec))
                        min_diff = min(min_diff,abs(best_topic.score-sim))
                        if best_topic.score<sim: # this is the best topic for this sample so far
                            best_topic = Topic(topic,sim)                        
                    for topic in topic_mappings: # check if this sample is better than the best so far
                        if topic==best_topic.name and label_max_instance[topic].score<best_topic.score:
                            max_instance = MaxInstance(sample.sample_info,best_topic.score, min_diff)
                            label_max_instance[topic] = max_instance
            for topic, max_instance in label_max_instance.items():
                if max_instance is not default_max_instance:
                    total += 1
                    result += os.linesep+'SIMILA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+topic+'\t'+str(topics_dic[topic])+'\t'+str(max_instance.score)+os.linesep
                    if max_instance.sample_info.sample_class==topic:
                        total_correct_sup += 1
                        sup_total_correct[topic] = sup_total_correct[topic]+1
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+topic+'\t'+str(topics_dic[topic])+'\t'+str(max_instance.score)+os.linesep
                    #else:
                    #    print(topic,max_instance.sample_info.sample_class)
                    if topic_mappings[max_instance.sample_info.sample_class]==topic_mappings[topic]:
                        total_correct_super += 1
                        super_total_correct[topic_mappings[topic]] = super_total_correct[topic_mappings[topic]]+1
                    if topic is not None: # update confusion matrix
                        sub_confusion_mat[sub_confusion_dic[max_instance.sample_info.sample_class]][sub_confusion_dic[topic]] += 1
                        super_confusion_mat[super_confusion_dic[topic_mappings[max_instance.sample_info.sample_class]]][super_confusion_dic[topic_mappings[topic]]] += 1
                    if total%100==0:
                        print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
                    # update topic definition
                    if min_bs_score<=max_instance.score and max_instance.diff>=min_bs_diff:
                        if weighted_bs==True:
                            sub_topics_vecs[topic] *= label_weighted_scores[topic]
                            sub_topics_vecs[topic] += max_instance.score*samples_dic[max_instance.sample_info.sample_name].sample_vec
                            label_weighted_scores[topic] += max_instance.score
                            sub_topics_vecs[topic] /= label_weighted_scores[topic]
                        else:
                            sub_topics_vecs[topic] *= label_weighted_scores[topic]
                            sub_topics_vecs[topic] += 1.0*samples_dic[max_instance.sample_info.sample_name].sample_vec
                            label_weighted_scores[topic] += 1
                            sub_topics_vecs[topic] /= label_weighted_scores[topic]
                    if max_instance.sample_info.sample_name in all_samples:
                        all_samples.remove(max_instance.sample_info.sample_name)
                    else: # a tie has happened, should no more happen after using max_topic_sim and max_topic_name
                        print(os.linesep+'TIE\t'+topic+'\t'+max_instance.sample_info.sample_name+os.linesep)
                        result+=os.linesep+'TIE\t'+topic+'\t'+max_instance.sample_info.sample_name+os.linesep                                                
                    label_max_instance[topic] = default_max_instance
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        #print confusion matrix
        ks = ['']*len(sub_confusion_dic)
        for k,v in sub_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in sub_confusion_mat[i]])+os.linesep
        ks = ['']*len(super_confusion_dic)
        for k,v in super_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in super_confusion_mat[i]])+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_bottomup_bootstrap_on_topics(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted=True):
    default_max_instance = MaxInstance(None,-1.0)
    for dim in dims:
        # final result
        result = ''
        # Initialize  correct counts
        sub_confusion_dic = {}
        super_confusion_dic = {}
        super_total_correct = {}
        sup_total_correct = {}
        sub_cnt = 0
        super_cnt = 0
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0
            sub_confusion_dic[k] = sub_cnt
            sub_cnt += 1
            if v not in super_confusion_dic:
                super_confusion_dic[v] = super_cnt
                super_cnt += 1            
        # allocate confusion matrices
        super_confusion_mat = np.zeros((super_cnt,super_cnt))
        sub_confusion_mat = np.zeros((sub_cnt,sub_cnt))
        #calcuate aggregated vector for all the topics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = float(not weighted)
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            vec += sample_weight*model[concept_name]
                        else:
                            vec += model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        if weighted==True:
                            vec += sample_weight*model['id'+titles[concept_name]+'di']
                        else:
                            vec += model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            vec /= accum_weight
            sub_topics_vecs[k] = vec            
        total_correct_sup = 0
        total_correct_super = 0
        # process samples
        # compute dense representation
        samples_dic = {}
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = float(not weighted)
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            sample_vec += sample_weight*model[concept_name]
                        else:
                            sample_vec += model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        if weighted==True:
                            sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                        else:
                            sample_vec += model['id'+titles[concept_name]+'di']                            
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            sample_vec /= accum_weight
            samples_dic[sample_info.sample_name] = Sample(sample_info, sample_vec)
        all_samples = set(samples_dic.keys())
        total = 0
        '''
        while len(all_samples)>0:
            topic_max_instance = dict(zip(topic_mappings.keys(),[default_max_instance for _ in topic_mappings.keys()]))
            for sample_name,sample in samples_dic.items():
                if sample_name in all_samples: # not assigned yet
                    max_topic_sim = -1.0 # to solve ties
                    max_topic_name = '' # to solve ties
                    for k,v in topics.items(): # loop on topics
                        if k not in topic_mappings:
                            continue
                        sim = 1 - spatial.distance.cosine(sub_topics_vecs[k],sample.sample_vec)
                        if topic_max_instance[k].score<sim: # this is a candidate topic, but need to make sure it is the best one so far or ties might happen
                            if max_topic_sim<sim:
                                max_topic_sim = sim
                                max_topic_name = k
                    if max_topic_sim>-1.0:
                        max_instance = MaxInstance(sample.sample_info,max_topic_sim)
                        topic_max_instance[max_topic_name] = max_instance
            for topic, max_instance in topic_max_instance.items():
                if max_instance.score>-1.0:
                    total += 1
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+topic+'\t'+str(topics_dic[topic])+'\t'+str(max_instance.score)+os.linesep
                    if max_instance.sample_info.sample_class==topic:
                        total_correct_sup += 1
                        sup_total_correct[topic] = sup_total_correct[topic]+1
                    #else:
                    #    print(topic,max_instance.sample_info.sample_class)
                    if topic_mappings[max_instance.sample_info.sample_class]==topic_mappings[topic]:
                        total_correct_super += 1
                        super_total_correct[topic_mappings[topic]] = super_total_correct[topic_mappings[topic]]+1
                    if topic is not None: # update confusion matrix
                        sub_confusion_mat[sub_confusion_dic[max_instance.sample_info.sample_class]][sub_confusion_dic[topic]] += 1
                        super_confusion_mat[super_confusion_dic[topic_mappings[max_instance.sample_info.sample_class]]][super_confusion_dic[topic_mappings[topic]]] += 1
                    if total%100==0:
                        print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
                    # update topic definition
                    sub_topics_vecs[topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec
                    topic_max_instance[topic] = default_max_instance
                    if max_instance.sample_info.sample_name in all_samples:
                        all_samples.remove(max_instance.sample_info.sample_name)
                    else: # a tie has happened, should no more happen after using max_topic_sim and max_topic_name
                        print(os.linesep+'TIE\t'+topic+'\t'+max_instance.sample_info.sample_name+os.linesep)
                        result+=os.linesep+'TIE\t'+topic+'\t'+max_instance.sample_info.sample_name+os.linesep
        '''
        while len(all_samples)>0:
            for topic in topics: # loop on topics
                if topic not in topic_mappings: # not a target topic
                    continue
                max_instance = default_max_instance
                for sample_name,sample in samples_dic.items(): # choose most similar sample
                    if sample_name in all_samples: # not assigned yet
                        sim = 1 - spatial.distance.cosine(sub_topics_vecs[topic],sample.sample_vec)
                        if max_instance.score<sim: # update most similar sample
                            max_instance = MaxInstance(sample.sample_info,sim)
                if max_instance is not default_max_instance:
                    total += 1                                           
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+topic+'\t'+str(topics_dic[topic])+'\t'+str(max_instance.score)+os.linesep
                    if max_instance.sample_info.sample_class==topic:
                        total_correct_sup += 1
                        sup_total_correct[topic] = sup_total_correct[topic]+1
                    #else:
                    #    print(topic,max_instance.sample_info.sample_class)
                    if topic_mappings[max_instance.sample_info.sample_class]==topic_mappings[topic]:
                        total_correct_super += 1
                        super_total_correct[topic_mappings[topic]] = super_total_correct[topic_mappings[topic]]+1
                    if topic is not None: # update confusion matrix
                        sub_confusion_mat[sub_confusion_dic[max_instance.sample_info.sample_class]][sub_confusion_dic[topic]] += 1
                        super_confusion_mat[super_confusion_dic[topic_mappings[max_instance.sample_info.sample_class]]][super_confusion_dic[topic_mappings[topic]]] += 1
                    if total%100==0:
                        print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
                    # update topic definition
                    sub_topics_vecs[topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec
                    if max_instance.sample_info.sample_name in all_samples:
                        all_samples.remove(max_instance.sample_info.sample_name)
                    else: # a tie has happened, should no more happen after using max_topic_sim and max_topic_name
                        print(os.linesep+'TIE\t'+topic+'\t'+max_instance.sample_info.sample_name+os.linesep)
                        result+=os.linesep+'TIE\t'+topic+'\t'+max_instance.sample_info.sample_name+os.linesep    
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        #print confusion matrix
        ks = ['']*len(sub_confusion_dic)
        for k,v in sub_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in sub_confusion_mat[i]])+os.linesep
        ks = ['']*len(super_confusion_dic)
        for k,v in super_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in super_confusion_mat[i]])+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_bootstrap_bottomup(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted=True):
    for dim in dims:
        # Initialize  correct counts
        sub_confusion_dic = {}
        super_confusion_dic = {}
        super_total_correct = {}
        sup_total_correct = {}
        sub_cnt = 0
        super_cnt = 0
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0
            sub_confusion_dic[k] = sub_cnt
            sub_cnt += 1
            if v not in super_confusion_dic:
                super_confusion_dic[v] = super_cnt
                super_cnt += 1            
        # allocate confusion matrices
        super_confusion_mat = np.zeros((super_cnt,super_cnt))
        sub_confusion_mat = np.zeros((sub_cnt,sub_cnt))
        #calcuate aggregated vector for all the topics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = float(not weighted)
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        if weighted==True:
                            vec += sample_weight*model[concept_name]
                        else:
                            vec += model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        if weighted==True:
                            vec += sample_weight*model['id'+titles[concept_name]+'di']
                        else:
                            vec += model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            vec /= accum_weight
            sub_topics_vecs[k] = vec            
        # final result
        result = ''
        total_correct_sup = 0
        total_correct_super = 0
        if bootstrap==True:
            candidates = list(range(len(samples)))
        else:
            candidates = []
        while True:
            for total,cur_sample in enumerate(samples):
                sample_info = SampleInfo()
                sample_info.max_all_sim = 0.0
                sample = cur_sample[0:cur_sample.find('\t')].split('\\')
                sample_info.sample_name = sample[len(sample)-1]
                sample_info.sample_class = sample[len(sample)-2]
                if sample_info.sample_class not in topic_mappings:
                    continue            
                sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
                sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
                accum_vector = []
                for concept in sample_vector.split(conc_sep):
                    if len(concept)>1:
                        concept_name = concept[0:concept.find(weight_sep)]
                        concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                        accum_vector.append((concept_weight,concept_name))
                accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
                sample_vec =  np.zeros(model.vector_size)
                accum_weight = float(not weighted)
                for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                    try:
                        if mode=='conc':
                            if weighted==True:
                                sample_vec += sample_weight*model[concept_name]
                            else:
                                sample_vec += model[concept_name]
                            accum_weight += sample_weight
                        elif mode=='w2v':
                            if weighted==True:
                                sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                                accum_weight += sample_weight
                            else:
                                sample_vec += model['id'+titles[concept_name]+'di']                            
                    except:
                        print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                        result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                        continue
                sample_vec /= accum_weight
                max_sim = 0
                max_topic = ''
                for k,v in topics.items(): # loop on topics
                    if k not in topic_mappings:
                        continue
                    sim = 1 - spatial.distance.cosine(sub_topics_vecs[k],sample_vec)
                    result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(sim)+os.linesep
                    if sim>sample_info.max_all_sim:
                        sample_info.max_all_sim = sim
                        sample_info.max_topic_name = k
                        sample_info.max_topic_id = topics_dic[k]
                if sample_info.max_topic_name is not None:
                    result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
                else:
                    result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\tNone\tNone\t'+str(sample_info.max_all_sim)+os.linesep
                if sample_info.sample_class==sample_info.max_topic_name:
                    total_correct_sup += 1
                    sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
                if sample_info.max_topic_name is not None and topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
                    total_correct_super += 1
                    super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1
                if sample_info.max_topic_name is not None: # update confusion matrix
                    sub_confusion_mat[sub_confusion_dic[sample_info.sample_class]][sub_confusion_dic[sample_info.max_topic_name]] += 1
                    super_confusion_mat[super_confusion_dic[topic_mappings[sample_info.sample_class]]][super_confusion_dic[topic_mappings[sample_info.max_topic_name]]] += 1
                if total%100==0:
                    print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
            if len(candidates)==0:
                break
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        #print confusion matrix
        ks = ['']*len(sub_confusion_dic)
        for k,v in sub_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in sub_confusion_mat[i]])+os.linesep
        ks = ['']*len(super_confusion_dic)
        for k,v in super_confusion_dic.items():
            ks[v] = k
        result += '\t'+'\t'.join(ks)+os.linesep
        for i in range(len(ks)):
            result += ks[i]+'\t'
            result += '\t'.join([str(val) for val in super_confusion_mat[i]])+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_topdown(samples=None, super_topics=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None):
    for dim in dims:
        # final result
        result = ''
        super_topics_vecs = {}
        if super_topics is not None:
            #calcuate vectors for all the super topics
            for k,v in super_topics.items():
                if k not in topic_mappings.values():
                    continue
                accum_weight = 0
                topic_accum_vector = v[0]
                topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
                vec =  np.zeros(model.vector_size)
                for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                    try:
                        if mode=='conc':
                            vec += sample_weight*model[concept_name]
                            accum_weight += sample_weight
                        elif mode=='w2v':
                            vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                    except:
                        print(os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep)
                        result+=os.linesep+'FATAL\t'+concept_name+'...topic concept not found'+os.linesep
                        continue
                if (vec==np.zeros(model.vector_size)).all()==True:
                    print(os.linesep+'FATAL\t'+k+'...no concepts found'+os.linesep)
                    return
                vec /= accum_weight
                super_topics_vecs[k] = vec
        #calcuate aggregated vector for all the subtopics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = 0
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...subtopic concept not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...subtopic concept not found'+os.linesep
                    continue
            if (vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+k+'...no concepts found'+os.linesep)
                return
            vec /= accum_weight
            sub_topics_vecs[k] = vec
            if super_topics is None:
                super_topic = topic_mappings[k]
                if super_topic not in super_topics_vecs: # new super topic
                    super_topics_vecs[super_topic] = (vec,1)
                else:
                    old_vec, count = super_topics_vecs[super_topic]
                    super_topics_vecs[super_topic] = (old_vec+vec,count+1)        
        if super_topics is None:
            # average super topics vectors
            for k,v in super_topics_vecs.items():
                vec, count = v
                super_topics_vecs[k] = vec/count
        # Initialize  correct counts
        super_total_correct = {}
        sup_total_correct = {}
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0        
        total_correct_sup = 0
        total_correct_super = 0
        discount = 0
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = 0.0
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        sample_vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            if (sample_vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+sample_info.sample_name+'...no concepts found'+os.linesep)
                discount += 1
                continue
            sample_vec /= accum_weight
            max_sim = 0
            max_topic = None
            for super_topic,super_vec in super_topics_vecs.items(): # loop on super topics
                sim = 1 - spatial.distance.cosine(super_vec,sample_vec)
                result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+super_topic+'\t'+str(sim)+os.linesep
                if sim>max_sim:
                    max_sim = sim
                    max_topic = super_topic            
            if max_topic is None:
                print('Sample not classified')
                continue
            if max_topic is not None and topic_mappings[sample_info.sample_class]==max_topic:
                total_correct_super += 1
                super_total_correct[max_topic] = super_total_correct[max_topic]+1
            for sub_topic,sub_vec in sub_topics_vecs.items(): # loop on sub topics of the super topic only
                if sub_topic not in topic_mappings:
                    continue
                if topic_mappings[sub_topic]!=max_topic:
                    continue                
                sim = 1 - spatial.distance.cosine(sub_vec,sample_vec)
                result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+sub_topic+'\t'+str(sim)+os.linesep
                if sim>sample_info.max_all_sim:
                    sample_info.max_all_sim = sim
                    sample_info.max_topic_name = sub_topic
                    sample_info.max_topic_id = topics_dic[sub_topic]
            if sample_info.max_topic_name is not None:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
            else:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\tNone\tNone\t'+str(sample_info.max_all_sim)+os.linesep
            if sample_info.sample_class==sample_info.max_topic_name:
                total_correct_sup += 1
                sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
            total -= discount
            if total%100==0:
                print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_topdown_bootstrap(samples=None, super_topics=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted_bs=True):
    default_max_instance = MaxInstance(None,-1.0)    
    for dim in dims:
        # final result
        result = ''
        super_topics_vecs = {}
        if super_topics is not None:
            #calcuate vectors for all the super topics
            for k,v in super_topics.items():
                if k not in topic_mappings.values():
                    continue
                accum_weight = 0
                topic_accum_vector = v[0]
                topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
                vec =  np.zeros(model.vector_size)
                for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                    try:
                        if mode=='conc':
                            vec += sample_weight*model[concept_name]
                            accum_weight += sample_weight
                        elif mode=='w2v':
                            vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                    except:
                        print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                        result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                        continue
                vec /= accum_weight
                super_topics_vecs[k] = vec                
        #calcuate aggregated vector for all the subtopics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = 0
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            vec /= accum_weight
            sub_topics_vecs[k] = vec
            if super_topics is None:
                super_topic = topic_mappings[k]
                if super_topic not in super_topics_vecs: # new super topic
                    super_topics_vecs[super_topic] = (vec,1)
                else:
                    old_vec, count = super_topics_vecs[super_topic]
                    super_topics_vecs[super_topic] = (old_vec+vec,count+1)        
        if super_topics is None:
            # average super topics vectors
            for k,v in super_topics_vecs.items():
                vec, count = v
                super_topics_vecs[k] = vec/count
        # Initialize  correct counts
        super_total_correct = {}
        sup_total_correct = {}
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0        
        total_correct_sup = 0
        total_correct_super = 0
        #process samples
        # compute dense representation
        samples_dic = {}
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = 0.0
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        sample_vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            sample_vec /= accum_weight
            samples_dic[sample_info.sample_name] = Sample(sample_info, sample_vec)
        all_samples = set(samples_dic.keys())
        label_max_instance = dict(zip(super_topics_vecs.keys(),[default_max_instance for _ in super_topics_vecs.keys()]))
        total = 0
        Topic = namedtuple('Topic',['name','score')
        while len(all_samples)>0:
            for sample_name,sample in samples_dic.items(): # choose most similar sample
                best_topic = Topic('',-1.0)
                for super_topic,super_vec in super_topics_vecs.items(): # loop on super topics
                    if super_topic in candidate_topics: # no sample assigned to this topic yet                        
                        sim = 1 - spatial.distance.cosine(super_vec,sample.sample_vec)
                        if best_topic.score<sim: # this is the best topic for this sample so far
                            best_topic = Topic(super_topic,sim)
                for super_topic in super_topics_vecs: # check if this sample is better than the best so far
                    if super_topic==best_topic.name and label_max_instance[super_topic].score<best_topic.score:
                        max_instance = MaxInstance(sample.sample_info,best_topic.score)
                        label_max_instance[super_topic] = max_instance            
            for super_topic, max_instance in label_max_instance.items():
                if max_instance is not default_max_instance:
                    total += 1                                           
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+super_topic+'\t'+str(max_instance.score)+os.linesep
                    if topic_mappings[max_instance.sample_info.sample_class]==super_topic:
                        total_correct_super += 1
                        super_total_correct[super_topic] = super_total_correct[super_topic]+1
                    # update topic definition
                    if weighted_bs==True:
                        super_topics_vecs[super_topic] += max_instance.score*samples_dic[max_instance.sample_info.sample_name].sample_vec
                    else:
                        super_topics_vecs[super_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec
                    max_sub_topic = None
                    max_sub_score = -1.0
                    for sub_topic,sub_vec in sub_topics_vecs.items(): # loop on sub topics of the super topic only
                        if sub_topic not in topic_mappings:
                            continue
                        if topic_mappings[sub_topic]!=super_topic:
                            continue                
                        sim = 1 - spatial.distance.cosine(sub_vec,samples_dic[max_instance.sample_info.sample_name].sample_vec)
                        if sim>max_sub_score:
                            max_sub_score = sim
                            max_sub_topic = sub_topic                            
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+max_sub_topic+'\t'+str(topics_dic[max_sub_topic])+'\t'+str(max_instance.sample_info.max_all_sim)+os.linesep
                    if max_instance.sample_info.sample_class==max_sub_topic:
                        total_correct_sup += 1
                        sup_total_correct[max_sub_topic] = sup_total_correct[max_sub_topic]+1
                    # update sub_topic definition
                    sub_topics_vecs[max_sub_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec                    
                    if total%100==0:
                        print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
                    if max_instance.sample_info.sample_name in all_samples:
                        all_samples.remove(max_instance.sample_info.sample_name)
                    else: # a tie has happened, should no more happen after using max_topic_sim and max_topic_name
                        print(os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep)
                        result+=os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep    
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_topdown_bootstrap_on_topics(samples=None, super_topics=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None):
    default_max_instance = MaxInstance(None,-1.0)    
    for dim in dims:
        # final result
        result = ''
        super_topics_vecs = {}
        if super_topics is not None:
            #calcuate vectors for all the super topics
            for k,v in super_topics.items():
                if k not in topic_mappings.values():
                    continue
                accum_weight = 0
                topic_accum_vector = v[0]
                topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
                vec =  np.zeros(model.vector_size)
                for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                    try:
                        if mode=='conc':
                            vec += sample_weight*model[concept_name]
                            accum_weight += sample_weight
                        elif mode=='w2v':
                            vec += sample_weight*model['id'+titles[concept_name]+'di']
                            accum_weight += sample_weight
                    except:
                        print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                        result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                        continue
                vec /= accum_weight
                super_topics_vecs[k] = vec                
        #calcuate aggregated vector for all the subtopics
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = 0
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            vec /= accum_weight
            sub_topics_vecs[k] = vec
            if super_topics is None:
                super_topic = topic_mappings[k]
                if super_topic not in super_topics_vecs: # new super topic
                    super_topics_vecs[super_topic] = (vec,1)
                else:
                    old_vec, count = super_topics_vecs[super_topic]
                    super_topics_vecs[super_topic] = (old_vec+vec,count+1)        
        if super_topics is None:
            # average super topics vectors
            for k,v in super_topics_vecs.items():
                vec, count = v
                super_topics_vecs[k] = vec/count
        # Initialize  correct counts
        super_total_correct = {}
        sup_total_correct = {}
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0        
        total_correct_sup = 0
        total_correct_super = 0
        #process samples
        # compute dense representation
        samples_dic = {}
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = 0.0
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        sample_vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            sample_vec /= accum_weight
            samples_dic[sample_info.sample_name] = Sample(sample_info, sample_vec)
        all_samples = set(samples_dic.keys())
        total = 0
        while len(all_samples)>0:
            for super_topic,super_vec in super_topics_vecs.items(): # loop on super topics
                if sample_name in all_samples: # not assigned yet
                    max_instance = default_max_instance
                    for sample_name,sample in samples_dic.items(): # choose most similar sample
                        if sample_name in all_samples: # not assigned yet
                            sim = 1 - spatial.distance.cosine(super_vec,sample.sample_vec)
                            if max_instance.score<sim: # update most similar sample
                                max_instance = MaxInstance(sample.sample_info,sim)
                    if max_instance is not default_max_instance:
                        total += 1                                           
                        result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+super_topic+'\t'+str(max_instance.score)+os.linesep
                        if topic_mappings[max_instance.sample_info.sample_class]==super_topic:
                            total_correct_super += 1
                            super_total_correct[super_topic] = super_total_correct[super_topic]+1
                        # update topic definition
                        super_topics_vecs[super_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec
                        max_sub_topic = None
                        max_sub_score = -1.0
                        for sub_topic,sub_vec in sub_topics_vecs.items(): # loop on sub topics of the super topic only
                            if sub_topic not in topic_mappings:
                                continue
                            if topic_mappings[sub_topic]!=super_topic:
                                continue                
                            sim = 1 - spatial.distance.cosine(sub_vec,samples_dic[max_instance.sample_info.sample_name].sample_vec)
                            if sim>max_sub_score:
                                max_sub_score = sim
                                max_sub_topic = sub_topic                            
                        result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+max_sub_topic+'\t'+str(topics_dic[max_sub_topic])+'\t'+str(max_instance.sample_info.max_all_sim)+os.linesep
                        if max_instance.sample_info.sample_class==max_sub_topic:
                            total_correct_sup += 1
                            sup_total_correct[max_sub_topic] = sup_total_correct[max_sub_topic]+1
                        # update sub_topic definition
                        sub_topics_vecs[max_sub_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec                    
                        if total%100==0:
                            print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
                        if max_instance.sample_info.sample_name in all_samples:
                            all_samples.remove(max_instance.sample_info.sample_name)
                        else: # a tie has happened, should no more happen after using max_topic_sim and max_topic_name
                            print(os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep)
                            result+=os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep    
                        label_max_instance[super_topic] = default_max_instance
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_topdown_aggr(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None):
    for dim in dims:
        # final result
        result = ''
        #calcuate aggregated vector for all the topics
        super_topics_vecs = {}
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = 0
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            if (vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+k+'...no concepts found'+os.linesep)
                return
            vec /= accum_weight
            sub_topics_vecs[k] = vec
            super_topic = topic_mappings[k]
            if super_topic not in super_topics_vecs: # new super topic
                super_topics_vecs[super_topic] = (vec,1)
            else:
                old_vec, count = super_topics_vecs[super_topic]
                super_topics_vecs[super_topic] = (old_vec+vec,count+1)        
        # average super topics vectors
        for k,v in super_topics_vecs.items():
            vec, count = v
            super_topics_vecs[k] = vec/count
        # Initialize  correct counts
        super_total_correct = {}
        sup_total_correct = {}
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0        
        total_correct_sup = 0
        total_correct_super = 0
        discount = 0
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = 0.0
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        sample_vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            if (sample_vec==np.zeros(model.vector_size)).all()==True:
                print(os.linesep+'FATAL\t'+sample_info.sample_name+'...no concepts found'+os.linesep)
                discount += 1
                continue
            sample_vec /= accum_weight
            max_sim = 0
            max_topic = None
            for super_topic,super_vec in super_topics_vecs.items(): # loop on super topics
                sim = 1 - spatial.distance.cosine(super_vec,sample_vec)
                result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+super_topic+'\t'+str(sim)+os.linesep
                if sim>max_sim:
                    max_sim = sim
                    max_topic = super_topic            
            if max_topic is None:
                print('Sample not classified')
                continue
            if max_topic is not None and topic_mappings[sample_info.sample_class]==max_topic:
                total_correct_super += 1
                super_total_correct[max_topic] = super_total_correct[max_topic]+1
            for sub_topic,sub_vec in sub_topics_vecs.items(): # loop on sub topics of the super topic only
                if sub_topic not in topic_mappings:
                    continue
                if topic_mappings[sub_topic]!=max_topic:
                    continue                
                sim = 1 - spatial.distance.cosine(sub_vec,sample_vec)
                result += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+sub_topic+'\t'+str(sim)+os.linesep
                if sim>sample_info.max_all_sim:
                    sample_info.max_all_sim = sim
                    sample_info.max_topic_name = sub_topic
                    sample_info.max_topic_id = topics_dic[sub_topic]
            if sample_info.max_topic_name is not None:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
            else:
                result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\tNone\tNone\t'+str(sample_info.max_all_sim)+os.linesep
            if sample_info.sample_class==sample_info.max_topic_name:
                total_correct_sup += 1
                sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
            total -= discount
            if total%100==0:
                print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_topdown_aggr_bootstrap(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None, weighted_bs=True):
    default_max_instance = MaxInstance(None,-1.0)
    for dim in dims:
        # final result
        result = ''
        #calcuate aggregated vector for all the topics
        super_topics_vecs = {}
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = 0
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            vec /= accum_weight
            sub_topics_vecs[k] = vec
            super_topic = topic_mappings[k]
            if super_topic not in super_topics_vecs: # new super topic
                super_topics_vecs[super_topic] = (vec,1)
            else:
                old_vec, count = super_topics_vecs[super_topic]
                super_topics_vecs[super_topic] = (old_vec+vec,count+1)        
        # average super topics vectors
        for k,v in super_topics_vecs.items():
            vec, count = v
            super_topics_vecs[k] = vec/count
        # Initialize  correct counts
        super_total_correct = {}
        sup_total_correct = {}
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0        
        total_correct_sup = 0
        total_correct_super = 0
        #process samples
        # compute dense representation
        samples_dic = {}
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = 0.0
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        sample_vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            sample_vec /= accum_weight
            samples_dic[sample_info.sample_name] = Sample(sample_info, sample_vec)
        all_samples = set(samples_dic.keys())
        label_max_instance = dict(zip(super_topics_vecs.keys(),[default_max_instance for _ in super_topics_vecs.keys()]))
        total = 0
        Topic = namedtuple('Topic',['name','score'])        
        #repeat
        #    for sample_i in samples:
        #        assign sample_i to label_i
        #    for each label
        #        choose highest matching sample
        #        add its vector
        #        remove the sample from the pool
        # until no more samples                
        while len(all_samples)>0:
            for sample_name,sample in samples_dic.items(): # choose most similar sample
                if sample_name in all_samples: # not assigned yet
                    best_topic = Topic('',-1.0)
                    for super_topic,super_vec in super_topics_vecs.items(): # loop on super topics
                        sim = 1 - spatial.distance.cosine(super_vec,sample.sample_vec)
                        if best_topic.score<sim: # this is the best topic for this sample so far
                            best_topic = Topic(super_topic,sim)
                    for super_topic in super_topics_vecs: # check if this sample is better than the best so far
                        if super_topic==best_topic.name and label_max_instance[super_topic].score<best_topic.score:
                            max_instance = MaxInstance(sample.sample_info,best_topic.score)
                            label_max_instance[super_topic] = max_instance            
            for super_topic, max_instance in label_max_instance.items():
                if max_instance is not default_max_instance:
                    total += 1                                           
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+super_topic+'\t'+str(max_instance.score)+os.linesep
                    if topic_mappings[max_instance.sample_info.sample_class]==super_topic:
                        total_correct_super += 1
                        super_total_correct[super_topic] = super_total_correct[super_topic]+1
                    # update topic definition
                    if weighted_bs==True:
                        super_topics_vecs[super_topic] += max_instance.score*samples_dic[max_instance.sample_info.sample_name].sample_vec
                    else:
                        super_topics_vecs[super_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec
                    max_sub_topic = None
                    max_sub_score = -1.0
                    for sub_topic,sub_vec in sub_topics_vecs.items(): # loop on sub topics of the super topic only
                        if sub_topic not in topic_mappings:
                            continue
                        if topic_mappings[sub_topic]!=super_topic:
                            continue                
                        sim = 1 - spatial.distance.cosine(sub_vec,samples_dic[max_instance.sample_info.sample_name].sample_vec)
                        if sim>max_sub_score:
                            max_sub_score = sim
                            max_sub_topic = sub_topic                            
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+max_sub_topic+'\t'+str(topics_dic[max_sub_topic])+'\t'+str(max_instance.sample_info.max_all_sim)+os.linesep
                    if max_instance.sample_info.sample_class==max_sub_topic:
                        total_correct_sup += 1
                        sup_total_correct[max_sub_topic] = sup_total_correct[max_sub_topic]+1
                    # update sub_topic definition
                    if weighted_bs==True:
                        sub_topics_vecs[max_sub_topic] += max_sub_score*samples_dic[max_instance.sample_info.sample_name].sample_vec
                    else:
                        sub_topics_vecs[max_sub_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec                    
                    if total%100==0:
                        print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
                    if max_instance.sample_info.sample_name in all_samples:
                        all_samples.remove(max_instance.sample_info.sample_name)
                    else: # a tie has happened, should no more happen after using max_topic_sim and max_topic_name
                        print(os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep)
                        result+=os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep                
                    label_max_instance[super_topic] = default_max_instance
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def classify_samples_with_densification_topdown_aggr_bootstrap_loop_on_topics(samples=None, topics=None, topics_dic=None, dims=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], topic_mappings=ng_topic_mappings, model=None, conc_sep=';;;;', weight_sep=',,,,', mode='conc', titles=None, outpath=None):
    default_max_instance = MaxInstance(None,-1.0)
    for dim in dims:
        # final result
        result = ''
        #calcuate aggregated vector for all the topics
        super_topics_vecs = {}
        sub_topics_vecs = {}
        for k,v in topics.items():
            if k not in topic_mappings:
                continue
            accum_weight = 0
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getWeight)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            vec /= accum_weight
            sub_topics_vecs[k] = vec
            super_topic = topic_mappings[k]
            if super_topic not in super_topics_vecs: # new super topic
                super_topics_vecs[super_topic] = (vec,1)
            else:
                old_vec, count = super_topics_vecs[super_topic]
                super_topics_vecs[super_topic] = (old_vec+vec,count+1)        
        # average super topics vectors
        for k,v in super_topics_vecs.items():
            vec, count = v
            super_topics_vecs[k] = vec/count
        # Initialize  correct counts
        super_total_correct = {}
        sup_total_correct = {}
        for k,v in topic_mappings.items():
            sup_total_correct[k] = 0
            super_total_correct[v] = 0        
        total_correct_sup = 0
        total_correct_super = 0
        #process samples
        # compute dense representation
        samples_dic = {}
        for total,cur_sample in enumerate(samples):
            sample_info = SampleInfo()
            sample_info.max_all_sim = 0.0
            sample = cur_sample[0:cur_sample.find('\t')].split('\\')
            sample_info.sample_name = sample[len(sample)-1]
            sample_info.sample_class = sample[len(sample)-2]
            if sample_info.sample_class not in topic_mappings:
                continue            
            sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
            sample_vector = cur_sample[cur_sample.rfind('\t')+1:]
            accum_vector = []
            for concept in sample_vector.split(conc_sep):
                if len(concept)>1:
                    concept_name = concept[0:concept.find(weight_sep)]
                    concept_weight = float(concept[concept.rfind(weight_sep)+len(weight_sep):])
                    accum_vector.append((concept_weight,concept_name))
            accum_vector = sorted(accum_vector, reverse=True, key=getWeight)[0:min(dim,len(accum_vector))]
            sample_vec =  np.zeros(model.vector_size)
            accum_weight = 0.0
            for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
                try:
                    if mode=='conc':
                        sample_vec += sample_weight*model[concept_name]
                        accum_weight += sample_weight
                    elif mode=='w2v':
                        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
                        accum_weight += sample_weight
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            sample_vec /= accum_weight
            samples_dic[sample_info.sample_name] = Sample(sample_info, sample_vec)
        all_samples = set(samples_dic.keys())
        total = 0
        while len(all_samples)>0:
            for super_topic,super_vec in super_topics_vecs.items(): # loop on super topics
                max_instance = default_max_instance
                for sample_name,sample in samples_dic.items(): # choose most similar sample
                    if sample_name in all_samples: # not assigned yet
                        sim = 1 - spatial.distance.cosine(super_vec,sample.sample_vec)
                        if max_instance.score<sim: # update most similar sample
                            max_instance = MaxInstance(sample.sample_info,sim)
                if max_instance is not default_max_instance:
                    total += 1                                           
                    #result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+super_topic+'\t'+str(max_instance.score)+os.linesep
                    if topic_mappings[max_instance.sample_info.sample_class]==super_topic:
                        total_correct_super += 1
                        super_total_correct[super_topic] = super_total_correct[super_topic]+1
                    # update topic definition
                    super_topics_vecs[super_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec
                    max_sub_topic = None
                    max_sub_score = -1.0
                    for sub_topic,sub_vec in sub_topics_vecs.items(): # loop on sub topics of the super topic only
                        if sub_topic not in topic_mappings:
                            continue
                        if topic_mappings[sub_topic]!=super_topic:
                            continue                
                        sim = 1 - spatial.distance.cosine(sub_vec,samples_dic[max_instance.sample_info.sample_name].sample_vec)
                        if sim>max_sub_score:
                            max_sub_score = sim
                            max_sub_topic = sub_topic                            
                    result += os.linesep+'TOPICA\t'+max_instance.sample_info.sample_name+'\t'+max_instance.sample_info.sample_class+'\t'+str(topics_dic[max_instance.sample_info.sample_class])+'\t'+max_sub_topic+'\t'+str(topics_dic[max_sub_topic])+'\t'+str(max_instance.sample_info.max_all_sim)+os.linesep
                    if max_instance.sample_info.sample_class==max_sub_topic:
                        total_correct_sup += 1
                        sup_total_correct[max_sub_topic] = sup_total_correct[max_sub_topic]+1
                    # update sub_topic definition
                    sub_topics_vecs[max_sub_topic] += samples_dic[max_instance.sample_info.sample_name].sample_vec
                    if total%100==0:
                        print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
                    if max_instance.sample_info.sample_name in all_samples:
                        all_samples.remove(max_instance.sample_info.sample_name)
                    else: # a tie has happened, should no more happen after using max_topic_sim and max_topic_name
                        print(os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep)
                        result+=os.linesep+'TIE\t'+super_topic+'\t'+max_instance.sample_info.sample_name+os.linesep    
        print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
        for k,v in sup_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
        for k,v in super_total_correct.items():
            result += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
        result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
        result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
        result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
        if os.path.isdir(outpath)==False:
            os.mkdir(outpath) 
        with open(outpath+'/'+str(dim),'w') as outp:
            outp.write(result)


def UnicodeDictReader(utf8_data, **kwargs):
    csv_reader = csv.DictReader(utf8_data, **kwargs)
    for row in csv_reader:
        yield {unicode(key): unicode(value, 'utf-8') for key, value in row.iteritems()}

def load_titles(inpath, to_keep=set(), load_redirects=True, 
    redirects_value='id', load_seealso=False):
    titles = {}
    redirects = {}
    seealso = {}
    print("loading titles and redirects and seealso")
    if sys.version_info[0]==3:
        records = csv.DictReader(open(inpath))
    else:
        records = UnicodeDictReader(open(inpath))
    count = 0
    for pair in records:
        count = count + 1
        if len(to_keep)==0 or pair['title'] in to_keep:
            titles[pair['title']] = pair['id']
            if load_redirects==True:
                extra = pair['redirect']
                if len(extra)>0:
                    keys = get_extra(extra)
                    for key in keys:
                        redirects[key] = pair[redirects_value]
            if load_seealso==True:
                extra = pair['seealso']
                if len(extra)>0:
                    keys = get_extra(extra)
                    if len(keys)>0:
                        seealso[pair['title']] = keys
            #redirectss = pair['redirect']
            #if len(redirectss)>0:
            #    redirectss = redirectss.replace('\,','\t')
            #    for redirect in redirectss.split(','):
            #        redirects[redirect.replace('\t',',')] = pair[redirects_value]
    print("loaded "+str(count)+" titles")
    return (titles,redirects,seealso)


def get_extra(keys):
    single_keys = set()
    keys = keys.replace('\,','\t')
    for key in keys.split(','):
        single_keys.add(key.replace('\t',','))
    return single_keys


def load_cosines_strings(inpath):
    # file format is
    #sample title#1   topic_title#1    sim_sample#1_topic#1  topic_title#2    sim_sample#1_topic#2
    #sample title#2   topic_title#1    sim_sample#2_topic#1  topic_title#2    sim_sample#2_topic#2
    titles_cosines = {} #{sample_title#1+tab+topic_title#1:sim_sample#1_topic#1,...}
    print("loading cosines")
    sys.stdout.flush()
    with open(inpath) as indata:
        cur_line = 0
        for line in indata:
            cur_line = cur_line + 1
            if cur_line%10000==0:
                print('processed (',cur_line,') samples')
                sys.stdout.flush()

            tokens = line.strip('\n').split('\t')            
            if len(tokens)%2==1:
                cosines = []    
                cur_smaple_title = ''
                cur_topic_title = ''   
                for cur_i, token in enumerate(tokens):
                    if cur_i==0:
                        cur_smaple_title = token
                    elif cur_i%2==1:
                        cur_topic_title = token
                    elif cur_i%2==0:
                        titles_cosines[cur_smaple_title+'\t'+cur_topic_title] = float(token)
            else:
                print('Invalid line format @'+str(count))
            
    print("loaded "+str(len(titles_cosines.keys()))+" sample titles")

    return titles_cosines

def load_cosines_2d(inpath):
    # file format is
    #sample title#1   topic_title#1    sim_sample#1_topic#1  topic_title#2    sim_sample#1_topic#2
    #sample title#2   topic_title#1    sim_sample#2_topic#1  topic_title#2    sim_sample#2_topic#2
    topic_titles = {}   #{topic_title#1:0,topic_title#2:1}
    sample_titles = {}   #{sample_title#1:0,sample_title#2:1}
    titles_cosines = [0 for x in range(300000)] #2D array of similarities
    print("loading cosines")
    sys.stdout.flush()
    '''
    cosines = [0 for i in range(8777)]
    titles_cosines = [cosines for i in range(268415)]
    topic_titles = {}
    for i in range(8777):
        topic_titles['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'+str(i)] = 1
    for i in range(268415):
            sample_titles['aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'+str(i)] = 1
    return titles_cosines[0:len(sample_titles.keys())], topic_titles, sample_titles #titles_cosines[]            
    '''    
    with open(inpath) as indata:
        cur_line = 0
        for line in indata:
            cur_line = cur_line + 1
            if cur_line%10000==0:
                print('processed (',cur_line,') samples')
                sys.stdout.flush()

            tokens = line.strip('\n').split('\t')            
            if len(tokens)%2==1:
                cosines = [float(0) for x in range(10000)]
                max_indx = 0
                for cur_i, token in enumerate(tokens):
                    if cur_i==0:
                        sample_titles[token] = cur_line-1
                    elif cur_line==1 and cur_i%2==1:
                        topic_titles[token] = int((cur_i-1)/2)
                    elif cur_i%2==0:
                        indx = int(cur_i/2)-1
                        if indx==10000:
                            max_indx = max_indx + 1
                            cosines.append(float(token))
                        else:
                            max_indx = max_indx + 1
                            cosines[indx] = float(token)
                
                indx = cur_line-1
                if indx==300000:
                    titles_cosines.append(cosines[0:max_indx])
                else:
                    titles_cosines[cur_line-1] = cosines[0:max_indx]
            else:
                print('Invalid line format @'+str(count))
            
    print("loaded "+str(len(topic_titles.keys()))+" topic titles")
    print("loaded "+str(len(sample_titles.keys()))+" sample titles")

    return titles_cosines[0:len(sample_titles.keys())], topic_titles, sample_titles #titles_cosines[]

def load_cosines(inpath, min_cosine=0.0):
    # file format is
    #sample title#1   topic_title#1    sim_sample#1_topic#1  topic_title#2    sim_sample#1_topic#2
    #sample title#2   topic_title#1    sim_sample#2_topic#1  topic_title#2    sim_sample#2_topic#2
    topic_titles = {}   #{topic_title#1:0,topic_title#2:1}
    titles_cosines = {} #{sample_title#1:{0:sim_sample#1_topic#1,1:sim_sample#1_topic#2...],sample_title#2:[0:sim_sample#2_topic#1,1:sim_sample#2_topic#2...]}
    print("loading cosines",inpath)
    sys.stdout.flush()
    with open(inpath) as indata:
        cur_line = 0
        for line in indata:
            cur_line = cur_line + 1
            if cur_line%10000==0:
                print('processed (',cur_line,') samples')
                sys.stdout.flush()
            tokens = line.strip('\n').split('\t')            
            if len(tokens)%2==1:
                cosines = {}    
                cur_title = ''            
                for cur_i, token in enumerate(tokens):
                    if cur_i==0:
                        cur_title = token
                    elif cur_line==1 and cur_i%2==1:
                        topic_titles[token] = int((cur_i-1)/2)
                    elif cur_i%2==0:
                        cosine = float(token)
                        if round(cosine,2)>=min_cosine:
                            cosines[int((cur_i-1)/2)] = cosine
                titles_cosines[cur_title] = cosines                        
            else:
                print('Invalid line format @'+str(count))
    print("loaded "+str(len(topic_titles.keys()))+" topic titles")
    print("loaded "+str(len(titles_cosines.keys()))+" sample titles")
    return titles_cosines, topic_titles #titles_cosines[]

def load_cosines_sparse(inpath):
    # file format is
    #sample_id   0  topic_title#1    2 topic_title#3    3  topic_title#4    1  topic_title#2
    #sample title#1   0    sim_sample#1_topic#1  3    sim_sample#1_topic#4
    topic_titles = {}   #{topic_title#1:0,topic_title#2:1}
    titles_cosines = {} #{sample_title#1:{0:sim_sample#1_topic#1,3:sim_sample#1_topic#4...],...}
    print("loading cosines",inpath)
    sys.stdout.flush()
    with open(inpath) as indata:
        cur_line = 0
        for line in indata:
            cur_line = cur_line + 1
            if cur_line%10000==0:
                print('processed (',cur_line,') samples')
                sys.stdout.flush()
            tokens = line.strip('\n').split('\t')            
            
            cosines = {}    
            cur_title = ''            
            for cur_i, token in enumerate(tokens):
                if cur_line==1: # 1st line is topic titles line
                    if cur_i>0: # not sample_title column
                        topic_title_tokens = token.split(':')                        
                        topic_titles[topic_title_tokens[1]] = int(topic_title_tokens[0])
                else:
                    if cur_i==0:
                        cur_title = token
                    else:
                        sample_title_tokens = token.split(':')
                        cosines[int(sample_title_tokens[0])] = float(sample_title_tokens[1])
                
            if cur_line>1:
                titles_cosines[cur_title] = cosines                        
            
    print("loaded "+str(len(topic_titles.keys()))+" topic titles")
    print("loaded "+str(len(titles_cosines.keys()))+" sample titles")
    return titles_cosines, topic_titles #titles_cosines[]

def load_cosines_spark(line, min_cosine=0.0):
    # file format is
    #sample title#1   topic_title#1    sim_sample#1_topic#1  topic_title#2    sim_sample#1_topic#2
    #sample title#2   topic_title#1    sim_sample#2_topic#1  topic_title#2    sim_sample#2_topic#2
    titles_cosines = {} #{sample_title#1:{0:sim_sample#1_topic#1,1:sim_sample#1_topic#2...],sample_title#2:[0:sim_sample#2_topic#1,1:sim_sample#2_topic#2...]}
    tokens = line.split('\t')
    if len(tokens)%2==1:
        cosines = {}    
        cur_title = ''            
        for cur_i, token in enumerate(tokens):
            if cur_i==0:
                cur_title = token
            elif cur_i%2==0:
                cosine = round(float(token),2)
                if cosine>=min_cosine:
                    cosines[int((cur_i-1)/2)] = cosine        
        titles_cosines[cur_title] = cosines                        
    else:
        print('Invalid line format @'+str(count))    
    return [titles_cosines]

def load_cosines_line(pair):
    line = pair[0]
    min_cosine = pair[1]
    # file format is
    #sample title#1   topic_title#1    sim_sample#1_topic#1  topic_title#2    sim_sample#1_topic#2
    #sample title#2   topic_title#1    sim_sample#2_topic#1  topic_title#2    sim_sample#2_topic#2
    titles_cosines = {} #{sample_title#1:{0:sim_sample#1_topic#1,1:sim_sample#1_topic#2...],sample_title#2:[0:sim_sample#2_topic#1,1:sim_sample#2_topic#2...]}
    tokens = line.split('\t')
    if len(tokens)%2==1:
        cosines = {}    
        cur_title = ''            
        for cur_i, token in enumerate(tokens):
            if cur_i==0:
                cur_title = token
            elif cur_i%2==0:
                cosine = round(float(token),2)
                if cosine>=min_cosine:
                    cosines[int((cur_i-1)/2)] = cosine        
        titles_cosines[cur_title] = cosines                        
    else:
        print('Invalid line format @'+str(count))    
    return titles_cosines

def get_topic_titles(line):
    # file format is
    #sample title#1   topic_title#1    sim_sample#1_topic#1  topic_title#2    sim_sample#1_topic#2
    #sample title#2   topic_title#1    sim_sample#2_topic#1  topic_title#2    sim_sample#2_topic#2
    topic_titles = {}   #{topic_title#1:0,topic_title#2:1}
    print("loading topic titles")
    tokens = line.strip('\n').split('\t')            
    if len(tokens)%2==1:
        for cur_i, token in enumerate(tokens):
            if cur_i%2==1:
                topic_titles[token] = int((cur_i-1)/2)
    else:
        print('Invalid line format @'+str(count))
    print("loaded "+str(len(topic_titles.keys()))+" topic titles")
    return topic_titles #titles_cosines[]

def load_cosines_all(inpath):
    # file format is
    #sample title#1   topic_title#1    sim_sample#1_topic#1  topic_title#2    sim_sample#1_topic#2
    #sample title#2   topic_title#1    sim_sample#2_topic#1  topic_title#2    sim_sample#2_topic#2
    topic_titles = {}   #{topic_title#1:0,topic_title#2:1}
    titles_cosines = {} #{sample_title#1:[sim_sample#1_topic#1,sim_sample#1_topic#2...],sample_title#2:[sim_sample#2_topic#1,sim_sample#2_topic#2...]}
    print("loading cosines")
    sys.stdout.flush()
    with open(inpath) as indata:
        cur_line = 0
        for line in indata:
            cur_line = cur_line + 1
            if cur_line%1000==0:
                print('processed (',cur_line,') samples')
                sys.stdout.flush()

            tokens = line.strip('\n').split('\t')            
            if len(tokens)%2==1:
                cosines = []    
                cur_title = ''            
                for cur_i, token in enumerate(tokens):
                    if cur_i==0:
                        cur_title = token
                    elif cur_line==1 and cur_i%2==1:
                        topic_titles[token] = cur_i-1
                    elif cur_i%2==0:
                        cosines.append(float(token))
                
                titles_cosines[cur_title] = cosines                        
            else:
                print('Invalid line format @'+str(count))
            
    print("loaded "+str(len(topic_titles.keys()))+" topic titles")
    print("loaded "+str(len(titles_cosines.keys()))+" sample titles")

    return titles_cosines, topic_titles #titles_cosines[]

def read_concept_dataset(input_path, mode='all', mappings={}):
    dataset = {}
    for line in open(input_path):
        tokens = line.strip().split('\t')
        for i,token in enumerate(tokens):
            if token=='': # empty line
                break
            if i==0:
                key = mappings.get(token, token)
                dataset[key] = []
            else:
                if mode=='all':
                    dataset[key].append(tokens[i])
                else:
                    multiword = tokens[i].find(' ')>0
                    if mode=='single' and multiword==False:
                        dataset[key].append(tokens[i])
                    elif mode=='multi' and multiword==True:
                        dataset[key].append(tokens[i])
    return dataset

def resolve_mention(title, titles, redirects, mode='w2v', prefix='id',postfix='di'):
    #print('resolving ('+title+')')
    try:
        iden = str(titles[title])
        if mode=='w2v':
            return prefix+iden+postfix
        else:
            return title
    except:
        try:
            return str(redirects[title]) # see if it is redirect            
        except:
            title = title.capitalize() # uppercase first letter of first word 
            try:
                iden = str(titles[title])
                if mode=='w2v':
                    return prefix+iden+postfix
                else:
                    return title
            except:
                title = title.title() # uppercase first letter of each word
                try:
                    iden = str(titles[title])
                    if mode=='w2v':
                        return prefix+iden+postfix
                    else:
                        return title
                except:
                    try:
                        return prefix+str(redirects[title])+postfix
                    except:
                        return ''

def get_nearest_before_refactor(model, instance, label_definitions, weighted=True):
    nearest = ''
    score = -1.0
    min_diff = 100.0
    for key,definition in label_definitions.items():
        if 1==1:# or key!='id18716923di': # algebra
            definition_vec = 1*model[definition[0]] # *1 to make new object
            for concept in definition[1:]:
                if weighted==False:
                    definition_vec += 1*model[concept[0]]
                else:
                    definition_vec += concept[1]*model[concept[0]]
            definition_vec = definition_vec/len(definition_vec)
            sim = abs(1 - spatial.distance.cosine(model[instance],definition_vec))
        else:
            sim = abs(1 - spatial.distance.cosine(model[instance],model[key]+model['id5176di'])) # algebra + calculus
        if score<sim:
            min_diff = sim - score
            score = sim
            nearest = key
    return nearest, score, min_diff

def get_nearest(model, instance, label_definitions, weighted=True):
    label_definitions_vecs = {}
    for key,definition in label_definitions.items():
        definition_vec = 1*model[definition[0]] # *1 to make new object
        aggr_sum = 1
        for concept in definition[1:]:
            if weighted==False:
                definition_vec += 1*model[concept[0]]
                aggr_sum += 1
            else:
                definition_vec += concept[1]*model[concept[0]]
                aggr_sum += concept[1]
        definition_vec = definition_vec/aggr_sum
        label_definitions_vecs[key] = definition_vec
    return get_nearest_vector(model, instance, label_definitions_vecs)


def get_nearest_vector(model, instance, label_definitions_vecs):
    nearest = ''
    score = -1.0
    min_diff = 100.0
    for key,definition_vec in label_definitions_vecs.items():
        sim = abs(1 - spatial.distance.cosine(model[instance],definition_vec))
        if score<sim:
            min_diff = sim - score
            score = sim
            nearest = key
    return nearest, score, min_diff

def print_confusion_matrix(confusion_matrix, ids=None):
    result = ''
    ks = list(confusion_matrix.keys())
    ks_names = list(ks)
    if ids is not None:
        for i,k in enumerate(ks_names):
            ks_names[i] = ids[k]
    result += '\t'+'\t'.join(ks_names)+os.linesep
    for i in range(len(ks_names)):
        result += ks_names[i]+'\t'
        result += '\t'.join([str(confusion_matrix[ks[i]][j]) for j in ks])+os.linesep        
    return result

def get_all_titles(titles=None, redirects=None, model=None, orig_titles=True, lower=True):
    """
    return a map of all wikipedia titles and redirects existing in the model 
    as keys and article id as values
    """
    all_pairs = []
    all_titles = {}
    for i,j in titles.items():
        all_pairs.append((i,'id'+j+'di',i))
    for i,j in redirects.items():
        all_pairs.append((i,'id'+titles[j]+'di',j))
    for i,id,j in all_pairs:
        if model is None or id in model:
            if lower==True:
                i = i.lower()
            if orig_titles==True:
                all_titles[i] = (id,j)
            else:
                all_titles[i] = (id,)
    return all_titles

def read_probase_file(infile):
    concept_entities = {}
    with open(infile,'r') as inp:
        for line in inp:
            if len(line)==0 or line[0]=='#':
                continue
            tokens = line.split('\t')
            if len(tokens)==3:
                if tokens[0] in concept_entities:
                    concept_entities[tokens[0]].append(tokens[1])
                else:
                    concept_entities[tokens[0]] = [tokens[1]]
    return concept_entities


def create_concept_dataset(inout_files=None, titles=None, format='probase'):
    """
    create concept dataset file suitable for concept clustering
    the function accepts an array of input/output files acting 
    as input data file and corresponding output  file
    the function will return coverage count for each file 
    i.e., (total input, found titles) pairs
    """
    coverage = []
    for infile,outfile in inout_files:
        total = 0
        found = 0
        notfound = 0
        if format=='probase':
            records = read_probase_file(infile)
            outputmap = {}
            for concept,entities in records.items():
                found_concept_entities = []
                for entity in entities:
                    total += 1                
                    try:
                        org_title = titles[entity][1]
                        found_concept_entities.append(org_title)
                        found += 1
                    except:
                        notfound += 1
                        pass
                outputmap[concept] = found_concept_entities
            with open(outfile,'w') as output:
                for concept, entities in outputmap.items():
                    output.write(concept+'\t'+'\t'.join(entities)+os.linesep)
        coverage.append((infile,total,found,notfound))
    return coverage


'''

import csv

def load_titles(inpath, to_keep, load_redirects=True, redirects_value='id'):
    titles = {}
    redirects = {}
    print("loading titles and redirects")
    records = csv.DictReader(open(inpath))
    count = 0
    for pair in records:
        count = count + 1
        if len(to_keep)==0 or pair['title'] in to_keep:
            titles[pair['title']] = pair['id']
            redirectss = pair['redirect']
            if len(redirectss)>0:
                redirectss = redirectss.replace('\,','\t')
                for redirect in redirectss.split(','):
                    redirects[redirect.replace('\t',',')] = pair[redirects_value]
        
    print("loaded "+str(count)+" titles")
    
    return (titles,redirects)

'''