import numpy as np
from scipy import spatial
import os

print("loading topics")
topics_inpath = 'tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.autos.motors'
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


class SampleInfo:
     def __init__(self):
         self.max_topic_id = None
         self.max_topic_name = None
         self.sample_class = None
         self.sample_name = None
         self.max_all_sim = 0.0


from commons import load_titles
titles_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/titles_redirects_ids_anno.csv'
titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')

samples_inpath = '20newsgroups.simple.esa.concepts.500.anno.resolved.autos.motors'
lines = []
with open(samples_inpath,'r') as samples_file:
  for line in samples_file:
   lines.append(line)

result = ''
for line in lines:
    sample_info = SampleInfo()
    sample_info.max_all_sim = 0.0
    sample = line[0:line.find('\t')].split('\\')
    sample_info.sample_name = sample[len(sample)-1]
    sample_info.sample_class = sample[len(sample)-2]
    sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
    sample_vector = line[line.rfind('\t')+1:]
    accum_vector = []
    for concept in sample_vector.split(';;;;'):
        if len(concept)>1:
            concept_name = concept[0:concept.find(',,,,')]
            concept_weight = float(concept[concept.rfind(',,,,')+4:])
            accum_vector.append((concept_weight,concept_name))            
    sample_vec =  np.zeros(model.vector_size)
    for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
        #if concept_name=="Pimp My Ride":
        #    print("testing concept: ("+concept_name+")")
        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
    max_sim = 0
    max_topic = ''
    for k,v in topics.items(): # loop on topics
        topic_accum_vector = v[0]
        vec =  np.zeros(model.vector_size)
        for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            vec += sample_weight*model['id'+titles[concept_name]+'di']
        sim = 1 - spatial.distance.cosine(vec,sample_vec)
        if sim>sample_info.max_all_sim:
            sample_info.max_all_sim = sim
            sample_info.max_topic_name = k
            sample_info.max_topic_id = topics_dic[k]
        #print(sample_info.sample_name,k,sim)
    print(os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep)
    result += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep



with open('results.txt') as resulta:
    resulta.write(results)
#-----------------------------------------------------------------------
print("loading topics")
topics_inpath = 'tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.autos.elecs'
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

samples_inpath = '20newsgroups.simple.esa.concepts.500.anno.resolved.autos.elecs'
lines = []
with open(samples_inpath,'r') as samples_file:
  for line in samples_file:
   lines.append(line)

result1 = ''
for line in lines:
    sample_info = SampleInfo()
    sample_info.max_all_sim = 0.0
    sample = line[0:line.find('\t')].split('\\')
    sample_info.sample_name = sample[len(sample)-1]
    sample_info.sample_class = sample[len(sample)-2]
    sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
    sample_vector = line[line.rfind('\t')+1:]
    accum_vector = []
    for concept in sample_vector.split(';;;;'):
        if len(concept)>1:
            concept_name = concept[0:concept.find(',,,,')]
            concept_weight = float(concept[concept.rfind(',,,,')+4:])
            accum_vector.append((concept_weight,concept_name))            
    sample_vec =  np.zeros(model.vector_size)
    for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
        #if concept_name=="Pimp My Ride":
        #    print("testing concept: ("+concept_name+")")
        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
    max_sim = 0
    max_topic = ''
    for k,v in topics.items(): # loop on topics
        topic_accum_vector = v[0]
        vec =  np.zeros(model.vector_size)
        for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            vec += sample_weight*model['id'+titles[concept_name]+'di']
        sim = 1 - spatial.distance.cosine(vec,sample_vec)
        if sim>sample_info.max_all_sim:
            sample_info.max_all_sim = sim
            sample_info.max_topic_name = k
            sample_info.max_topic_id = topics_dic[k]
        #print(sample_info.sample_name,k,sim)
    print(os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep)
    result1 += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep

with open('results1.txt','w') as resulta:
    resulta.write(result1)

#---------------------------------------------------------------------

print("loading topics")
topics_inpath = 'tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved'
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

samples_inpath = '20newsgroups.simple.esa.concepts.500.anno.resolved'
lines = []
with open(samples_inpath,'r') as samples_file:
  for line in samples_file:
   lines.append(line)

result1 = ''
totcal_correct = 0
for total,line in enumerate(lines):
    sample_info = SampleInfo()
    sample_info.max_all_sim = 0.0
    sample = line[0:line.find('\t')].split('\\')
    sample_info.sample_name = sample[len(sample)-1]
    sample_info.sample_class = sample[len(sample)-2]
    sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
    sample_vector = line[line.rfind('\t')+1:]
    accum_vector = []
    for concept in sample_vector.split(';;;;'):
        if len(concept)>1:
            concept_name = concept[0:concept.find(',,,,')]
            concept_weight = float(concept[concept.rfind(',,,,')+4:])
            accum_vector.append((concept_weight,concept_name))            
    sample_vec =  np.zeros(model.vector_size)
    for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
        #if concept_name=="Pimp My Ride":
        #    print("testing concept: ("+concept_name+")")
        sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
    max_sim = 0
    max_topic = ''
    for k,v in topics.items(): # loop on topics
        topic_accum_vector = v[0]
        vec =  np.zeros(model.vector_size)
        for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            vec += sample_weight*model['id'+titles[concept_name]+'di']
        sim = 1 - spatial.distance.cosine(vec,sample_vec)
        if sim>sample_info.max_all_sim:
            sample_info.max_all_sim = sim
            sample_info.max_topic_name = k
            sample_info.max_topic_id = topics_dic[k]
        #print(sample_info.sample_name,k,sim)
    #print(os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep)
    result1 += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
    if topics_dic[sample_info.sample_class]==sample_info.max_topic_id:
        totcal_correct += 1
    if total%100==0:
        print(str(totcal_correct)+'/'+str(total))

with open('results-all.txt','w') as resulta:
    resulta.write(result1)


print(totcal_correct)



#---------------------------------------------------------------------

import numpy as np
from scipy import spatial
import os

class SampleInfo:
     def __init__(self):
         self.max_topic_id = None
         self.max_topic_name = None
         self.sample_class = None
         self.sample_name = None
         self.max_all_sim = 0.0


from commons import load_titles
titles_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/titles_redirects_ids_anno.csv'
titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')

def getKey(item):
    return item[0]


print("loading topics")
topics_inpath = 'tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved'
topics = {}
topics_dic = {}
top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
topic_mappings = {'rec.motorcycles':'autos.sports', 'comp.windows.x':'computer', 'talk.politics.guns':'politics', 'talk.politics.mideast':'politics', 'talk.religion.misc':'religion', 'rec.autos':'autos.sports', 'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports', 'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer', 'sci.space':'science', 'talk.politics.misc':'politics', 'sci.electronics':'science', 'comp.graphics':'computer', 'comp.os.ms.windows.misc':'computer', 'sci.med':'science', 'sci.crypt':'science', 'soc.religion.christian':'religion', 'alt.atheism':'religion', 'misc.forsale':'sale'}
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

samples_inpath = '20newsgroups.simple.esa.concepts.500.anno.resolved'
lines = []
with open(samples_inpath,'r') as samples_file:
  for line in samples_file:
   lines.append(line)

for dim in [1,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500]:
    super_total_correct = {}
    sup_total_correct = {}
    for k,v in topic_mappings.items():
        sup_total_correct[k] = 0
        super_total_correct[v] = 0
    result1 = ''
    total_correct_sup = 0
    total_correct_super = 0
    for total,line in enumerate(lines):
        sample_info = SampleInfo()
        sample_info.max_all_sim = 0.0
        sample = line[0:line.find('\t')].split('\\')
        sample_info.sample_name = sample[len(sample)-1]
        sample_info.sample_class = sample[len(sample)-2]
        sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
        sample_vector = line[line.rfind('\t')+1:]
        accum_vector = []
        for concept in sample_vector.split(';;;;'):
            if len(concept)>1:
                concept_name = concept[0:concept.find(',,,,')]
                concept_weight = float(concept[concept.rfind(',,,,')+4:])
                accum_vector.append((concept_weight,concept_name))
        accum_vector = sorted(accum_vector, reverse=True, key=getKey)[0:min(dim,len(accum_vector))]
        sample_vec =  np.zeros(model.vector_size)
        for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            sample_vec += sample_weight*model['id'+titles[concept_name]+'di']
        max_sim = 0
        max_topic = ''
        for k,v in topics.items(): # loop on topics
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getKey)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                #if concept_name=="Pimp My Ride":
                #    print("testing concept: ("+concept_name+")")
                vec += sample_weight*model['id'+titles[concept_name]+'di']
            sim = 1 - spatial.distance.cosine(vec,sample_vec)
            result1 += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(sim)+os.linesep
            if sim>sample_info.max_all_sim:
                sample_info.max_all_sim = sim
                sample_info.max_topic_name = k
                sample_info.max_topic_id = topics_dic[k]
            #print(sample_info.sample_name,k,sim)
        #print(os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep)
        result1 += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
        if sample_info.sample_class==sample_info.max_topic_name:
            total_correct_sup += 1
            sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
        if topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
            total_correct_super += 1
            super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1
        if total%100==0:
            print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
    print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
    for k,v in sup_total_correct.items():
        result1 += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
    for k,v in super_total_correct.items():
        result1 += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
    result1 += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
    result1 += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
    result1 += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
    with open('results-all.txt.'+str(dim),'w') as resulta:
        resulta.write(result1)
    




#---------------------------------------------------------------------

import numpy as np
from scipy import spatial
import os

class SampleInfo:
     def __init__(self):
         self.max_topic_id = None
         self.max_topic_name = None
         self.sample_class = None
         self.sample_name = None
         self.max_all_sim = 0.0


def getKey(item):
    return item[0]


print("loading topics")
topics_inpath = 'tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved'
topics = {}
topics_dic = {}
top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
topic_mappings = {'rec.motorcycles':'autos.sports', 'comp.windows.x':'computer', 'talk.politics.guns':'politics', 'talk.politics.mideast':'politics', 'talk.religion.misc':'religion', 'rec.autos':'autos.sports', 'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports', 'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer', 'sci.space':'science', 'talk.politics.misc':'politics', 'sci.electronics':'science', 'comp.graphics':'computer', 'comp.os.ms.windows.misc':'computer', 'sci.med':'science', 'sci.crypt':'science', 'soc.religion.christian':'religion', 'alt.atheism':'religion', 'misc.forsale':'sale'}
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

samples_inpath = '20newsgroups.simple.esa.concepts.500.resolved'
lines = []
with open(samples_inpath,'r') as samples_file:
  for line in samples_file:
   lines.append(line)

for dim in [1,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500]:
    super_total_correct = {}
    sup_total_correct = {}
    for k,v in topic_mappings.items():
        sup_total_correct[k] = 0
        super_total_correct[v] = 0
    result1 = ''
    total_correct_sup = 0
    total_correct_super = 0
    for total,line in enumerate(lines):
        sample_info = SampleInfo()
        sample_info.max_all_sim = 0.0
        sample = line[0:line.find('\t')].split('\\')
        sample_info.sample_name = sample[len(sample)-1]
        sample_info.sample_class = sample[len(sample)-2]
        sample_info.sample_name = sample_info.sample_name+'_'+sample_info.sample_class
        sample_vector = line[line.rfind('\t')+1:]
        accum_vector = []
        for concept in sample_vector.split(';;;;'):
            if len(concept)>1:
                concept_name = concept[0:concept.find(',,,,')]
                concept_weight = float(concept[concept.rfind(',,,,')+4:])
                accum_vector.append((concept_weight,concept_name))
        accum_vector = sorted(accum_vector, reverse=True, key=getKey)[0:min(dim,len(accum_vector))]
        sample_vec =  np.zeros(model.vector_size)
        for i,(sample_weight,concept_name) in enumerate(accum_vector): # loop on each concept in sample
            #if concept_name=="Pimp My Ride":
            #    print("testing concept: ("+concept_name+")")
            try:
                sample_vec += sample_weight*model[concept_name]
            except:
                print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                result1+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                continue
        max_sim = 0
        max_topic = ''
        for k,v in topics.items(): # loop on topics
            topic_accum_vector = v[0]
            topic_accum_vector = sorted(topic_accum_vector, reverse=True, key=getKey)[0:min(dim,len(topic_accum_vector))]
            vec =  np.zeros(model.vector_size)
            for i,(sample_weight,concept_name) in enumerate(topic_accum_vector): # loop on each concept in sample
                #if concept_name=="Pimp My Ride":
                #    print("testing concept: ("+concept_name+")")
                try:
                    vec += sample_weight*model[concept_name]
                except:
                    print(os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep)
                    result1+=os.linesep+'FATAL\t'+concept_name+'...not found'+os.linesep
                    continue
            sim = 1 - spatial.distance.cosine(vec,sample_vec)
            result1 += os.linesep+'SIMILA\t'+sample_info.sample_name+'\t'+k+'\t'+str(sim)+os.linesep
            if sim>sample_info.max_all_sim:
                sample_info.max_all_sim = sim
                sample_info.max_topic_name = k
                sample_info.max_topic_id = topics_dic[k]
            #print(sample_info.sample_name,k,sim)
        #print(os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep)
        result1 += os.linesep+'TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str(topics_dic[sample_info.sample_class])+'\t'+sample_info.max_topic_name+'\t'+str(sample_info.max_topic_id)+'\t'+str(sample_info.max_all_sim)+os.linesep
        if sample_info.sample_class==sample_info.max_topic_name:
            total_correct_sup += 1
            sup_total_correct[sample_info.max_topic_name] = sup_total_correct[sample_info.max_topic_name]+1
        if topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
            total_correct_super += 1
            super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1
        if total%100==0:
            print(str(total_correct_sup+total_correct_super)+'/'+str(2*total))
    print(os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+os.linesep)
    for k,v in sup_total_correct.items():
        result1 += os.linesep+'@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep    
    for k,v in super_total_correct.items():
        result1 += os.linesep+'@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep    
    result1 += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/total)+os.linesep
    result1 += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep
    result1 += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/total)+os.linesep
    with open('results-all.txt.'+str(dim),'w') as resulta:
        resulta.write(result1)

'''
tail -n 60 results-all.anno/results-all.txt.10
@10 - leaf:comp.windows.x - 686
@10 - leaf:talk.religion.misc - 247
@10 - leaf:rec.sport.hockey - 936
@10 - leaf:talk.politics.guns - 842
@10 - leaf:comp.graphics - 253
@10 - leaf:sci.space - 724
@10 - leaf:rec.sport.baseball - 832
@10 - leaf:comp.sys.mac.hardware - 375
@10 - leaf:rec.autos - 696
@10 - leaf:talk.politics.mideast - 835
@10 - leaf:sci.med - 649
@10 - leaf:comp.sys.ibm.pc.hardware - 686
@10 - leaf:comp.os.ms.windows.misc - 510
@10 - leaf:misc.forsale - 108
@10 - leaf:talk.politics.misc - 250
@10 - leaf:sci.crypt - 503
@10 - leaf:alt.atheism - 592
@10 - leaf:soc.religion.christian - 321
@10 - leaf:rec.motorcycles - 306
@10 - leaf:sci.electronics - 545
@10 - super:politics - 2558
@10 - super:autos.sports - 3138
@10 - super:computer - 4470
@10 - super:science - 2609
@10 - super:religion - 2290
@10 - super:sale - 108
@10 - leaves:10896 - 0.5449907467613665
@10 - supers:15173 - 0.7589156204671635
@10:26069 - 0.651953183614265
computer = 4470/5000.0          0.894
politics = 2558/3000.0          0.8526666666666667
auto = 3138/4000.0              0.7845
religion = 2290/3000.0          0.7633333333333333
sale = 108/1000.0               0.108
science = 2609/4000.0           0.65225
'''
'''
tail -n 60 results-all.conc/results-all.txt.30
@30 - leaf:sci.space - 669
@30 - leaf:comp.sys.ibm.pc.hardware - 670
@30 - leaf:misc.forsale - 223
@30 - leaf:talk.politics.mideast - 804
@30 - leaf:soc.religion.christian - 427
@30 - leaf:rec.sport.baseball - 902
@30 - leaf:talk.politics.guns - 852
@30 - leaf:rec.autos - 722
@30 - leaf:comp.os.ms.windows.misc - 493
@30 - leaf:rec.motorcycles - 174
@30 - leaf:talk.religion.misc - 205
@30 - leaf:sci.med - 646
@30 - leaf:rec.sport.hockey - 862
@30 - leaf:sci.crypt - 482
@30 - leaf:alt.atheism - 598
@30 - leaf:talk.politics.misc - 157
@30 - leaf:sci.electronics - 510
@30 - leaf:comp.windows.x - 658
@30 - leaf:comp.sys.mac.hardware - 402
@30 - leaf:comp.graphics - 266
@30 - super:science - 2456
@30 - super:politics - 2480
@30 - super:computer - 4329
@30 - super:sale - 223
@30 - super:religion - 2363
@30 - super:autos.sports - 2987
@30 - leaves:10722 - 0.5362877006952433
@30 - supers:14838 - 0.7421597559145701
@30:25560 - 0.6392237283049067
computer = 4329/5000.0          0.8658
politics = 2480/3000.0          0.8266666666666667
auto = 2987/4000.0              0.74675
religion = 2363/3000.0          0.7876666666666666
sale = 223/1000.0               0.223
science = 2456/4000.0           0.614
'''


#---------------------------------------------------------------------
import sys
import gensim
import numpy as np
from commons import ng_topic_mappings
from commons import load_topics
from commons import load_samples
from commons import load_titles
from commons import classify_samples_with_densification_bottomup
from commons import classify_samples_with_densification_bottomup_bootstrap
from commons import classify_samples_with_densification_topdown
#from commons import classify_samples
from commons import SampleInfo
from commons import getWeight

ng_topic_mappings_autos_motors = {'rec.motorcycles':'autos.sports','rec.autos':'autos.sports'}
ng_topic_mappings_autos_elecs = {'rec.autos':'autos.sports','sci.electronics':'science'}
ng_topic_mappings_guns_mideast_misc = {'talk.politics.guns':'politics', 'talk.politics.mideast':'politics', 'talk.politics.misc':'politics'}
ng_topic_mappings_christ_atheism_misc = {'talk.religion.misc':'religion', 'soc.religion.christian':'religion', 'alt.atheism':'religion'}
ng_topic_mappings_christ_atheism = {'soc.religion.christian':'religion', 'alt.atheism':'religion'}
ng_topic_mappings_comp = {'comp.windows.x':'computer', 'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer', 'comp.graphics':'computer', 'comp.os.ms.windows.misc':'computer'}
ng_topic_mappings_wnds_mac = {'comp.windows.x':'computer', 'comp.sys.mac.hardware':'computer'}
ng_topic_mappings_ibm_mac = {'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer'}
ng_topic_mappings_wnds_hw = {'comp.windows.x':'computer', 'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer', 'comp.os.ms.windows.misc':'computer'}
ng_topic_mappings_sport = {'rec.motorcycles':'autos.sports','rec.autos':'autos.sports', 'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports'}
ng_topic_mappings_base_hockey = {'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports'}
ng_topic_mappings_base_hockey_autos = {'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports','rec.autos':'autos.sports'}
ng_topic_mappings_science = {'sci.space':'science', 'sci.electronics':'science', 'sci.med':'science', 'sci.crypt':'science'}

#/home/walid/work/github/semantic_searcher/
#/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/
#/scratch/wshalaby/doc2vec/models/word2vec/

mode = sys.argv[1] # conc or w2v
'''yes'''#mode = 'conc'
#mode = 'w2v'

topics_inpath = sys.argv[2]
#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved.before.new.map'
#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.resolved'
#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.resolved'
#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.500.leaves'
#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.seealsoasso.uniq.1.0.5.1.accum.augm.seealso_asso.500.newmap.resolved'
#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.subnewmap.resolved'
'''yes'''#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.subnewmap.v2.resolved'
#topics_inpath = '/media/vol2/walid/work/github/semantic_searcher/tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.anno.subnewmap.v2.seealsoasso.uniq.1.0.5.1.accum.augm.seealso_asso.500.resolved'

'''yes'''#super_topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.supers.500.anno.subnewmap.v2.resolved'
#super_topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.supers.500'

#concepts
#topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.500.subnewmap.resolved'

samples_inpath = sys.argv[3]
#samples_inpath = '../20newsgroups.simple.esa.concepts.500.anno.resolved.before.new.map'
#samples_inpath = '../20newsgroups.simple.esa.concepts.500.resolved'
#samples_inpath = '../20newsgroups.simple.esa.concepts.500.anno.resolved'
#samples_inpath = '../20newsgroups.simple.esa.concepts.500'
#samples_inpath = '../20newsgroups.simple.esa.concepts.500.anno.seealsoasso.uniq.1.0.5.1.accum.augm.seealso_asso.500.newmap.resolved'
#samples_inpath = '../20newsgroups.simple.esa.concepts.500.anno.subnewmap.resolved'
'''yes'''#samples_inpath = '../20newsgroups.simple.esa.concepts.500.anno.subnewmap.v2.resolved'
#samples_inpath = '/media/vol2/walid/work/github/semantic_searcher/20newsgroups.simple.esa.concepts.500.anno.subnewmap.v2.seealsoasso.uniq.1.0.5.1.accum.augm.seealso_asso.500.resolved'

#concepts
#samples_inpath = '../20newsgroups.simple.esa.concepts.500.subnewmap.resolved'

model_path = sys.argv[4]
#model_path = "../concepts-10iter-dim500-wind5-skipgram1.model"
#model_path = "../w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.model"
'''yes'''
mode=w2v
#model_path = "/scratch/wshalaby/doc2vec/models/word2vec/w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.model"
#model_path = '/media/vol2/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.model'
#model_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.model'
'''yes'''
#model_path = '/scratch/wshalaby/doc2vec/concepts-self-dic-sentence-per-article/concepts-self-dic-10iter-dim500-wind9-skipgram1.model'

titles_path = sys.argv[5]
#titles_path = '../titles_redirects_ids.csv'
'''yes'''#titles_path = '/scratch/wshalaby/doc2vec/titles_redirects_ids.csv'

outpath = sys.argv[6]
#outpath = 'densification-top-down-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-top-down-aggr-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-sport-politics-top-down-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-sport-politics-top-down-aggr-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-politics-religion-top-down-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-politics-religion-top-down-aggr-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-sport-religion-top-down-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-sport-religion-top-down-aggr-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'

#outpath = 'densification-top-down-aggr-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-top-down-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-sport-politics-top-down-aggr-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-sport-politics-top-down-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-politics-religion-top-down-aggr-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-politics-religion-top-down-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-sport-religion-top-down-aggr-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-sport-religion-top-down-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'


#outpath = 'densification-sports-w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-autos-hockey-baseball-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-10iter-dim500-wind9-cnt1-skipgram1'
'''yes'''#outpath = 'densification-guns-mideast-misc-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-autos-motors-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-base-hockey-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-autos-hockey-baseball-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-sports-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
#outpath = 'densification-sports-concepts-self-dic-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-sports-concepts-self-dic-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-sports-concepts-self-dic-notanno-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-guns-mideast-misc-concepts-self-dic-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-concepts-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-autos-motors-concepts-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-bottom-up-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-guns-mideast-misc-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-autos-motors-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-wnds-mac-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-wnds-hw-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-ibm-mac-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-base-hockey-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-base-hockey-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-base-hockey-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-guns-mideast-misc-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-guns-mideast-misc-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-autos-motors-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-autos-motors-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-autos-motors-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1.seealsoasso.uniq.1.0.5.1.accum.augm.seealso_asso.500'
#outpath = 'densification-comp-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-wnds-hw-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-wnds-mac-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-ibm-mac-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-sport-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-autos-hockey-baseball-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-sport-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-christ-atheism-misc-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-christ-atheism-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-science-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-autos-elecs-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-christ-atheism-misc-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-christ-atheism-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-comp-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-sport-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-science-bottom-up-newsubmap-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-bottom-up-w2v-concepts-10iter-dim500-wind7-skipgram1'
#outpath = 'densification-bottom-up-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-bottom-up-newsubmap-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
#outpath = 'densification-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-new-mappings-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-new-mappings-top-down-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-top-down-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-new-mappings-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1.seealsoasso.uniq.1.0.5.1.accum.augm.seealso_asso.500'
#outpath = 'densification-autos-motors-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-new-mappings-autos-motors-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-autos-elecs-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'densification-new-mappings-autos-elecs-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
#outpath = 'esa-no-densification'
#outpath = 'esa-no-densification.resolved'
#outpath = 'esa-no-densification.autos.motors'
#outpath = 'esa-no-densification.autos.elecs'
#outpath = 'esa-no-densification.religion.christ.atheism.misc'
#outpath = 'esa-no-densification.religion.christ.atheism'
#outpath = 'esa-no-densification.politics.guns.mideast.misc'
#outpath = 'esa-no-densification.comp'
#outpath = 'esa-no-densification.wnds.mac'
#outpath = 'esa-no-densification.wnds.hw'
#outpath = 'esa-no-densification.ibm.mac'
#outpath = 'esa-no-densification.sport'
#outpath = 'esa-no-densification.science'
#outpath = 'esa-no-densification.base.hockey'
#outpath = 'esa-no-densification.autos.base.hockey'

#outpath = 'esa-no-densification-top-down'
#outpath = 'esa-no-densification-sport-politics-top-down'
#outpath = 'esa-no-densification-politics-religion-top-down'
#outpath = 'esa-no-densification-sport-religion-top-down'

'''yes new-esa'''
cat 20newsgroups_esa_con500_in0_out5_alltext.txt |grep 104582_rec.motorcycles
topics_inpath = '20newsgroups-topics_esa_con500_len500_in1_out5_alltext.txt'
samples_inpath = '20newsgroups_esa_con500_len500_in1_out5_alltext.txt'
super_topics_inpath = '20newsgroups-super-topics_esa_con500_len500_in1_out5_alltext.txt'
topics_new, topics_dic_new = load_topics(topics_inpath)
samples_new = load_samples(samples_inpath)
super_topics_new, super_topics_dic_new = load_topics(super_topics_inpath)

topic_mappings = ng_topic_mappings_base_hockey
outpath = 'densification-base-hockey-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-base-hockey-bottom-up-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
outpath = 'densification-base-hockey-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-base-hockey-bottom-up-bootstrap-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup_bootstrap(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims, weighted_bs=True, min_bs_score=0,min_bs_diff=0)

outpath = 'esa-base-hockey-bottom-up-new-esa-len500-in1'
classify_samples_bottomup(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, outpath=outpath)

topic_mappings = ng_topic_mappings_guns_mideast_misc
outpath = 'densification-guns-mideast-misc-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-guns-mideast-misc-bottom-up-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
outpath = 'densification-guns-mideast-misc-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-guns-mideast-misc-bottom-up-bootstrap-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup_bootstrap(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims, weighted_bs=True, min_bs_score=0,min_bs_diff=0)

outpath = 'esa-guns-mideast-misc-bottom-up-new-esa-len500-in1'
classify_samples_bottomup(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, outpath=outpath)

topic_mappings = ng_topic_mappings_autos_motors
outpath = 'densification-autos-motors-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-autos-motors-bottom-up-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
outpath = 'densification-autos-motors-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-autos-motors-bottom-up-bootstrap-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup_bootstrap(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims, weighted_bs=True, min_bs_score=0,min_bs_diff=0)

outpath = 'esa-autos-motors-bottom-up-new-esa-len500-in1'
classify_samples_bottomup(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, outpath=outpath)

topic_mappings = ng_topic_mappings
outpath = 'densification-20ng-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-20ng-bottom-up-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
outpath = 'densification-20ng-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-20ng-bottom-up-bootstrap-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_bottomup_bootstrap(samples=samples_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims, weighted_bs=True, min_bs_score=0,min_bs_diff=0)

ng_topic_mappings_sport_politics = {}
ng_topic_mappings_sport_politics.update(ng_topic_mappings_sport)
ng_topic_mappings_sport_politics.update(ng_topic_mappings_guns_mideast_misc)
outpath = 'densification-sport-politics-topdown-aggr-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-sport-politics-topdown-aggr-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_topdown_aggr(samples=samples_new, topics=topics_new, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_politics, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

outpath = 'densification-sport-politics-topdown-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-sport-politics-topdown-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_topdown(samples=samples_new, super_topics=super_topics_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=ng_topic_mappings_sport_politics, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

outpath = 'esa-sport-politics-topdown-new-esa-len500-in1'
classify_samples_topdown(samples=samples_new, super_topics=super_topics_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, outpath=outpath)

ng_topic_mappings_sport_religion = {}
ng_topic_mappings_sport_religion.update(ng_topic_mappings_sport)
ng_topic_mappings_sport_religion.update(ng_topic_mappings_christ_atheism_misc)
outpath = 'densification-sport-religion-topdown-aggr-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-sport-religion-topdown-aggr-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_topdown_aggr(samples=samples_new, topics=topics_new, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_religion, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

outpath = 'densification-sport-religion-topdown-w2v-4.4-concepts-from-plain-anno-titles-new-esa-len500-in1-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-sport-religion-topdown-new-esa-len500-in1-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
classify_samples_with_densification_topdown(samples=samples_new, super_topics=super_topics_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=ng_topic_mappings_sport_religion, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

outpath = 'esa-sport-religion-topdown-new-esa-len500-in1'
classify_samples_topdown(samples=samples_new, super_topics=super_topics_new, topics=topics_new, topics_dic=topics_dic_new, topic_mappings=topic_mappings, outpath=outpath)

'''yes new-esa'''

import os
for dim in [50,100]:
    for window in [1,2,3,4,5,6,7,8,9,10]:
        model_path = '../models/concepts-10iter-dim'+str(dim)+'-wind'+str(window)+'-skipgram1.model'
        if os.path.isfile(model_path):
            model = gensim.models.Word2Vec.load(model_path)
            outpath = 'densification-concepts-10iter-dim'+str(dim)+'-wind'+str(window)+'-skipgram1'
            classify_samples_with_densification(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath)
        else:
            print('not found',model_path)

'''yes'''
model = gensim.models.Word2Vec.load(model_path)
model.init_sims(replace=True)

topics, topics_dic = load_topics(topics_inpath)

super_topics, super_topics_dic = load_topics(super_topics_inpath)

samples = load_samples(samples_inpath)

titles, redirects, seealso = load_titles(titles_path, load_redirects=True, redirects_value='title')
dims = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500]
'''yes'''

classify_samples_with_densification_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_topdown_aggr(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

'''yes'''
ng_topic_mappings_sport_politics = {}
ng_topic_mappings_sport_politics.update(ng_topic_mappings_sport)
ng_topic_mappings_sport_politics.update(ng_topic_mappings_guns_mideast_misc)
#classify_samples_with_densification_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_politics, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_topdown_aggr(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_politics, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

ng_topic_mappings_sport_religion = {}
ng_topic_mappings_sport_religion.update(ng_topic_mappings_sport)
ng_topic_mappings_sport_religion.update(ng_topic_mappings_christ_atheism_misc)
#classify_samples_with_densification_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_religion, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_topdown_aggr(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_religion, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
'''yes'''

ng_topic_mappings_politics_religion = {}
ng_topic_mappings_politics_religion.update(ng_topic_mappings_guns_mideast_misc)
ng_topic_mappings_politics_religion.update(ng_topic_mappings_christ_atheism_misc)
classify_samples_with_densification_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_politics_religion, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_topdown_aggr(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_politics_religion, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_wnds_mac, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_ibm_mac, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_wnds_hw, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_autos_elecs, model=model, mode=mode, titles=titles, outpath=outpath)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_base_hockey_autos, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

'''yes'''
topic_mappings = ng_topic_mappings_autos_motors
outpath = 'densification-autos-motors-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-autos-motors-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'

topic_mappings = ng_topic_mappings_guns_mideast_misc
outpath = 'densification-guns-mideast-misc-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-guns-mideast-misc-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'

topic_mappings = ng_topic_mappings_base_hockey
outpath = 'densification-base-hockey-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-base-hockey-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'

topic_mappings = ng_topic_mappings
outpath = 'densification-20ng-bottom-up-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-20ng-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'

classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup_bootstrap(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims, weighted_bs=True, min_bs_score=0.95,min_bs_diff=0.50)
classify_samples_with_densification_bottomup_bootstrap(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims, weighted_bs=True, min_bs_score=0,min_bs_diff=0)

ng_topic_mappings_sport_religion = {}
ng_topic_mappings_sport_religion.update(ng_topic_mappings_sport)
ng_topic_mappings_sport_religion.update(ng_topic_mappings_christ_atheism_misc)
topic_mappings = ng_topic_mappings_sport_religion
outpath = 'densification-sport-religion-top-down-aggr-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-sport-religion-top-down-aggr-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'

ng_topic_mappings_sport_politics = {}
ng_topic_mappings_sport_politics.update(ng_topic_mappings_sport)
ng_topic_mappings_sport_politics.update(ng_topic_mappings_guns_mideast_misc)
topic_mappings = ng_topic_mappings_sport_politics
outpath = 'densification-sport-politics-top-down-aggr-bootstrap-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1'
outpath = 'densification-sport-politics-top-down-aggr-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'

classify_samples_with_densification_topdown_aggr(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_topdown_aggr_bootstrap(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)

'''yes'''

'''etr'''
ids = {}
for k in model.vocab():
 ids[v] = k


def flush_vocabulary_as_associations(model, outpath):
 import os
 with open(outpath,'w') as outp:
  for concept in model.vocab:
   if concept[0:2]=='id' and concept[-2:]=='di':
    id = concept.replace('id','').replace('di','')
    if id.isdigit():
     outp.write(ids[id]+'/\/\\1'+os.linesep)

flush_vocabulary_as_associations(model, '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/wiki_asso_etr.txt')

Institute of Physics and Power Engineering
advanced certified patient account representative (acpar)
muslim_skill
dealer business systems_skill
mtv news_skill
mobile app_skill
reservec_skill
jad_skill
some college accounting_school
beauty_school
ocean_school
ascent hearing care_company

topics v1 no titles
    samples no titles
    14 --> 8040 (best)
    9 --> 8036

topics v2 no titles
    samples no titles
    14 --> 8067
    45 --> 8226 (best), 40 --> 8183, 50 --> 8132
    11 --> 8207
    12 --> 8145
    samples titles
    11 --> 8052

topics v1 titles
    samples no titles
    6 --> 7448
    14 --> 5897

topics v2 no titles
    samples no titles
    12 --> 8058 (best)
    14 --> 7873
   

topics v2+added company,skill...etc in the front no titles
    samples no titles
    45 --> 8274

topics v2 titles
    samples titles
    12 --> 7860 (best)
    samples no titles

topics v3 no titles, samples no titles
    45 --> 8247
topics v4 no titles, samples no titles
    12 --> 8329
    11 --> 8323
topics v5 no titles, samples no titles
    12 --> 8328
    11 --> 8316

topics v6 no titles, samples no titles
@11 - leaves:8338 - 0.8372326538809117
@12 - leaves:8321 - 0.8355256551862636
@40 - leaves:8346 - 0.8380359473842756
@45 - leaves:8382 - 0.8416507681494126
@50 - leaves:8320 - 0.8354252434983432
@60 - leaves:8322 - 0.8356260668741842

topics v7 no titles, samples no titles
@11 - leaves:8204 - 0.8237774876995683

topics v8 no titles, samples no titles
@11 - leaves:8304 - 0.8338186564916156

topics v9 no titles, samples no titles
@45 - leaves:8337 - 0.8371322421929913

topics v10 no titles from v6, samples no titles
@40 - leaves:8418 - 0.8452655889145496
@45 - leaves:8429 - 0.8463701174816749
@60 - leaves:8438 - 0.8472738226729591

topics v11 no titles, samples no titles
@40 - leaves:8425 - 0.8459684707299929
@45 - leaves:8424 - 0.8458680590420725
@60 - leaves:8426 - 0.8460688824179134

topics v16 no titles, samples no titles
@25 - leaves:8408 - 0.844261472035345
@30 - leaves:8426 - 0.8460688824179134
@35 - leaves:8434 - 0.8468721759212773

topics v17 no titles, samples no titles
@30 - leaves:8426 - 0.8460688824179134
@35 - leaves:8426 - 0.8460688824179134

topics v18 no titles, samples no titles
@35 - leaves:8419 - 0.8453660006024701

companies
outpath = 'densification-dbcompanies-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v20'
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/dbpedia_companies_training_esa_con500_in1_out5_alltext.txt_01'

outpath = 'densification-dbcompanies-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v20_ps2'
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/dbpedia_companies_training_esa_con500_in1_out5_alltext_ps2.txt_01'

all_titles_lower = get_all_titles(titles=titles, redirects=redirects, model=model)

coverage = create_concept_dataset([('/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/company1000.txt','/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/company1000-concept.txt'),('/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/school_university1000.txt','/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/school_university1000-concept.txt')], all_titles_lower)
coverage = create_concept_dataset([('/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/probase_sample1000.txt','/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/probase_sample1000-concept.txt')], all_titles_lower)
coverage = create_concept_dataset([('/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/probase_all.txt','/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/probase_all-concept.txt')], all_titles_lower)

f= open('/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/etr/data/dbpedia_companies_samples.csv','r')
dp = csv.reader(f)
for i,j in dp:                                                                                                                             
 dbpedia_companies.add(i.lower())                                                                                                          

all_count = 0
count = 0
correct = set()
incorrect = set()
for i in dbpedia_companies:
    if i in all_titles_lower:
        id = all_titles_lower[i][0]
        if id in model:
            all_count += 1
            max_sim = -100
            max_topic = ''
            for j in ['Skill', 'School', 'Company', 'Job']:
                sim = 1 - spatial.distance.cosine(model[id],model['id'+titles[j]+'di'])
                if sim>max_sim:
                    max_sim = sim
                    max_topic = j
            if max_topic=='Company':
                count += 1
                correct.add(i)
            else:
                incorrect.add(i)
                if i=='the behemoth':
                    print(id,max_topic,max_sim)


topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/etr_topics_esa_con500_in0_out5_alltext_v20.txt'

topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/etr.txt'
topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/etr_topics_esa_con500_in0_out5_alltext_v1.txt'
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/training_entity_type_v4_esa_con500_in0_out5_alltext.txt_01'
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/training_entity_type_v4_esa_con500_in0_out5_alltext_all.txt'
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/training_entity_type_v4_esa_con500_in1_out5_alltext_ps2.txt_01'
topics, topics_dic = load_topics(topics_inpath)

topics from samples
cat /media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/training_entity_type_v4_esa_con500_in1_out5_alltext_ps2.txt_01 |grep -P 'job_title\\'|head -n 10 > tmp4.txt
topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/training_entity_type_v4_esa_con500_in1_out5_alltext_ps2_topics_samples.txt_01'
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/training_entity_type_v4_esa_con500_in1_out5_alltext_ps2.txt_01'
outpath = 'densification-etr-bottom-up-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v20_ps2_topics_samples'
topics, topics_dic = load_topics(topics_inpath, multi=True)

samples = load_samples(samples_inpath)

topic_mappings_etr = {'company':'etr','skill':'etr','job_title':'etr','school':'etr'}

outpath = 'densification-etr-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1'
outpath = 'densification-etr-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v20_ps2'

context
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/etr/data/training_entity_type_v3_wiki_context_min100_hits3_high_ctg_wtitle_quote_esa_con500_in1_out5_alltext_ps2.txt_01'
topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/etr_topics_esa_con500_in0_out5_alltext_v20.txt'
outpath = 'densification-etr-cxt-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v20_ps2'

topics from samples with context
cat /media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/etr/data/training_entity_type_v3_wiki_context_min100_hits3_high_ctg_wtitle_quote_esa_con500_in1_out5_alltext_ps2.txt_01 |grep -P 'job_title\\'|head -n 10 > tmp4.txt
topics_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/solr-4.10.2/solr/example/training_entity_type_v3_wiki_context_min100_hits3_high_ctg_wtitle_quote_esa_con500_in1_out5_alltext_ps2_topics_samples.txt_01'
samples_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/etr/data/training_entity_type_v3_wiki_context_min100_hits3_high_ctg_wtitle_quote_esa_con500_in1_out5_alltext_ps2.txt_01'
outpath = 'densification-etr-cxt-bottom-up-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v20_ps2_topics_samples'

dims = [1]
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=topic_mappings_etr, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup_bootstrap(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=topic_mappings_etr, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims, weighted_bs=True, min_bs_score=0,min_bs_diff=0)

tail -n 20 densification-etr-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v4/*|grep leaves:8
tail -n 12 densification-etr-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v6/45
cat densification-etr-bottom-up-bootstrap-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1_v6/45|grep school|grep company|grep TOPICA > tmp.txt
cat tmp.txt | tr ' ' '\n' | tr '\t' '\n' | sort | uniq -ic | sort -n > count.txt
cat training_entity_type_v4.csv|grep company|grep corporation

cat training_entity_type_v4_esa_con500_in0_out5_alltext.txt_01|grep -P 'skill\\'|sed 's/;;;;/\n/g' | sed 's/,,,,/\n/g' > tmp.txt
cat tmp.txt | sort | uniq -ic | sort -n > count.txt
cat count.txt |grep -P '[a-z]+'|tail -n 100

cat etr_topics_esa_con500_in0_out5_alltext_v21.txt|grep -P 'skill\t'|sed 's/;;;;/\n/g' | sed 's/,,,,/\n/g' > tmp.txt
cat tmp.txt | sort | uniq -ic | sort -n > count-topics.txt
cat count-topics.txt |grep -P '[a-z]+'|tail -n 100

@60 - leaves:8438 - 0.8472738226729591

@60 - supers:9960 - 1.0001004116879204

@60:18398 - 0.9236871171804398
        company skill   school  job_title
company 2104.0  278.0   412.0   3.0
skill   194.0   1255.0  107.0   1.0
school  188.0   102.0   5077.0  3.0
job_title       86.0    107.0   41.0    2.0
        etr
etr     9960.0

@45 - leaves:125441 - 0.8148270844700808

@45 - supers:148893 - 0.967164237274924

@45:274334 - 0.8909956608725024
        company skill   school  job_title
company 29938.0 4387.0  5377.0  49.0
skill   2778.0  18402.0 1279.0  852.0
school  3552.0  1807.0  76944.0 140.0
job_title       1125.0  1499.0  607.0   157.0
        etr
etr     148893.0

'''etr'''



classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_science, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_autos_elecs, model=model, mode=mode, titles=titles, outpath=outpath)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_christ_atheism_misc, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_comp, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_bottomup(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_christ_atheism, model=model, mode=mode, titles=titles, outpath=outpath,dims=dims)
classify_samples_with_densification_topdown(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath)

topics, topics_dic = load_topics(topics_inpath, conc_sep=';', weight_sep=',')
super_topics, super_topics_dic = load_topics(super_topics_inpath, conc_sep=';', weight_sep=',')
samples = load_samples(samples_inpath)

classify_samples_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_politics, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_politics_religion, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples_topdown(samples=samples, super_topics=super_topics, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport_religion, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)

classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, outpath=outpath)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, conc_sep=';', weight_sep=',', outpath=outpath)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_wnds_hw, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_wnds_mac, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_ibm_mac, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_autos_elecs, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_autos_motors, conc_sep=';', weight_sep=',', outpath=outpath)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_base_hockey, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_base_hockey_autos, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_guns_mideast_misc, conc_sep=';', weight_sep=',', outpath=outpath)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_christ_atheism_misc, conc_sep=';', weight_sep=',', outpath=outpath)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_christ_atheism, conc_sep=';', weight_sep=',', outpath=outpath)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_comp, conc_sep=';', weight_sep=',', outpath=outpath)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_sport, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)
classify_samples(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings_science, conc_sep=';', weight_sep=',', outpath=outpath,dims=dims)

topics_inpath = '../tree.20newsgroups.simple.esa.concepts.newrefine.leaves.10.anno.resolved'
samples_inpath = '../20newsgroups.simple.esa.concepts.10.anno.resolved'
outpath = 'densification-new-mappings-w2v-plain-anno-titles-10iter-dim500-wind5-skipgram1'
topics, topics_dic = load_topics(topics_inpath)
samples = load_samples(samples_inpath)
classify_samples_with_densification(samples=samples, topics=topics, topics_dic=topics_dic, topic_mappings=ng_topic_mappings, model=model, mode=mode, titles=titles, outpath=outpath, dims=[10])

import gensim
model_path = '/scratch/wshalaby/doc2vec/google/freebase-vectors-skipgram1000-en.bin'
model = gensim.models.word2vec.Word2Vec.load_word2vec_format(model_path,binary=True)
model.init_sims(replace=True)

model_path1 = '/scratch/wshalaby/doc2vec/google/freebase-vectors-skipgram1000.bin'
model1 = gensim.models.word2vec.Word2Vec.load_word2vec_format(model_path,binary=True)
model1.init_sims(replace=True)

for i in model.vocab:
    if i.find('in')>0 and i.find('sport')>0 and i.find('2005')>0:
        print(i)

ids = {}
for k,v in titles.items():
    ids[v] = k


for ind,i in enumerate(model.most_similar('id'+titles['Yamaha Virago']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])


model.similarity('id'+titles['Vancouver Canucks']+'di','id'+titles['Edmonton Oilers']+'di')
model.similarity('id'+titles['Yamaha TZR250']+'di','id'+titles['Yamaha WR250F']+'di')







"""
model_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-4.4-concepts-20160820-no-redirects-dic-10iter-dim500-wind7-cnt1-skipgram1.model
0 ('id7529378di', 0.8142267465591431) Facebook
1 ('id9988187di', 0.7824112176895142) Twitter
2 ('id1270655di', 0.7274441719055176) Myspace
3 ('id4434812di', 0.717693567276001) Viral video
4 ('id186266di', 0.7051114439964294) ITunes
5 ('id31591547di', 0.6590680480003357) Instagram
6 ('id7482183di', 0.6539322137832642) Vimeo
7 ('id27401317di', 0.6529773473739624) SoundCloud
8 ('id22347942di', 0.6386541128158569) Vevo
9 ('id28682di', 0.6329643130302429) Streaming media
10 ('id18701436di', 0.6319628953933716) Tumblr
11 ('id5897742di', 0.624632716178894) Social media
12 ('id218232di', 0.6190321445465088) ITunes Store
13 ('id18856di', 0.6170687675476074) MTV
14 ('id90138di', 0.6132311820983887) Music video
15 ('id90451di', 0.6120947599411011) Amazon.com
16 ('id33645di', 0.6112318634986877) Blog
17 ('id1092923di', 0.6058774590492249) Google
18 ('id37180810di', 0.6011822819709778) Reply girl
19 ('id20148343di', 0.6010741591453552) Spotify


ids = {}
for k,v in titles.items():
    ids[v] = k


def topn(concept='Harvard University', topn=20, concepts_only=True):
    for ind,i in enumerate(model.most_similar('id'+titles[concept]+'di',topn=20)):
        id = i[0].replace('id','').replace('di','')
        if id in ids:
            print(ind,i,ids[id])
        elif concepts_only==False:
            print(ind,i)
 

0 ('id34273di', 0.882897138595581) Yale University
1 ('id23922di', 0.8451585173606873) Princeton University
2 ('id6310di', 0.8445795774459839) Columbia University
3 ('id18879di', 0.8435715436935425) Massachusetts Institute of Technology
4 ('id61114di', 0.8104631900787354) Boston University
5 ('id260879di', 0.8013678789138794) Harvard College
6 ('id7954422di', 0.7936483025550842) Cornell University
7 ('id26977di', 0.7867215275764465) Stanford University
8 ('id4157di', 0.7847708463668823) Brown University
9 ('id7954455di', 0.7828654646873474) New York University
10 ('id32127di', 0.7778875827789307) University of Chicago
11 ('id31922di', 0.775860607624054) University of California, Berkeley
12 ('id31797di', 0.7717559933662415) University of Oxford
13 ('id8418di', 0.7702051401138306) Dartmouth College
14 ('id31793di', 0.7692397236824036) University of Pennsylvania
15 ('id158984di', 0.7663556933403015) Harvard Law School
16 ('id25978572di', 0.7631617188453674) University of Cambridge
17 ('id21031297di', 0.7521637678146362) Doctor of Philosophy
18 ('id60355di', 0.752018392086029) Amherst College
19 ('id38420di', 0.7468933463096619) Johns Hopkins University

for ind,i in enumerate(model.most_similar('id'+titles['Black hole']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])

0 ('id21869di', 0.7934496998786926) Neutron star
1 ('id29320146di', 0.7225103378295898) Event horizon
2 ('id54244di', 0.7219958901405334) Gravitational singularity
3 ('id8651di', 0.715164303779602) Dark matter
4 ('id12558di', 0.7106876969337463) Galaxy
5 ('id215706di', 0.7018566131591797) Supermassive black hole
6 ('id27680di', 0.6954707503318787) Supernova
7 ('id173724di', 0.6896173357963562) Hawking radiation
8 ('id19604228di', 0.6850820183753967) Dark energy
9 ('id19319238di', 0.6850706934928894) Relativistic star
10 ('id34043di', 0.6801064610481262) Wormhole
11 ('id23922614di', 0.6797451972961426) Primordial black hole
12 ('id33501di', 0.6783536076545715) White dwarf
13 ('id48400112di', 0.6737918257713318) Accretion disk
14 ('id28758di', 0.6732418537139893) Spacetime
15 ('id2589714di', 0.6731758117675781) Milky Way
16 ('id1508445di', 0.6728496551513672) Ergosphere
17 ('id619926di', 0.6712267398834229) Gravitational collapse
18 ('id510340di', 0.6709612011909485) Stellar black hole
19 ('id25239di', 0.6705248951911926) Quasar
for ind,i in enumerate(model.most_similar('id'+titles['Gemstone']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
1 ('id10045di', 0.7259731888771057) Emerald
2 ('id43551di', 0.7126421332359314) Ruby
4 ('id29469di', 0.6883571147918701) Sapphire
5 ('id12240di', 0.6716592907905579) Gold
6 ('id15739di', 0.6661626100540161) Jewellery
7 ('id25233di', 0.665136456489563) Quartz
8 ('id31415di', 0.6558405160903931) Topaz
9 ('id45162di', 0.6534045338630676) Peridot
10 ('id21776060di', 0.6515100598335266) Prince of Burma
11 ('id304300di', 0.6505627036094666) Cabochon
12 ('id24007di', 0.6482283473014832) Pearl
13 ('id180211di', 0.6461643576622009) Precious metal
14 ('id1366di', 0.6422092318534851) Amethyst
15 ('id5974di', 0.6405693292617798) Corundum
16 ('id1523di', 0.6356081962585449) Agate
17 ('id4910di', 0.6324129700660706) Beryl
18 ('id44653di', 0.6215612292289734) Lapis lazuli
19 ('id37506di', 0.6188913583755493) Garnet
for ind,i in enumerate(model.most_similar('id'+titles['FIFA']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])

0 ('id32332di', 0.7963888645172119) UEFA
2 ('id11370di', 0.705990731716156) FIFA World Cup
3 ('id715249di', 0.6936690807342529) Asian Football Confederation
4 ('id42278di', 0.6769502758979797) Sepp Blatter
5 ('id46136di', 0.6675164699554443) The Football Association
7 ('id29728916di', 0.6635397672653198) List of first association football internationals per country: 1940–1962
8 ('id36576030di', 0.6602137684822083) List of national teams with no AFC Asian Cup appearances
9 ('id11052di', 0.6576012969017029) List of presidents of FIFA
10 ('id715265di', 0.6575682759284973) Confederation of African Football
11 ('id511237di', 0.6518605351448059) CONMEBOL
12 ('id295167di', 0.6511364579200745) CONCACAF
14 ('id725851di', 0.6472413539886475) International Football Association Board
15 ('id501260di', 0.6424123048782349) Referee (association football)
16 ('id26336148di', 0.6349858045578003) 2010 FIFA World Cup officials
18 ('id25210872di', 0.6343994736671448) Portugal–Spain 2018 FIFA World Cup bid
19 ('id26349067di', 0.6314557194709778) FIFA Council
"""
"""
/scratch/wshalaby/doc2vec/models/word2vec/w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.model
for ind,i in enumerate(model.most_similar('id'+titles['FIFA']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])

0 ('id42278di', 0.7188396453857422) Sepp Blatter
1 ('id32332di', 0.6752723455429077) UEFA
2 ('id715249di', 0.6019099950790405) Asian Football Confederation
3 ('id501260di', 0.5982575416564941) Referee (association football)
4 ('id295167di', 0.5815110802650452) CONCACAF
5 ('id715265di', 0.5718251466751099) Confederation of African Football
6 ('id11052di', 0.568651020526886) List of presidents of FIFA
7 ('id725851di', 0.5532274842262268) International Football Association Board
8 ('id11370di', 0.5491757988929749) FIFA World Cup
9 ('id46812210di', 0.5481243133544922) 2015 FIFA corruption case
10 ('id15998477di', 0.5437405109405518) Assistant referee (association football)
11 ('id26349067di', 0.5406002402305603) FIFA Council
12 ('id3476965di', 0.5381742119789124) World Football Elo Ratings
13 ('id249510di', 0.5376392602920532) UEFA European Championship
14 ('id214478di', 0.5371350049972534) Michel Platini
15 ('id279712di', 0.5333603620529175) Oceania Football Confederation
16 ('id876236di', 0.5333590507507324) List of men's national association football teams
17 ('id579541di', 0.5330038666725159) João Havelange
18 ('id3811166di', 0.526019275188446) International Federation of Football History &amp; Statistics
19 ('id592115di', 0.525469183921814) FIFA Confederations Cup

for ind,i in enumerate(model.most_similar('id'+titles['YouTube']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])

0 ('id4434812di', 0.7225453853607178) Viral video
1 ('id7482183di', 0.618456244468689) Vimeo
2 ('id22347942di', 0.5517860054969788) Vevo
3 ('id1305348di', 0.5474522113800049) Video blog
4 ('id7190236di', 0.532200813293457) Dailymotion
5 ('id27401317di', 0.505139172077179) SoundCloud
6 ('id4923012di', 0.49702465534210205) Video hosting service
7 ('id32242976di', 0.49634692072868347) Google+
8 ('id10708073di', 0.49086934328079224) Web series
9 ('id1616492di', 0.47187042236328125) Internet meme
13 ('id37743450di', 0.44715774059295654) List of most viewed YouTube videos
14 ('id23446580di', 0.44610968232154846) M3Radio
15 ('id39418910di', 0.431934118270874) Multi-channel network
16 ('id18701436di', 0.4304552972316742) Tumblr
17 ('id31591547di', 0.42782479524612427) Instagram
18 ('id35198146di', 0.42740821838378906) Let's Play (video gaming)
19 ('id40084424di', 0.4221181273460388) Portal A Interactive
for ind,i in enumerate(model.most_similar('id'+titles['Harvard University']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])

0 ('id360816di', 0.8078027963638306) John F. Kennedy School of Government
1 ('id5685di', 0.6513446569442749) Cambridge, Massachusetts
2 ('id260879di', 0.6357944011688232) Harvard College
3 ('id242977di', 0.6104052066802979) Radcliffe College
4 ('id455732di', 0.5770295858383179) Harvard Society of Fellows
5 ('id302401di', 0.5655266642570496) The Harvard Crimson
6 ('id1074613di', 0.5570036172866821) Harvard Divinity School
7 ('id574990di', 0.5486043691635132) The Harvard Lampoon
8 ('id158984di', 0.5323041677474976) Harvard Law School
9 ('id18879di', 0.5299345254898071) Massachusetts Institute of Technology
10 ('id203844di', 0.5264247059822083) Phillips Exeter Academy
11 ('id414193di', 0.523749589920044) Radcliffe Institute for Advanced Study
12 ('id1274446di', 0.5175685882568359) Harvard Graduate School of Arts and Sciences
13 ('id174145di', 0.5156001448631287) Phillips Academy
14 ('id1975520di', 0.5131611824035645) Nieman Fellowship
15 ('id81223di', 0.511757493019104) Brandeis University
16 ('id321515di', 0.5045640468597412) Harvard Crimson
17 ('id1503200di', 0.5042493343353271) Hasty Pudding Theatricals
18 ('id1446327di', 0.5017040371894836) Hasty Pudding Club
19 ('id10306321di', 0.5011023879051208) John A. Paulson School of Engineering and Applied Sciences
for ind,i in enumerate(model.most_similar('id'+titles['Black hole']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])

0 ('id29320146di', 0.6648939251899719) Event horizon
1 ('id21869di', 0.6355272531509399) Neutron star
2 ('id54244di', 0.6233801245689392) Gravitational singularity
3 ('id34043di', 0.6158322691917419) Wormhole
4 ('id173724di', 0.6017568111419678) Hawking radiation
5 ('id12558di', 0.6006947755813599) Galaxy
6 ('id240972di', 0.5963913202285767) White hole
7 ('id5752038di', 0.5950264930725098) BBHS
8 ('id215706di', 0.5932334661483765) Supermassive black hole
9 ('id19604228di', 0.589704155921936) Dark energy
10 ('id25239di', 0.5887642502784729) Quasar
11 ('id8651di', 0.587800920009613) Dark matter
12 ('id8111079di', 0.5808608531951904) Gravitational wave
13 ('id86061di', 0.5808006525039673) Cygnus X-1
14 ('id11439di', 0.5799174308776855) Faster-than-light
15 ('id28758di', 0.5756555795669556) Spacetime
16 ('id619926di', 0.5755188465118408) Gravitational collapse
17 ('id191717di', 0.5751460790634155) X-ray binary
18 ('id405711di', 0.574928879737854) Superluminal motion
19 ('id206122di', 0.5747954249382019) Big Crunch
for ind,i in enumerate(model.most_similar('id'+titles['Gemstone']+'di',topn=20)):
 id = i[0].replace('id','').replace('di','')
 if id in ids:
  print(ind,i,ids[id])

0 ('id10045di', 0.624985933303833) Emerald
1 ('id8082di', 0.6215639114379883) Diamond
2 ('id43551di', 0.6126499176025391) Ruby
3 ('id31415di', 0.5835344791412354) Topaz
4 ('id29469di', 0.582021951675415) Sapphire
5 ('id4910di', 0.5680612325668335) Beryl
6 ('id304300di', 0.5504646897315979) Cabochon
7 ('id1571250di', 0.5484564304351807) Diamond color
8 ('id1366di', 0.5451520681381226) Amethyst
9 ('id1075641di', 0.5415037274360657) Regards ring
10 ('id37506di', 0.5388010740280151) Garnet
11 ('id8972771di', 0.5360362529754639) Dearest ring
12 ('id7158di', 0.5319223403930664) Carat (mass)
13 ('id180211di', 0.5272282361984253) Precious metal
14 ('id60652di', 0.5247260332107544) Lustre (mineralogy)
15 ('id24007di', 0.5207903981208801) Pearl
16 ('id12240di', 0.5207083225250244) Gold
17 ('id37346986di', 0.5199640989303589) Henry Heydenryk, Jr.
18 ('id3156395di', 0.5199630856513977) Cacholong
19 ('id7729004di', 0.517630934715271) Gems &amp; Gemology
"""


"""
|grep -oP '@[0-9]+ - supers:[0-9]+'|sed 's/supers://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n

sport-vs-politics
cat esa-no-densification-sport-politics-top-down.txt|grep -oP '@[0-9]+ - supers:[0-9]+'|sed 's/supers://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-sport-politics-top-down-aggr-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1.txt|grep -oP '@[0-9]+ - supers:[0-9]+'|sed 's/supers://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-sport-politics-top-down-aggr-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1.txt|grep -oP '@[0-9]+ - supers:[0-9]+'|sed 's/supers://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n

sport-vs-religion
cat esa-no-densification-sport-religion-top-down.txt|grep -oP '@[0-9]+ - supers:[0-9]+'|sed 's/supers://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-sport-religion-top-down-aggr-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1.txt|grep -oP '@[0-9]+ - supers:[0-9]+'|sed 's/supers://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-sport-religion-top-down-aggr-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1.txt|grep -oP '@[0-9]+ - supers:[0-9]+'|sed 's/supers://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n

autos-vs-motors
cat esa-no-densification.autos.motors.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-autos-motors-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-autos-motors-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n


hockey-vs-baseball
cat esa-no-densification.base.hockey.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-base-hockey-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-base-hockey-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n

guns-vs-mideast-misc
cat esa-no-densification.politics.guns.mideast.misc.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-guns-mideast-misc-bottom-up-newsubmap-v2-w2v-plain-anno-titles-10iter-dim500-wind9-skipgram1.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n
cat densification-guns-mideast-misc-bottom-up-w2v-4.4-concepts-from-plain-anno-titles-newsubmap-v2-10iter-dim500-wind9-cnt1-skipgram1.txt|grep -oP '@[0-9]+ - leaves:[0-9]+'|sed 's/leaves://g'|sed 's/ - /\t/g'|sed 's/@//g'|sort -n

java -jar dense-maxmatch.jar 5 1 0.85 0.0 sport_politics 2 autos.sports religion
java -jar dense-maxmatch.jar 5 1 0.85 0.0 sport_politics 2 autos.sports politics
"""

