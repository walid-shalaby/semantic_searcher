# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 01:33:40 2016

@author: wshalaby
"""

import numpy as np
from scipy import spatial
import os

mode = 'average_sub_super'
mode = 'average_conc_anno'

class SampleInfo:
     def __init__(self):
         self.max_topic_id = None
         self.max_topic_name = None
         self.sample_class = None
         self.sample_name = None
         self.max_all_sim = 0.0

if mode == 'average_conc_anno':
    for dim in [1,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500]:
        inpath1 = '/home/wshalaby/work/github/semantic_searcher/src/python/results-all.anno/results-all.txt.'+str(dim)
        inpath2 = '/home/wshalaby/work/github/semantic_searcher/src/python/results-all.conc/results-all.txt.'+str(dim)
        outpath = '/home/wshalaby/work/github/semantic_searcher/src/python/results-all.conc.anno/results-all.txt.'+str(dim)
        #top_topics = {'religion':1, 'comp':2, 'talk':3, 'rec':4, 'misc':5, 'sci':6}
        topic_mappings = {'rec.motorcycles':'autos.sports', 'comp.windows.x':'computer', 'talk.politics.guns':'politics', 'talk.politics.mideast':'politics', 'talk.religion.misc':'religion', 'rec.autos':'autos.sports', 'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports', 'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer', 'sci.space':'science', 'talk.politics.misc':'politics', 'sci.electronics':'science', 'comp.graphics':'computer', 'comp.os.ms.windows.misc':'computer', 'sci.med':'science', 'sci.crypt':'science', 'soc.religion.christian':'religion', 'alt.atheism':'religion', 'misc.forsale':'sale'}
        with open(outpath,'w') as outp:
            with open(inpath1,'r') as inp:
                sims = {}
                total_correct_sup = 0
                total_correct_super = 0
                for line in inp:
                    if line.find('SIMILA')==0:
                        tokens = line.strip('\n').split('\t')
                        sample_name = tokens[1]
                        topic_name = tokens[2]
                        sim = float(tokens[3])
                        if sample_name in sims:
                            dic = sims[sample_name]
                            dic[topic_name] = sim
                            sims[sample_name] = dic
                        else:
                            sims[sample_name] = {topic_name:sim}
                
                print('total samples from file1',len(sims.keys()))


                with open(inpath2,'r') as inp:
                    for line in inp:
                        if line.find('SIMILA')==0:
                            tokens = line.strip('\n').split('\t')
                            sample_name = tokens[1]
                            topic_name = tokens[2]
                            sim = float(tokens[3])
                            if sample_name in sims:
                                dic = sims[sample_name]
                                if topic_name in dic:
                                    dic[topic_name] = (dic[topic_name]+sim)/2.0 # average similarities
                                    #dic[topic_name] = max(dic[topic_name],sim) # take max similarities
                                    sims[sample_name] = dic
                                else:
                                    print('Oops! topic not found (',topic_name,') with sample (',sample_name,')')                                
                            else:
                                print('Oops! sample not found (',sample_name,')')
                    
                print('total samples',len(sims.keys()))
                
                total_correct_sub = 0
                total_correct_super = 0
                sub_total_correct = {}
                super_total_correct = {}
                for k,v in topic_mappings.items():
                    sub_total_correct[k] = 0
                    super_total_correct[v] = 0
                
                total = len(sims.keys())
                for sample,sim_dic in sims.items():
                    print(len(sim_dic.keys()))
                    sample_info = SampleInfo()
                    sample_info.sample_name = sample
                    sample_info.sample_class = sample[sample.find('_')+1:]
                    sample_info.max_all_sim = 0.0
                    for topic,sim in sim_dic.items():
                        outp.write('SIMILA\t'+sample_info.sample_name+'\t'+topic+'\t'+str(sim)+os.linesep)
                        if sim>sample_info.max_all_sim:
                            sample_info.max_all_sim = sim
                            sample_info.max_topic_name = topic                        
                    outp.write('TOPICA\t'+sample_info.sample_name+'\t'+sample_info.sample_class+'\t'+str('')+'\t'+sample_info.max_topic_name+'\t'+str('')+'\t'+str(sample_info.max_all_sim)+os.linesep)
                    if sample_info.sample_class==sample_info.max_topic_name:
                        total_correct_sub += 1
                        sub_total_correct[sample_info.max_topic_name] = sub_total_correct[sample_info.max_topic_name]+1
                    if topic_mappings[sample_info.sample_class]==topic_mappings[sample_info.max_topic_name]:
                        total_correct_super += 1
                        super_total_correct[topic_mappings[sample_info.max_topic_name]] = super_total_correct[topic_mappings[sample_info.max_topic_name]]+1                

                outp.write('@'+str(dim)+':'+str(total_correct_sub+total_correct_super)+os.linesep)
                for k,v in sub_total_correct.items():
                    outp.write('@'+str(dim)+' - leaf:'+k+' - '+str(v)+os.linesep)
                for k,v in super_total_correct.items():
                    outp.write('@'+str(dim)+' - super:'+k+' - '+str(v)+os.linesep)
                outp.write('@'+str(dim)+' - leaves:'+str(total_correct_sub)+' - '+str(1.0*total_correct_sub/total)+os.linesep)
                outp.write('@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/total)+os.linesep)
                outp.write('@'+str(dim)+':'+str(total_correct_sub+total_correct_super)+' - '+str(0.5*(total_correct_sub+total_correct_super)/total)+os.linesep)
                
elif mode == 'average_sub_super':
    for dim in [10]:
        inpath = '/home/wshalaby/work/github/semantic_searcher/src/python/average-results/results-all.txt.'+str(dim)
        topic_mappings = {'rec.motorcycles':'autos.sports', 'comp.windows.x':'computer', 'talk.politics.guns':'politics', 'talk.politics.mideast':'politics', 'talk.religion.misc':'religion', 'rec.autos':'autos.sports', 'rec.sport.baseball':'autos.sports', 'rec.sport.hockey':'autos.sports', 'comp.sys.mac.hardware':'computer', 'comp.sys.ibm.pc.hardware':'computer', 'sci.space':'science', 'talk.politics.misc':'politics', 'sci.electronics':'science', 'comp.graphics':'computer', 'comp.os.ms.windows.misc':'computer', 'sci.med':'science', 'sci.crypt':'science', 'soc.religion.christian':'religion', 'alt.atheism':'religion', 'misc.forsale':'sale'}
        with open(inpath,'r') as inp:
            sims = {}
            total_correct_sup = 0
            total_correct_super = 0
            for line in inp:
                if line.find('SIMILA')==0:
                    tokens = line.strip('\n').split('\t')
                    sample_name = tokens[1]
                    topic_name = tokens[2]
                    sim = float(tokens[3])
                    if sample_name in sims:
                        sims[sample_name].append((topic_name,sim))
                    else:
                        sims[sample_name] = [(topic_name,sim)]
            #print(sims['177011_talk.politics.misc'])
            print('total samples',len(sims.keys()))
            result = ''
            for sample,sim_pair in sims.items():
                #'TOPICA  177011_talk.politics.misc   talk.politics.misc      talk.politics.mideast       0.380011076762'
                #{'science': (0.828194102922, 4, 'sci.med', 0.327494936098), 'computer': (1.020058743689, 5, 'comp.windows.x', 0.243021569569), 'politics': (1.013431074027, 3, 'talk.politics.mideast', 0.380011076762), 'religion': (0.950879110849, 3, 'talk.religion.misc', 0.370208276284), 'sale': (0.17933435195, 1, 'misc.forsale', 0.17933435195), 'autos.sports': (0.530534400626, 4, 'rec.motorcycles', 0.147207743275)}
                #{'religion': (0.950879110849, 3, 'talk.religion.misc', 0.370208276284), 'autos.sports': (0.530534400626, 4, 'rec.motorcycles', 0.147207743275), 'computer': (1.020058743689, 5, 'comp.windows.x', 0.243021569569), 'politics': (1.013431074027, 3, 'talk.politics.mideast', 0.380011076762), 'sale': (0.17933435195, 1, 'misc.forsale', 0.17933435195), 'science': (0.828194102922, 4, 'sci.med', 0.327494936098)}
                #'TOPICA  177011_talk.politics.misc   talk.politics.misc      talk.religion.misc      0.370208276284'
                

                #if sample='177011_talk.politics.misc':
                #    continue
                super_sims = {}
                for topic,sim in sim_pair:
                    super_topic = topic_mappings[topic]
                    if super_topic in super_sims:
                        cur_sim,cur_count,cur_max_sub_topic,cur_max_sim = super_sims[super_topic]
                        if sim>cur_max_sim:
                            cur_max_sim = sim
                            cur_max_sub_topic = topic
                        super_sims[super_topic] = (cur_sim+sim,cur_count+1,cur_max_sub_topic,cur_max_sim)
                    else:
                        super_sims[super_topic] = (sim,1,topic,sim)
                #print(super_sims)
                max_mean_sim = -10.0
                max_sim = 0.0
                max_topic = ''
                for k,v in super_sims.items():
                    mean_topic_sim = v[0]/v[1]
                    if mean_topic_sim>max_mean_sim:
                        max_mean_sim = mean_topic_sim
                        max_topic = v[2]
                        max_sim = v[3]

                sample_class = sample[sample.find('_')+1:]
                if sample_class==max_topic:
                    total_correct_sup += 1
                if topic_mappings[sample_class]==topic_mappings[max_topic]:
                    total_correct_super += 1
                result += os.linesep+'TOPICA\t'+sample+'\t'+sample_class+'\t\t'+max_topic+'\t\t'+str(max_sim)+os.linesep
                #break

            result += os.linesep+'@'+str(dim)+' - leaves:'+str(total_correct_sup)+' - '+str(1.0*total_correct_sup/len(sims.keys()))+os.linesep
            result += os.linesep+'@'+str(dim)+' - supers:'+str(total_correct_super)+' - '+str(1.0*total_correct_super/len(sims.keys()))+os.linesep
            result += os.linesep+'@'+str(dim)+':'+str(total_correct_sup+total_correct_super)+' - '+str(0.5*(total_correct_sup+total_correct_super)/len(sims.keys()))+os.linesep
            with open('results-all.txt'+str(dim)+'.supermean','w') as resulta:
                resulta.write(result)
            
