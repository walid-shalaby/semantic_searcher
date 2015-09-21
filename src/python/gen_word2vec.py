# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:35:12 2015

@author: wshalaby
"""
#count = 0
words_dic = {}
output_path = None
model = None
topn = 10

def generate_synonyms(entity):
    import sys
    import os
    global count, words_dic, output_path, model, topn
    #count = count + 1
    
    entity = entity.lower()
    sep = entity.rfind(',')
    sample = entity[0:sep] \
    .replace("\"", "") \
    .replace("/", "") \
    .replace("*", "") \
    .replace("|", "") \
    .replace("?", "") \
    .replace("<", "") \
    .replace(">", "") \
    .replace("\\", "") \
    .replace(":", "") \
    .replace("-", " ") \
    .replace("~", "")
    label = entity[sep+1:-1].replace("\"", "")
    sample_synonyms = []
    #print("%d- %s,%s" % (count,sample,label))
    for word in sample.split(" "):
        synonyms = []
        if word in words_dic.keys():
            synonyms = words_dic.get(word)
        else:
            try:
                synonyms = model.most_similar(word, topn=topn)
            except:
                print("Exception: "+str(sys.exc_info()))
                pass
            
        words_dic[word] = synonyms
        
        for syn in synonyms:
            sample_synonyms.append(syn[0])
            
    if not os.path.exists(os.path.join(output_path,label)):
        os.makedirs(os.path.join(output_path,label))
    
    with open(os.path.join(os.path.join(output_path,label),sample),"w") as output_file:
        output = sample + os.linesep
        for syn in sample_synonyms:
            output = output + syn + " "
        
        output_file.write(output)
        output_file.close()
    
def generate_word2vec(input_path, model_path, out_path, n):
    from gensim.models import word2vec
    import sys
    import os
    import multiprocessing
    from multiprocessing import Pool
    
    global model, output_path, topn
    
    topn = n
    output_path = out_path
    
    model = word2vec.Word2Vec.load(model_path)
    model.init_sims(replace=True)
    print("word2vec model loaded")
    p = Pool(multiprocessing.cpu_count())
    p.map(generate_synonyms, open(input_path))
    p.close()
    p.join()
    '''
    for line in open(input_path):
        count = count + 1
        sep = line.rfind(',')
        sample = line[0:sep] \
        .replace("\"", "") \
        .replace("/", "") \
        .replace("*", "") \
        .replace("|", "") \
        .replace("?", "") \
        .replace("<", "") \
        .replace(">", "") \
        .replace("\\", "") \
        .replace(":", "") \
        .replace("-", " ") \
        .replace("~", "")
        label = line[sep+1:-1].replace("\"", "")
        sample_synonyms = []
        print("%d- %s,%s" % (count,sample,label))
        for word in sample.split(" "):
            synonyms = []
            if word in words_dic.keys():
                synonyms = words_dic.get(word)
            else:
                try:
                    synonyms = model.most_similar(sample)
                except:
                    print("Exception: "+str(sys.exc_info()))
                    pass
                
            words_dic[word] = synonyms
            
            for syn in synonyms:
                sample_synonyms.append(syn[0])
                
        if not os.path.exists(os.path.join(output_path,label)):
            os.makedirs(os.path.join(output_path,label))
        
        with open(os.path.join(os.path.join(output_path,label),sample),"w") as output_file:
            output = sample + os.linesep
            for syn in sample_synonyms:
                output = output + syn + " "
            
            output_file.write(output)
            output_file.close()
    '''
        
def get_word2vec(model_path):
    import gensim
    import time
    print(time.strftime('%m/%d/%y %H:%M:%S'))
    model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)
    print(time.strftime('%m/%d/%y %H:%M:%S'))    
    #print(model.similarity('java developer','software engineer'))
    print(model.similarity('developer','engineer'))
    print(model.similarity('developer','programmer'))
    #print(model.most_similar('software_engineer'))
    token = input('Enter word or sentence (0 to exit): ')
    print(model.most_similar(token))
    
def main():
    import sys
    
    if(len(sys.argv)!=6):
        print('Usage: python gen_word2vec.py input-path output-path path-to-word2vec-model synonyms-size [--py|--mllib]')
    else:
        if(sys.argv[5]=='--py'):
            generate_word2vec(sys.argv[1], sys.argv[3], sys.argv[2], sys.argv[4])
        elif(sys.argv[5]=='--mllib'):
            from commons import generate_w2v_syn
            generate_w2v_syn(sys.argv[1], sys.argv[3], sys.argv[2], sys.argv[4])
        
        print("Done generating word2vec!")
    
main()