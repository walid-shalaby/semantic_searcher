# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(walid-shalaby)s
"""

# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(walid-shalaby)s
"""

#https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1#.wqg7w8ih2
from os import listdir
from os.path import isfile, join
from time import gmtime, strftime
import gensim

inpath = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/corpus"
outpath = "/home/wshalaby/work/github/solr-4.10.2/solr/example/clef-2010/samples/doc2vec.model"

    
# Preparing the data for Gensim Doc2vec
# train on sentences
#class LabeledLineSentence(object):
#    def __init__(self, filename):
#        self.filename = filename
#    def __iter__(self):
#        for uid, line in enumerate(open(filename)):
#            yield LabeledSentence(words=line.split(), labels=[‘SENT_%s’ % uid])
            
# train on documents
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# Load the labels and data
docLabels = []
docLabels = [f for f in listdir(inpath)]
data = []
for doc in docLabels:
    fin = open(inpath+"/"+doc,'r')
    text = fin.read()
    data.append(text)
    fin.close()
    
# Training the model
LabeledSentence = gensim.models.doc2vec.LabeledSentence    
it = LabeledLineSentence(data, docLabels)#DocIt.DocIterator(data, docLabels)          

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model = gensim.models.Doc2Vec(size=300, window=10, min_count=50, workers=12,alpha=0.025, min_alpha=0.025) # use fixed learning rate
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model.build_vocab(it)
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
for epoch in range(10):
    print(iteration+str(epoch))
    model.train(it)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it)
    model.save(outpath)
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))    
model.save(outpath)
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print("done")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model = gensim.models.Doc2Vec.load(outpath)
print(model.docvecs.most_similar("US7470671B2"))
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
