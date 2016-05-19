# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(walid-shalaby)s
"""

from subprocess import call
from subprocess import Popen
import time
import os
import requests

#distance = ['cosine','euclidean']
distance = ['cosine']
#minwikilen = [0,5000,1000,500,100,2000]
minwikilen = [5000]
#minseealsolen = [2000,1000,500,200,100,3000,4000,5000,8000,10000]
minseealsolen = [0,5000]
#ngram = [3,4,2,1,100]
ngram = [3]
#seealsongram = [3,4,2,1,100]
seealsongram = [3]
#supp = [20,15,10,5,1]
#conf = [2.0,1.0,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05,0.00]
conf = [0.0]
supp = [1]
#field = ['alltext','text']
field = ['alltext','alltext_nonorm']
#conc = [100,300,500,1000,2000,3000,4000]
#conc = [100,200,300,400,500,600,700,800,900,1000,1200,1500,2000,2500,3000,3500,4000,4500] # alternating concepts
conc = [500]
maxhits = [5000]
#dataset = [('MC-30','EN-MC-30.txt'),('RG-65','EN-RG-65.txt'),('WS-353','WS-353.txt'),('WS-Sim','WS-Sim.txt'),('WS-Rel','WS-Rel.txt'),('MEN-TEST','MEN-TEST.txt'),('MTurk-287','MTurk-287.txt'),('MTurk-771','MTurk-771.txt'),('TOEFL','TOEFL.txt'),('semeval2014','semeval2014.txt')]
dataset = [('MC-30','EN-MC-30.txt'),('RG-65','EN-RG-65.txt'),('WS-353','WS-353.txt'),('WS-Sim','WS-Sim.txt'),('WS-Rel','WS-Rel.txt'),('MEN-TEST','MEN-TEST.txt'),('MTurk-287','MTurk-287.txt'),('MTurk-771','MTurk-771.txt')]
#method = [('Explicit','_Explicit_'),('MSA_seealso','_seealso_'),('MSA_seealso_asso','_seealso_asso_')]
method = [('MSA_seealso','_seealso_'),('MSA_seealso_asso','_seealso_asso_'),('Explicit','_Explicit_')]
command = '--input /home/wshalaby/work/github/solr-4.10.2/solr/example/{0} --output /home/wshalaby/work/github/solr-4.10.2/solr/example/results/cikm/{1}/{6}_gram{2}_seealsogram{3}_len{8}_seelen{9}{12}sup{10}_conf{13}_con{7}_{11}_{5}.txt --max-title-ngrams {2} --max-seealso-ngrams {3} --method {4} --field {5} --max-hits {6} --concepts-num {7} --min-len {8} --min-seealso-len {9} --min-asso-cnt {10} --min-confidence {13} --relax-same-title off --abs-explicit off --title-search off --relax-cache off --relax-seealso off --relax-disambig off --relax-listof off --relax-ner off --relax-categories off --relatedness-expr on --distance {11} --wikiurl http://localhost:5678/solr/collection1/'
for h in maxhits:
	for sl in minseealsolen:
		for e in distance:
			for f in field:
			   for sg in seealsongram:
				for l in minwikilen:
					for g in ngram:
						for c in conc:
							for s in supp:				
								for d in dataset:
									for m in method:
                                                                                for o in conf:											
										        print command.format(d[1],d[0],g,sg,m[0],f,h,c,l,sl,s,e,m[1],o)
										        #print command.format(d[1],d[0],g,100,m[0],f,h,c,l,sl,s,e,m[1]) # take all see also lengths and grams but alternate support
        										#print command.format(d[1],d[0],100,100,m[0],f,h,c,0,0,1,e,m[1]) # take all and change # concepts only
