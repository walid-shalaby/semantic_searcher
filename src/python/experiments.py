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

#minwikilen = [2000,1000,500,200,100,3000,4000,5000,8000,10000]
minwikilen = [5000]
#ngram = [100,3,2,1,4]
ngram = [3]
#supp = [2,1,3]
supp = [1]
#field = ['text','line1']
field = ['text']
#conc = [500,300,3000,2000,4000,1000,100]
conc = [500]
#dataset = [('TOEFL','TOEFL.txt'),('MEN-TEST','MEN-TEST.txt'),('semeval2014','semeval2014.txt'),('WS-353','WS-353.txt'),('WS-Sim','WS-Sim.txt'),('WS-Rel','WS-Rel.txt'),('RG-65','EN-RG-65.txt'),('MC-30','EN-MC-30.txt'),('MTurk-287','MTurk-287.txt'),('MTurk-771','MTurk-771.txt')]
dataset = [('MC-301','EN-MC-30.txt')]
#method = [('Explicit','_Explicit_'),('MSA_seealso','_seealso_'),('MSA_seealso_asso','_seealso_asso_')]
method = [('MSA_seealso','_seealso_')]
#url = 'http://localhost:8987/solr/collection1/browse?q=cat%2Ctiger&hmaxhits={4}&hmaxngrams={0}&hseealsomaxngrams={0}&hminwikilen={1}&hwikifield={2}&hshowids=0&hshowweight=0&hminassocnt={3}&hshowassocounts=n&hrelaxsearch=y&hrelatednessexpr=y&hexperin={5}&hexperout=results%2f{6}%2f{6}{8}{4}k_gram{0}_seealsogram{0}_len{1}_{2}_asso{3}_con{4}_q.txt&hrelaxcategories=y&hrelaxsametitle=n&hrelaxlistof=n&hrelaxdisambig=n&hrelaxner=y&conceptsmethod={7}&conceptsno={4}&measure_relatedness=on&hdistance=cosine&hwikiextraq=AND+NOT+title%3Alist*+AND+NOT+title%3Aindex*+AND+NOT+title%3A*disambiguation*'
url = 'http://localhost:8990/solr/collection1/browse?q=cat%2Ctiger&hmaxhits={4}&hmaxngrams={0}&hseealsomaxngrams={0}&hminwikilen={1}&hminseealsolen={1}&hwikifield={2}&hsim=cosine&hshowids=0&hshowpids=0&hshowweight=0&hminassocnt={3}&hshowassocounts=0&hshowtype=0&hshowdocno=0&hshowlen=0&hshowtable=0&hrelaxcache=on&hrelaxsearch=on&hrelatednessexpr=on&hexperin={5}&hexperout=results%2f{6}%2f{6}{8}{4}k_gram{0}_seealsogram{0}_len{1}_{2}_asso{3}_con{4}_q.txt&hrelaxcategories=on&hrelaxsametitle=&hrelaxlistof=&hrelaxdisambig=&hrelaxner=on&conceptsmethod={7}&conceptsno={4}&measure_relatedness=on&hwikiextraq=AND+NOT+title%3Alist*+AND+NOT+title%3Aindex*+AND+NOT+title%3A*disambiguation*'
for f in field:
	for l in minwikilen:
		for g in ngram:
			for c in conc:
				for s in supp:				
					for d in dataset:
						for m in method:
							print url.format(g,l,f,s,c,d[1],d[0],m[1],m[0])
							p = Popen(['curl', url.format(g,l,f,s,c,d[1],d[0],m[0],m[1])])
							p.wait()
							#Popen(['curl', url.format(g,l,f,s,c,d[1],d[0])])
							#os.spawnl(os.P_NOWAIT,'curl '+url.format(g,l,f,s,c,d[1],d[0]))
							#requests.get(url.format(g,l,f,s,c,d[1],d[0]))
							#call(['curl', url.format(g,l,f,s,c,d[1],d[0])])

						#time.sleep(10*60)

