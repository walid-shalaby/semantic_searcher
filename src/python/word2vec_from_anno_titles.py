# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import gensim
from gensim import utils
import os
from time import gmtime, strftime
import multiprocessing
import sys

'''
records = csv.DictReader(open('wiki_ids_titles_redirects.csv'))
for r in records:
 print(r['id'])
 print(r['title'])
 print(r['redirect'])

'''
titles = {}
redirects = {}
'''
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles_redirects.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/wiki_ids_titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"

keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"
keeponly_path = ""
'''
in_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles.txt'
in_path = '/home/wshalaby/work/github/semantic_searcher/src/python/sample-wiki.txt'
in_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-plain-anno-titles.txt'
in_path = '/home/wshalaby/work/github/semantic_searcher/enwiki20160820-plain-anno-titles-no-redirects.txt'
in_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles-no-redirects.txt'
in_path = '/home/walid/work/github/semantic_searcher/enwiki20160820-plain-anno-titles-no-redirects.txt'
in_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-plain-anno-titles-no-redirects.txt'
in_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles-no-redirects.txt.4.4mtitles'
in_path = '/home/walid/work/github/semantic_searcher/enwiki20160820-plain-anno-titles-no-redirects.txt.4.4mtitles'
in_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki-w2v-4.4-concepts-20160820-no-redirects-dic.txt'
in_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-concepts-from-plain-anno-titles.txt.4.4mtitles'
in_path = '/home/walid/work/github/semantic_searcher/enwiki20160820-concepts-from-plain-anno-titles.txt.4.4mtitles'
in_path = '/media/vol2/walid/work/github/semantic_searcher/enwiki20160820-plain-anno-titles-no-redirects-concepts-from-anno-titles.txt.4.4mtitles'
in_path = '/media/vol2/walid/work/github/semantic_searcher/enwiki20160820-words-from-plain-anno-titles.txt.4.4mtitles'
'''
outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts.500.30out.model'
outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter.500.30out.model'
outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts.model'
outpath = '/home/wshalaby/work/github/semantic_searcher/src/python/w2v-plain-anno-titles-10iter.model'
outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-10iter.model'
outpath = '/home/wshalaby/work/github/semantic_searcher/w2v-plain-anno-titles-10iter.model'
outpath = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-10iter.model'
'''
sample1 = "FIBA European Champions Cup 1962–63#REDIRECT  ##$@@$##1962–63 FIBA European Champions Cup##$@@$##   \n FIBA European Champions Cup 1962-63#REDIRECT  ##$@@$##1962–63 FIBA European Champions Cup##$@@$##   \n 1962-63 FIBA European Champions Cup#REDIRECT  ##$@@$##1962–63 FIBA European Champions Cup##$@@$##   \n The 1962–63 FIBA European Champions Cup was the sixth season of the  ##$@@$##FIBA European Champions Cup##$@@$##  . It was won by  ##$@@$##PBC CSKA Moscow##$@@$##  , after they beat  ##$@@$##Real Madrid Baloncesto##$@@$##  in a three legged  ##$@@$##Euroleague Finals##$@@$##  , after the two first games ended with an  ##$@@$##aggregate score##$@@$##   ##$@@$##two-legged tie##$@@$##  . CSKA won the third and decisive game, by a score of 99–80, and thus won its second European Champions Cup. \n <h2>First round</h2> \n <li>series decided over one game in Casablanca.</li> \n <h2>Round of 16</h2> \n <li>Automatically qualified to the quarter finals:</li> \n <li>  ##$@@$##BC Dinamo Tbilisi##$@@$##  (title holder)</li> \n <h2>Quarterfinals</h2> \n <li>A tie-break was played in Madrid on 2 April 1963:  ##$@@$##Real Madrid Baloncesto##$@@$##  -  ##$@@$##Budapesti Honvéd (basketball)##$@@$##  77–65.</li> \n <h2>Semifinals</h2> \n <h2>Finals</h2> \n Finals. \n First leg Frontón “Fiesta Alegre”;Attendance 5,000, 23 July 1963<br> \n Second leg  ##$@@$##Luzhniki Palace of Sports##$@@$##  ;Attendance 20,000, 31 July 1963 \n <li>Third leg  ##$@@$##Luzhniki Palace of Sports##$@@$##  ;Attendance 20,000, 1 August 1963,  ##$@@$##Moscow##$@@$##  ,  ##$@@$##Soviet Union##$@@$##  </li> \n <h2>References</h2> \n <h2>External links</h2> \n <li>European Cup 1962–63</li> \n"
sample2 = "Benjamin Bonzi (born 9 June 1996) is a  ##$@@$##france##$@@$##   ##$@@$##tennis##$@@$##  player. Bonzi along with  ##$@@$##Quentin Halys##$@@$##  won the  ##$@@$##2014 French Open – Boys' Doubles##$@@$##  after defeating  ##$@@$##Lucas Miedler##$@@$##  and Akira Santillan in the final, 6–3, 6–3.  \n <h2>Tour finals</h2> \n <h3>Doubles</h3> \n <h2>Junior Grand Slam finals</h2> \n <h3>Boys' Doubles</h3> \n <h2>External links</h2> \n <li>ATP Profile</li> \n"

'''
titles = {'France':15,'tennis':13,'Quentin Halys':14,'Lucas Miedler':17}
out = ''
for x in sample2.split('\n'):
 out += wikiTitle.sub(make_title_tag,x)
 
out
'''

from commons import load_titles
import re
wikiTitle = re.compile(r'\#\#\$@@\$\#\#(.*?)\#\#\$@@\$\#\#')
#PAT_ALPHANUMERIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
PAT_ALPHANUMERIC = re.compile('\w+', re.UNICODE)
def make_title_tag(match):
    global titles, redirects
    title = match.group(1)
    if len(title)>0:
        try:
            return ' '+str(titles[title])+' '
        except:
            try:
                return ' '+str(redirects[title])+' ' # see if it is redirect
            except:
                if len(title)>1:
                    title = title[0].upper()+title[1:]
                else:
                    title = title[0].upper()
                try:
                    return ' '+str(titles[title])+' '
                except:
                    try:
                        return ' '+str(redirects[title])+' '
                    except:
                        print(title)
                        return ' '+title+' '
    else:
        return ' '

def w2v_trim_rule(word, count, min_count):
    if word.find('id')==0 and word.find('di')==len(word)-2: # it is a concept
        return utils.RULE_KEEP
    else:
        return utils.RULE_DEFAULT

class MySentences(object):
    global titles
    def __init__(self, dirname, mode='no-redirects'):
        self.dirname = dirname
        self.mode = mode
 
    def __iter__(self):
        #for fname in os.listdir(self.dirname):
        #for line in open(os.path.join(self.dirname, fname)):
        if self.mode!='no-redirects':
            for doc in open(self.dirname):
                outcol = []
                lines = doc.split("\n")
                for line in lines:
                    line = wikiTitle.sub(make_title_tag, line).lower() # replace titles with id
                    for match in PAT_ALPHANUMERIC.finditer(line):
                        token = match.group()
                        if len(token)>1:
                            outcol.append(token)
                    
                yield outcol
        else:
            for doc in open(self.dirname):
                outcol = []
                lines = doc.split("\n")
                for line in lines:
                    #line = wikiTitle.sub(make_title_tag, line).lower() # replace titles with id
                    for token in line.split(' '):
                        if len(token)>1:
                            outcol.append(token)
                    
                yield outcol

print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
'''
to_keep = set()
if keeponly_path!="":
    print("loading keey only")
    with open(keeponly_path) as keeponly_titles:
        for title in keeponly_titles:
            to_keep.add(title.replace(os.linesep,""))
print("loaded "+str(len(to_keep))+" keeponly titles")

#titles,redirects = load_titles(titles_path, to_keep)
print("keeping "+str(len(titles))+" titles only")
print("loaded "+str(len(redirects))+" redirects")
'''
print("sentencing "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))    
sentences = MySentences(in_path) # a memory-friendly iterator
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#model = gensim.models.Word2Vec(sentences)
#'<page>][$#@@#$][    <title>VESA</title>][$#@@#$][    <ns>0</ns>][$#@@#$][    <id>65350</id>][$#@@#$][    <revision>][$#@@#$][      <id>622146990</id>][$#@@#$][      <parentid>611644107</parentid>][$#@@#$][      <timestamp>2014-08-21T03:48:49Z</timestamp>][$#@@#$][      <contributor>][$#@@#$][        <username>Huw Powell</username>][$#@@#$][        <id>275605</id>][$#@@#$][      </contributor>][$#@@#$][      <comment>/* top */</comment>][$#@@#$][      <model>wikitext</model>][$#@@#$][      <format>text/x-wiki</format>][$#@@#$][      <text xml:space="preserve">{{multiple issues|][$#@@#$][{{expert-subject|date=August 2011}}][$#@@#$][{{refimprove|date=August 2011}}][$#@@#$][}}][$#@@#$][][$#@@#$][\'\'\'VESA\'\'\' ({{IPAc-en|\'|v|i?|s|?}}), or the \'\'\'Video Electronics Standards Association\'\'\', is an international [[non-profit corporation]] [[standards body]] for [[computer graphics]] formed in 1988 by [[NEC Home Electronics]], maker of the [[Multisync monitor|MultiSync monitor line]], and eight [[video display adapter]] manufacturers: [[ATI Technologies]], [[Genoa Systems]], [[Orchid Technology]], Renaissance GRX, [[STB Systems]], [[Tecmar]], [[Video 7]] and [[Western Digital]]/[[Paradise Systems]].&lt;ref&gt;NEC Forms Video Standards Group, \'\'InfoWorld\'\', Nov 14, 1988&lt;/ref&gt;][$#@@#$][][$#@@#$][VESA\'s initial goal was to produce a standard for 800x600 [[SVGA]] resolution video displays. Since then VESA has issued a number of standards, mostly relating to the function of video [[peripheral]]s in personal computers.][$#@@#$][][$#@@#$][In November 2010, VESA announced a cooperative agreement with the [[Wireless Gigabit Alliance]] (WiGig)  for sharing technology expertise and specifications to develop multi-gigabit wireless [[DisplayPort]] capabilities. DisplayPort is a VESA technology that provides digital display connectivity.][$#@@#$][][$#@@#$][== Standards ==][$#@@#$][* [[VESA Feature Connector]] (VFC), obsolete connector that was often present on older videocards, used as an 8-bit video bus to other devices][$#@@#$][* [[Feature connector|VESA Advanced Feature Connector]] (VAFC), newer version of the above VFC that widens the 8-bit bus to either a 16-bit or 32-bit bus][$#@@#$][* [[VESA Local Bus]] (VLB), once used as a fast video bus (akin to the more recent [[Accelerated Graphics Port|AGP]])][$#@@#$][* [[VESA BIOS Extensions]] (VBE), used for enabling standard support for advanced video modes (at high resolutions and color depths)][$#@@#$][* [[Display Data Channel]] (DDC), a data link protocol which allows a host device to control an attached display and communicate EDID, DPMS, MCCS and similar messages][$#@@#$][* [[Extended display identification data|E-EDID]], a data format for display identification data that defines supported resolutions and video timings][$#@@#$][* [[Monitor Control Command Set]] (MCCS), a message protocol for controlling display parameters such as brightness, contrast, display orientation from the host device][$#@@#$][* [[DisplayID]], display identification data format, which is a replacement for E-EDID.][$#@@#$][* [[VESA Display Power Management Signaling]] (DPMS), which allows monitors to be queried on the types of power saving modes they support][$#@@#$][* [[Digital Packet Video Link]] (DPVL), a display link standard that allows to update only portions of the screen][$#@@#$][* [[VESA Stereo]], a standard 3-pin connector for synchronization of [[stereoscopy|stereoscopic]] images with [[LC shutter glasses]]][$#@@#$][* [[Flat Display Mounting Interface]] (FDMI), which defines &quot;VESA mounts&quot;][$#@@#$][* [[Generalized Timing Formula]] (GTF), video timings standard][$#@@#$][* [[Coordinated Video Timings]] (CVT), a replacement for GTF][$#@@#$][* [[VESA Video Interface Port]] (VIP), a digital video interface standard][$#@@#$][* [[DisplayPort]] Standard, a digital video interface standard][$#@@#$][* [[VESA Enhanced Video Connector]], an obsolete standard for reducing the number of cables around computers.][$#@@#$][][$#@@#$][==Criticisms==][$#@@#$][VESA has been criticized for their policy of charging non-members for some of their published standards. Some people{{Who|date=June 2011}} believe the practice of charging for specifications has undermined the purpose of the VESA organization. According to Kendall Bennett, developer of the VBE/AF standard, the VESA Software Standards Committee was closed down due to a lack of interest resulting from charging high prices for specifications.&lt;ref&gt;[http://lkml.org/lkml/2000/1/26/28 Re: vm86 in kernel]&lt;/ref&gt;  At that time no VESA standards were available for free. Although VESA now hosts some free standards documents, the free collection does not include newly developed standards. Even for obsolete standards, the free collection is incomplete. As of 2010, current standards documents from VESA cost hundreds, or thousands, of dollars each.  Some older standards are not available for free, or for purchase. As of 2010, the free downloads require mandatory registration.&lt;ref&gt;[https://fs16.formsite.com/VESA/form714826558/secure_index.html VESA PUBLIC STANDARDS DOWNLOAD REGISTRATION]&lt;/ref&gt;  While not all standards bodies provide specifications freely available for download, many do, including: [[ITU]], [[JEDEC]], [[Digital Display Working Group|DDWG]], and [[HDMI]] (through HDMI 1.3a).][$#@@#$][][$#@@#$][At the time [[DisplayPort]] was announced, VESA was criticized for developing the specification in secret and having a track record of developing unsuccessful digital interface standards, including [[VESA Plug and Display|Plug &amp; Display]] and [[Digital Flat Panel]].&lt;ref&gt;[http://digitimes.com/displays/a20051007PR200.html Commentary: Will VESA survive DisplayPort?]&lt;/ref&gt;][$#@@#$][][$#@@#$][==References==][$#@@#$][{{reflist}}][$#@@#$][][$#@@#$][==External links==][$#@@#$][* [http://www.vesa.org/ Group home page]][$#@@#$][][$#@@#$][{{DEFAULTSORT:Vesa}}][$#@@#$][[[Category:VESA| ]]][$#@@#$][[[Category:Computer display standards]]</text>][$#@@#$][      <sha1>l8dcsrx5tth2olb77vlpeoviv1isaas</sha1>][$#@@#$][    </revision>][$#@@#$][  </page>'
#sentences = [['VESA_-_Plug_-_and_-_Display','Digital_-_Flat_-_Panel'],['Plug_-_&amp;_-_Display','Digital_-_Flat_-_Panel']]
#sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences

sys.stdout.flush()
'''
size = 500
iter = 10
window = 5
sg = 1
mincnt = 5
outpath = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
print("training "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model = gensim.models.Word2Vec(sentences, trim_rule=w2v_trim_rule, min_count=mincnt, iter=iter, window=window, size=size, workers=multiprocessing.cpu_count()-1, sg=sg)
print("saving: ("+outpath+')'+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model.save(outpath)
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
sys.stdout.flush()
'''
for size in [500,200,300,100,400,50]:#[500,200]:#[500,400,300,250,200,100,50]:
    for sg in [1]:#[1,0]:
        for iter in [10]:#[1,0]:
            for window in [9,11]:#[5,7,3,2,10]:
                for mincnt in [1]:
                    #outpath = '/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-10iter-dim'+str(size)+'-wind'+str(window)+'-skipgram'+str(sg)+'.model'
                    outpath = '/home/wshalaby/work/github/semantic_searcher/w2v-plain-anno-titles-10iter-dim'+str(size)+'-wind'+str(window)+'-skipgram'+str(sg)+'.model'
                    outpath = '/scratch/wshalaby/doc2vec/models/word2vec/w2v-plain-anno-titles-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
                    outpath = '/home/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
                    outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-4.4-concepts-20160820-no-redirects-dic-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
                    outpath = '/scratch/wshalaby/doc2vec/models/word2vec/w2v-4.4-concepts-from-plain-anno-titles-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
                    outpath = '/home/walid/work/github/semantic_searcher/w2v-4.4-concepts-from-plain-anno-titles-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
                    outpath = '/media/vol2/walid/work/github/semantic_searcher/w2v-titles-no-redirects-and-concepts-from-anno-titles.txt.4.4mtitles-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
                    outpath = '/media/vol2/walid/work/github/semantic_searcher/enwiki20160820-words-from-plain-anno-titles.txt.4.4mtitles-'+str(iter)+'iter-dim'+str(size)+'-wind'+str(window)+'-cnt'+str(mincnt)+'-skipgram'+str(sg)+'.model'
                    print("training "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
                    sys.stdout.flush()
                    model = gensim.models.Word2Vec(sentences, trim_rule=w2v_trim_rule, min_count=mincnt, iter=iter, window=window, size=size, workers=multiprocessing.cpu_count()-1, sg=sg)
                    print("saving: ("+outpath+')'+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
                    model.save(outpath)
                    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
                    sys.stdout.flush()

print('Done!')
