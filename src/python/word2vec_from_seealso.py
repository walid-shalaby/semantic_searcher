# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""

import gensim
import os
from time import gmtime, strftime
from commons import load_titles

titles = {}
redirects = {}
seealso = {}
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects_seealso.csv"

keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"
keeponly_path = ""
#concepts_path = "/home/wshalaby/work/github/semantic_searcher/src/python/concepts"
#concepts_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820.txt'
concepts_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/concepts.txt'

concepts_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki-concepts-20160820.txt'
concepts_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki-concepts-20160820-no-redirects.txt'
concepts_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-titles-seealso-pairs-no-redirects.txt'

class MySentences(object):
    global titles
    def __init__(self, dirname, mode='no-redirects'):
        self.dirname = dirname
        self.mode = mode
 
    def __iter__(self):
        if self.mode=='no-redirects':
            #for fname in os.listdir(self.dirname):
            #for line in open(os.path.join(self.dirname, fname)):
            for line in open(self.dirname):
                outcol = []
                for item in line.strip('\n').split(' '):
                    if len(item)>0:
                        item = item.replace('___',' ')
                        if item in titles.keys():
                            outcol.append(item)
                        elif len(item)>1:
                            print('|'+item+'|')
                yield outcol
        
print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))

to_keep = set()
if keeponly_path!="":
    print("loading keey only")
    with open(keeponly_path) as keeponly_titles:
        for title in keeponly_titles:
            to_keep.add(title.replace(os.linesep,""))
print("loaded "+str(len(to_keep))+" keeponly titles")

titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=False)
print("keeping "+str(len(titles))+" titles only")
print("loaded "+str(len(redirects))+" redirects")

print("sentencing "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))    
sentences = MySentences(concepts_path) # a memory-friendly iterator
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
model = gensim.models.Word2Vec(sentences)
print("training "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#'<page>][$#@@#$][    <title>VESA</title>][$#@@#$][    <ns>0</ns>][$#@@#$][    <id>65350</id>][$#@@#$][    <revision>][$#@@#$][      <id>622146990</id>][$#@@#$][      <parentid>611644107</parentid>][$#@@#$][      <timestamp>2014-08-21T03:48:49Z</timestamp>][$#@@#$][      <contributor>][$#@@#$][        <username>Huw Powell</username>][$#@@#$][        <id>275605</id>][$#@@#$][      </contributor>][$#@@#$][      <comment>/* top */</comment>][$#@@#$][      <model>wikitext</model>][$#@@#$][      <format>text/x-wiki</format>][$#@@#$][      <text xml:space="preserve">{{multiple issues|][$#@@#$][{{expert-subject|date=August 2011}}][$#@@#$][{{refimprove|date=August 2011}}][$#@@#$][}}][$#@@#$][][$#@@#$][\'\'\'VESA\'\'\' ({{IPAc-en|\'|v|i?|s|?}}), or the \'\'\'Video Electronics Standards Association\'\'\', is an international [[non-profit corporation]] [[standards body]] for [[computer graphics]] formed in 1988 by [[NEC Home Electronics]], maker of the [[Multisync monitor|MultiSync monitor line]], and eight [[video display adapter]] manufacturers: [[ATI Technologies]], [[Genoa Systems]], [[Orchid Technology]], Renaissance GRX, [[STB Systems]], [[Tecmar]], [[Video 7]] and [[Western Digital]]/[[Paradise Systems]].&lt;ref&gt;NEC Forms Video Standards Group, \'\'InfoWorld\'\', Nov 14, 1988&lt;/ref&gt;][$#@@#$][][$#@@#$][VESA\'s initial goal was to produce a standard for 800x600 [[SVGA]] resolution video displays. Since then VESA has issued a number of standards, mostly relating to the function of video [[peripheral]]s in personal computers.][$#@@#$][][$#@@#$][In November 2010, VESA announced a cooperative agreement with the [[Wireless Gigabit Alliance]] (WiGig)  for sharing technology expertise and specifications to develop multi-gigabit wireless [[DisplayPort]] capabilities. DisplayPort is a VESA technology that provides digital display connectivity.][$#@@#$][][$#@@#$][== Standards ==][$#@@#$][* [[VESA Feature Connector]] (VFC), obsolete connector that was often present on older videocards, used as an 8-bit video bus to other devices][$#@@#$][* [[Feature connector|VESA Advanced Feature Connector]] (VAFC), newer version of the above VFC that widens the 8-bit bus to either a 16-bit or 32-bit bus][$#@@#$][* [[VESA Local Bus]] (VLB), once used as a fast video bus (akin to the more recent [[Accelerated Graphics Port|AGP]])][$#@@#$][* [[VESA BIOS Extensions]] (VBE), used for enabling standard support for advanced video modes (at high resolutions and color depths)][$#@@#$][* [[Display Data Channel]] (DDC), a data link protocol which allows a host device to control an attached display and communicate EDID, DPMS, MCCS and similar messages][$#@@#$][* [[Extended display identification data|E-EDID]], a data format for display identification data that defines supported resolutions and video timings][$#@@#$][* [[Monitor Control Command Set]] (MCCS), a message protocol for controlling display parameters such as brightness, contrast, display orientation from the host device][$#@@#$][* [[DisplayID]], display identification data format, which is a replacement for E-EDID.][$#@@#$][* [[VESA Display Power Management Signaling]] (DPMS), which allows monitors to be queried on the types of power saving modes they support][$#@@#$][* [[Digital Packet Video Link]] (DPVL), a display link standard that allows to update only portions of the screen][$#@@#$][* [[VESA Stereo]], a standard 3-pin connector for synchronization of [[stereoscopy|stereoscopic]] images with [[LC shutter glasses]]][$#@@#$][* [[Flat Display Mounting Interface]] (FDMI), which defines &quot;VESA mounts&quot;][$#@@#$][* [[Generalized Timing Formula]] (GTF), video timings standard][$#@@#$][* [[Coordinated Video Timings]] (CVT), a replacement for GTF][$#@@#$][* [[VESA Video Interface Port]] (VIP), a digital video interface standard][$#@@#$][* [[DisplayPort]] Standard, a digital video interface standard][$#@@#$][* [[VESA Enhanced Video Connector]], an obsolete standard for reducing the number of cables around computers.][$#@@#$][][$#@@#$][==Criticisms==][$#@@#$][VESA has been criticized for their policy of charging non-members for some of their published standards. Some people{{Who|date=June 2011}} believe the practice of charging for specifications has undermined the purpose of the VESA organization. According to Kendall Bennett, developer of the VBE/AF standard, the VESA Software Standards Committee was closed down due to a lack of interest resulting from charging high prices for specifications.&lt;ref&gt;[http://lkml.org/lkml/2000/1/26/28 Re: vm86 in kernel]&lt;/ref&gt;  At that time no VESA standards were available for free. Although VESA now hosts some free standards documents, the free collection does not include newly developed standards. Even for obsolete standards, the free collection is incomplete. As of 2010, current standards documents from VESA cost hundreds, or thousands, of dollars each.  Some older standards are not available for free, or for purchase. As of 2010, the free downloads require mandatory registration.&lt;ref&gt;[https://fs16.formsite.com/VESA/form714826558/secure_index.html VESA PUBLIC STANDARDS DOWNLOAD REGISTRATION]&lt;/ref&gt;  While not all standards bodies provide specifications freely available for download, many do, including: [[ITU]], [[JEDEC]], [[Digital Display Working Group|DDWG]], and [[HDMI]] (through HDMI 1.3a).][$#@@#$][][$#@@#$][At the time [[DisplayPort]] was announced, VESA was criticized for developing the specification in secret and having a track record of developing unsuccessful digital interface standards, including [[VESA Plug and Display|Plug &amp; Display]] and [[Digital Flat Panel]].&lt;ref&gt;[http://digitimes.com/displays/a20051007PR200.html Commentary: Will VESA survive DisplayPort?]&lt;/ref&gt;][$#@@#$][][$#@@#$][==References==][$#@@#$][{{reflist}}][$#@@#$][][$#@@#$][==External links==][$#@@#$][* [http://www.vesa.org/ Group home page]][$#@@#$][][$#@@#$][{{DEFAULTSORT:Vesa}}][$#@@#$][[[Category:VESA| ]]][$#@@#$][[[Category:Computer display standards]]</text>][$#@@#$][      <sha1>l8dcsrx5tth2olb77vlpeoviv1isaas</sha1>][$#@@#$][    </revision>][$#@@#$][  </page>'
#sentences = [['VESA_-_Plug_-_and_-_Display','Digital_-_Flat_-_Panel'],['Plug_-_&amp;_-_Display','Digital_-_Flat_-_Panel']]
#sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences

for sg in [1,0]:
    for size in [500,400,300,250,200,100,50]:
        for window in [2]:#,3,5,7,10]:        
            #outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts.500.30out.model'
            #outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts-10iter.500.30out.model'
            #outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/concepts.model'
            outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/seealso-pairs-10iter-dim'+str(size)+'-wind'+str(window)+'-skipgram'+str(sg)+'.model'
            model = gensim.models.Word2Vec(sentences, min_count=1, iter=10, window=window, size=size, workers=2, sg=sg)
            print("saving: ("+outpath+')'+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            model.save(outpath)
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

print('Done!')
