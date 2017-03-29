# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""
#nohup python3 resolve_redirects.py concepts-all
#nohup python3 resolve_redirects.py concepts-dic
#nohup python3 resolve_redirects.py concepts-all-raw
#nohup python3 resolve_redirects.py concepts-dic-raw
#nohup python3 resolve_redirects.py raw
#nohup python3 resolve_redirects.py raw-ids /scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles.txt.4.4mtitles /scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-concepts-from-plain-anno-titles.txt.4.4mtitles
#nohup python3 resolve_redirects.py raw-words /scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles.txt.4.4mtitles /scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-words-from-plain-anno-titles.txt.4.4mtitles
#nohup python3 resolve_redirects.py raw-concepts

import re
import os
import sys
from time import gmtime, strftime
from commons import load_titles
from commons import get_extra
import csv

titles = {}
redirects = {}
seealso = {}
titles_path = "/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles_redirects.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects.csv"
titles_path = "/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles_redirects_seealso.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects_seealso.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles_redirects_seealso.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/titles_redirects_ids.csv"

keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"
keeponly_path = ""
#concepts_path = "/home/wshalaby/work/github/semantic_searcher/src/python/concepts"


concepts_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki-concepts-20160820.txt'
outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki-concepts-20160820-no-redirects.txt'

concepts_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820.txt'
outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820-no-redirects-all.txt'
outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820-no-redirects-dic.txt'

raw_and_concepts_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820-no-redirects-dic.txt'
raw_and_concepts_outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820-no-redirects-dic-ids.txt'

raw_path = '/home/wshalaby/work/github/semantic_searcher/src/python/enwiki20160820-plain-anno-titles-sample.txt'
raw_outpath = '/home/wshalaby/work/github/semantic_searcher/src/python/enwiki20160820-plain-anno-titles-no-redirects-sample.txt'

raw_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-plain-anno-titles.txt'
raw_outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-plain-anno-titles-no-redirects.txt'

raw_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles.txt'
raw_outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles-no-redirects.txt'
raw_concepts_outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-raw-20160820-no-redirects-dic.txt'

seealso_outpath = '/home/wshalaby/work/github/semantic_searcher/src/python/enwiki20160820-titles-seealso-no-redirects.txt'
seealso_outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-titles-seealso-no-redirects.txt'
seealso_outpath = '/home/wshalaby/work/github/semantic_searcher/src/python/enwiki20160820-titles-seealso-pairs-no-redirects.txt'
seealso_outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-titles-seealso-pairs-no-redirects.txt'

wikiTitle = re.compile(r'\#\#\$@@\$\#\#(.*?)\#\#\$@@\$\#\#')
#PAT_ALPHANUMERIC = re.compile('(((?![\d])\w)+)', re.UNICODE)
PAT_ALPHANUMERIC = re.compile('\w+', re.UNICODE)
PAT_CONCEPT = re.compile('id\d+di', re.UNICODE)
def make_title_tag(match):
    global titles, redirects
    title = match.group(1)
    if len(title)>0:
        title = title.replace('&', '&amp;').replace('"', '&quot;')
        hashchar = title.find('#') # if internal anchor, resolve to main title
        if hashchar!=-1:
            title = title[0:hashchar]

        title = title.strip().replace('_-_-_',' ')
        try:
            return ' id'+str(titles[title])+'di '
        except:
            try:
                return ' id'+str(redirects[title])+'di ' # see if it is redirect
            except:
                if len(title)>1:
                    title = title[0].upper()+title[1:]
                else:
                    title = title[0].upper()
                try:
                    return ' id'+str(titles[title])+'di '
                except:
                    try:
                        return ' id'+str(redirects[title])+'di '
                    except:
                        print(title+'...not found')
                        return ' '+title+' '
    else:
        return ' '
   
def make_title_words(match):
    global titles, redirects
    title = match.group(1)
    if len(title)>0:
        title = title.replace('&', '&amp;').replace('"', '&quot;')
        hashchar = title.find('#') # if internal anchor, resolve to main title
        if hashchar!=-1:
            title = title[0:hashchar]

        title = title.strip().replace('_-_-_',' ')
        if title in titles:
            return ' '+title+' '
        else:
            if title in redirects:
                return ' '+str(redirects[title])+' ' # see if it is redirect
            else:
                if len(title)>1:
                    title = title[0].upper()+title[1:]
                else:
                    title = title[0].upper()
                if title in titles:
                    return ' '+title+' '
                else:
                    if title in redirects:
                        return ' '+str(redirects[title])+' ' # see if it is redirect
                    else:
                        print(title+'...not found')
                        return ' '+title+' '
    else:
        return ' '
   
class RedirectsResolver(object):
    global titles, redirects, seealso
    def __init__(self, dirname,mode='concepts',format='seealso-pairs'):
        self.dirname = dirname
        self.mode = mode
        self.format = format        
 
    def resolve_value(self,item):
        item = item.replace('&#039;',"'").replace('&', '&amp;').replace('"', '&quot;')
        item = item.strip()

        if self.mode=='concepts-all' or self.mode=='concepts-all-raw':
            newitem = item
        elif self.mode=='concepts-dic' or self.mode=='concepts-dic-raw':
            newitem = ''
        lookup_item = item
        if lookup_item in titles:
            newitem = item
        else:
            try:
                newitem = redirects[lookup_item]
            except:
                if len(item)>1:
                    item = item[0].upper()+item[1:]
                else:
                    item = item[0].upper()
                lookup_item = item
                if lookup_item in titles:
                    newitem = item
                else:
                    try:
                        newitem = redirects[lookup_item]
                    except:
                        print(item+'...not found')
        return newitem

    def __iter__(self):
        if self.mode in ('concepts-all','concepts-dic','concepts-all-raw','concepts-dic-raw'):

            #for fname in os.listdir(self.dirname):
            #for line in open(os.path.join(self.dirname, fname)):
            for line in open(self.dirname):
                outlines = ''
                if self.mode in ('concepts-all','concepts-dic'):
                    col = line.strip(' \n').split("][$#@@#$][") # each represent wiki article            
                elif self.mode in ('concepts-all-raw','concepts-dic-raw'):
                    col = line.strip(' \n').split('\\n')
                for items in col:
                    outline = ''
                    if self.mode in ('concepts-all','concepts-dic'):
                        items = items.split(' ')
                    elif self.mode in ('concepts-all-raw','concepts-dic-raw'):
                        items = [item.strip('##$@@$##') for item in items.strip(' \n').split() if item.find('##$@@$##')!=-1]
                    for item in items: # each is a concept
                        if len(item)>0:
                            newitem = self.resolve_value(item.replace('_-_-_',' '))
                            if len(newitem)>0:
                                outline += newitem.replace(' ','_-_-_') + ' '
                            elif item.find('#')>0: # might be an anchor
                                newitem = self.resolve_value(item[:item.find('#')].replace('_-_-_',' '))
                                if len(newitem)>0:
                                    outline += newitem.replace(' ','_-_-_') + ' '
                            
                            '''
                            lookup_item = item.replace('___',' ')
                            if lookup_item in titles.keys():
                                outline += item + ' '
                            else:
                                try:
                                    outline += redirects[lookup_item].replace(' ','___') + ' '
                                except:
                                    if len(item)>1:
                                        item = item[0].upper()+item[1:]
                                    else:
                                        item = item[0].upper()
                                    lookup_item = item.replace('___',' ')
                                    if lookup_item in titles.keys():
                                        outline += item + ' '
                                    else:
                                        try:
                                            outline += redirects[lookup_item].replace(' ','___') + ' '
                                        except:
                                            print(item)
                            '''
                    if len(outline)>0:
                        outlines += outline + '][$#@@#$]['
                
                yield outlines
        
        elif self.mode in ('raw','raw-ids','raw-words'):
            for doc in open(self.dirname):
                outcol = []
                outlines = ''
                lines = doc.split("\n")
                for line in lines:
                    outline = ''
                    if self.mode=='raw-words':
                        line = wikiTitle.sub(make_title_words, line).lower() # replace titles with id
                    elif self.mode in ('raw','raw-ids'):
                        line = wikiTitle.sub(make_title_tag, line).lower() # replace titles with id
                    if self.mode in ('raw','raw-words'):
                        matches = PAT_ALPHANUMERIC.finditer(line)
                    elif self.mode=='raw-ids':
                        matches = PAT_CONCEPT.finditer(line)
                    for match in matches:
                        token = match.group()
                        if len(token)>1:
                            outline += token + ' '
                    if len(outline)>0:
                        outlines += outline + '\n'
            
                yield outlines

        elif self.mode=='raw-concepts':
            for doc in open(self.dirname):
                outcol = []
                outlines = ''
                lines = doc.split("][$#@@#$][")
                for line in lines:
                    outline = ''
                    tokens = line.split() # replace titles with id, assumed to be titles and not redirects
                    for token in tokens:
                        if len(token)>1:
                            outline += 'id'+titles[token.replace('_-_-_',' ')] + 'di '
                    if len(outline)>0:
                        # @wshalaby for concepts, output all article concepts in single line
                        outlines += outline + ' '
                        #outlines += outline + '\n'
            
                yield outlines

        elif self.mode=='seealso':
            #for title,seealsos in seealso.items():
            records = csv.DictReader(open(self.dirname))
            for pair in records:
                outlines = ''
                title = pair['title']
                seealso = pair['seealso']
                if len(seealso)>0:
                    seealsos = get_extra(seealso)            
                    newtitle = title.replace(' ','___')
                    resolved_values = set()
                    for item in seealsos:
                        if len(item)>0:
                            newitem = self.resolve_value(item)
                            if len(newitem)>0:
                                resolved_values.add(newitem)
                    if self.format=='seealso-pairs':
                        for value1 in resolved_values:
                            newvalue1 = value1.replace(' ','___')
                            outlines +=  newtitle + ' ' + newvalue1 + os.linesep
                            outlines +=  newvalue1 + ' ' + newtitle + os.linesep
                            for value2 in resolved_values:
                                newvalue2 = value2.replace(' ','___')
                                if value1!=value2:                            
                                    outlines +=  newvalue1 + ' ' + newvalue2 + os.linesep
                                    outlines +=  newvalue2 + ' ' + newvalue1 + os.linesep
                    else:
                        for i in range(0, len(resolved_values)):
                            resolved_values_list = list(resolved_values)
                            value1 = resolved_values_list[i].replace(' ','___')
                            if i%6==0: # we will use a context window of size 5 when building the skipgram model
                                outlines +=  newtitle + ' '
                            outlines +=  value1 + ' '

                        if len(outlines)>0:
                            outlines += os.linesep

                yield outlines
            
def main():
    global titles, redirects, seealso
    print('python3 resolve_redirects.py concepts|raw')
    print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    to_keep = set()
    if keeponly_path!="":
        print("loading keey only")
        with open(keeponly_path) as keeponly_titles:
            for title in keeponly_titles:
                to_keep.add(title.replace(os.linesep,""))
    print("loaded "+str(len(to_keep))+" keeponly titles")

    mode = sys.argv[1] # concepts or raw or seealso
    if mode in ('concepts-all','concepts-dic','concepts-all-raw','concepts-dic-raw'):
        titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
    elif mode=='raw' or mode=='raw-concepts' or mode=='raw-ids':
        titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='id')
    elif mode=='raw-words':
        titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title')
    elif mode=='seealso':
        titles, redirects, seealso = load_titles(titles_path, to_keep, load_redirects=True, redirects_value='title', load_seealso=True)

    print("keeping "+str(len(titles))+" titles only")
    print("loaded "+str(len(redirects))+" redirects")
    print("loaded "+str(len(seealso))+" titles with seealso")

    #print("loading titles")
    #records = csv.DictReader(open(titles_path))
    #count = 0
    #for pair in records:
    #    count = count + 1
    #    if len(to_keep)==0 or pair['title'] in to_keep:
    #        titles.add(pair['title'])
        
    #print("loaded "+str(count)+" titles and keeping "+str(len(titles))+" only")

    if mode in ('concepts-all','concepts-dic','concepts-all-raw','concepts-dic-raw'):
        if len(sys.argv)==2:
            if mode.find('raw')==-1:
                final_inpath = concepts_path
                final_outputpath = outpath
            else:
                final_inpath = raw_path
                final_outputpath = raw_concepts_outpath
        else:
            final_inpath = sys.argv[2]
            final_outputpath = sys.argv[3]
            print("input: "+final_inpath)
            print("output: "+final_outputpath)
        format = ''
    elif mode in ('raw','raw-ids','raw-words'):
        if len(sys.argv)==2:
            final_inpath = raw_path
            final_outputpath = raw_outpath
        else:
            final_inpath = sys.argv[2]
            final_outputpath = sys.argv[3]
            print("input: "+final_inpath)
            print("output: "+final_outputpath)
        format = ''
    elif mode=='raw-concepts':
        if len(sys.argv)==2:
            final_inpath = raw_and_concepts_path
            final_outputpath = raw_and_concepts_outpath
        else:
            final_inpath = sys.argv[2]
            final_outputpath = sys.argv[3]
            print("input: "+final_inpath)
            print("output: "+final_outputpath)
        format = ''
    elif mode=='seealso':
        format = 'seealso-pairs'
        #format = 'seealso-not-pairs'
        #final_inpath = '/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles_redirects_seealso.csv'
        final_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/python/wiki_ids_titles_redirects_seealso-sample.csv'
        final_inpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects_seealso.csv'
        final_outputpath = seealso_outpath

    pure_articles = RedirectsResolver(final_inpath, mode, format) # a memory-friendly iterator
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    with open(final_outputpath,'w') as output: 
        for article in pure_articles:
            if len(article)>0:
                output.write(article+os.linesep)

    #'<page>][$#@@#$][    <title>VESA</title>][$#@@#$][    <ns>0</ns>][$#@@#$][    <id>65350</id>][$#@@#$][    <revision>][$#@@#$][      <id>622146990</id>][$#@@#$][      <parentid>611644107</parentid>][$#@@#$][      <timestamp>2014-08-21T03:48:49Z</timestamp>][$#@@#$][      <contributor>][$#@@#$][        <username>Huw Powell</username>][$#@@#$][        <id>275605</id>][$#@@#$][      </contributor>][$#@@#$][      <comment>/* top */</comment>][$#@@#$][      <model>wikitext</model>][$#@@#$][      <format>text/x-wiki</format>][$#@@#$][      <text xml:space="preserve">{{multiple issues|][$#@@#$][{{expert-subject|date=August 2011}}][$#@@#$][{{refimprove|date=August 2011}}][$#@@#$][}}][$#@@#$][][$#@@#$][\'\'\'VESA\'\'\' ({{IPAc-en|\'|v|i?|s|?}}), or the \'\'\'Video Electronics Standards Association\'\'\', is an international [[non-profit corporation]] [[standards body]] for [[computer graphics]] formed in 1988 by [[NEC Home Electronics]], maker of the [[Multisync monitor|MultiSync monitor line]], and eight [[video display adapter]] manufacturers: [[ATI Technologies]], [[Genoa Systems]], [[Orchid Technology]], Renaissance GRX, [[STB Systems]], [[Tecmar]], [[Video 7]] and [[Western Digital]]/[[Paradise Systems]].&lt;ref&gt;NEC Forms Video Standards Group, \'\'InfoWorld\'\', Nov 14, 1988&lt;/ref&gt;][$#@@#$][][$#@@#$][VESA\'s initial goal was to produce a standard for 800x600 [[SVGA]] resolution video displays. Since then VESA has issued a number of standards, mostly relating to the function of video [[peripheral]]s in personal computers.][$#@@#$][][$#@@#$][In November 2010, VESA announced a cooperative agreement with the [[Wireless Gigabit Alliance]] (WiGig)  for sharing technology expertise and specifications to develop multi-gigabit wireless [[DisplayPort]] capabilities. DisplayPort is a VESA technology that provides digital display connectivity.][$#@@#$][][$#@@#$][== Standards ==][$#@@#$][* [[VESA Feature Connector]] (VFC), obsolete connector that was often present on older videocards, used as an 8-bit video bus to other devices][$#@@#$][* [[Feature connector|VESA Advanced Feature Connector]] (VAFC), newer version of the above VFC that widens the 8-bit bus to either a 16-bit or 32-bit bus][$#@@#$][* [[VESA Local Bus]] (VLB), once used as a fast video bus (akin to the more recent [[Accelerated Graphics Port|AGP]])][$#@@#$][* [[VESA BIOS Extensions]] (VBE), used for enabling standard support for advanced video modes (at high resolutions and color depths)][$#@@#$][* [[Display Data Channel]] (DDC), a data link protocol which allows a host device to control an attached display and communicate EDID, DPMS, MCCS and similar messages][$#@@#$][* [[Extended display identification data|E-EDID]], a data format for display identification data that defines supported resolutions and video timings][$#@@#$][* [[Monitor Control Command Set]] (MCCS), a message protocol for controlling display parameters such as brightness, contrast, display orientation from the host device][$#@@#$][* [[DisplayID]], display identification data format, which is a replacement for E-EDID.][$#@@#$][* [[VESA Display Power Management Signaling]] (DPMS), which allows monitors to be queried on the types of power saving modes they support][$#@@#$][* [[Digital Packet Video Link]] (DPVL), a display link standard that allows to update only portions of the screen][$#@@#$][* [[VESA Stereo]], a standard 3-pin connector for synchronization of [[stereoscopy|stereoscopic]] images with [[LC shutter glasses]]][$#@@#$][* [[Flat Display Mounting Interface]] (FDMI), which defines &quot;VESA mounts&quot;][$#@@#$][* [[Generalized Timing Formula]] (GTF), video timings standard][$#@@#$][* [[Coordinated Video Timings]] (CVT), a replacement for GTF][$#@@#$][* [[VESA Video Interface Port]] (VIP), a digital video interface standard][$#@@#$][* [[DisplayPort]] Standard, a digital video interface standard][$#@@#$][* [[VESA Enhanced Video Connector]], an obsolete standard for reducing the number of cables around computers.][$#@@#$][][$#@@#$][==Criticisms==][$#@@#$][VESA has been criticized for their policy of charging non-members for some of their published standards. Some people{{Who|date=June 2011}} believe the practice of charging for specifications has undermined the purpose of the VESA organization. According to Kendall Bennett, developer of the VBE/AF standard, the VESA Software Standards Committee was closed down due to a lack of interest resulting from charging high prices for specifications.&lt;ref&gt;[http://lkml.org/lkml/2000/1/26/28 Re: vm86 in kernel]&lt;/ref&gt;  At that time no VESA standards were available for free. Although VESA now hosts some free standards documents, the free collection does not include newly developed standards. Even for obsolete standards, the free collection is incomplete. As of 2010, current standards documents from VESA cost hundreds, or thousands, of dollars each.  Some older standards are not available for free, or for purchase. As of 2010, the free downloads require mandatory registration.&lt;ref&gt;[https://fs16.formsite.com/VESA/form714826558/secure_index.html VESA PUBLIC STANDARDS DOWNLOAD REGISTRATION]&lt;/ref&gt;  While not all standards bodies provide specifications freely available for download, many do, including: [[ITU]], [[JEDEC]], [[Digital Display Working Group|DDWG]], and [[HDMI]] (through HDMI 1.3a).][$#@@#$][][$#@@#$][At the time [[DisplayPort]] was announced, VESA was criticized for developing the specification in secret and having a track record of developing unsuccessful digital interface standards, including [[VESA Plug and Display|Plug &amp; Display]] and [[Digital Flat Panel]].&lt;ref&gt;[http://digitimes.com/displays/a20051007PR200.html Commentary: Will VESA survive DisplayPort?]&lt;/ref&gt;][$#@@#$][][$#@@#$][==References==][$#@@#$][{{reflist}}][$#@@#$][][$#@@#$][==External links==][$#@@#$][* [http://www.vesa.org/ Group home page]][$#@@#$][][$#@@#$][{{DEFAULTSORT:Vesa}}][$#@@#$][[[Category:VESA| ]]][$#@@#$][[[Category:Computer display standards]]</text>][$#@@#$][      <sha1>l8dcsrx5tth2olb77vlpeoviv1isaas</sha1>][$#@@#$][    </revision>][$#@@#$][  </page>'
    #sentences = [['VESA_-_Plug_-_and_-_Display','Digital_-_Flat_-_Panel'],['Plug_-_&amp;_-_Display','Digital_-_Flat_-_Panel']]
    #sentences = [['first', 'sentence'], ['second', 'sentence']]
    # train word2vec on the two sentences
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print('Done!')

main()









'''
before seealso

import re
import os
import sys
from time import gmtime, strftime
from commons import load_titles

titles = {}
redirects = {}
titles_path = "/home/wshalaby/work/github/semantic_searcher/src/python/wiki_ids_titles_redirects.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles.csv"
titles_path = "/scratch/wshalaby/doc2vec/wiki_ids_titles_redirects.csv"
titles_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wiki_ids_titles_redirects.csv"

keeponly_path = "/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/patents/innovation-analytics/data/esa/DatalessClassification/data/wikipedia-2016.500.30out.lst"
keeponly_path = ""
#concepts_path = "/home/wshalaby/work/github/semantic_searcher/src/python/concepts"

concepts_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki-concepts-20160820.txt'
outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki-concepts-20160820-no-redirects.txt'

concepts_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820.txt'
outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki-concepts-20160820-no-redirects.txt'

raw_path = '/home/wshalaby/work/github/semantic_searcher/src/python/enwiki20160820-plain-anno-titles-sample.txt'
raw_outpath = '/home/wshalaby/work/github/semantic_searcher/src/python/enwiki20160820-plain-anno-titles-no-redirects-sample.txt'

raw_path = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles.txt'
raw_outpath = '/scratch/wshalaby/EntityTypeRecognition/etr/data/data-prep/enwiki20160820-plain-anno-titles-no-redirects.txt'

raw_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-plain-anno-titles.txt'
raw_outpath = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/enwiki20160820-plain-anno-titles-no-redirects.txt'

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
   
class RedirectsResolver(object):
    global titles
    def __init__(self, dirname,mode='concepts'):
        self.dirname = dirname
        self.mode = mode
 
    def __iter__(self):
        if self.mode=='concepts':

            #for fname in os.listdir(self.dirname):
            #for line in open(os.path.join(self.dirname, fname)):
            for line in open(self.dirname):
                outlines = ''
                col = line.strip(' \n').split("][$#@@#$][") # each represent wiki article            
                for items in col:
                    outline = ''
                    for item in items.split(' '): # each is a concept
                        if len(item)>0:
                            lookup_item = item.replace('___',' ')
                            if lookup_item in titles.keys():
                                outline += item + ' '
                            else:
                                try:
                                    outline += redirects[lookup_item].replace(' ','___') + ' '
                                except:
                                    if len(item)>1:
                                        item = item[0].upper()+item[1:]
                                    else:
                                        item = item[0].upper()
                                    lookup_item = item.replace('___',' ')
                                    if lookup_item in titles.keys():
                                        outline += item + ' '
                                    else:
                                        try:
                                            outline += redirects[lookup_item].replace(' ','___') + ' '
                                        except:
                                            print(item)
                    
                    if len(outline)>0:
                        outlines += outline + '][$#@@#$]['
                
                yield outlines
        elif self.mode=='raw':
            for doc in open(self.dirname):
                outcol = []
                outlines = ''
                lines = doc.split("\n")
                for line in lines:
                    outline = ''
                    line = wikiTitle.sub(make_title_tag, line).lower() # replace titles with id
                    for match in PAT_ALPHANUMERIC.finditer(line):
                        token = match.group()
                        if len(token)>1:
                            outline += token + ' '
                    if len(outline)>0:
                        outlines += outline + '\n'
                
                yield outlines

print('python3 resolve_redirects.py concepts|raw')
print("starting "+strftime("%Y-%m-%d %H:%M:%S", gmtime()))

to_keep = set()
if keeponly_path!="":
    print("loading keey only")
    with open(keeponly_path) as keeponly_titles:
        for title in keeponly_titles:
            to_keep.add(title.replace(os.linesep,""))
print("loaded "+str(len(to_keep))+" keeponly titles")

mode = sys.argv[1] # concepts or raw
if mode=='concepts':
    titles, redirects = load_titles(titles_path, to_keep, True,'title')
elif mode=='raw':
    titles,redirects = load_titles(titles_path, to_keep)

print("keeping "+str(len(titles))+" titles only")
print("loaded "+str(len(redirects))+" redirects")

#print("loading titles")
#records = csv.DictReader(open(titles_path))
#count = 0
#for pair in records:
#    count = count + 1
#    if len(to_keep)==0 or pair['title'] in to_keep:
#        titles.add(pair['title'])
    
#print("loaded "+str(count)+" titles and keeping "+str(len(titles))+" only")

if mode=='concepts':
    fial_inpath = concepts_path
    final_outputpath = outpath
elif mode=='raw':
    fial_inpath = raw_path
    final_outputpath = raw_outpath

pure_articles = RedirectsResolver(fial_inpath, mode) # a memory-friendly iterator
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
with open(final_outputpath,'w') as output: 
    for article in pure_articles:
        output.write(article+os.linesep)

#'<page>][$#@@#$][    <title>VESA</title>][$#@@#$][    <ns>0</ns>][$#@@#$][    <id>65350</id>][$#@@#$][    <revision>][$#@@#$][      <id>622146990</id>][$#@@#$][      <parentid>611644107</parentid>][$#@@#$][      <timestamp>2014-08-21T03:48:49Z</timestamp>][$#@@#$][      <contributor>][$#@@#$][        <username>Huw Powell</username>][$#@@#$][        <id>275605</id>][$#@@#$][      </contributor>][$#@@#$][      <comment>/* top */</comment>][$#@@#$][      <model>wikitext</model>][$#@@#$][      <format>text/x-wiki</format>][$#@@#$][      <text xml:space="preserve">{{multiple issues|][$#@@#$][{{expert-subject|date=August 2011}}][$#@@#$][{{refimprove|date=August 2011}}][$#@@#$][}}][$#@@#$][][$#@@#$][\'\'\'VESA\'\'\' ({{IPAc-en|\'|v|i?|s|?}}), or the \'\'\'Video Electronics Standards Association\'\'\', is an international [[non-profit corporation]] [[standards body]] for [[computer graphics]] formed in 1988 by [[NEC Home Electronics]], maker of the [[Multisync monitor|MultiSync monitor line]], and eight [[video display adapter]] manufacturers: [[ATI Technologies]], [[Genoa Systems]], [[Orchid Technology]], Renaissance GRX, [[STB Systems]], [[Tecmar]], [[Video 7]] and [[Western Digital]]/[[Paradise Systems]].&lt;ref&gt;NEC Forms Video Standards Group, \'\'InfoWorld\'\', Nov 14, 1988&lt;/ref&gt;][$#@@#$][][$#@@#$][VESA\'s initial goal was to produce a standard for 800x600 [[SVGA]] resolution video displays. Since then VESA has issued a number of standards, mostly relating to the function of video [[peripheral]]s in personal computers.][$#@@#$][][$#@@#$][In November 2010, VESA announced a cooperative agreement with the [[Wireless Gigabit Alliance]] (WiGig)  for sharing technology expertise and specifications to develop multi-gigabit wireless [[DisplayPort]] capabilities. DisplayPort is a VESA technology that provides digital display connectivity.][$#@@#$][][$#@@#$][== Standards ==][$#@@#$][* [[VESA Feature Connector]] (VFC), obsolete connector that was often present on older videocards, used as an 8-bit video bus to other devices][$#@@#$][* [[Feature connector|VESA Advanced Feature Connector]] (VAFC), newer version of the above VFC that widens the 8-bit bus to either a 16-bit or 32-bit bus][$#@@#$][* [[VESA Local Bus]] (VLB), once used as a fast video bus (akin to the more recent [[Accelerated Graphics Port|AGP]])][$#@@#$][* [[VESA BIOS Extensions]] (VBE), used for enabling standard support for advanced video modes (at high resolutions and color depths)][$#@@#$][* [[Display Data Channel]] (DDC), a data link protocol which allows a host device to control an attached display and communicate EDID, DPMS, MCCS and similar messages][$#@@#$][* [[Extended display identification data|E-EDID]], a data format for display identification data that defines supported resolutions and video timings][$#@@#$][* [[Monitor Control Command Set]] (MCCS), a message protocol for controlling display parameters such as brightness, contrast, display orientation from the host device][$#@@#$][* [[DisplayID]], display identification data format, which is a replacement for E-EDID.][$#@@#$][* [[VESA Display Power Management Signaling]] (DPMS), which allows monitors to be queried on the types of power saving modes they support][$#@@#$][* [[Digital Packet Video Link]] (DPVL), a display link standard that allows to update only portions of the screen][$#@@#$][* [[VESA Stereo]], a standard 3-pin connector for synchronization of [[stereoscopy|stereoscopic]] images with [[LC shutter glasses]]][$#@@#$][* [[Flat Display Mounting Interface]] (FDMI), which defines &quot;VESA mounts&quot;][$#@@#$][* [[Generalized Timing Formula]] (GTF), video timings standard][$#@@#$][* [[Coordinated Video Timings]] (CVT), a replacement for GTF][$#@@#$][* [[VESA Video Interface Port]] (VIP), a digital video interface standard][$#@@#$][* [[DisplayPort]] Standard, a digital video interface standard][$#@@#$][* [[VESA Enhanced Video Connector]], an obsolete standard for reducing the number of cables around computers.][$#@@#$][][$#@@#$][==Criticisms==][$#@@#$][VESA has been criticized for their policy of charging non-members for some of their published standards. Some people{{Who|date=June 2011}} believe the practice of charging for specifications has undermined the purpose of the VESA organization. According to Kendall Bennett, developer of the VBE/AF standard, the VESA Software Standards Committee was closed down due to a lack of interest resulting from charging high prices for specifications.&lt;ref&gt;[http://lkml.org/lkml/2000/1/26/28 Re: vm86 in kernel]&lt;/ref&gt;  At that time no VESA standards were available for free. Although VESA now hosts some free standards documents, the free collection does not include newly developed standards. Even for obsolete standards, the free collection is incomplete. As of 2010, current standards documents from VESA cost hundreds, or thousands, of dollars each.  Some older standards are not available for free, or for purchase. As of 2010, the free downloads require mandatory registration.&lt;ref&gt;[https://fs16.formsite.com/VESA/form714826558/secure_index.html VESA PUBLIC STANDARDS DOWNLOAD REGISTRATION]&lt;/ref&gt;  While not all standards bodies provide specifications freely available for download, many do, including: [[ITU]], [[JEDEC]], [[Digital Display Working Group|DDWG]], and [[HDMI]] (through HDMI 1.3a).][$#@@#$][][$#@@#$][At the time [[DisplayPort]] was announced, VESA was criticized for developing the specification in secret and having a track record of developing unsuccessful digital interface standards, including [[VESA Plug and Display|Plug &amp; Display]] and [[Digital Flat Panel]].&lt;ref&gt;[http://digitimes.com/displays/a20051007PR200.html Commentary: Will VESA survive DisplayPort?]&lt;/ref&gt;][$#@@#$][][$#@@#$][==References==][$#@@#$][{{reflist}}][$#@@#$][][$#@@#$][==External links==][$#@@#$][* [http://www.vesa.org/ Group home page]][$#@@#$][][$#@@#$][{{DEFAULTSORT:Vesa}}][$#@@#$][[[Category:VESA| ]]][$#@@#$][[[Category:Computer display standards]]</text>][$#@@#$][      <sha1>l8dcsrx5tth2olb77vlpeoviv1isaas</sha1>][$#@@#$][    </revision>][$#@@#$][  </page>'
#sentences = [['VESA_-_Plug_-_and_-_Display','Digital_-_Flat_-_Panel'],['Plug_-_&amp;_-_Display','Digital_-_Flat_-_Panel']]
#sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print('Done!')

'''