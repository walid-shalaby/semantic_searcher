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
docprefix = ""#"topic_"

def format_plain_training_corpus_tsv(inpath,outpath):
    import os
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    with open(inpath) as infile:
        for line in infile:
            tokens = line.replace(os.linesep,"").split("\t")
            id = tokens[0]
            title = tokens[1]
            text = tokens[2]
        
            if not os.path.exists(outpath+"/"+docprefix+id):
                out = open(outpath+"/"+docprefix+id,"w")
                out.write(title+os.linesep+text)
                out.close()
        infile.close()
        
def format_plain_training_corpus_json(inpath,outpath,outformat='flat'):
    import os
    import ijson
    if outformat=='flat':
        output = open(outpath,"w")
    elif not os.path.exists(outpath):
        os.mkdir(outpath)
        
    with open(inpath) as infile:
        parser = ijson.parse(infile)
        doc = {}
        for prefix, event, value in parser:
            if (prefix,event) == ('response.docs.item','map_key'):
                if value=='id' and 'id' in doc:# and 'title' in doc and 'abstract' in doc and 'claims' in doc:
                    if outformat=='flat' or not os.path.exists(outpath+"/"+docprefix+doc['id']):
                        content = ''
                        if 'title' in doc:
                            content += doc['title']+os.linesep
                        if 'abstract' in doc:
                            content += doc['abstract']+os.linesep
                        if 'text' in doc:
                            content += doc['text']+os.linesep
                        #if 'claims' in doc:
                        #    content += doc['claims']+os.linesep
                        if 'claim1' in doc:
                            content += doc['claim1']+os.linesep
                        if len(content)>0 and (1==1 or ('title' not in doc or len(content)>len(doc['title']+os.linesep))):
                            if('title' not in doc):
                                print(doc['id']+'...title missing')
                        
                        if outformat=='flat':
                            output.write(doc['id']+"\t"+content.replace(os.linesep," ")+os.linesep)
                            doc = {}
                        else:
                            out = open(outpath+"/"+docprefix+doc['id'],"w")
                            out.write(content)
                            out.close()
                            doc = {}
            elif prefix.endswith('.id'):
                doc['id'] = value
            elif prefix.endswith('.title'):
                doc['title'] = value
            elif prefix.endswith('.abstract'):
                doc['abstract'] = value
            elif prefix.endswith('.text'):
                doc['text'] = value
            #elif prefix.endswith('.claims'):
            #    doc['claim'] = value
            elif prefix.endswith('.claim1'):
                doc['claim1'] = value
        if 'id' in doc:# and 'title' in doc and 'abstract' in doc and 'claims' in doc:
            if outformat=='flat' or not os.path.exists(outpath+"/"+docprefix+doc['id']):
                content = ''
                if 'title' in doc:
                    content += doc['title']+os.linesep
                if 'abstract' in doc:
                    content += doc['abstract']+os.linesep
                if 'text' in doc:
                    content += doc['text']+os.linesep
                #if 'claims' in doc:
                #    content += doc['claims']+os.linesep
                if 'claim1' in doc:
                    content += doc['claim1']+os.linesep
                if len(content)>0 and (1==1 or ('title' not in doc or len(content)>len(doc['title']+os.linesep))):
                    if('title' not in doc):
                        print(doc['id']+'...title missing')
                    if outformat=='flat':
                        output.write(doc['id']+"\t"+content.replace(os.linesep," ")+os.linesep)
                        doc = {}
                    else:
                        out = open(outpath+"/"+docprefix+doc['id'],"w")
                        out.write(content)
                        out.close()
                        doc = {}
        infile.close()
       #data = json.load(infile)
        #for doc in data["response"]["docs"]:
        #    out = open(outpath+"/"+doc["id"],"w")
        #    out.write(doc["title"]+os.linesep+doc["abstract"])
        #    out.close()
        
import sys
if sys.argv[1]=="tsv":
    format_plain_training_corpus_tsv(sys.argv[2],sys.argv[3])
else:
    format_plain_training_corpus_json(sys.argv[3],sys.argv[4],sys.argv[2])

