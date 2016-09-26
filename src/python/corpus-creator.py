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

def format_plain_training_corpus(inpath,outpath):
    import os
    import ijson
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    
    with open(inpath) as infile:
        parser = ijson.parse(infile)
        doc = {}
        for prefix, event, value in parser:
            if (prefix,event) == ('response.docs.item','map_key'):
                if value=='id' and 'id' in doc:# and 'title' in doc and 'abstract' in doc and 'claims' in doc:
                    if not os.path.exists(outpath+"/"+docprefix+doc['id']):
                        content = ''
                        if 'title' in doc:
                            content += doc['title']+os.linesep
                        if 'abstract' in doc:
                            content += doc['abstract']+os.linesep
                        if 'text' in doc:
                            content += doc['text']+os.linesep
                        if 'claims' in doc:
                            content += doc['claims']+os.linesep
                        if len(content)>0 and ('title' not in doc or len(content)>len(doc['title']+os.linesep)):
                            if('title' not in doc):
                                print(doc['id']+'...title missing')
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
            elif prefix.endswith('.claims'):
                doc['claims'] = value
       
        if 'id' in doc:# and 'title' in doc and 'abstract' in doc and 'claims' in doc:
            if not os.path.exists(outpath+"/"+docprefix+doc['id']):
                content = ''
                if 'title' in doc:
                    content += doc['title']+os.linesep
                if 'abstract' in doc:
                    content += doc['abstract']+os.linesep
                if 'text' in doc:
                    content += doc['text']+os.linesep
                if 'claims' in doc:
                    content += doc['claims']+os.linesep
                if len(content)>0 and ('title' not in doc or len(content)>len(doc['title']+os.linesep)):
                    if('title' not in doc):
                        print(doc['id']+'...title missing')
                    out = open(outpath+"/"+docprefix+doc['id'],"w")
                    out.write(content)
                    out.close()
        infile.close()
       #data = json.load(infile)
        #for doc in data["response"]["docs"]:
        #    out = open(outpath+"/"+doc["id"],"w")
        #    out.write(doc["title"]+os.linesep+doc["abstract"])
        #    out.close()
        
import sys
format_plain_training_corpus(sys.argv[1],sys.argv[2])

