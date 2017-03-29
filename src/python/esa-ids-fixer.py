# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(walid-shalaby)s
"""

#python3 esa-ids-fixer.py wiki_ids_titles.csv wikipedia-2016.500.30out.dat wiki_esa_ids.tsv

import sys
import csv
import os
full_wiki_ids = {}
ids = {}
full_wiki = csv.DictReader(open(sys.argv[1]))
for record in full_wiki:
    full_wiki_ids[record["title"]] = record["id"]

with open(sys.argv[2]) as esa_data:
    count1 = 0
    count2 = 0
    for line in esa_data:
        tokens = line.replace(os.linesep,"").split("\t")
        id = tokens[0]
        title = tokens[1]
        try:
            ids[id] = full_wiki_ids[title]
            count1 = count1 + 1
        except:
            count2 = count2 + 1
            continue
        print(count1+count2)
    print(count1)
    print(count2)
with open(sys.argv[3],"w") as output:
    for id1,id2 in ids.items():
        output.write(id1+"\t"+id2+os.linesep)
    output.close()