# -*- coding: utf-8 -*-
"""
Created on Mon Sep 6 22:33:40 2016

@author: wshalaby
"""
#python3 resolve_missing.py > still_missing.txt

import os


resolved_missing_path = "skipgram_esa_mappings.txt"

out_path = "skipgram_esa_mappings_no_duplicates.py"

titles = {}
with open(out_path,'w') as fout:
    with open(resolved_missing_path,'r') as fin:
        for line in fin:
            line = line.strip(os.linesep)
            tokens = line.split('\t')
            if len(tokens)!=2:
                print(line+'...invalid format')
                continue
            else:
                org = tokens[0]
                resolve = tokens[1]
                if org in titles.keys() and titles[org]!=resolve and org!=resolve:
                    print(org+'\t'+titles[org]+'\t'+resolve)
                else:
                    titles[org] = resolve

    fout.write('skipgram_esa_mappings_dic = {')
    for k, v in titles.items():
        fout.write(k+': '+v+','+os.linesep)
    fout.write('}')
