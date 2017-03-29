import gensim
from scipy import spatial
import os

titles_path = '/scratch/wshalaby/doc2vec/titles_redirects_ids.csv'
model_path = '/scratch/wshalaby/doc2vec/models/word2vec/w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.model'
input_path = '/scratch/wshalaby/doc2vec/DOTA.tsv'
input_path = '/scratch/wshalaby/doc2vec/Battig.tsv'

titles_path = '/media/vol2/walid/work/github/semantic_searcher/titles_redirects_ids.csv'
input_path = '/media/vol2/walid/work/github/semantic_searcher/DOTA.tsv'
input_path = '/media/vol2/walid/work/github/semantic_searcher/Battig.tsv'
model_path = '/media/vol2/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.model'

titles_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/titles_redirects_ids.csv'
model_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.model'

titles, redirects, seealso = load_titles(titles_path, load_redirects=True, redirects_value='id')
output_path = None
mappings = {}
ids = None
compact_vecs = False
weighted = False

input_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/probase_sample1000-concept.txt'
input_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/probase_all-concept.txt'
output_path = '/media/wshalaby/b667e28e-5e90-4884-8810-5d897c9e56ce/work/github/semantic_searcher/data-concept/probase_sample1000-concept-classes.txt'
#mappings = {'position':'Job','company':'Company','school':'School','skill':'Skill'}
#mappings = {'position':'Project manager','company':'Google','school':'Boston University','skill':'Problem solving'}
mappings = {'position':'Accountant','company':'Google','school':'Boston University','skill':'Productivity'}
titles, redirects, seealso = load_titles(titles_path, load_redirects=True, redirects_value='title')
ids = {'id'+v+'di':k for k,v in titles.items()}
compact_vecs = True
weighted = False

mode = 'enrich'
mode = 'bootstrap'

from commons import load_titles
from commons import read_concept_dataset
from commons import resolve_mention
from commons import get_nearest
from commons import print_confusion_matrix

model = gensim.models.Word2Vec.load(model_path)
model.init_sims(replace=True)

processing_mode = ['all', 'single', 'multi']
processing_mode = ['all']
run_clustering(processing_mode=processing_mode,min_bs_score=0.0, min_bs_diff=0.0)

def run_clustering(processing_mode=['all', 'single', 'multi'],min_bs_score=0.0, min_bs_diff=0.0):
    for processing in processing_mode:
        print(processing)
        raw_dataset = read_concept_dataset(input_path,processing, mappings=mappings)
        c1 = 0
        c2 = 0
        dataset = {}
        for label, instances in raw_dataset.items():
            c1 = c1 + 1
            #print(label+str(instances))
            newlabel = resolve_mention(label, titles, redirects)
            if newlabel!='':
                c2 = c2 + 1
                dataset[newlabel] = []
                for instance in instances:
                    c1 = c1 + 1
                    newinstance = resolve_mention(instance, titles, redirects)
                    if newinstance!='':
                        c2 = c2 + 1
                        dataset[newlabel].append(newinstance)
                    else:
                        print('Can\'t resolve instance:({0}) under label:({1})'.format(instance,label))
            else:
                print('Can\'t resolve label:'+label)
        print(c1)
        print(c2)
        keys = list(dataset.keys())
        if mode=='enrich':
            cluster_with_enrichment(dataset, keys, model, [0], output_path=output_path)
            #cluster_with_enrichment(dataset, keys, model, [0,1,2])
        elif mode=='bootstrap':
            cluster_with_bootstrap(dataset, keys, model, min_bs_score=min_bs_score, min_bs_diff=min_bs_diff, output_path=output_path, compact_vecs=compact_vecs)

def cluster_with_bootstrap(dataset, keys, model, min_bs_score=0.0, min_bs_diff=0.0, output_path=None, compact_vecs=False):
    from collections import namedtuple
    MaxInstance = namedtuple('MaxInstance',['name','label','score','diff'])
    default_max_instance = MaxInstance('','',-1.0, 100.0)
    #Label = namedtuple('Label',['definitions','max_instance'])
    #label_meta = dict(zip(keys,[Label([v],default_max_instance) for v in keys]))
    label_definitions = {}
    if compact_vecs==True:
        label_definitions = dict(zip(keys,[(1*model[v],1) for v in keys]))
    else:
        label_definitions = dict(zip(keys,[[v] for v in keys]))
    label_max_instance = dict(zip(keys,[default_max_instance for _ in keys]))
    confusion_matrix = dict(zip(keys,[{k:v for k,v in dict(zip(keys,[0]*len(keys))).items()} for _ in range(len(keys))]))
    all_values = set([value for values in dataset.values() for value in values])
    correct_count = 0
    count = len(all_values)
    total = 0
    cur_itera = 0
    clusterings = []
    output = None
    if output_path is not None:
        output = open(output_path, 'w')        
    while len(all_values)>0:
        for label, instances in dataset.items():
            if label in model:
                for instance in instances:
                    if instance in all_values: # not assigned yet
                        if instance in model:                        
                            if compact_vecs==True:
                                nearest, score, min_diff = get_nearest_vector(model, instance, {k:v[0] for k,v in label_definitions.items()})
                            else:
                                nearest, score, min_diff = get_nearest(model, instance, label_definitions, weighted=weighted)
                            if nearest!='':
                                if label_max_instance[nearest].score<score:
                                    max_instance = MaxInstance(instance,label,score, min_diff)
                                    label_max_instance[nearest] = max_instance                                
                            else:
                                print('Can\'t assign instance:({0}) under label:({1})'.format(instance,label))    
                        else:
                            print('Can\'t find instance:({0}) under label:({1})'.format(instance,label))
                            if instance in all_values: # remove it 
                                all_values.remove(instance)
                                count -= 1
            else:
                print('Can\'t find label:'+label)
        for label, max_instance in label_max_instance.items():
            if max_instance.score>0:
                total += 1
                #print(max_instance,label)
                confusion_matrix[max_instance.label][label] += 1
                if max_instance.label==label:
                    correct_count += 1
                if min_bs_score<=max_instance.score and max_instance.diff>=min_bs_diff:
                    if compact_vecs==True:
                        vec, aggr_sum = label_definitions[label]
                        factor = 1
                        if weighted==True:
                            factor = max_instance.score
                        vec = aggr_sum*vec + factor*model[max_instance.name]
                        aggr_sum += factor
                        vec /= aggr_sum
                        label_definitions[label] = (vec, aggr_sum)
                    else:
                        label_definitions[label].append((max_instance.name,max_instance.score))
                label_max_instance[label] = default_max_instance
                all_values.remove(max_instance.name)
                if output is not None:
                    clusterings.append((max_instance.name,max_instance.label,label,max_instance.score, cur_itera))
            #else:
            #    print('instances are assigned for label:'+label)
        if total%100==0:
            print(print_confusion_matrix(confusion_matrix, ids))
            print(str(correct_count)+'/'+str(total))
            print(correct_count/total)    
        cur_itera += 1
        if output is not None:
            for name, label, classs, score, itera in clusterings:
                if ids is not None:
                    name, label, classs = [i for i in [ids[name],ids[label],ids[classs]]]
                output.write('\t'.join([name, label, classs, str(score), str(itera)])+os.linesep)
            clusterings = []
        #if cur_itera==20:
        #    break
    print(print_confusion_matrix(confusion_matrix, ids))
    print(count)
    print(str(correct_count)+'/'+str(total))
    print(correct_count/count)
    if output is not None:
        output.close()

def cluster_with_enrichment(dataset, keys, model, counts, output_path=None):
    for enrich_num in counts:
        print('enrich_num='+str(enrich_num))
        label_definitions = dict(zip(keys,[[v] for v in keys]))
        if enrich_num>0:
            for key in keys:
                #if key=='id18973446di':
                #    continue
                top_similars = model.most_similar(key)
                for top_similar,score in top_similars:
                    if top_similar[0:2]=='id' and top_similar[-2:]=='di':
                        label_definitions[key].append((top_similar,score))
                        if len(label_definitions[key])-1==enrich_num:
                            break
            if len(label_definitions[key])-1!=enrich_num:
                print('enrich not satisfied for:'+key)
        print(label_definitions)
        confusion_matrix = dict(zip(keys,[{k:v for k,v in dict(zip(keys,[0]*len(keys))).items()} for _ in range(len(keys))]))
        count = 0
        correct_count = 0
        clusterings = []
        for label, instances in dataset.items():
            if label in model:
                for instance in instances:
                    if instance in model:
                        count += 1
                        nearest, score, _ = get_nearest(model, instance, label_definitions, weighted=weighted)
                        #if label=='id18716923di':                        
                        #    sim1 = 1 - spatial.distance.cosine(model[instance],model['id18716923di'])
                        #    sim2 = 1 - spatial.distance.cosine(model[instance],model['id18973446di'])
                        #    print(instance,sim1,sim2)
                        if nearest!='':
                            confusion_matrix[label][nearest] += 1
                            if output_path is not None:
                                clusterings.append((instance,label,nearest,score))
                            if label==nearest:
                                correct_count += 1
                        else:
                            print('Can\'t assign instance:({0}) under label:({1})'.format(instance,label))    
                    else:
                        print('Can\'t find instance:({0}) under label:({1})'.format(instance,label))
            else:
                print('Can\'t find label:'+label)
            if count%100==0:
                print(print_confusion_matrix(confusion_matrix, ids))
        print(print_confusion_matrix(confusion_matrix, ids))
        print(count)
        print(correct_count)
        print(correct_count/count)
        if output_path is not None:
            with open(output_path, 'w') as output:
                for name, label, classs, score in clusterings:
                    if ids is not None:
                        name, label, classs = [i for i in [ids[name],ids[label],ids[classs]]]
                    output.write('\t'.join([name, label, classs, str(score)])+os.linesep)



'''
#/scratch/wshalaby/doc2vec/models/word2vec/w2v-4.4-concepts-from-plain-anno-titles-10iter-dim500-wind9-cnt1-skipgram1.model
DOTA, enrich
all
Can't resolve instance:(brittany dog) under label:(dogs)
Can't resolve instance:(hyeonmi cha) under label:(beverages)
465
463
enrich_num=0
{'id18955875di': ['id18955875di'], 'id250515di': ['id250515di'], 'id25778403di': ['id25778403di'], 'id13692155di': ['id13692155di'], 'id10406di': ['id10406di'], 'id18973446di': ['id18973446di'], 'id32410di': ['id32410di'], 'id18716923di': ['id18716923di'], 'id33978di': ['id33978di'], 'id18839di': ['id18839di'], 'id4699587di': ['id4699587di'], 'id22760983di': ['id22760983di'], 'id7984di': ['id7984di'], 'id22986di': ['id22986di'], 'id4269567di': ['id4269567di']}
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id250515di      id25778403di    id13692155di    id10406di       id18973446di    id32410di       id18716923di    id33978di      id18839di       id4699587di     id22760983di    id7984di        id22986di       id4269567di
id18955875di    25      3       0       0       0       0       0       0       0       0       0       0       1       0       0
id250515di      0       29      0       0       0       0       0       0       0       0       1       0       0       0       0
id25778403di    0       0       26      0       0       0       3       0       0       0       0       0       1       0       0
id13692155di    0       0       0       16      12      0       0       1       0       0       0       1       0       0       0
id10406di       0       0       0       0       29      0       0       0       0       0       0       0       0       0       1
id18973446di    0       0       0       0       1       28      0       1       0       0       0       0       0       0       0
id32410di       0       0       0       0       0       0       29      0       0       0       0       0       1       0       0
id18716923di    0       0       0       0       0       6       0       24      0       0       0       0       0       0       0
id33978di       0       5       0       0       0       0       1       0       24      0       0       0       0       0       0
id18839di       0       0       0       0       5       0       0       0       0       25      0       0       0       0       0
id4699587di     0       0       0       0       0       0       0       0       0       0       28      0       0       0       0
id22760983di    0       0       0       0       5       0       0       0       0       0       0       25      0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       29      0       0
id22986di       0       0       0       4       7       0       0       0       0       0       0       0       0       19      0
id4269567di     0       0       0       0       0       0       0       0       0       0       0       0       0       0       29

445
385
0.8651685393258427
single
315
315
enrich_num=0
{'id18955875di': ['id18955875di'], 'id250515di': ['id250515di'], 'id25778403di': ['id25778403di'], 'id13692155di': ['id13692155di'], 'id10406di': ['id10406di'], 'id18973446di': ['id18973446di'], 'id32410di': ['id32410di'], 'id18716923di': ['id18716923di'], 'id33978di': ['id33978di'], 'id18839di': ['id18839di'], 'id4699587di': ['id4699587di'], 'id22760983di': ['id22760983di'], 'id7984di': ['id7984di'], 'id22986di': ['id22986di'], 'id4269567di': ['id4269567di']}
        id18955875di    id250515di      id25778403di    id13692155di    id10406di       id18973446di    id32410di       id18716923di    id33978di      id18839di       id4699587di     id22760983di    id7984di        id22986di       id4269567di
id18955875di    20      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id250515di      0       19      0       0       0       0       0       0       0       0       1       0       0       0       0
id25778403di    0       0       17      0       0       0       2       0       0       0       0       0       1       0       0
id13692155di    0       0       0       12      8       0       0       0       0       0       0       0       0       0       0
id10406di       0       0       0       0       20      0       0       0       0       0       0       0       0       0       0
id18973446di    0       0       0       0       1       18      0       1       0       0       0       0       0       0       0
id32410di       0       0       0       0       0       0       19      0       0       0       0       0       1       0       0
id18716923di    0       0       0       0       0       4       0       16      0       0       0       0       0       0       0
id33978di       0       1       0       0       0       0       1       0       18      0       0       0       0       0       0
id18839di       0       0       0       0       1       0       0       0       0       19      0       0       0       0       0
id4699587di     0       0       0       0       0       0       0       0       0       0       20      0       0       0       0
id22760983di    0       0       0       0       2       0       0       0       0       0       0       18      0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       20      0       0
id22986di       0       0       0       3       4       0       0       0       0       0       0       0       0       13      0
id4269567di     0       0       0       0       0       0       0       0       0       0       0       0       0       0       20

300
269
0.8966666666666666
multi
Can't resolve instance:(brittany dog) under label:(dogs)
Can't resolve instance:(hyeonmi cha) under label:(beverages)
165
163
enrich_num=0
{'id18955875di': ['id18955875di'], 'id250515di': ['id250515di'], 'id25778403di': ['id25778403di'], 'id13692155di': ['id13692155di'], 'id10406di': ['id10406di'], 'id18973446di': ['id18973446di'], 'id32410di': ['id32410di'], 'id18716923di': ['id18716923di'], 'id33978di': ['id33978di'], 'id18839di': ['id18839di'], 'id4699587di': ['id4699587di'], 'id22760983di': ['id22760983di'], 'id7984di': ['id7984di'], 'id22986di': ['id22986di'], 'id4269567di': ['id4269567di']}
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id250515di      id25778403di    id13692155di    id10406di       id18973446di    id32410di       id18716923di    id33978di      id18839di       id4699587di     id22760983di    id7984di        id22986di       id4269567di
id18955875di    5       3       0       0       0       0       0       0       0       0       0       0       1       0       0
id250515di      0       10      0       0       0       0       0       0       0       0       0       0       0       0       0
id25778403di    0       0       9       0       0       0       1       0       0       0       0       0       0       0       0
id13692155di    0       0       0       4       4       0       0       1       0       0       0       1       0       0       0
id10406di       0       0       0       0       9       0       0       0       0       0       0       0       0       0       1
id18973446di    0       0       0       0       0       10      0       0       0       0       0       0       0       0       0
id32410di       0       0       0       0       0       0       10      0       0       0       0       0       0       0       0
id18716923di    0       0       0       0       0       2       0       8       0       0       0       0       0       0       0
id33978di       0       4       0       0       0       0       0       0       6       0       0       0       0       0       0
id18839di       0       0       0       0       4       0       0       0       0       6       0       0       0       0       0
id4699587di     0       0       0       0       0       0       0       0       0       0       8       0       0       0       0
id22760983di    0       0       0       0       3       0       0       0       0       0       0       7       0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       9       0       0
id22986di       0       0       0       1       3       0       0       0       0       0       0       0       0       6       0
id4269567di     0       0       0       0       0       0       0       0       0       0       0       0       0       0       9

145
116
0.8


DOTA, bootstrap
all
Can't resolve instance:(brittany dog) under label:(dogs)
Can't resolve instance:(hyeonmi cha) under label:(beverages)
465
463
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id250515di      id25778403di    id13692155di    id10406di       id18973446di    id32410di       id18716923di    id33978di      id18839di       id4699587di     id22760983di    id7984di        id22986di       id4269567di
id18955875di    28      2       0       0       0       0       0       0       0       0       0       0       0       0       0
id250515di      0       28      0       0       0       0       0       0       14      0       0       0       0       0       0
id25778403di    0       0       30      0       0       0       0       0       0       0       0       0       0       1       0
id13692155di    0       0       0       24      0       0       0       1       0       0       0       0       0       1       0
id10406di       0       0       0       3       30      1       0       0       0       0       0       1       0       1       0
id18973446di    0       0       0       1       0       28      0       28      0       0       0       0       0       0       0
id32410di       0       0       0       0       0       0       30      0       0       0       0       0       0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       29      0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       29      0       0
id22986di       0       0       0       0       0       0       0       0       0       0       0       0       0       27      0
id4269567di     0       0       0       0       0       0       0       0       0       0       0       0       0       0       29

443
387
0.873589164785553
single
315
315
        id18955875di    id250515di      id25778403di    id13692155di    id10406di       id18973446di    id32410di       id18716923di    id33978di      id18839di       id4699587di     id22760983di    id7984di        id22986di       id4269567di
id18955875di    20      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id250515di      0       20      0       0       0       0       0       0       1       0       0       0       0       0       0
id25778403di    0       0       19      0       0       0       0       0       0       0       0       0       0       1       0
id13692155di    0       0       0       18      0       0       0       0       0       0       0       0       0       1       0
id10406di       0       0       0       0       20      1       0       0       0       0       0       1       0       0       0
id18973446di    0       0       0       0       0       18      0       19      0       0       0       0       0       0       0
id32410di       0       0       1       0       0       0       20      0       0       0       0       0       0       0       0
id18716923di    0       0       0       0       0       1       0       1       0       0       0       0       0       0       0
id33978di       0       0       0       0       0       0       0       0       19      0       0       0       0       0       0
id18839di       0       0       0       0       0       0       0       0       0       20      0       0       0       0       0
id4699587di     0       0       0       0       0       0       0       0       0       0       20      0       0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       19      0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       20      0       0
id22986di       0       0       0       0       0       0       0       0       0       0       0       0       0       18      0
id4269567di     0       0       0       0       0       0       0       0       0       0       0       0       0       0       20

298
272
0.912751677852349
multi
Can't resolve instance:(brittany dog) under label:(dogs)
Can't resolve instance:(hyeonmi cha) under label:(beverages)
165
163
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id250515di      id25778403di    id13692155di    id10406di       id18973446di    id32410di       id18716923di    id33978di      id18839di       id4699587di     id22760983di    id7984di        id22986di       id4269567di
id18955875di    7       1       0       0       0       0       0       0       0       0       0       0       0       0       0
id250515di      0       9       0       0       0       0       0       0       3       0       0       0       0       0       0
id25778403di    0       0       10      0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       0       0       0       0       0       0       0       0       0       0       0       0       1       0
id10406di       0       0       0       7       8       0       0       0       0       0       0       0       0       0       0
id18973446di    0       0       0       0       0       8       0       1       0       0       0       0       0       0       0
id32410di       0       0       0       0       1       0       10      0       0       0       0       0       0       0       0
id18716923di    0       0       0       1       0       2       0       9       0       0       0       0       0       0       0
id33978di       0       0       0       0       0       0       0       0       7       0       0       0       0       0       0
id18839di       0       0       0       0       0       0       0       0       0       10      0       0       0       0       0
id4699587di     1       0       0       0       0       0       0       0       0       0       8       0       0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       10      0       0       0
id7984di        1       0       0       0       0       0       0       0       0       0       0       0       9       0       0
id22986di       0       0       0       2       0       0       0       0       0       0       0       0       0       9       0
id4269567di     0       0       0       0       1       0       0       0       0       0       0       0       0       0       9

145
123
0.8482758620689655

Battig, enrich
enrich
all
92
92
enrich_num=0
{'id3410di': ['id3410di'], 'id10843di': ['id10843di'], 'id32410di': ['id32410di'], 'id1031396di': ['id1031396di'], 'id4699587di': ['id4699587di'], 'id38180di': ['id38180di'], 'id5791492di': ['id5791492di'], 'id30677di': ['id30677di'], 'id18838di': ['id18838di'], 'id18955875di': ['id18955875di']}
Can't find instance:(id4499087di) under label:(id1031396di)
        id3410di        id10843di       id32410di       id1031396di     id4699587di     id38180di       id5791492di     id30677di       id18838di      id18955875di
id3410di        7       0       0       0       2       0       0       0       1       0
id10843di       0       0       0       1       0       0       6       0       0       0
id32410di       0       0       10      0       0       0       0       0       0       0
id1031396di     0       0       0       8       0       0       0       0       0       0
id4699587di     0       0       0       0       6       0       0       0       0       0
id38180di       0       0       0       3       0       7       0       0       0       0
id5791492di     0       0       0       0       0       0       10      0       0       0
id30677di       0       0       0       2       0       0       0       4       0       0
id18838di       3       0       1       0       2       0       2       0       2       0
id18955875di    0       0       0       0       0       0       0       0       0       4

81
58
0.7160493827160493

Battig, bootstrap
all
92
92
Can't find instance:(id4499087di) under label:(id1031396di)
        id3410di        id10843di       id32410di       id1031396di     id4699587di     id38180di       id5791492di     id30677di       id18838di      id18955875di
id3410di        6       0       0       0       0       0       0       0       3       0
id10843di       0       0       0       0       0       0       0       0       0       0
id32410di       0       0       10      0       0       0       0       0       0       0
id1031396di     0       0       0       8       0       0       0       1       0       0
id4699587di     0       1       0       0       6       0       0       0       0       0
id38180di       0       0       0       0       0       10      0       0       0       0
id5791492di     0       6       0       0       0       0       10      0       0       0
id30677di       0       0       0       0       0       0       0       5       0       0
id18838di       4       0       0       0       0       0       0       0       7       0
id18955875di    0       0       0       0       0       0       0       0       0       4

81
66
0.8148148148148148
========================================

'/media/vol2/walid/work/github/semantic_searcher/w2v-plain-anno-titles-4.4-10iter-dim500-wind9-cnt1-skipgram1.model'
DOTA, enrich
all
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
465
463
enrich_num=0
{'id32410di': ['id32410di'], 'id18839di': ['id18839di'], 'id13692155di': ['id13692155di'], 'id250515di': ['id250515di'], 'id10406di': ['id10406di'], 'id4699587di': ['id4699587di'], 'id18955875di': ['id18955875di'], 'id4269567di': ['id4269567di'], 'id25778403di': ['id25778403di'], 'id18973446di': ['id18973446di'], 'id18716923di': ['id18716923di'], 'id22986di': ['id22986di'], 'id22760983di': ['id22760983di'], 'id7984di': ['id7984di'], 'id33978di': ['id33978di']}
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
Can't find instance:(id23316667di) under label:(id18955875di)
        id32410di       id18839di       id13692155di    id250515di      id10406di       id4699587di     id18955875di    id4269567di     id25778403di   id18973446di    id18716923di    id22986di       id22760983di    id7984di        id33978di
id32410di       30      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id18839di       0       30      0       0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       0       25      0       4       1       0       0       0       0       0       0       0       0       0
id250515di      0       0       0       28      0       1       1       0       0       0       0       0       0       0       0
id10406di       0       0       0       0       30      0       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       0       0       28      0       0       0       0       0       0       0       0       0
id18955875di    0       0       0       1       0       0       26      0       0       0       2       0       0       0       0
id4269567di     0       0       0       0       0       0       0       29      0       0       0       0       0       0       0
id25778403di    0       0       0       0       0       0       0       0       30      0       0       0       0       0       0
id18973446di    0       0       0       0       0       0       1       0       0       29      0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       0       24      6       0       0       0       0
id22986di       0       0       3       0       1       0       0       0       1       0       0       25      0       0       0
id22760983di    0       1       0       0       0       0       0       0       0       0       0       0       29      0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       0       29      0
id33978di       0       0       0       3       0       0       0       0       0       0       0       0       0       0       27

445
401
0.9011235955056179
single
315
315
enrich_num=0
{'id32410di': ['id32410di'], 'id18839di': ['id18839di'], 'id13692155di': ['id13692155di'], 'id250515di': ['id250515di'], 'id10406di': ['id10406di'], 'id4699587di': ['id4699587di'], 'id18955875di': ['id18955875di'], 'id4269567di': ['id4269567di'], 'id25778403di': ['id25778403di'], 'id18973446di': ['id18973446di'], 'id18716923di': ['id18716923di'], 'id22986di': ['id22986di'], 'id22760983di': ['id22760983di'], 'id7984di': ['id7984di'], 'id33978di': ['id33978di']}
        id32410di       id18839di       id13692155di    id250515di      id10406di       id4699587di     id18955875di    id4269567di     id25778403di   id18973446di    id18716923di    id22986di       id22760983di    id7984di        id33978di
id32410di       20      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id18839di       0       20      0       0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       0       18      0       2       0       0       0       0       0       0       0       0       0       0
id250515di      0       0       0       19      0       1       0       0       0       0       0       0       0       0       0
id10406di       0       0       0       0       20      0       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       0       0       20      0       0       0       0       0       0       0       0       0
id18955875di    0       0       0       0       0       0       20      0       0       0       0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       20      0       0       0       0       0       0       0
id25778403di    0       0       0       0       0       0       0       0       20      0       0       0       0       0       0
id18973446di    0       0       0       0       0       0       1       0       0       19      0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       0       16      4       0       0       0       0
id22986di       0       0       2       0       0       0       0       0       1       0       0       17      0       0       0
id22760983di    0       1       0       0       0       0       0       0       0       0       0       0       19      0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       0       20      0
id33978di       0       0       0       2       0       0       0       0       0       0       0       0       0       0       18

300
274
0.9133333333333333
multi
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
165
163
enrich_num=0
{'id32410di': ['id32410di'], 'id18839di': ['id18839di'], 'id13692155di': ['id13692155di'], 'id250515di': ['id250515di'], 'id10406di': ['id10406di'], 'id4699587di': ['id4699587di'], 'id18955875di': ['id18955875di'], 'id4269567di': ['id4269567di'], 'id25778403di': ['id25778403di'], 'id18973446di': ['id18973446di'], 'id18716923di': ['id18716923di'], 'id22986di': ['id22986di'], 'id22760983di': ['id22760983di'], 'id7984di': ['id7984di'], 'id33978di': ['id33978di']}
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
Can't find instance:(id23316667di) under label:(id18955875di)
        id32410di       id18839di       id13692155di    id250515di      id10406di       id4699587di     id18955875di    id4269567di     id25778403di   id18973446di    id18716923di    id22986di       id22760983di    id7984di        id33978di
id32410di       10      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id18839di       0       10      0       0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       0       7       0       2       1       0       0       0       0       0       0       0       0       0
id250515di      0       0       0       9       0       0       1       0       0       0       0       0       0       0       0
id10406di       0       0       0       0       10      0       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       0       0       8       0       0       0       0       0       0       0       0       0
id18955875di    0       0       0       1       0       0       6       0       0       0       2       0       0       0       0
id4269567di     0       0       0       0       0       0       0       9       0       0       0       0       0       0       0
id25778403di    0       0       0       0       0       0       0       0       10      0       0       0       0       0       0
id18973446di    0       0       0       0       0       0       0       0       0       10      0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       0       8       2       0       0       0       0
id22986di       0       0       1       0       1       0       0       0       0       0       0       8       0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       0       10      0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       0       9       0
id33978di       0       0       0       1       0       0       0       0       0       0       0       0       0       0       9

145
127
0.8758620689655172

DOTA, enrich-weighted
all
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
465
463
enrich_num=0
{'id18955875di': ['id18955875di'], 'id22986di': ['id22986di'], 'id13692155di': ['id13692155di'], 'id4699587di': ['id4699587di'], 'id33978di': ['id33978di'], 'id10406di': ['id10406di'], 'id18839di': ['id18839di'], 'id250515di': ['id250515di'], 'id18716923di': ['id18716923di'], 'id18973446di': ['id18973446di'], 'id4269567di': ['id4269567di'], 'id7984di': ['id7984di'], 'id22760983di': ['id22760983di'], 'id32410di': ['id32410di'], 'id25778403di': ['id25778403di']}
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id22986di       id13692155di    id4699587di     id33978di       id10406di       id18839di       id250515di      id18716923di   id18973446di    id4269567di     id7984di        id22760983di    id32410di       id25778403di
id18955875di    26      0       0       0       0       0       0       1       2       0       0       0       0       0       0
id22986di       0       25      3       0       0       1       0       0       0       0       0       0       0       0       1
id13692155di    0       0       25      1       0       4       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       28      0       0       0       0       0       0       0       0       0       0       0
id33978di       0       0       0       0       27      0       0       3       0       0       0       0       0       0       0
id10406di       0       0       0       0       0       30      0       0       0       0       0       0       0       0       0
id18839di       0       0       0       0       0       0       30      0       0       0       0       0       0       0       0
id250515di      1       0       0       1       0       0       0       28      0       0       0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       6       24      0       0       0       0       0
id18973446di    1       0       0       0       0       0       0       0       0       29      0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       0       0       0       29      0       0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       29      0       0       0
id22760983di    0       0       0       0       0       0       1       0       0       0       0       0       29      0       0
id32410di       0       0       0       0       0       0       0       0       0       0       0       0       0       30      0
id25778403di    0       0       0       0       0       0       0       0       0       0       0       0       0       0       30

445
401
0.9011235955056179
single
315
315
enrich_num=0
{'id18955875di': ['id18955875di'], 'id22986di': ['id22986di'], 'id13692155di': ['id13692155di'], 'id4699587di': ['id4699587di'], 'id33978di': ['id33978di'], 'id10406di': ['id10406di'], 'id18839di': ['id18839di'], 'id250515di': ['id250515di'], 'id18716923di': ['id18716923di'], 'id18973446di': ['id18973446di'], 'id4269567di': ['id4269567di'], 'id7984di': ['id7984di'], 'id22760983di': ['id22760983di'], 'id32410di': ['id32410di'], 'id25778403di': ['id25778403di']}
        id18955875di    id22986di       id13692155di    id4699587di     id33978di       id10406di       id18839di       id250515di      id18716923di   id18973446di    id4269567di     id7984di        id22760983di    id32410di       id25778403di
id18955875di    20      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id22986di       0       17      2       0       0       0       0       0       0       0       0       0       0       0       1
id13692155di    0       0       18      0       0       2       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       20      0       0       0       0       0       0       0       0       0       0       0
id33978di       0       0       0       0       18      0       0       2       0       0       0       0       0       0       0
id10406di       0       0       0       0       0       20      0       0       0       0       0       0       0       0       0
id18839di       0       0       0       0       0       0       20      0       0       0       0       0       0       0       0
id250515di      0       0       0       1       0       0       0       19      0       0       0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       4       16      0       0       0       0       0
id18973446di    1       0       0       0       0       0       0       0       0       19      0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       0       0       0       20      0       0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       20      0       0       0
id22760983di    0       0       0       0       0       0       1       0       0       0       0       0       19      0       0
id32410di       0       0       0       0       0       0       0       0       0       0       0       0       0       20      0
id25778403di    0       0       0       0       0       0       0       0       0       0       0       0       0       0       20

300
274
0.9133333333333333
multi
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
165
163
enrich_num=0
{'id18955875di': ['id18955875di'], 'id22986di': ['id22986di'], 'id13692155di': ['id13692155di'], 'id4699587di': ['id4699587di'], 'id33978di': ['id33978di'], 'id10406di': ['id10406di'], 'id18839di': ['id18839di'], 'id250515di': ['id250515di'], 'id18716923di': ['id18716923di'], 'id18973446di': ['id18973446di'], 'id4269567di': ['id4269567di'], 'id7984di': ['id7984di'], 'id22760983di': ['id22760983di'], 'id32410di': ['id32410di'], 'id25778403di': ['id25778403di']}
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id22986di       id13692155di    id4699587di     id33978di       id10406di       id18839di       id250515di      id18716923di   id18973446di    id4269567di     id7984di        id22760983di    id32410di       id25778403di
id18955875di    6       0       0       0       0       0       0       1       2       0       0       0       0       0       0
id22986di       0       8       1       0       0       1       0       0       0       0       0       0       0       0       0
id13692155di    0       0       7       1       0       2       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       8       0       0       0       0       0       0       0       0       0       0       0
id33978di       0       0       0       0       9       0       0       1       0       0       0       0       0       0       0
id10406di       0       0       0       0       0       10      0       0       0       0       0       0       0       0       0
id18839di       0       0       0       0       0       0       10      0       0       0       0       0       0       0       0
id250515di      1       0       0       0       0       0       0       9       0       0       0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       2       8       0       0       0       0       0
id18973446di    0       0       0       0       0       0       0       0       0       10      0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       0       0       0       9       0       0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       9       0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       0       10      0       0
id32410di       0       0       0       0       0       0       0       0       0       0       0       0       0       10      0
id25778403di    0       0       0       0       0       0       0       0       0       0       0       0       0       0       10

145
127
0.8758620689655172


DOTA, bootstrap
all
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
465
463
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
Can't find instance:(id23316667di) under label:(id18955875di)
        id32410di       id18839di       id13692155di    id250515di      id10406di       id4699587di     id18955875di    id4269567di     id25778403di   id18973446di    id18716923di    id22986di       id22760983di    id7984di        id33978di
id32410di       30      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id18839di       0       30      0       0       0       0       0       0       0       1       0       0       0       0       0
id13692155di    0       0       25      0       0       0       0       0       0       0       0       1       0       0       0
id250515di      0       0       0       30      0       0       1       0       0       0       0       0       0       0       1
id10406di       0       0       0       0       30      0       0       0       0       0       0       1       0       0       0
id4699587di     0       0       1       0       0       28      0       0       0       0       0       0       0       0       0
id18955875di    0       0       0       0       0       0       26      0       0       0       0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       29      0       0       0       0       0       0       0
id25778403di    0       0       0       0       0       0       0       0       30      0       0       1       0       0       0
id18973446di    0       0       0       0       0       0       0       0       0       27      0       0       0       0       0
id18716923di    0       0       1       0       0       0       0       0       0       2       30      0       0       0       0
id22986di       0       0       0       0       0       0       2       0       0       0       0       27      0       0       0
id22760983di    0       0       1       0       0       0       0       0       0       0       0       0       30      0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       0       29      0
id33978di       0       0       0       0       0       0       0       0       0       0       0       0       0       0       29

443
430
0.9706546275395034
single
315
315
        id32410di       id18839di       id13692155di    id250515di      id10406di       id4699587di     id18955875di    id4269567di     id25778403di   id18973446di    id18716923di    id22986di       id22760983di    id7984di        id33978di
id32410di       20      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id18839di       0       20      0       0       0       0       0       0       0       1       0       0       0       0       0
id13692155di    0       0       17      0       0       0       0       0       0       0       0       1       0       0       0
id250515di      0       0       0       20      0       0       0       0       0       0       0       0       0       0       0
id10406di       0       0       0       0       20      0       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       0       0       20      0       0       0       0       0       0       0       0       0
id18955875di    0       0       0       0       0       0       20      0       0       0       0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       20      0       0       0       0       0       0       0
id25778403di    0       0       0       0       0       0       0       0       20      0       0       1       0       0       0
id18973446di    0       0       0       0       0       0       0       0       0       18      0       0       0       0       0
id18716923di    0       0       1       0       0       0       0       0       0       1       20      0       0       0       0
id22986di       0       0       0       0       0       0       0       0       0       0       0       18      0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       0       20      0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       0       20      0
id33978di       0       0       0       0       0       0       0       0       0       0       0       0       0       0       20

298
293
0.9832214765100671
multi
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
165
163
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
Can't find instance:(id23316667di) under label:(id18955875di)
        id32410di       id18839di       id13692155di    id250515di      id10406di       id4699587di     id18955875di    id4269567di     id25778403di   id18973446di    id18716923di    id22986di       id22760983di    id7984di        id33978di
id32410di       10      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id18839di       0       10      0       0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       0       9       0       0       0       0       0       0       0       0       1       0       0       0
id250515di      0       0       0       9       0       0       0       0       0       0       0       0       0       0       0
id10406di       0       0       0       0       10      0       1       0       0       0       0       1       0       0       0
id4699587di     0       0       1       0       0       8       0       0       0       0       0       0       0       0       0
id18955875di    0       0       0       1       0       0       6       0       0       0       0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       9       0       0       0       0       0       0       0
id25778403di    0       0       0       0       0       0       0       0       10      0       0       0       0       0       0
id18973446di    0       0       0       0       0       0       1       0       0       10      0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       0       0       10      0       0       0       0
id22986di       0       0       0       0       0       0       1       0       0       0       0       8       0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       0       10      0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       0       0       9       0
id33978di       0       0       0       0       0       0       0       0       0       0       0       0       0       0       10

145
138
0.9517241379310345

bootstrap-weighted
all
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
465
463
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id22986di       id13692155di    id4699587di     id33978di       id10406di       id18839di       id250515di      id18716923di   id18973446di    id4269567di     id7984di        id22760983di    id32410di       id25778403di
id18955875di    26      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id22986di       2       27      0       0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       1       25      0       0       0       0       0       0       0       0       0       0       0       0
id4699587di     0       0       1       28      0       0       0       0       0       0       0       0       0       0       0
id33978di       0       0       0       0       29      0       0       0       0       0       0       0       0       0       0
id10406di       0       1       0       0       0       30      0       0       0       0       0       0       0       0       0
id18839di       0       0       0       0       0       0       30      0       0       1       0       0       0       0       0
id250515di      1       0       0       0       1       0       0       30      0       0       0       0       0       0       0
id18716923di    0       0       1       0       0       0       0       0       29      2       0       0       0       0       0
id18973446di    0       0       0       0       0       0       0       0       1       27      0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       0       0       0       29      0       0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       29      0       0       0
id22760983di    0       0       1       0       0       0       0       0       0       0       0       0       30      0       0
id32410di       0       0       0       0       0       0       0       0       0       0       0       0       0       30      0
id25778403di    0       1       0       0       0       0       0       0       0       0       0       0       0       0       30

443
429
0.9683972911963883
single
315
315
        id18955875di    id22986di       id13692155di    id4699587di     id33978di       id10406di       id18839di       id250515di      id18716923di   id18973446di    id4269567di     id7984di        id22760983di    id32410di       id25778403di
id18955875di    20      0       0       0       0       0       0       0       0       0       0       0       0       0       0
id22986di       0       17      0       0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       2       17      0       0       0       0       0       0       0       0       0       0       0       0
id4699587di     0       0       0       20      0       0       0       0       0       0       0       0       0       0       0
id33978di       0       0       0       0       20      0       0       0       0       0       0       0       0       0       0
id10406di       0       0       0       0       0       20      0       0       0       0       0       0       0       0       0
id18839di       0       0       0       0       0       0       20      0       0       1       0       0       0       0       0
id250515di      0       0       0       0       0       0       0       20      0       0       0       0       0       0       0
id18716923di    0       0       1       0       0       0       0       0       20      1       0       0       0       0       0
id18973446di    0       0       0       0       0       0       0       0       0       18      0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       0       0       0       20      0       0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       20      0       0       0
id22760983di    0       0       0       0       0       0       0       0       0       0       0       0       20      0       0
id32410di       0       0       0       0       0       0       0       0       0       0       0       0       0       20      0
id25778403di    0       1       0       0       0       0       0       0       0       0       0       0       0       0       20

298
292
0.9798657718120806
multi
Can't resolve instance:(hyeonmi cha) under label:(beverages)
Can't resolve instance:(brittany dog) under label:(dogs)
165
163
Can't find instance:(id23316667di) under label:(id18955875di)
Can't find instance:(id16944957di) under label:(id4699587di)
Can't find instance:(id28644615di) under label:(id4699587di)
        id18955875di    id22986di       id13692155di    id4699587di     id33978di       id10406di       id18839di       id250515di      id18716923di   id18973446di    id4269567di     id7984di        id22760983di    id32410di       id25778403di
id18955875di    6       0       0       0       0       0       0       0       0       0       0       0       0       0       0
id22986di       1       8       0       0       0       0       0       0       0       0       0       0       0       0       0
id13692155di    0       1       6       0       0       0       0       0       0       0       0       0       0       0       0
id4699587di     0       0       1       8       0       0       0       0       0       0       0       0       0       0       0
id33978di       0       0       0       0       10      0       0       0       0       0       0       0       0       0       0
id10406di       0       1       2       0       0       10      0       0       0       0       0       0       0       0       0
id18839di       0       0       0       0       0       0       10      0       0       0       0       0       0       0       0
id250515di      1       0       0       0       0       0       0       10      0       0       0       0       0       0       0
id18716923di    0       0       0       0       0       0       0       0       9       0       0       0       0       0       0
id18973446di    1       0       0       0       0       0       0       0       1       10      0       0       0       0       0
id4269567di     0       0       0       0       0       0       0       0       0       0       9       0       0       0       0
id7984di        0       0       0       0       0       0       0       0       0       0       0       9       0       0       0
id22760983di    0       0       1       0       0       0       0       0       0       0       0       0       10      0       0
id32410di       0       0       0       0       0       0       0       0       0       0       0       0       0       10      0
id25778403di    0       0       0       0       0       0       0       0       0       0       0       0       0       0       10

145
135
0.9310344827586207

Battig, enrich
all
92
92
enrich_num=0
{'id5791492di': ['id5791492di'], 'id4699587di': ['id4699587di'], 'id3410di': ['id3410di'], 'id18838di': ['id18838di'], 'id30677di': ['id30677di'], 'id38180di': ['id38180di'], 'id32410di': ['id32410di'], 'id10843di': ['id10843di'], 'id18955875di': ['id18955875di'], 'id1031396di': ['id1031396di']}
Can't find instance:(id4499087di) under label:(id1031396di)
        id5791492di     id4699587di     id3410di        id18838di       id30677di       id38180di       id32410di       id10843di       id18955875di   id1031396di
id5791492di     10      0       0       0       0       0       0       0       0       0
id4699587di     0       6       0       0       0       0       0       0       0       0
id3410di        1       0       9       0       0       0       0       0       0       0
id18838di       2       0       0       7       0       0       1       0       0       0
id30677di       0       0       0       0       6       0       0       0       0       0
id38180di       0       0       0       1       0       9       0       0       0       0
id32410di       0       0       0       0       0       0       10      0       0       0
id10843di       6       0       0       0       0       0       0       0       0       1
id18955875di    0       0       0       0       0       0       0       0       4       0
id1031396di     0       0       0       0       2       0       0       0       0       6

81
67
0.8271604938271605

Battig, bootstrap
all
92
92
Can't find instance:(id4499087di) under label:(id1031396di)
        id5791492di     id4699587di     id3410di        id18838di       id30677di       id38180di       id32410di       id10843di       id18955875di   id1031396di
id5791492di     10      0       0       0       0       0       0       6       0       0
id4699587di     0       6       0       0       0       0       0       0       0       0
id3410di        0       0       9       0       0       0       0       0       0       0
id18838di       0       0       1       10      0       0       0       0       0       0
id30677di       0       0       0       0       6       0       0       0       0       0
id38180di       0       0       0       0       0       9       0       0       0       0
id32410di       0       0       0       0       0       0       10      0       0       0
id10843di       0       0       0       0       0       0       0       0       0       0
id18955875di    0       0       0       0       0       0       0       0       4       0
id1031396di     0       0       0       0       0       1       0       1       0       8

81
72
0.8888888888888888


etr
       id355343di      id4918223di     id28022di       id16888425di
id355343di      288     209     44      123
id4918223di     79      55      15      666
id28022di       0       2       610     12
id16888425di    57      141     168     90

2559
1043
0.4075810863618601

enrich
all
2948
2948
enrich_num=0
{'id61114di': ['id61114di'], 'id239050di': ['id239050di'], 'id1467948di': ['id1467948di'], 'id1092923di': ['id1092923di']}
        Boston University       Project manager Problem solving Google
Boston University       689     0       0       5
Project manager 97      359     105     54
Problem solving 35      142     490     62
Google  54      34      43      775

2944
2313
0.7856657608695652

        Boston University       Productivity    Google  Accountant
Boston University       683     0       4       7
Productivity    26      465     81      157
Google  47      48      762     49
Accountant      48      128     51      388

2944
2298
0.7805706521739131

        Boston University       Productivity    Google  Accountant
Boston University       854     0       5       8
Productivity    323     3854    1229    1723
Google  3898    4267    22534   3882
Accountant      359     1089    530     2053

46608
29295
0.6285401647785788

bootstrap
        Boston University       Problem solving Accountant      Google
Boston University       612     4       3       5
Problem solving 8       388     170     19
Accountant      22      123     372     15
Google  13      117     39      649

2559
2021
0.7897616256350137
        Boston University       Project manager Problem solving Google
Boston University       602     16      3       4
Project manager 31      289     197     14
Problem solving 10      190     374     14
Google  14      136     32      633

2559
1898
0.7416959749902305

        Boston University       Productivity    Google  Accountant
Boston University       613     3       5       4
Productivity    12      449     16      187
Google  13      66      654     85
Accountant      13      117     14      308

2559
2024
0.7909339585775693
'''

