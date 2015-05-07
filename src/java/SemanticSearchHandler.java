/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.solr.handler.component;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.apache.commons.lang.ArrayUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;
import org.apache.solr.common.params.CommonParams;
import org.apache.solr.common.params.ModifiableSolrParams;
import org.apache.solr.common.util.NamedList;
import org.apache.solr.common.util.SimpleOrderedMap;
import org.apache.solr.core.PluginInfo;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.response.SolrQueryResponse;

/**
 *
 * Refer SOLR-281
 *
 */

enum ENUM_SEMANTIC_METHOD {
  e_UNKNOWN, 
  e_ESA,
  e_ESA_ANCHORS,
  e_ESA_SEE_ALSO,
  e_ESA_SEE_ALSO_ASSO,
  e_ESA_ANCHORS_SEE_ALSO, 
  e_ESA_ANCHORS_SEE_ALSO_ASSO
}

enum ENUM_CONCEPT_TYPE {
  e_UNKNOWN, 
  e_TITLE,
  e_ANCHOR,
  e_SEE_ALSO,
  e_ASSOCIATION
}

class CachedConceptInfo {
  public String[] category; // first two categories of the title 
  
  public CachedConceptInfo() {
  }
  
  public CachedConceptInfo(String n, String cat1, String cat2) {
    category = new String[2];
    category[0] = cat1;
    category[1] = cat2;
  }
}

class CachedAssociationInfo {
  ArrayList<Integer[]> associations = null;
  public CachedAssociationInfo() {
    associations = new ArrayList<Integer[]>();
  }
  
  public void addAssociation(int idx, int c) {
    // each association is represented by its index and its count
    associations.add(new Integer[]{idx,c});
  }
}

class SemanticConcept implements Comparable<SemanticConcept> {
  public CachedConceptInfo cachedInfo = null;
  public String name = "";
  public String ner = ""; // NE label ("P" --> person, "o" --> organization, "L" --> location, "M" --> misc)
  public float weight = 0.0f;
  public int id = 0;
  public int parent_id = 0;
  public int asso_cnt = 0;
  ENUM_CONCEPT_TYPE e_concept_type = ENUM_CONCEPT_TYPE.e_UNKNOWN;
  
  public SemanticConcept() {    
  }
  
  public SemanticConcept(String name, CachedConceptInfo cachedInfo, String ne, float w, 
    int id, int parent_id, int asso, ENUM_CONCEPT_TYPE type) {
    this.name = name;
    this.cachedInfo = cachedInfo;
    ner = ne;
    weight = w;
    e_concept_type = type;
    this.id = id;
    this.parent_id = parent_id;
    this.asso_cnt = asso;
  }

  public SemanticConcept(SemanticConcept c) {
    this.name = c.name;
    this.cachedInfo = c.cachedInfo;
    ner = c.ner;
    weight = c.weight;
    e_concept_type = c.e_concept_type;
    this.id = c.id;
    this.parent_id = c.parent_id;
    this.asso_cnt = c.asso_cnt;
  }

  @Override
  public int compareTo(SemanticConcept o) {
    if (((SemanticConcept)o).weight>this.weight)
      return 1;
    else if (((SemanticConcept)o).weight<this.weight)
      return -1;
    else return 0;
    
  }

  public SimpleOrderedMap<Object> getInfo() {
    SimpleOrderedMap<Object> conceptInfo = new SimpleOrderedMap<Object>();
    conceptInfo.add("weight", weight);
    conceptInfo.add("id", id);
    conceptInfo.add("p_id", parent_id);
    conceptInfo.add("asso_cnt", asso_cnt);
    conceptInfo.add("type", getConceptTypeString(e_concept_type));
    
    return conceptInfo;
  }
  

  private String getConceptTypeString(ENUM_CONCEPT_TYPE e) {
    if(e==ENUM_CONCEPT_TYPE.e_TITLE)
      return "title";
    else if(e==ENUM_CONCEPT_TYPE.e_ANCHOR)
      return "anchor";
    else if(e==ENUM_CONCEPT_TYPE.e_SEE_ALSO)
      return "seealso";
    else if(e==ENUM_CONCEPT_TYPE.e_ASSOCIATION)
      return "association";
    else return "unknown";
  }
}

public class SemanticSearchHandler extends SearchHandler
{
  
  /* 
   *
   */
  @Override
  public void init(PluginInfo info) {
    super.init(info);
    
    // cache Wiki see also associations for fast retrieval
    cacheAssociationsInfo();

    // cache Wiki titles with some required information for fast retrieval
    cacheConceptsInfo();
    //cachedConceptsInfo = new HashMap<String,CachedConceptInfo>();
    
  }

  private boolean hidden_relax_see_also = false;
  
  private boolean hidden_relax_ner = false;
  
  private boolean hidden_relax_categories = false;
  
  private boolean hidden_display_wiki_hits = false;
  
  private boolean hidden_relax_search = false;
  
  private String hidden_wiki_search_field = "text";
  
  private int hidden_max_hits = 0;
  
  private int hidden_min_wiki_length = 0;
  
  private int hidden_min_asso_cnt = 1;
  
  private int hidden_max_title_ngrams = 3;
  
  private boolean hidden_relax_disambig = false;
  
  private boolean hidden_relax_listof = false;
  
  private boolean hidden_relax_same_title = false;
  
  private boolean hidden_relatedness_experiment = false;
  
  private HashMap<String,CachedConceptInfo> cachedConceptsInfo = null;
  
  private HashMap<String,Integer> titleIntMapping = null;
  private HashMap<Integer,String> titleStrMapping = null;
  
  private HashMap<Integer,CachedAssociationInfo> cachedAssociationsInfo = null;
  
  @Override
  public void handleRequestBody(SolrQueryRequest req, SolrQueryResponse rsp) throws Exception
  {
    NamedList<Object> semanticConceptsInfo = null;
    ENUM_SEMANTIC_METHOD e_Method = ENUM_SEMANTIC_METHOD.e_UNKNOWN;
    boolean enable_title_search = false;
    boolean measure_relatedness = false;
    
    // get method of semantic concepts retrievals
    String tmp = req.getParams().get("conceptsmethod");
    if(tmp!=null)
      e_Method = getSemanticMethod(tmp);
    
    // get measure relatedness flag 
    tmp = req.getParams().get("measure_relatedness");
    if(tmp!=null && tmp.compareTo("on")==0)
      measure_relatedness = true;
    
    // get semantic relatedness experiment flag 
    tmp = req.getParams().get("hrelatednessexpr");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relatedness_experiment = true;
    else
      hidden_relatedness_experiment = false;
    
    // get enable title search flag 
    tmp = req.getParams().get("titlesearch");
    if(tmp!=null && tmp.compareTo("on")==0)
      enable_title_search = true;
    
    // get force see also flag 
    tmp = req.getParams().get("hrelaxseealso");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_see_also = true;
    else
      hidden_relax_see_also = false;

    // get relax NER flag 
    tmp = req.getParams().get("hrelaxner");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_ner = true;
    else
      hidden_relax_ner = false;
    
    // get relax categories flag 
    tmp = req.getParams().get("hrelaxcategories");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_categories = true;
    else
      hidden_relax_categories = false;
    
    // get relax search flag 
    tmp = req.getParams().get("hrelaxsearch");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_search = true;
    else
      hidden_relax_search = false;

    // get relax list of filter flag 
    tmp = req.getParams().get("hrelaxlistof");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_listof = true;
    else
      hidden_relax_listof = false;
    
    // get relax same title flag 
    tmp = req.getParams().get("hrelaxsametitle");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_same_title = true;
    else
      hidden_relax_same_title = false;
    
    // get relax disambiguation filter flag 
    tmp = req.getParams().get("hrelaxdisambig");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_disambig = true;
    else
      hidden_relax_disambig = false;
    
    // get maximum hits in initial wiki search
    tmp = req.getParams().get("hmaxhits");
    if(tmp!=null)
      hidden_max_hits = Integer.parseInt(tmp);
    else
      hidden_max_hits = Integer.MAX_VALUE;
    
    // get maximum hits in initial wiki search
    tmp = req.getParams().get("hminassocnt");
    if(tmp!=null)
      hidden_min_asso_cnt = Integer.parseInt(tmp);
    else
      hidden_min_asso_cnt = 1;
    
    // get minimum wiki article length to search
    tmp = req.getParams().get("hminwikilen");
    if(tmp!=null)
      hidden_min_wiki_length = Integer.parseInt(tmp);
    else
      hidden_min_wiki_length = 0;
    
    // get maximum ngrams of wiki titles
    tmp = req.getParams().get("hmaxngrams");
    if(tmp!=null)
      hidden_max_title_ngrams = Integer.parseInt(tmp);
    
    // get wiki search field
    tmp = req.getParams().get("hwikifield");
    if(tmp!=null)
      hidden_wiki_search_field = tmp;
    
    
    // get number of semantic concepts to retrieve 
    String concept = req.getParams().get(CommonParams.Q);
    int concepts_num = 0;
    if (req.getParams().getInt("conceptsno")!=null){
      concepts_num = req.getParams().getInt("conceptsno");
    }
    
    if(concepts_num>0) {
      if(measure_relatedness==false) {
        semanticConceptsInfo = doSemanticSearch(concept, concepts_num, e_Method, 
            enable_title_search, req);
      }
      else {
        semanticConceptsInfo = doSemanticRelatedness(concept, concepts_num, e_Method, enable_title_search);
        
        // clear search string
        ModifiableSolrParams params = new ModifiableSolrParams(req.getParams());
        params.set(CommonParams.Q, "");
        req.setParams(params);
      }
    }
    
    super.handleRequestBody(req, rsp);
    
    // return back found semantic concepts in response
    rsp.getValues().add("semantic_concepts",semanticConceptsInfo);
  }
  
  protected NamedList<Object> doSemanticSearch(String concept, int concepts_num, 
      ENUM_SEMANTIC_METHOD e_Method, boolean enable_title_search, SolrQueryRequest req) {
    NamedList<Object> semanticConceptsInfo = null;
    HashMap<String,SemanticConcept> relatedConcepts = null;
    
    // retrieve related semantic concepts      
    relatedConcepts = new HashMap<String,SemanticConcept>();
    retrieveRelatedConcepts(concept, relatedConcepts, hidden_max_hits, e_Method, enable_title_search);
    
    // remove irrelevant concepts
    filterRelatedConcepts(relatedConcepts);
    
    if(relatedConcepts.size()>0) {        
      // sort concepts
      SemanticConcept sem[] = new SemanticConcept[relatedConcepts.size()];
      sem = (SemanticConcept[])relatedConcepts.values().toArray(sem);
      Arrays.sort(sem);
      
      // add concepts to query and to response
      String newQuery = concept;
      semanticConceptsInfo = new SimpleOrderedMap<Object>();
      for(int i=0,j=0; i<concepts_num && i<sem.length; j++) {
        // remove a concept that exactly match original concept
        if(hidden_relax_same_title==false && concept.compareToIgnoreCase(sem[j].name)==0) {
          System.out.println(sem[j].name+"...Removed!");
          continue;
        }
        newQuery += " OR \"" + sem[j].name + "\"";
        
        SimpleOrderedMap<Object> conceptInfo = sem[j].getInfo();
        //semanticConceptsInfo.add(sem[j].name, sem[j].weight);
        semanticConceptsInfo.add(sem[j].name, conceptInfo);
        i++;
      }
      
      // add related concepts to the query
      ModifiableSolrParams params = new ModifiableSolrParams(req.getParams());
      if(hidden_relax_search==false)
        params.set(CommonParams.Q, newQuery);
      else
        params.set(CommonParams.Q, "");
      req.setParams(params);
    }
    return semanticConceptsInfo;
  }
  
  protected NamedList<Object> doSemanticRelatedness(String searchConcepts, int concepts_num, 
      ENUM_SEMANTIC_METHOD e_Method, boolean enable_title_search) {
    NamedList<Object> semanticConceptsInfo = null;
    
    if(hidden_relatedness_experiment==true) {
      // open concepts file and loop on concepts
      BufferedReader in;
      try {
        in = new BufferedReader(new FileReader("./concepts.txt"));
        FileWriter out = new FileWriter("./concepts-sim.txt");
        if(in!=null && out!=null) {        
          String line;        
          line = in.readLine();
          while(line!=null) {
            // expected format (concept1,concept2)
            String[] concepts = line.split(",");
            
            // calculating similarity between each two concepts
            HashMap<String,SemanticConcept> relatedConcepts1 = new HashMap<String,SemanticConcept>();
            HashMap<String,SemanticConcept> relatedConcepts2 = new HashMap<String,SemanticConcept>();
            
            double similarity = getRelatedness(concepts[0], concepts[1], 
                concepts_num, e_Method, enable_title_search, 
                relatedConcepts1, relatedConcepts2);
            
            out.write(concepts[0]+","+concepts[1]+","+similarity+"\n");
            
            line = in.readLine();
            }
          in.close();
          out.close();
        }
      } catch (IOException e) {
        throw new RuntimeException();
      }
    }
    else {
      String[] concepts = searchConcepts.split(",");
      if(concepts.length==2) {
        HashMap<String,SemanticConcept> relatedConcepts1 = new HashMap<String,SemanticConcept>();
        HashMap<String,SemanticConcept> relatedConcepts2 = new HashMap<String,SemanticConcept>();
        
        double similarity = getRelatedness(concepts[0], concepts[1], 
            concepts_num, e_Method, enable_title_search, 
            relatedConcepts1, relatedConcepts2);
        
        if(relatedConcepts1.size()>0 || relatedConcepts2.size()>0) {          
          semanticConceptsInfo = new SimpleOrderedMap<Object>();
          
          semanticConceptsInfo.add("similarity", similarity);
        
          // add concepts related to concept 1 to response
          SimpleOrderedMap<Object> conceptInfo1 = new SimpleOrderedMap<Object>();
          int ind=0;
          for(SemanticConcept c : relatedConcepts1.values()) {
            SimpleOrderedMap<Object> conceptInfo = c.getInfo();
            conceptInfo1.add(c.name, conceptInfo);
            ind++;
            if(ind==concepts_num || ind==relatedConcepts1.size())
              break;
          }
          semanticConceptsInfo.add(concepts[0], conceptInfo1);
          
          // add concepts related to concept 2 to response
          SimpleOrderedMap<Object> conceptInfo2 = new SimpleOrderedMap<Object>();
          ind=0;
          for(SemanticConcept c : relatedConcepts2.values()) {
            SimpleOrderedMap<Object> conceptInfo = c.getInfo();
            conceptInfo2.add(c.name, conceptInfo);
            ind++;
            if(ind==concepts_num || ind==relatedConcepts2.size())
              break;
          }
          semanticConceptsInfo.add(concepts[1], conceptInfo2);
        }
      }
      else { // invalid format (expected concept1,concept2)
        
      }
    }
    return semanticConceptsInfo;
  }
  
  protected double getRelatedness(String concept1, String concept2, int concepts_num, 
      ENUM_SEMANTIC_METHOD e_Method, boolean enable_title_search, 
      HashMap<String,SemanticConcept> relatedConcepts1, 
      HashMap<String,SemanticConcept> relatedConcepts2) {
    double similarity = Double.MAX_VALUE;
    
    // retrieve related semantic concepts for concept 1
    System.out.println("Retrieving for concept: ("+concept1+")");
    retrieveRelatedConcepts(concept1, relatedConcepts1, hidden_max_hits, e_Method, enable_title_search);
    
    // remove irrelevant concepts
    filterRelatedConcepts(relatedConcepts1);
    
    // retrieve related semantic concepts for concept 2
    System.out.println("Retrieving for concept: ("+concept2+")");
    retrieveRelatedConcepts(concept2, relatedConcepts2, hidden_max_hits, e_Method, enable_title_search);
    
    // remove irrelevant concepts
    filterRelatedConcepts(relatedConcepts2);
    
    if(relatedConcepts1.size()>0 || relatedConcepts2.size()>0) {
      ArrayList<SemanticConcept> toAdd = new ArrayList<SemanticConcept>(); 
      // add concepts appearing in concept 1 and not concept 2
      for(String s : relatedConcepts1.keySet()) {
        if(relatedConcepts2.get(s)==null) {
          SemanticConcept c = new SemanticConcept(relatedConcepts1.get(s));
          c.weight = 0;
          toAdd.add(c);            
        }
      }
      for(int i=0; i<toAdd.size(); i++)
        relatedConcepts2.put(toAdd.get(i).name, toAdd.get(i));
      
      toAdd.clear();
      
      // add concepts appearing in concept 2 and not concept 1
      for(String s : relatedConcepts2.keySet()) {
        if(relatedConcepts1.get(s)==null) {
          SemanticConcept c = new SemanticConcept(relatedConcepts2.get(s));
          c.weight = 0;
          toAdd.add(c);
        }
      }
      for(int i=0; i<toAdd.size(); i++)
        relatedConcepts1.put(toAdd.get(i).name, toAdd.get(i));
      
      toAdd.clear();
      
      // calculate cosine similarity
      double norm1=0, norm2=0, dot_product=0;
      for(SemanticConcept c1 : relatedConcepts1.values()) {
        SemanticConcept c2 = relatedConcepts2.get(c1.name);
        dot_product += c1.weight*c2.weight;
        norm1 += Math.pow(c1.weight, 2.0);
        norm2 += Math.pow(c2.weight, 2.0);
      }
      similarity = dot_product/(Math.sqrt(norm1)*Math.sqrt(norm2));
    }
    
    return similarity;
  }
  /*
   * @param concept source concept for which we search for related concepts
   * @param relatedConcepts related concepts retrieved
   * @param maxhits maximum hits in the initial wiki search
   * @param e_Method method to use for semantic concept retrieval (ESA,ESA_anchors, ESA_seealso, ESA_anchors_seealso)
   * @param enable_title_search whether we search in wiki titles as well as text or not
   */
  protected void retrieveRelatedConcepts(String concept, HashMap<String,SemanticConcept> relatedConcepts, 
      int max_hits, ENUM_SEMANTIC_METHOD e_Method, boolean enable_title_search) {
    //TODO: 
    /* do we need to intersect with technical dictionary
     * do we need to score see also based on cross-reference/see also graph similarity (e.g., no of common titles in the see also graph)
     * do we need to filter out titles with places, nationality,N_N,N_N_N,Adj_N_N...etc while indexing
     * do we need to look at cross-references
     * do we need to search in title too with boosting factor then remove exact match at the end
     * do we need to add "" here or in the request
     */
    String indexPath = "./wiki_index";
    try {
      // open the index
      IndexReader indexReader = DirectoryReader.open(FSDirectory.open(new File(indexPath)));
      IndexSearcher searcher = new IndexSearcher(indexReader);
      Analyzer stdAnalyzer = new StandardAnalyzer();
      QueryParser parser = new QueryParser(hidden_wiki_search_field, stdAnalyzer);
      Query query = null;
      try {
        
        parser.setAllowLeadingWildcard(true);
        String queryString = "padded_length:["+String.format("%09d", hidden_min_wiki_length)+" TO *]";
        if(enable_title_search) {
          queryString += " AND (title:"+concept+" OR "+hidden_wiki_search_field+":"+concept+")";
        }
        else {
          queryString += " AND ("+hidden_wiki_search_field+":"+concept+")";
        }
        query = parser.parse(queryString);
      } catch (ParseException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      } //
      TopDocs topDocs = searcher.search(query, max_hits);
      if(topDocs.totalHits > 0) {
        ScoreDoc[] hits = topDocs.scoreDocs;
        int cur_id = 1;
        int cur_parent_id = 0;
        for(int i = 0 ; i < hits.length; i++) {
          boolean relevant = false;
          IndexableField multiTitles[] = indexReader.document(hits[i].doc).getFields("title");
          String ner = indexReader.document(hits[i].doc).getField("title_ne").stringValue();
          CachedConceptInfo cachedInfo = null;
          CachedAssociationInfo cachedAssoInfo = null;
          for(int t=0; t<multiTitles.length; t++) {
            IndexableField f = multiTitles[t];
            //System.out.println(f.stringValue());
            // check if relevant concept
            boolean relevantTitle = true;//TODO: do we need to call isRelevantConcept(f.stringValue());
            if(relevantTitle==true) {
              relevant = true;
              
              // check if already there
              SemanticConcept sem = relatedConcepts.get(f.stringValue());
              if(sem==null) { // new concept
                if(t==0) { // first title
                  cachedInfo = cachedConceptsInfo.get(f.stringValue());
                  if(cachedInfo==null) {
                    System.out.println(f.stringValue()+"...title not found!");
                    cachedInfo = new CachedConceptInfo(f.stringValue(), "", "");
                  }
                  
                  sem = new SemanticConcept(f.stringValue(), cachedInfo, ner, hits[i].score,
                      cur_id, 0, 0, ENUM_CONCEPT_TYPE.e_TITLE);
                  cur_parent_id = cur_id;
                  
                  // get its associations
                  Integer I = titleIntMapping.get(f.stringValue());
                  if(I!=null) {
                    cachedAssoInfo = cachedAssociationsInfo.get(I);
                  }
                  else
                    System.out.println(f.stringValue()+"...title not in mappings!");
                }
                else { // anchor 
                  sem = new SemanticConcept(f.stringValue(), cachedInfo, ner, hits[i].score, 
                    cur_id, cur_parent_id, 0, ENUM_CONCEPT_TYPE.e_ANCHOR);
                }
                cur_id++;
              }
              else { // existing concept, update its weight to higher weight
                cachedInfo = sem.cachedInfo;
                sem.weight = sem.weight>hits[i].score?sem.weight:hits[i].score;
              }
              relatedConcepts.put(sem.name, sem);
              if(e_Method==ENUM_SEMANTIC_METHOD.e_UNKNOWN || 
                  (e_Method!=ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS && 
                  e_Method!=ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS_SEE_ALSO && 
                  e_Method!=ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS_SEE_ALSO_ASSO)) // only one title is retrieved, we don't use anchors
                break;
            }
            else
              System.out.println(f.stringValue()+"...title not relevant!");
          }  
          //System.out.println();
          if(hidden_relax_see_also==false || relevant) {
            // force see also is enabled OR,
            // the original title or one of its anchors is relevant
            // in this case we can add its see_also
            if(e_Method==ENUM_SEMANTIC_METHOD.e_ESA_SEE_ALSO || 
                e_Method==ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS_SEE_ALSO) { // add See also to the hit list
              
              IndexableField[] multiSeeAlso = indexReader.document(hits[i].doc).getFields("see_also");
              IndexableField[] multiSeeAlsoNE = indexReader.document(hits[i].doc).getFields("see_also_ne");
            
              for(int s=0; s<multiSeeAlso.length; s++) {
                //System.out.println(f.stringValue());
                // check if relevant concept
                boolean relevantTitle = true; //TODO: do we need to call isRelevantConcept(multiSeeAlso[s].stringValue());
                if(relevantTitle==true) {
                  // check if already there
                  SemanticConcept sem = relatedConcepts.get(multiSeeAlso[s].stringValue()); 
                  if(sem==null) { // new concept
                    cachedInfo = cachedConceptsInfo.get(multiSeeAlso[s].stringValue());
                    if(cachedInfo==null) {
                      System.out.println(multiSeeAlso[s].stringValue()+"...see_also not found!");
                      cachedInfo = new CachedConceptInfo(multiSeeAlso[s].stringValue(), "", "");
                    }
                    
                    // get see also association info
                    int asso_cnt = 0;
                    if(cachedAssoInfo!=null) {
                      Integer I = titleIntMapping.get(multiSeeAlso[s].stringValue());
                      if(I!=null) {
                        for(Integer[] info :cachedAssoInfo.associations) {
                          if(info[0].intValue()==I.intValue()) {
                            asso_cnt = info[1].intValue();
                            break;
                          }
                        }
                        if(asso_cnt==0)
                          System.out.println(multiSeeAlso[s].stringValue()+"...see_also not in associations!");
                      }
                      else
                        System.out.println(multiSeeAlso[s].stringValue()+"...see_also not in mappings!");
                    }
                    if(asso_cnt==0 || asso_cnt>=hidden_min_asso_cnt) { // support > minimum support
                      sem = new SemanticConcept(multiSeeAlso[s].stringValue(), 
                          cachedInfo, multiSeeAlsoNE[s].stringValue(), 
                          hits[i].score, cur_id, cur_parent_id, asso_cnt, ENUM_CONCEPT_TYPE.e_SEE_ALSO);
                      cur_id++;
                    }
                  }
                  else { // existing concept, update its weight to higher weight
                    cachedInfo = sem.cachedInfo;
                    sem.weight = sem.weight>hits[i].score?sem.weight:hits[i].score;
                  }
                  if(sem!=null)
                    relatedConcepts.put(sem.name, sem);            
                }
                else
                  System.out.println(multiSeeAlso[s].stringValue()+"...see-also not relevant!");
                //System.out.println();
              }              
            }
            else if(e_Method==ENUM_SEMANTIC_METHOD.e_ESA_SEE_ALSO_ASSO || 
                e_Method==ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS_SEE_ALSO_ASSO) { // add see also using association mining
              Integer key = titleIntMapping.get(indexReader.document(hits[i].doc).getFields("title")[0].stringValue());
              CachedAssociationInfo assoInfo = cachedAssociationsInfo.get(key);
              if(assoInfo==null) {
                System.out.println(indexReader.document(hits[i].doc).getFields("title")[0].stringValue() + "..no associations cached!");
              }
              else {
                for(int a=0; a<assoInfo.associations.size(); a++) {
                  Integer[] assos = assoInfo.associations.get(a);
                  if(assos[1]>=hidden_min_asso_cnt) // support > minimum support
                  {
                    String assoStr = titleStrMapping.get(assos[0]);
                    
                    // check if relevant concept
                    boolean relevantTitle = true; //TODO: do we need to call isRelevantConcept(multiSeeAlso[s].stringValue());
                    if(relevantTitle==true) {
                      // check if already there
                      SemanticConcept sem = relatedConcepts.get(assoStr); 
                      if(sem==null) { // new concept
                        cachedInfo = cachedConceptsInfo.get(assoStr);
                        if(cachedInfo==null) {
                          System.out.println(assoStr+"...see_also not found!");
                          cachedInfo = new CachedConceptInfo(assoStr, "", "");
                        }
                        
                        sem = new SemanticConcept(assoStr, 
                            cachedInfo, "M", 
                            hits[i].score, cur_id, cur_parent_id, assos[1], ENUM_CONCEPT_TYPE.e_SEE_ALSO);
                        cur_id++;
                      }
                      else { // existing concept, update its weight to higher weight
                        cachedInfo = sem.cachedInfo;
                        sem.weight = sem.weight>hits[i].score?sem.weight:hits[i].score;
                      }
                      relatedConcepts.put(sem.name, sem);
                    }
                    else
                      System.out.println(assoStr+"...see-also not relevant!");
                  }
                  else
                    System.out.println(assos[0]+"...see-also below threshold!");
                }
              }
            }
          }
        }
      }
      else 
        System.out.println("No semantic results found :(");
        
    } catch (IOException e) {
      // TODO Auto-generated catch block
      
      e.printStackTrace();
    }
  }
  
  /*
   * remove concepts that are irrelevant (only 1-2-3 word phrases are allowed)
   * @param relatedConcepts related concepts to be filtered
   */
  protected void filterRelatedConcepts(HashMap<String,SemanticConcept> relatedConcepts) {
    ArrayList<String> toRemove = new ArrayList<String>();
    for (SemanticConcept concept : relatedConcepts.values()) {
      if(isRelevantConcept(concept.name)==false) {
        System.out.println(concept.name+"("+concept.id+")...Removed!");
        toRemove.add(concept.name);
      }
      else if(hidden_relax_ner==false && ArrayUtils.contains(new String[]{"P","L","O"},concept.ner)) { // check if not allowed NE
        System.out.println(concept.name+"("+concept.id+")...Removed (NER)!");
        toRemove.add(concept.name);
      }
      else if(hidden_relax_categories==false) { // check if not allowed category
        for(int i=0; i<concept.cachedInfo.category.length; i++) {
          String c = concept.cachedInfo.category[i].toLowerCase();
          if(c.contains("companies") || 
              c.contains("manufacturers") || 
              c.contains("publishers") || 
              c.contains("births") || 
              c.contains("deaths") || 
              c.contains("anime") || 
              c.contains("movies") || 
              c.contains("theaters") || 
              c.contains("games") ||
              c.contains("channels")) {
            System.out.println(concept.name+"("+concept.id+")...Removed (CATEGORY)!");
            toRemove.add(concept.name);
            break;
          }
        }
      }      
    }
    
    for(String concept : toRemove) {
      relatedConcepts.remove(concept);
    }
  }
  
  /*
   * remove concepts that are irrelevant (only 1-2-3 word phrases are allowed)
   * @param concept concept to be evaluated
   */
  protected boolean isRelevantConcept(String concept) {
    //TODO: 
    /* handle list of: (List of solar powered products),(List of types of solar cells)
     * handle File: (File:Jason Robinson.jpg)
     * handle names, places, adjectives
     * remove titles with (disambiguation)
     * keep only titles in technical dictionary
     */
    String re = "\\S+(\\s\\S+){0,"+String.valueOf(hidden_max_title_ngrams-1)+"}";
    boolean relevant = true;
    if(concept.toLowerCase().matches(re)==false) {
      relevant = false;
    }
    if(hidden_relax_listof==false) {
      re = "list of.*";
      if(concept.toLowerCase().matches(re)==true) {
        relevant = false;
      }
    }
    if(hidden_relax_disambig==false) {
      re = ".*\\(disambiguation\\)";
      if(concept.toLowerCase().matches(re)==true) {
        relevant = false;
      }    
    }
    return relevant; 
  }
  
  private ENUM_SEMANTIC_METHOD getSemanticMethod(String conceptMethod) {
    if(conceptMethod.compareTo("ESA")==0)
      return ENUM_SEMANTIC_METHOD.e_ESA;
    else if (conceptMethod.compareTo("ESA_anchors")==0)
      return ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS;
    else if (conceptMethod.compareTo("ESA_seealso")==0)
      return ENUM_SEMANTIC_METHOD.e_ESA_SEE_ALSO;
    else if (conceptMethod.compareTo("ESA_anchors_seealso")==0)
      return ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS_SEE_ALSO;
    else if (conceptMethod.compareTo("ESA_seealso_asso")==0)
      return ENUM_SEMANTIC_METHOD.e_ESA_SEE_ALSO_ASSO;
    else if (conceptMethod.compareTo("ESA_anchors_seealso_asso")==0)
      return ENUM_SEMANTIC_METHOD.e_ESA_ANCHORS_SEE_ALSO_ASSO;    
    else return ENUM_SEMANTIC_METHOD.e_UNKNOWN;
  }
  
  /**
   * cache Wiki titles with some required information for fast retrieval
   */
  private void cacheConceptsInfo() {
    String indexPath = "./wiki_index";//reader.nextLine();
    try {
      // open the index
      IndexReader indexReader = DirectoryReader.open(FSDirectory.open(new File(indexPath)));
      IndexSearcher searcher = new IndexSearcher(indexReader);
      Analyzer stdAnalyzer = new StandardAnalyzer();
      QueryParser parser = new QueryParser("title", stdAnalyzer);
      Query query = null;
      try {
        
        parser.setAllowLeadingWildcard(true);
        String queryString;
        queryString = "*";
        
        query = parser.parse(queryString);
      } catch (ParseException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      } //
      TopDocs topDocs = searcher.search(query, Integer.MAX_VALUE);
      if(topDocs.totalHits > 0) {
        cachedConceptsInfo = new HashMap<String,CachedConceptInfo>();
        String title = "";
        String cat1 = "";
        String cat2 = "";
        IndexableField[] multi = null;
        CachedConceptInfo cachedInfo = null;
        for(ScoreDoc d : topDocs.scoreDocs) {
          // retrieve title
          title = indexReader.document(d.doc).getFields("title")[0].stringValue();
          
          // retrieve categories
          multi = indexReader.document(d.doc).getFields("category");
          cat1 = multi.length>0? multi[0].stringValue():"";
          cat2 = multi.length>1? multi[1].stringValue():"";
          
          cachedInfo = new CachedConceptInfo(title, cat1, cat2);
          cachedConceptsInfo.put(title,  cachedInfo);
        }
        System.out.println("cached ("+cachedConceptsInfo.size()+") concepts.");
      }
    } catch (IOException e) {
      // TODO Auto-generated catch block
      
      e.printStackTrace();
    }
  }
  
  // cache all title associations into memory for fast access
  private void cacheAssociationsInfo() {
    try {
      BufferedReader f;      
      f = new BufferedReader(new FileReader("./wiki_associations.txt"));
      
      int curassoc=0, key=0, idx, count;
      String line;
      CachedAssociationInfo associationInfo = null;
      
      titleIntMapping = new HashMap<String,Integer>(4000000);
      titleStrMapping = new HashMap<Integer,String>(4000000);
      
      cachedAssociationsInfo = new HashMap<Integer,CachedAssociationInfo>(4000000);
      
      line = f.readLine();
      String[] association = new String[2];
      while(line!=null) {
        String[] associations = line.split("#\\$#");
        for(int i=0; i<associations.length; i++) {
          
          association = associations[i].split("/\\\\/\\\\");
          
          // look it up
          Integer index = titleIntMapping.get(association[0]);
          if(index==null) { // add it
            titleIntMapping.put(association[0], new Integer(curassoc));
            titleStrMapping.put(new Integer(curassoc), association[0]);
            idx = curassoc;
            curassoc++;
          }
          else {
            idx = index.intValue();
          }
          if(i==0) {
            associationInfo = new CachedAssociationInfo();
            key = idx;
          }
          else { // add it to associations
            associationInfo.addAssociation(idx,Integer.parseInt(association[1]));
          }
        }
        if(associations.length>0)
          cachedAssociationsInfo.put(new Integer(key), associationInfo);
        
        line = f.readLine();
      }
      f.close();
      
    } catch (IOException e) {
        throw new RuntimeException();
    }    
  }
}

