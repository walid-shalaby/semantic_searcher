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

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import jdk.nashorn.internal.ir.LiteralNode.ArrayLiteralNode.ArrayUnit;

import org.apache.commons.lang.ArrayUtils;
import org.apache.derby.iapi.services.io.ArrayUtil;
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

enum ENUM_SEMANTIC_METHODS {
  e_UNKNOWN, 
  e_ESA,
  e_ESA_ANCHORS,
  e_ESA_SEE_ALSO,
  e_ESA_ANCHORS_SEE_ALSO
}

class SemanticConcept implements Comparable<SemanticConcept> {
  public String name;
  public String ner; // NE label ("P" --> person, "o" --> organization, "L" --> location, "M" --> misc)
  public String[] category; // first two categories of the title 
  public float weight;  
  
  public SemanticConcept() {
    name = "";
    ner = "";
    weight = 0.0f;
  }
  
  public SemanticConcept(String n, String ne, String cat1, String cat2, float w) {
    name = n;
    ner = ne;
    weight = w;
    category = new String[2];
    category[0] = cat1;
    category[1] = cat2;
  }

  @Override
  public int compareTo(SemanticConcept o) {
    if (((SemanticConcept)o).weight>this.weight)
      return 1;
    else if (((SemanticConcept)o).weight<this.weight)
      return -1;
    else return 0;
    
  }
}

public class SemanticSearchHandler extends SearchHandler
{
  
  private boolean hidden_relax_see_also = false;
  
  private boolean hidden_relax_ner = false;
  
  private boolean hidden_relax_categories = false;
  
  private boolean hidden_display_wiki_hits = false;

  private int hidden_max_hits = 0;
  
  private int hidden_max_title_ngrams = 3;
  
  @Override
  public void handleRequestBody(SolrQueryRequest req, SolrQueryResponse rsp) throws Exception
  {
    HashMap<String,SemanticConcept> relatedConcepts = null;
    NamedList<Object> semanticConceptsInfo = null;
    ENUM_SEMANTIC_METHODS e_Method = ENUM_SEMANTIC_METHODS.e_UNKNOWN;
    boolean enable_title_search = false;
    
    // get method of semantic concepts retrievals
    String tmp = req.getParams().get("conceptsmethod");
    if(tmp!=null)
      e_Method = getSemanticMethod(tmp);
    
    // get enable title search flag 
    tmp = req.getParams().get("titlesearch");
    if(tmp!=null && tmp.compareTo("on")==0)
      enable_title_search = true;
    
    // get force see also flag 
    tmp = req.getParams().get("hrelaxseealso");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_see_also = true;

    // get relax NER flag 
    tmp = req.getParams().get("hrelaxner");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_ner = true;
    
    // get relax categories flag 
    tmp = req.getParams().get("hrelaxcategories");
    if(tmp!=null && tmp.compareTo("y")==0)
      hidden_relax_categories = true;

    // get maximum hits in initial wiki search
    tmp = req.getParams().get("hmaxhits");
    if(tmp!=null)
      hidden_max_hits = Integer.parseInt(tmp);
    else
      hidden_max_hits = Integer.MAX_VALUE;
    
    // get maximum ngrams of wiki titles
    tmp = req.getParams().get("hmaxngrams");
    if(tmp!=null)
      hidden_max_title_ngrams = Integer.parseInt(tmp);
    
    // get number of semantic concepts to retrieve 
    String concept = req.getParams().get(CommonParams.Q);
    int concepts_num = 0;
    if (req.getParams().getInt("conceptsno")!=null){
      concepts_num = req.getParams().getInt("conceptsno");
    }
    
    if(concepts_num>0) {
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
          if(concept.compareToIgnoreCase(sem[j].name)==0) {
            System.out.println(sem[j].name+"...Removed!");
            continue;
          }
          newQuery += " OR \"" + sem[j].name + "\"";
          semanticConceptsInfo.add(sem[j].name, sem[j].weight);
          i++;
        }
        
        // add related concepts to the query
        ModifiableSolrParams params = new ModifiableSolrParams(req.getParams());
        params.set(CommonParams.Q, newQuery);
        req.setParams(params);
      }      
    }
    
    super.handleRequestBody(req, rsp);
    
    // return back found semantic concepts in response
    rsp.getValues().add("semantic_concepts",semanticConceptsInfo);
  }
  
  /*
   * @param concept source concept for which we search for related concepts
   * @param relatedConcepts related concepts retrieved
   * @param maxhits maximum hits in the initial wiki search
   * @param e_Method method to use for semantic concept retrieval (ESA,ESA_anchors, ESA_seealso, ESA_anchors_seealso)
   * @param enable_title_search whether we search in wiki titles as well as text or not
   */
  protected void retrieveRelatedConcepts(String concept, HashMap<String,SemanticConcept> relatedConcepts, 
      int max_hits, ENUM_SEMANTIC_METHODS e_Method, boolean enable_title_search) {
    //TODO: 
    /* do we need to intersect with technical dictionary
     * do we need to score see also based on cross-reference/see also graph similarity (e.g., no of common titles in the see also graph)
     * do we need to filter out titles with places, nationality,N_N,N_N_N,Adj_N_N...etc while indexing
     * do we need to look at cross-references
     * do we need to search in title too with boosting factor then remove exact match at the end
     * do we need to add "" here or in the request
     */
    String indexPath = "./wiki_index";//reader.nextLine();
    try {
      // open the index
      IndexReader indexReader = DirectoryReader.open(FSDirectory.open(new File(indexPath)));
      IndexSearcher searcher = new IndexSearcher(indexReader);
      Analyzer stdAnalyzer = new StandardAnalyzer();
      QueryParser parser = new QueryParser("text", stdAnalyzer);
      Query query = null;
      try {
        
        parser.setAllowLeadingWildcard(true);
        String queryString;
        if(enable_title_search) {
          queryString = "title:"+concept+" OR text:"+concept;
        }
        else {
          queryString = "text:"+concept;
        }
        query = parser.parse(queryString);
      } catch (ParseException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      } //
      TopDocs topDocs = searcher.search(query, max_hits);
      if(topDocs.totalHits > 0) {
        ScoreDoc[] hits = topDocs.scoreDocs;
        for(int i = 0 ; i < hits.length; i++) {
          boolean relevant = false;
          IndexableField multiTitles[] = indexReader.document(hits[i].doc).getFields("title");
          String ner = indexReader.document(hits[i].doc).getField("title_ne").stringValue();
          
          // get category information
          IndexableField multiCategories[] = indexReader.document(hits[i].doc).getFields("category");
          String cat1 = multiCategories.length>0? multiCategories[0].stringValue():"";
          String cat2 = multiCategories.length>1? multiCategories[1].stringValue():"";
          
          for(IndexableField f : multiTitles) {
            //System.out.println(f.stringValue());
            // check if relevant concept
            boolean relevantTitle = true;//TODO: do we need to call isRelevantConcept(f.stringValue());
            if(relevantTitle==true) {
              relevant = true;
              
              // check if already there
              SemanticConcept sem = relatedConcepts.get(f.stringValue()); 
              if(sem==null) { // new concept
                sem = new SemanticConcept(f.stringValue(), ner, cat1, cat2, hits[i].score);
              }
              else { // existing concept, update its weight to higher weight
                sem.weight = sem.weight>hits[i].score?sem.weight:hits[i].score;
              }
              relatedConcepts.put(sem.name, sem);
              if(e_Method==ENUM_SEMANTIC_METHODS.e_UNKNOWN || 
                  (e_Method!=ENUM_SEMANTIC_METHODS.e_ESA_ANCHORS && 
                  e_Method!=ENUM_SEMANTIC_METHODS.e_ESA_ANCHORS_SEE_ALSO)) // only one title is retrieved, we don't use anchors
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
            if(e_Method==ENUM_SEMANTIC_METHODS.e_ESA_SEE_ALSO || 
                e_Method==ENUM_SEMANTIC_METHODS.e_ESA_ANCHORS_SEE_ALSO) { // add See also to the hit list
              
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
                    sem = new SemanticConcept(multiSeeAlso[s].stringValue(), 
                        multiSeeAlsoNE[s].stringValue(), "", "", hits[i].score);
                  }
                  else { // existing concept, update its weight to higher weight
                    sem.weight = sem.weight>hits[i].score?sem.weight:hits[i].score;
                  }
                  relatedConcepts.put(sem.name, sem);            
                }
                else
                  System.out.println(multiSeeAlso[s].stringValue()+"...see-also not relevant!");
                //System.out.println();
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
        System.out.println(concept.name+"...Removed!");
        toRemove.add(concept.name);
      }
      else if(hidden_relax_ner==false && ArrayUtils.contains(new String[]{"P","L","O"},concept.ner)) { // check if not allowed NE
        System.out.println(concept.name+"...Removed (NER)!");
        toRemove.add(concept.name);
      }
      else if(hidden_relax_categories==false) { // check if not allowed category
        for(int i=0; i<concept.category.length; i++) {
          if(concept.category[i].contains("companies") || 
              concept.category[i].contains("manufacturers") || 
              concept.category[i].contains("publishers") || 
              concept.category[i].contains("births") || 
              concept.category[i].contains("deaths") || 
              concept.category[i].contains("anime") || 
              concept.category[i].contains("movies") || 
              concept.category[i].contains("theaters") || 
              concept.category[i].contains("games") ||
              concept.category[i].contains("channels")) {
            System.out.println(concept.name+"...Removed (CATEGORY)!");
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
    re = "list of.*";
    if(concept.toLowerCase().matches(re)==true) {
      relevant = false;
    }
    re = ".*\\(disambiguation\\)";
    if(concept.toLowerCase().matches(re)==true) {
      relevant = false;
    }    
    return relevant; 
  }
  
  private ENUM_SEMANTIC_METHODS getSemanticMethod(String conceptMethod) {
    if(conceptMethod.compareTo("ESA")==0)
      return ENUM_SEMANTIC_METHODS.e_ESA;
    else if (conceptMethod.compareTo("ESA_anchors")==0)
      return ENUM_SEMANTIC_METHODS.e_ESA_ANCHORS;
    else if (conceptMethod.compareTo("ESA_seealso")==0)
      return ENUM_SEMANTIC_METHODS.e_ESA_SEE_ALSO;
    else if (conceptMethod.compareTo("ESA_anchors_seealso")==0)
      return ENUM_SEMANTIC_METHODS.e_ESA_ANCHORS_SEE_ALSO;
    else return ENUM_SEMANTIC_METHODS.e_UNKNOWN;
  }  
}

