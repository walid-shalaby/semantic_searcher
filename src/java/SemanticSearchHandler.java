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
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringEscapeUtils;
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
import org.apache.solr.client.solrj.impl.HttpSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import org.apache.solr.common.params.CommonParams;
import org.apache.solr.common.params.ModifiableSolrParams;
import org.apache.solr.common.util.NamedList;
import org.apache.solr.common.util.SimpleOrderedMap;
import org.apache.solr.core.PluginInfo;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.response.SolrQueryResponse;
import org.apache.solr.search.SyntaxError;

/**
 *
 * Refer SOLR-281
 *
 */

enum ENUM_SEMANTIC_METHOD {
  e_UNKNOWN, 
  e_MSA,
  e_MSA_ANCHORS,
  e_MSA_SEE_ALSO,
  e_MSA_SEE_ALSO_ASSO,
  e_MSA_ANCHORS_SEE_ALSO, 
  e_MSA_ANCHORS_SEE_ALSO_ASSO
}

enum ENUM_DISTANCE_METRIC {
  e_COSINE, 
  e_COSINE_BIN,
  e_COSINE_NORM,
  e_Eucleadian,
  e_WO
}

enum ENUM_CONCEPT_TYPE {
  e_UNKNOWN, 
  e_TITLE,
  e_ANCHOR,
  e_SEE_ALSO,
  e_ASSOCIATION
}

enum ENUM_ANALYTIC_TYPE {
  e_UNKNOWN, 
  e_Search,
  e_TECH_EXPLORE,
  e_TECH_LANDSCAPE,
  e_CI,
  e_PRIOR,
  e_RELATEDNESS
}

class CachedConceptInfo {
  public String[] category; // first two categories of the title 
  public String docno;
  public int length;

  public CachedConceptInfo() {
  }
  
  public CachedConceptInfo(String title, int len, String no, String cat1, String cat2) {
    docno = no;
    length = len;
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
  public int ignore = 0;
    
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
    if (((SemanticConcept)o).weight<this.weight)
      return 1;
    else if (((SemanticConcept)o).weight>this.weight)
      return -1;
    else {
       if (this.asso_cnt==0) {
         if(((SemanticConcept)o).asso_cnt==0)
           return 0;
         else 
           return 1;
       }
       else if (((SemanticConcept)o).asso_cnt==0)
        return -1;
      else if  (((SemanticConcept)o).asso_cnt<this.asso_cnt)
        return 1;
      else if (((SemanticConcept)o).asso_cnt>this.asso_cnt)
        return -1;
      else 
        return 0;
    }
  }

  public SimpleOrderedMap<Object> getInfo() {
    SimpleOrderedMap<Object> conceptInfo = new SimpleOrderedMap<Object>();
    conceptInfo.add("weight", weight);
    conceptInfo.add("id", id);
    conceptInfo.add("p_id", parent_id);
    conceptInfo.add("asso_cnt", asso_cnt);
    conceptInfo.add("type", getConceptTypeString(e_concept_type));
    conceptInfo.add("docno", cachedInfo.docno);
    conceptInfo.add("length", cachedInfo.length);
    conceptInfo.add("ignore", ignore);
    
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

class ConfigParams {
  public ENUM_SEMANTIC_METHOD e_Method = ENUM_SEMANTIC_METHOD.e_MSA_SEE_ALSO;
  public ENUM_DISTANCE_METRIC e_Distance = ENUM_DISTANCE_METRIC.e_COSINE;
  public ENUM_ANALYTIC_TYPE e_analytic_type = ENUM_ANALYTIC_TYPE.e_UNKNOWN;
  public boolean enable_title_search = false;
  public boolean enable_search_all = false;
  public boolean enable_show_records = false;
  public boolean enable_search_title = false;
  public boolean enable_search_abstract = false;
  public boolean enable_search_description = false;
  public boolean enable_search_claims = false;
  public boolean enable_extract_all = false;
  public boolean enable_extract_title = false;
  public boolean enable_extract_abstract = false;
  public boolean enable_extract_description = false;
  public boolean enable_extract_claims = false;
  public boolean measure_relatedness = false;
  public boolean hidden_relax_see_also = false;
  public boolean hidden_relax_ner = false;
  public boolean hidden_relax_categories = false;
  public boolean hidden_relax_search = true;
  public boolean hidden_include_q = false;
  public boolean hidden_relax_cache = true;
  public boolean hidden_relax_disambig = false;
  public boolean hidden_relax_listof = false;
  public boolean hidden_relax_same_title = false;
  public boolean hidden_relatedness_experiment = false;
  public boolean abs_explicit = false;
  public int hidden_max_hits = 1000;
  public int hidden_min_wiki_length = 0;
  public int hidden_min_seealso_length = 0;
  public int hidden_min_asso_cnt = 1;
  public int hidden_max_title_ngrams = 3;
  public int hidden_max_seealso_ngrams = 3;
  public int concepts_num = 10;
  public int ci_patents_num = 10;
  public String hidden_wiki_search_field = "text";
  public String hidden_wiki_extra_query = "AND NOT title:list* AND NOT title:index* AND NOT title:*disambiguation*";
  public String experiment_in_path = "";
  public String experiment_out_path = "";
  public HashMap<String,ArrayList<String>> fqs = null;
  public HashSet<String> ignored_concepts = null;
  
  public ConfigParams() {
    
  }
}

public class SemanticSearchHandler extends SearchHandler
{
  private HashMap<String,CachedConceptInfo> cachedConceptsInfo = null;
  
  private HashMap<String,Integer> titleIntMapping = null;
  private HashMap<Integer,String> titleStrMapping = null;
  
  private HashMap<Integer,CachedAssociationInfo> cachedAssociationsInfo = null;
  
  IndexReader indexReader;
  
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
    
    // open the index
    String indexPath = "./wiki_index";
    try {
      indexReader = DirectoryReader.open(FSDirectory.open(new File(indexPath)));
    } catch (IOException e) {
      e.printStackTrace();
      throw new RuntimeException();
    }
  }

  @Override
  public void handleRequestBody(SolrQueryRequest req, SolrQueryResponse rsp) throws Exception
  {
    NamedList<Object> semanticConceptsInfo = null;
    
    String escapedQ = StringEscapeUtils.escapeJavaScript(StringEscapeUtils.escapeHtml(req.getParams().get(CommonParams.Q)));
    
    if(escapedQ!=null && escapedQ.length()>0)
    {
      String tmp = req.getParams().get("analytic");
      String tmp1 = StringEscapeUtils.escapeJavaScript(StringEscapeUtils.escapeHtml(req.getParams().get("q1")));
      if(tmp!=null && tmp.length()>0 && tmp.compareToIgnoreCase("relatedness")==0 && tmp1!=null && tmp1.length()>0)
        escapedQ += "___"+tmp1;
        
      // log concept to file
      FileWriter f = new FileWriter("./msa_log/"+escapedQ.substring(0,escapedQ.length()>255?255:escapedQ.length()));
      f.close();
     
      ConfigParams params = new ConfigParams(); 
      
      ModifiableSolrParams qparams = new ModifiableSolrParams(req.getParams());
      
      // parse request
      ParseRequestParams(req, params, qparams);
      
      //req.setParams(qparams);
      
      //if(params.hidden_relax_search==false) {
      //  super.handleRequestBody(req, rsp);
      //}

      if(params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_Search) {
        super.handleRequestBody(req, rsp);
      }
      else if(params.concepts_num>0 && 
          (params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_TECH_EXPLORE || 
          params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_TECH_LANDSCAPE || 
          params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_PRIOR)) {
        String orgQ = req.getParams().get(CommonParams.Q);
        semanticConceptsInfo = doSemanticSearch(orgQ, params);
        if(params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_TECH_EXPLORE) {
          
        }
        if(params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_TECH_LANDSCAPE) {
          // construct new query with concepts added to it
          String newQuery;
          if(params.hidden_include_q==true)
            newQuery = getNewQuery(req.getParams().get(CommonParams.Q), semanticConceptsInfo);
          else
            newQuery = getNewQuery("", semanticConceptsInfo);
          
          fillTechLandscapeQParams(newQuery, new String[]{"{!ex=fqs}class", "{!ex=fqs}class_year"}, params, qparams);
          
          req.setParams(qparams);
          super.handleRequestBody(req, rsp);
          qparams.set(CommonParams.Q, orgQ);
          qparams.remove("fq");
        }
        if(params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_PRIOR) {
          // construct new query with concepts added to it
          String newQuery;
          if(params.hidden_include_q==true)
            newQuery = getNewQuery(req.getParams().get(CommonParams.Q), semanticConceptsInfo);
          else
            newQuery = getNewQuery("", semanticConceptsInfo);
          
          qparams.set(CommonParams.Q, newQuery);
          qparams.set("df", getQuerySearchFields(params));
          
          req.setParams(qparams);
          super.handleRequestBody(req, rsp);
          qparams.set(CommonParams.Q, orgQ);
          qparams.remove("fq");
        }
      }
      else if(params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_CI) {
        String orgQ = req.getParams().get(CommonParams.Q);
        
        // construct a new query of given assignee innovations and retrieve them
        HttpSolrServer server = new HttpSolrServer("http://localhost:8990/solr/collection1/");
        ModifiableSolrParams innoQParams = new ModifiableSolrParams();
        innoQParams.set(CommonParams.Q, orgQ.toLowerCase());
        innoQParams.set("defType","edismax");
        innoQParams.set("df", "assignee");
        innoQParams.set(CommonParams.ROWS, params.ci_patents_num);
        innoQParams.set("sort", "publication_date desc");
        
        // add target extract field
        String[] extractFields = getQueryExtractFields(params);
        innoQParams.set("fl", extractFields);
        QueryResponse innoQResp = server.query(innoQParams);
        
        // loop on results and construct a new query to explore technologies
        SolrDocumentList results = innoQResp.getResults();
        if(results!=null && results.size()>0) {
          String techQ = "";
          for(int i=0; i<results.size(); i++) {
            SolrDocument doc = results.get(i);
            for(String field : extractFields) {
              techQ += (String)doc.getFieldValue(field) + " ";
            }              
          }
          semanticConceptsInfo = doSemanticSearch(techQ, params);
          System.out.println(techQ);
          
          // construct new query with concepts added to it
          String newQuery;
          if(params.hidden_include_q==true)
            newQuery = getNewQuery(orgQ, semanticConceptsInfo);
          else
            newQuery = getNewQuery("", semanticConceptsInfo);
                    
          fillTechLandscapeQParams(newQuery, new String[]{"{!ex=fqs}class", "{!ex=fqs}class_year", "{!ex=fqs}assignee"}, params, qparams);         
          
          req.setParams(qparams);
          super.handleRequestBody(req, rsp);
          qparams.set(CommonParams.Q, orgQ);
          qparams.remove("fq");
        }
      }
      else if(params.e_analytic_type==ENUM_ANALYTIC_TYPE.e_RELATEDNESS) {
        semanticConceptsInfo = doSemanticRelatedness(req.getParams().get(CommonParams.Q), 
              req.getParams().get("q1"), params);
        }

      // clear search string
      //if(params.hidden_relax_search==true || params.measure_relatedness==true)
      //  qparams.set(CommonParams.ROWS, 0);
      
      // return back found semantic concepts in response
      if(semanticConceptsInfo!=null)
        rsp.getValues().add("semantic_concepts",semanticConceptsInfo);
    }
    
  }
  
  private void fillTechLandscapeQParams(String newQuery, String[] facets, ConfigParams params,
      ModifiableSolrParams qparams) {
    qparams.set(CommonParams.Q, newQuery);
    if(params.enable_show_records==false)
      qparams.set(CommonParams.ROWS, 0);
    qparams.set("facet", true);
    qparams.set("facet.field", facets);
    if(params.fqs!=null) {
      for(String fq : params.fqs.keySet()) {
        ArrayList<String> fqvalues = params.fqs.get(fq);
        String values = fq+":"+fqvalues.get(0);
        for(int i=1; i<fqvalues.size();i++) {
          values += " OR "+fq+":"+fqvalues.get(i);
        }
        qparams.add("fq", "{!tag=fqs}"+values);
      }
    }
    //qparams.set("facet.range", "publication_date");
    //qparams.set("f.publication_date.facet.range.start", "1975-01-01T00:00:00Z/YEAR");
    //qparams.set("f.publication_date.facet.range.end", "NOW/YEAR");
    //qparams.set("f.publication_date.facet.range.gap", "+1YEAR");
    
    qparams.set("df", getQuerySearchFields(params));
  }

  private String getNewQuery(String org,
      NamedList<Object> semanticConceptsInfo) {
    String newQuery = org;
    if(semanticConceptsInfo!=null){ 
      for(int i=0; i<semanticConceptsInfo.size(); i++) {
        SimpleOrderedMap<Object> obj = (SimpleOrderedMap<Object>)semanticConceptsInfo.getVal(i);
        Integer ignore = (Integer)obj.get("ignore");
         if(ignore.intValue()==0) { // shouldn't be ignored
          if(newQuery.isEmpty())
            newQuery = "\"" + semanticConceptsInfo.getName(i) + "\"";
          else
            newQuery += " OR \"" + semanticConceptsInfo.getName(i) + "\"";
          }
      }
    }
    return newQuery;
  }

  private String[] getQuerySearchFields(ConfigParams params) {
    ArrayList<String> qf = new ArrayList<String>();
    if(params.enable_search_all)
      qf.add("text");
    if(params.enable_search_title)
      qf.add("title");
    if(params.enable_search_abstract)
      qf.add("abstract");
    if(params.enable_search_description)
      qf.add("description");
    if(params.enable_search_claims)
      qf.add("claims");
    
    return qf.toArray(new String[qf.size()]);
  }

  private String[] getQueryExtractFields(ConfigParams params) {
    ArrayList<String> qf = new ArrayList<String>();
    if(params.enable_extract_all)
      qf.add("text");          
    if(params.enable_extract_title)
      qf.add("title");
    if(params.enable_extract_abstract)
      qf.add("abstract");
    if(params.enable_extract_description)
      qf.add("description");
    if(params.enable_extract_claims)
      qf.add("claims");
    
    return qf.toArray(new String[qf.size()]);
  }

  private void ParseRequestParams(SolrQueryRequest req, ConfigParams params,
      ModifiableSolrParams qparams) throws SyntaxError {
    
    qparams.set(CommonParams.Q, req.getParams().get(CommonParams.Q));
    
    // get method of semantic concepts retrievals
    String tmp = req.getParams().get("conceptsmethod");
    if(tmp!=null) {
      params.e_Method = getSemanticMethod(tmp);
      if(params.e_Method==ENUM_SEMANTIC_METHOD.e_UNKNOWN) {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (conceptsmethod)");
      }
    }
    // get measure relatedness flag 
    tmp = req.getParams().get("measure_relatedness");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.measure_relatedness = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (measure_relatedness)");
      }
    }
    
    // get semantic relatedness experiment flag 
    tmp = req.getParams().get("hrelatednessexpr");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relatedness_experiment = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelatednessexpr)");
      }
    }
    else
      params.hidden_relatedness_experiment = false;
    
 // get enable title search flag 
    tmp = req.getParams().get("titlesearch");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_title_search = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (titlesearch)");
      }
    }
    // get force see also flag 
    tmp = req.getParams().get("hrelaxseealso");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_see_also = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxseealso)");
      }
    }    
    else
      params.hidden_relax_see_also = false;

    // get ignored concepts if any
    String[] ignsem = req.getParams().getFieldParams(null, "ignsem");
    if(ignsem!=null) {
      params.ignored_concepts = new HashSet<String>();
      for(String s : ignsem)
      params.ignored_concepts.add(s);
    }
    
    // get fqs if any
    //params.fqs  = req.getParams().getFieldParams(null, "fqs");
    String[] fqs = req.getParams().getFieldParams(null, "fqs");
    if(fqs!=null) {
      params.fqs = new HashMap<String,ArrayList<String>>();
      
      for(String fq : fqs) {
        String[] fqToken = fq.split(":");
        ArrayList<String> values = params.fqs.get(fqToken[0]);
        if(values==null)
          values = new ArrayList<String>();
        
        values.add(fqToken[1]);
        params.fqs.put(fqToken[0], values);
      }
    }
    
    // get relax NER flag 
    tmp = req.getParams().get("hrelaxner");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_ner = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxner)");
      }
    }
    else
      params.hidden_relax_ner = false;
    
    // get relax categories flag 
    tmp = req.getParams().get("hrelaxcategories");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_categories = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxcategories)");
      }
    }
    else
      params.hidden_relax_categories = false;
    
    // get relax search flag 
    tmp = req.getParams().get("hrelaxsearch");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_search = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxsearch)");
      }
    }
    else
      params.hidden_relax_search = false;

    // get absolute explicit concepts flag 
    tmp = req.getParams().get("habsexplicit");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.abs_explicit = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (habsexplicit)");
      }
    }
    else
      params.hidden_include_q = false;
    
    // get include q flag 
    tmp = req.getParams().get("hincludeq");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_include_q = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hincludeq)");
      }
    }
    else
      params.hidden_include_q = false;

    // get show records flag 
    tmp = req.getParams().get("showrecords");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_show_records = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (showrecords)");
      }
    }
    else
      params.enable_show_records = false;

    // get search all flag 
    tmp = req.getParams().get("searchall");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_search_all = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (searchall)");
      }
    }
    else
      params.enable_search_all = false;

    // get search title flag 
    tmp = req.getParams().get("searchtitle");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_search_title = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (searchtitle)");
      }
    }
    else
      params.enable_search_title = false;

    // get search abstract flag 
    tmp = req.getParams().get("searchabstract");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_search_abstract = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (searchabstract)");
      }
    }
    else
      params.enable_search_abstract = false;
    

    // get search description flag 
    tmp = req.getParams().get("searchdescription");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_search_description = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (searchdescription)");
      }
    }
    else
      params.enable_search_description = false;
    
    // get search claims flag 
    tmp = req.getParams().get("searchclaims");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_search_claims = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (searchclaims)");
      }
    }
    else
      params.enable_search_claims = false;
    
    // get extract all flag 
    tmp = req.getParams().get("extractall");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_extract_all = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (extractall)");
      }
    }
    else
      params.enable_extract_all = false;

    // get extract title flag 
    tmp = req.getParams().get("extracttitle");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_extract_title = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (extracttitle)");
      }
    }
    else
      params.enable_extract_title = false;

    // get extract abstract flag 
    tmp = req.getParams().get("extractabstract");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_extract_abstract = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (extractabstract)");
      }
    }
    else
      params.enable_extract_abstract = false;
    

    // get extract description flag 
    tmp = req.getParams().get("extractdescription");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_extract_description = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (extractdescription)");
      }
    }
    else
      params.enable_extract_description = false;
    
    // get extract claims flag 
    tmp = req.getParams().get("extractclaims");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.enable_extract_claims = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (extractclaims)");
      }
    }
    else
      params.enable_extract_claims = false;
    
    // get relax caching flag 
    tmp = req.getParams().get("hrelaxcache");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_cache = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxcache)");
      }
    }
    else
      params.hidden_relax_cache = false;

    // get relax list of filter flag 
    tmp = req.getParams().get("hrelaxlistof");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_listof = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxlistof)");
      }
    }
    else
      params.hidden_relax_listof = false;
    
    // get relax same title flag 
    tmp = req.getParams().get("hrelaxsametitle");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_same_title = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxsametitle)");
      }
    }
    else
      params.hidden_relax_same_title = false;
    
    // get relax disambiguation filter flag 
    tmp = req.getParams().get("hrelaxdisambig");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_disambig = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxdisambig)");
      }
    }
    else
      params.hidden_relax_disambig = false;
    
    // get method of experiment input file
    tmp = req.getParams().get("hexperin");
    if(tmp!=null && tmp.length()>0) {
      params.experiment_in_path = StringEscapeUtils.escapeHtml(tmp);
      qparams.set("hexperin", params.experiment_in_path);
    }
    // get method of experiment input file
    tmp = req.getParams().get("hexperout");
    if(tmp!=null && tmp.length()>0) {
      params.experiment_out_path = StringEscapeUtils.escapeHtml(tmp);        
      qparams.set("hexperout", params.experiment_out_path);
    }
    
    // get distance metric method
    tmp = req.getParams().get("hsim");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareToIgnoreCase("cosine")==0)
        params.e_Distance = ENUM_DISTANCE_METRIC.e_COSINE;
      else if(tmp.compareToIgnoreCase("bin")==0)
        params.e_Distance = ENUM_DISTANCE_METRIC.e_COSINE_BIN;
      else if(tmp.compareToIgnoreCase("cosinenorm")==0)
        params.e_Distance = ENUM_DISTANCE_METRIC.e_COSINE_NORM;
      else if(tmp.compareToIgnoreCase("wo")==0)
        params.e_Distance = ENUM_DISTANCE_METRIC.e_WO;
      else if(tmp.compareToIgnoreCase("eucleadian")==0)
        params.e_Distance = ENUM_DISTANCE_METRIC.e_Eucleadian;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hsim)");
      }
    }
    
    // get analytic type
    tmp = req.getParams().get("analytic");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareToIgnoreCase("explore")==0)
        params.e_analytic_type = ENUM_ANALYTIC_TYPE.e_TECH_EXPLORE;
      else if(tmp.compareToIgnoreCase("landscape")==0)
        params.e_analytic_type = ENUM_ANALYTIC_TYPE.e_TECH_LANDSCAPE;
      else if(tmp.compareToIgnoreCase("competitive")==0)
        params.e_analytic_type = ENUM_ANALYTIC_TYPE.e_CI;
      else if(tmp.compareToIgnoreCase("prior")==0)
        params.e_analytic_type = ENUM_ANALYTIC_TYPE.e_PRIOR;
      else if(tmp.compareToIgnoreCase("relatedness")==0)
        params.e_analytic_type = ENUM_ANALYTIC_TYPE.e_RELATEDNESS;
      else if(tmp.compareToIgnoreCase("search")==0)
        params.e_analytic_type = ENUM_ANALYTIC_TYPE.e_Search;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (analytic)");
      }
    }
    
    // get maximum hits in initial wiki search
    tmp = req.getParams().get("hmaxhits");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.hidden_max_hits = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hmaxhits)");
      }
    }
    else
      params.hidden_max_hits = Integer.MAX_VALUE;
    
    // get maximum hits in initial wiki search
    tmp = req.getParams().get("hminassocnt");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.hidden_min_asso_cnt = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hminassocnt)");
      }
    }
//    else
//      params.hidden_min_asso_cnt = 1;
    
    // get minimum wiki article length to search
    tmp = req.getParams().get("hminwikilen");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.hidden_min_wiki_length = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hminwikilen)");
      }      
    }
//    else
//      params.hidden_min_wiki_length = 0;
    
    // get minimum seealso article length to search
    tmp = req.getParams().get("hminseealsolen");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.hidden_min_seealso_length = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hminseealsolen)");
      }
    }
//    else
//      params.hidden_min_seealso_length = 0;
    
    // get maximum ngrams of wiki titles
    tmp = req.getParams().get("hmaxngrams");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.hidden_max_title_ngrams = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hmaxngrams)");
      }
    }      
    
    // get maximum seealso ngrams of wiki titles
    tmp = req.getParams().get("hseealsomaxngrams");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.hidden_max_seealso_ngrams = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hseealsomaxngrams)");
      }
    }
    
    // get wiki search field
    tmp = req.getParams().get("hwikifield");
    if(tmp!=null)
      params.hidden_wiki_search_field = tmp;
    

    // get wiki extra query
    tmp = req.getParams().get("hwikiextraq");
    if(tmp!=null)
      params.hidden_wiki_extra_query = tmp;
    
    // get number of CI patents to retrieve
    tmp = req.getParams().get("patentsno");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.ci_patents_num = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (ci_patents_num)");
      }
    }
    
    // get number of semantic concepts to retrieve
    tmp = req.getParams().get("conceptsno");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.concepts_num = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (conceptsno)");
      }
    }
  }

  protected NamedList<Object> doSemanticSearch(String concept, ConfigParams params) {
    NamedList<Object> semanticConceptsInfo = null;
    HashMap<String,SemanticConcept> relatedConcepts = null;
    
    // retrieve related semantic concepts      
    relatedConcepts = new HashMap<String,SemanticConcept>();
    retrieveRelatedConcepts(concept, relatedConcepts, params);
    
    // remove irrelevant concepts
    filterRelatedConcepts(relatedConcepts, params);
    
    if(relatedConcepts.size()>0) {        
      // sort concepts
      SemanticConcept sem[] = new SemanticConcept[relatedConcepts.size()];
      sem = (SemanticConcept[])relatedConcepts.values().toArray(sem);
      Arrays.sort(sem, Collections.reverseOrder());
      
      // add concepts to response
      
      semanticConceptsInfo = new SimpleOrderedMap<Object>();
      for(int i=0,j=0; i<params.concepts_num && i<sem.length; j++) {
        // remove a concept that exactly match original concept
        if(params.hidden_relax_same_title==false && concept.compareToIgnoreCase(sem[j].name)==0) {
          System.out.println(sem[j].name+"...Removed!");
          continue;
        }
        if(params.ignored_concepts!=null && params.ignored_concepts.contains(sem[j].name)) {
          // set ignored flag
          sem[j].ignore = 1;
        }
        SimpleOrderedMap<Object> conceptInfo = sem[j].getInfo();
        //semanticConceptsInfo.add(sem[j].name, sem[j].weight);
        semanticConceptsInfo.add(sem[j].name, conceptInfo);
        if(!params.abs_explicit || sem[j].e_concept_type==ENUM_CONCEPT_TYPE.e_TITLE)
            i++;
      }
      
      // add related concepts to the query
      //if(params.hidden_relax_search==false)
      //  qparams.set(CommonParams.Q, newQuery);
      
      //req.setParams(qparams);
    }
    return semanticConceptsInfo;
  }
  
  protected NamedList<Object> doSemanticRelatedness(String sentence1, String sentence2, ConfigParams params) {
    NamedList<Object> semanticConceptsInfo = null;
    
    if(params.hidden_relatedness_experiment==true) {
      // open concepts file and loop on concepts
      BufferedReader in;
      try {
        in = new BufferedReader(new FileReader("./"+params.experiment_in_path));
        FileWriter out = new FileWriter("./"+params.experiment_out_path);
        if(in!=null && out!=null) {
          String line;        
          line = in.readLine();
          while(line!=null) {
            // expected format (concept1,concept2)
            String[] concepts = line.split(",");
            
            // calculating similarity between each two concepts
            HashMap<String,SemanticConcept> relatedConcepts1 = new HashMap<String,SemanticConcept>();
            HashMap<String,SemanticConcept> relatedConcepts2 = new HashMap<String,SemanticConcept>();
            
            out.write(concepts[0]);
            for(int i=1; i<concepts.length; i++) {
              double similarity = getRelatedness(concepts[0], concepts[i], 
                  relatedConcepts1, relatedConcepts2, params);
              
              out.write(","+concepts[i]+","+similarity);
              relatedConcepts2.clear();
            }
            out.write("\n");
            out.flush();
            
            line = in.readLine();
            }
          in.close();
          out.close();
        }
      } catch (IOException e) {
        e.printStackTrace();
        throw new RuntimeException();
      }
    }
    else {
      if(sentence1.length()>0 && sentence2.length()>0) {
        HashMap<String,SemanticConcept> relatedConcepts1 = new HashMap<String,SemanticConcept>();
        HashMap<String,SemanticConcept> relatedConcepts2 = new HashMap<String,SemanticConcept>();
        
        double relatedness = getRelatedness(sentence1, sentence2, 
            relatedConcepts1, relatedConcepts2, params);
        
        semanticConceptsInfo = new SimpleOrderedMap<Object>();
        
        System.out.println("score: "+relatedness);
        relatedness = mapRelatednessScore(relatedness);        
        
        semanticConceptsInfo.add("relatedness", String.format("%.2f", relatedness));
      
        
        if(relatedConcepts1.size()>0 || relatedConcepts2.size()>0) {          
          // add concepts related to concept 1 to response
          SimpleOrderedMap<Object> conceptInfo1 = new SimpleOrderedMap<Object>();
          int ind=0;
          for(SemanticConcept c : relatedConcepts1.values()) {
            SimpleOrderedMap<Object> conceptInfo = c.getInfo();
            conceptInfo1.add(c.name, conceptInfo);
            ind++;
            if(ind==params.concepts_num || ind==relatedConcepts1.size())
              break;
          }
          semanticConceptsInfo.add(sentence1, conceptInfo1);
          
          // add concepts related to concept 2 to response
          SimpleOrderedMap<Object> conceptInfo2 = new SimpleOrderedMap<Object>();
          ind=0;
          for(SemanticConcept c : relatedConcepts2.values()) {
            SimpleOrderedMap<Object> conceptInfo = c.getInfo();
            conceptInfo2.add(c.name, conceptInfo);
            ind++;
            if(ind==params.concepts_num || ind==relatedConcepts2.size())
              break;
          }
          semanticConceptsInfo.add(sentence2, conceptInfo2);
        }
      }
      else { // invalid format (expected concept1,concept2)
        
      }
    }
    return semanticConceptsInfo;
  }
  
  private double mapRelatednessScore(double score) {
    double relatedness = score;
    double max = 0;
    double min = 0;
    double rmin = 0;
    double rmax = 0;
    if(relatedness>=0.35) { // [3.5-4]
      rmin = 0.35;
      rmax = 1;
      min = 3.5;
      max = 4;
    }
    else if(relatedness>=0.2) { // [2-3.5]
      rmin = 0.2;
      rmax = 0.349999999;
      min = 2;
      max = 3.5;
    }
    else if(relatedness>=0.01) { // [1-2]
      rmin = 0.01;
      rmax = 0.1999999999;
      min = 1;
      max = 2;
    }
    else { // [0-1]
      rmin = 0.0;
      rmax = 0.00999999999;
      min = 0;
      max = 1;
    }
    return 1+min+(max-min)*(relatedness-rmin)/(rmax-rmin);
  }

  protected double getRelatedness(String concept1, String concept2, 
      HashMap<String,SemanticConcept> relatedConcepts1, 
      HashMap<String,SemanticConcept> relatedConcepts2, 
      ConfigParams params) {
    double similarity = 0;
    
    // retrieve related semantic concepts for concept 1
    System.out.println("Retrieving for concept: ("+concept1+")");
    retrieveRelatedConcepts(concept1, relatedConcepts1, params);
    
    // remove irrelevant concepts
    filterRelatedConcepts(relatedConcepts1, params);
    
    // retrieve related semantic concepts for concept 2
    System.out.println("Retrieving for concept: ("+concept2+")");
    retrieveRelatedConcepts(concept2, relatedConcepts2, params);
    
    // remove irrelevant concepts
    filterRelatedConcepts(relatedConcepts2, params);
    
    if(relatedConcepts1.size()>0 || relatedConcepts2.size()>0) {
      // keep only required number of concepts
      SemanticConcept[] sem = new SemanticConcept[relatedConcepts1.size()];
      sem = (SemanticConcept[])relatedConcepts1.values().toArray(sem);
      Arrays.sort(sem, Collections.reverseOrder());
      relatedConcepts1.clear();
      for(int i=0,j=0; i<sem.length && j<params.concepts_num; i++) {
        relatedConcepts1.put(sem[i].name, sem[i]);
        if(sem[i].e_concept_type==ENUM_CONCEPT_TYPE.e_TITLE)
          j++;
      }
      
      sem = new SemanticConcept[relatedConcepts2.size()];
      sem = (SemanticConcept[])relatedConcepts2.values().toArray(sem);
      Arrays.sort(sem, Collections.reverseOrder());
      relatedConcepts2.clear();
      for(int i=0,j=0; i<sem.length && j<params.concepts_num; i++) {
        relatedConcepts2.put(sem[i].name, sem[i]);
        if(sem[i].e_concept_type==ENUM_CONCEPT_TYPE.e_TITLE)
          j++;
      }
      if(params.e_Distance==ENUM_DISTANCE_METRIC.e_COSINE || 
          params.e_Distance==ENUM_DISTANCE_METRIC.e_COSINE_BIN || 
          params.e_Distance==ENUM_DISTANCE_METRIC.e_COSINE_NORM) {
        similarity = cosineSim(relatedConcepts1, relatedConcepts2, params);
      }
      else if(params.e_Distance==ENUM_DISTANCE_METRIC.e_Eucleadian) {
        similarity = 1.0/(1.0+eucleadianDist(relatedConcepts1, relatedConcepts2, params));        
      }
      else if(params.e_Distance==ENUM_DISTANCE_METRIC.e_WO) {
        try {
          similarity = WeightedOverlapSim(relatedConcepts1, relatedConcepts2, params);
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    }
    return similarity;
  }
  private double eucleadianDist(
      HashMap<String,SemanticConcept> relatedConcepts1,
      HashMap<String,SemanticConcept> relatedConcepts2, ConfigParams params) {
    double distance = 0;
    // normalize both concepts vectors
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
    
    // calculate eucleadian similarity
    for(SemanticConcept c1 : relatedConcepts1.values()) {
      SemanticConcept c2 = relatedConcepts2.get(c1.name);
      distance += Math.pow(c1.weight-c2.weight, 2.0);      
    }
    distance = Math.sqrt(distance);
    
    return distance;
  }

  private double WeightedOverlapSim(
      HashMap<String,SemanticConcept> relatedConcepts1,
      HashMap<String,SemanticConcept> relatedConcepts2, ConfigParams params) throws Exception {
    
    // sort the two vectors by weight
    SemanticConcept concepts1[] = new SemanticConcept[relatedConcepts1.size()];
    concepts1 = (SemanticConcept[])relatedConcepts1.values().toArray(concepts1);
    Arrays.sort(concepts1, Collections.reverseOrder());
    
    SemanticConcept concepts2[] = new SemanticConcept[relatedConcepts2.size()];
    concepts2 = (SemanticConcept[])relatedConcepts2.values().toArray(concepts2);
    Arrays.sort(concepts2, Collections.reverseOrder());
    
    // get overlap between two vectors
    Set<String> overlap = relatedConcepts1.keySet();
    overlap.retainAll(relatedConcepts2.keySet());
    System.out.println("common concepts: ("+String.valueOf(overlap.size())+")");
    
    int[] ranks1 = new int[overlap.size()];
    int[] ranks2 = new int[overlap.size()];
    int pos = 0;
    for(String s : overlap) {
      // get rank in first vector
      for(int i=0; i<concepts1.length; i++)
        if(s.compareTo(concepts1[i].name)==0) {
          ranks1[pos] = i+1;
          break;
        }
      
      // get rank in second vector
      for(int i=0; i<concepts2.length; i++)
        if(s.compareTo(concepts2[i].name)==0) {
          ranks2[pos] = i+1;
          break;
        }
      
      pos++;
    }
    if(pos<overlap.size())
      throw new Exception("overlap mismatch!");
    
    double num = 0.0, den = 0.0;
    for(int i=0; i<ranks1.length; i++) {
      num += 1.0/(ranks1[i]+ranks2[i]);
      den += 1.0/(2*(Math.min(ranks1[i],ranks2[i])));
    }
    return (num/den); 
  }

  private double cosineSim(HashMap<String,SemanticConcept> relatedConcepts1, 
      HashMap<String,SemanticConcept> relatedConcepts2, ConfigParams params) {
    double similarity=0, norm1=0, norm2=0, dot_product=0;
    if(params.e_Distance==ENUM_DISTANCE_METRIC.e_COSINE || 
        params.e_Distance==ENUM_DISTANCE_METRIC.e_COSINE_NORM) {

      if(params.e_Distance==ENUM_DISTANCE_METRIC.e_COSINE_NORM) { // normalize weights
        float maxWeight = Collections.max(relatedConcepts1.values()).weight;
        for(String s : relatedConcepts1.keySet()) {
          SemanticConcept sem = relatedConcepts1.get(s);
          sem.weight = sem.weight/maxWeight;
        }
         
        maxWeight = Collections.max(relatedConcepts2.values()).weight;
        for(String s : relatedConcepts2.keySet()) {
          SemanticConcept sem = relatedConcepts2.get(s);
          sem.weight = sem.weight/maxWeight;
        }
      }
      
      // normalize both concepts vectors
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
      for(SemanticConcept c1 : relatedConcepts1.values()) {
        SemanticConcept c2 = relatedConcepts2.get(c1.name);
        dot_product += c1.weight*c2.weight;
        norm1 += Math.pow(c1.weight, 2.0);
        norm2 += Math.pow(c2.weight, 2.0);
      }
      similarity = dot_product/(Math.sqrt(norm1)*Math.sqrt(norm2));
    }
    else if(params.e_Distance==ENUM_DISTANCE_METRIC.e_COSINE_BIN) {
      // calculate cosine similarity
      for(SemanticConcept c1 : relatedConcepts1.values()) {
        SemanticConcept c2 = relatedConcepts2.get(c1.name);
        if(c2!=null) {
          dot_product += 1;
        }
      }
      norm1 = relatedConcepts1.size();
      norm2 = relatedConcepts2.size();
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
      ConfigParams params) {
    //TODO: 
    /* do we need to intersect with technical dictionary
     * do we need to score see also based on cross-reference/see also graph similarity (e.g., no of common titles in the see also graph)
     * do we need to filter out titles with places, nationality,N_N,N_N_N,Adj_N_N...etc while indexing
     * do we need to look at cross-references
     * do we need to search in title too with boosting factor then remove exact match at the end
     * do we need to add "" here or in the request
     */
    concept = QueryParser.escape(concept.toLowerCase());
    
    try {
      IndexSearcher searcher = new IndexSearcher(indexReader);
      Analyzer stdAnalyzer = new StandardAnalyzer();
      QueryParser parser = new QueryParser(params.hidden_wiki_search_field, stdAnalyzer);
      Query query = null;
      try {
        
        parser.setAllowLeadingWildcard(true);
        String queryString = "title_length:[000 TO "+String.format("%03d", params.hidden_max_title_ngrams)+"] "
            + "AND padded_length:["+String.format("%09d", params.hidden_min_wiki_length)+" TO *]";
        if(params.enable_title_search) {
          queryString += " AND (title:"+concept+" OR "+params.hidden_wiki_search_field+":"+concept+")";
        }
        else {
          queryString += " AND ("+params.hidden_wiki_search_field+":"+concept+")";
        }
        if(params.hidden_wiki_extra_query.length()>0)
          queryString += " "+ params.hidden_wiki_extra_query;
        
        query = parser.parse(queryString);
      } catch (ParseException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      } //
      TopDocs topDocs = searcher.search(query, params.hidden_max_hits);
      if(topDocs.totalHits > 0) {
        ScoreDoc[] hits = topDocs.scoreDocs;
        int cur_id = 1;
        int cur_parent_id = 0;
        for(int i = 0 ; i < hits.length; i++) {
          boolean relevant = false;
          IndexableField title = indexReader.document(hits[i].doc).getField("title");
          String ner = indexReader.document(hits[i].doc).getField("title_ne").stringValue();
          CachedConceptInfo cachedInfo = null;
          CachedAssociationInfo cachedAssoInfo = null;
          
          //System.out.println(title.stringValue());
          // check if relevant concept
          boolean relevantTitle = true;//TODO: do we need to call isRelevantConcept(f.stringValue());
          if(relevantTitle==true) {
            relevant = true;            
            // check if already there
            SemanticConcept sem = relatedConcepts.get(title.stringValue());
            if(sem==null) { // new concept              
              cachedInfo = cachedConceptsInfo.get(title.stringValue());
              if(cachedInfo==null) {
                System.out.println(title.stringValue()+"...title not found!");
                if(params.hidden_relax_cache==true)
                  cachedInfo = new CachedConceptInfo(title.stringValue(), 0, "", "", "");
              }
              if(cachedInfo!=null) {
                sem = new SemanticConcept(title.stringValue(), cachedInfo, ner, hits[i].score,
                    cur_id, 0, 0, ENUM_CONCEPT_TYPE.e_TITLE);
                cur_parent_id = cur_id;
                cur_id++;
                // get its associations
                if(params.e_Method!=ENUM_SEMANTIC_METHOD.e_MSA && 
                    params.e_Method!=ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS) {
                  Integer I = titleIntMapping.get(title.stringValue());
                  if(I!=null) {
                    cachedAssoInfo = cachedAssociationsInfo.get(I);
                  }
                  else
                    System.out.println(title.stringValue()+"...title not in mappings!");
                }                              
              }
            }
            else { // existing concept, update its weight to higher weight
              cachedInfo = sem.cachedInfo;
              sem.weight = sem.weight>hits[i].score?sem.weight:hits[i].score;
              cur_parent_id = sem.id;
            }
            if(sem!=null)
              relatedConcepts.put(sem.name, sem);            
          }
          else
            System.out.println(title.stringValue()+"...title not relevant!");
          if(relevant==true && (params.e_Method==ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS || 
                  params.e_Method==ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS_SEE_ALSO || 
                      params.e_Method==ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS_SEE_ALSO_ASSO)) // retrieve anchors
          {
            relevant = false;
            IndexableField Anchors[] = indexReader.document(hits[i].doc).getFields("title_anchors");
            for(int t=0; t<Anchors.length; t++) {
              IndexableField f = Anchors[t];
              //System.out.println(f.stringValue());
              // check if relevant concept
              relevantTitle = true;//TODO: do we need to call isRelevantConcept(f.stringValue());
              if(relevantTitle==true) {
                relevant = true;
                
                // check if already there
                SemanticConcept sem = relatedConcepts.get(f.stringValue());
                if(sem==null) { // new concept                  
                    sem = new SemanticConcept(f.stringValue(), cachedInfo, ner, hits[i].score-0.0001f, 
                      cur_id, cur_parent_id, 0, ENUM_CONCEPT_TYPE.e_ANCHOR);
                  cur_id++;
                }
                else { // existing concept, update its weight to higher weight
                  cachedInfo = sem.cachedInfo;
                  sem.weight = sem.weight>hits[i].score-0.0001f?sem.weight:hits[i].score-0.0001f;
                }
                if(sem!=null)
                  relatedConcepts.put(sem.name, sem);                
              }
              else
                System.out.println(f.stringValue()+"...title not relevant!");
            }
          }
          
          //System.out.println();
          if(params.hidden_relax_see_also==false || relevant) {
            // force see also is enabled OR,
            // the original title or one of its anchors is relevant
            // in this case we can add its see_also
            if(params.e_Method==ENUM_SEMANTIC_METHOD.e_MSA_SEE_ALSO || 
                params.e_Method==ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS_SEE_ALSO) { // add See also to the hit list
              
              IndexableField[] multiSeeAlso = indexReader.document(hits[i].doc).getFields("see_also");
              IndexableField[] multiSeeAlsoNE = indexReader.document(hits[i].doc).getFields("see_also_ne");
              IndexableField[] multiSeeAlsoLength = indexReader.document(hits[i].doc).getFields("seealso_length");              
            
              for(int s=0; s<multiSeeAlso.length; s++) {
                //System.out.println(f.stringValue());
                // check if relevant concept
                relevantTitle = true; //TODO: do we need to call isRelevantConcept(multiSeeAlso[s].stringValue());
                /*String re = "\\S+(\\s\\S+){0,"+String.valueOf(params.hidden_max_seealso_ngrams-1)+"}";
                if(multiSeeAlso[s].stringValue().toLowerCase().matches(re)==false)
                  relevantTitle = false;
                */
                if(Integer.parseInt(multiSeeAlsoLength[s].stringValue())>params.hidden_max_seealso_ngrams)
                  relevantTitle = false;
                
                if(relevantTitle==true) {
                  // check if already there
                  SemanticConcept sem = relatedConcepts.get(multiSeeAlso[s].stringValue()); 
                  if(sem==null) { // new concept
                    cachedInfo = cachedConceptsInfo.get(multiSeeAlso[s].stringValue());
                    if(cachedInfo==null) {
                      System.out.println(multiSeeAlso[s].stringValue()+"...see_also not found!");
                      if(params.hidden_relax_cache==true)
                        cachedInfo = new CachedConceptInfo(multiSeeAlso[s].stringValue(), 0, "", "", "");
                    } 
                    if(cachedInfo!=null) {  
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
                      if((cachedInfo.length==0 || cachedInfo.length>=params.hidden_min_seealso_length) && 
                          (asso_cnt==0 || asso_cnt>=params.hidden_min_asso_cnt)) { // support > minimum support
                        sem = new SemanticConcept(multiSeeAlso[s].stringValue(), 
                            cachedInfo, multiSeeAlsoNE[s].stringValue(), 
                            hits[i].score-0.0002f, cur_id, cur_parent_id, asso_cnt, ENUM_CONCEPT_TYPE.e_SEE_ALSO);
                        cur_id++;                      
                      }
                    }
                  }
                  else { // existing concept, update its weight to higher weight
                    cachedInfo = sem.cachedInfo;
                    sem.weight = sem.weight>hits[i].score-0.0002f?sem.weight:hits[i].score-0.0002f;
                  }
                  if(sem!=null)
                    relatedConcepts.put(sem.name, sem);            
                }
                //else
                  //System.out.println(multiSeeAlso[s].stringValue()+"...see-also not relevant!");
                //System.out.println();
              }              
            }
            else if(params.e_Method==ENUM_SEMANTIC_METHOD.e_MSA_SEE_ALSO_ASSO || 
                params.e_Method==ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS_SEE_ALSO_ASSO) { // add see also using association mining
              Integer key = titleIntMapping.get(indexReader.document(hits[i].doc).getFields("title")[0].stringValue());
              CachedAssociationInfo assoInfo = cachedAssociationsInfo.get(key);
              if(assoInfo==null) {
                System.out.println(indexReader.document(hits[i].doc).getFields("title")[0].stringValue() + "..no associations cached!");
              }
              else {
                for(int a=0; a<assoInfo.associations.size(); a++) {
                  Integer[] assos = assoInfo.associations.get(a);
                  if(assos[1]>=params.hidden_min_asso_cnt) // support > minimum support
                  {
                    String assoStr = titleStrMapping.get(assos[0]);
                    
                    // check if relevant concept
                    relevantTitle = true; //TODO: do we need to call isRelevantConcept(multiSeeAlso[s].stringValue());
                    if(getTitleLength(assoStr)>params.hidden_max_seealso_ngrams)
                      relevantTitle = false;
                    if(relevantTitle==true) {
                      // check if already there
                      SemanticConcept sem = relatedConcepts.get(assoStr); 
                      if(sem==null) { // new concept
                        cachedInfo = cachedConceptsInfo.get(assoStr);
                        if(cachedInfo==null) {
                          System.out.println(assoStr+"...see_also not found!");
                          if(params.hidden_relax_cache==true)
                            cachedInfo = new CachedConceptInfo(assoStr, 0, "", "", "");
                        }
                        if(cachedInfo!=null)
                        {  
                          if(cachedInfo.length==0 || cachedInfo.length>=params.hidden_min_seealso_length) { 
                              sem = new SemanticConcept(assoStr, 
                          
                                cachedInfo, "M", 
                              hits[i].score-0.0002f, cur_id, cur_parent_id, assos[1], ENUM_CONCEPT_TYPE.e_SEE_ALSO);
                            cur_id++;
                          }
                        }
                      }
                      else { // existing concept, update its weight to higher weight
                        cachedInfo = sem.cachedInfo;
                        sem.weight = sem.weight>hits[i].score-0.0002f?sem.weight:hits[i].score-0.0002f;
                      }
                      if(sem!=null)
                        relatedConcepts.put(sem.name, sem);
                    }
                    //else
                      //System.out.println(assoStr+"...see-also not relevant!");
                  }
                  //else
                    //System.out.println(assos[0]+"...see-also below threshold!");
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
  
  private int getTitleLength(String title) {
      // TODO Auto-generated method stub
      int index = title.length(), index1, index2;
      if((index1=title.indexOf(','))==-1)
          index1 = index;
      if((index2=title.indexOf('('))==-1)
          index2 = index;
      
      index = Math.min(index1, index2);
      String s[] = title.substring(0,index).split(" ");
      return s.length;    
  }

  /*
   * remove concepts that are irrelevant (only 1-2-3 word phrases are allowed)
   * @param relatedConcepts related concepts to be filtered
   */
  protected void filterRelatedConcepts(HashMap<String,SemanticConcept> relatedConcepts, 
      ConfigParams params) {
    ArrayList<String> toRemove = new ArrayList<String>();
    for (SemanticConcept concept : relatedConcepts.values()) {
      if(isRelevantConcept(params.hidden_relax_listof,params.hidden_relax_disambig,params.hidden_max_title_ngrams,concept.name)==false) {
        System.out.println(concept.name+"("+concept.id+")...Removed!");
        toRemove.add(concept.name);
      }
      else if(params.hidden_relax_ner==false && ArrayUtils.contains(new String[]{"P","L","O"},concept.ner)) { // check if not allowed NE
        System.out.println(concept.name+"("+concept.id+")...Removed (NER)!");
        toRemove.add(concept.name);
      }
      else if(params.hidden_relax_categories==false) { // check if not allowed category
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
  protected boolean isRelevantConcept(boolean hidden_relax_listof, boolean hidden_relax_disambig,  
      int hidden_max_title_ngrams, String concept) {
    //TODO: 
    /* handle list of: (List of solar powered products),(List of types of solar cells)
     * handle File: (File:Jason Robinson.jpg)
     * handle names, places, adjectives
     * remove titles with (disambiguation)
     * keep only titles in technical dictionary
     */
    boolean relevant = true;
    String re;
    /*String re = "\\S+(\\s\\S+){0,"+String.valueOf(hidden_max_title_ngrams-1)+"}";
    if(concept.toLowerCase().matches(re)==false) {
      relevant = false;
    }*/
    if(relevant==true && hidden_relax_listof==false) {
      re = "list of.*";
      if(concept.toLowerCase().matches(re)==true) {
        relevant = false;
      }
    }
    if(relevant==true && hidden_relax_disambig==false) {
      re = ".*\\(disambiguation\\)";
      if(concept.toLowerCase().matches(re)==true) {
        relevant = false;
      }    
    }
    return relevant; 
  }
  
  private ENUM_SEMANTIC_METHOD getSemanticMethod(String conceptMethod) {
    if(conceptMethod.compareTo("Explicit")==0)
      return ENUM_SEMANTIC_METHOD.e_MSA;
    else if (conceptMethod.compareTo("MSA_anchors")==0)
      return ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS;
    else if (conceptMethod.compareTo("MSA_seealso")==0)
      return ENUM_SEMANTIC_METHOD.e_MSA_SEE_ALSO;
    else if (conceptMethod.compareTo("MSA_anchors_seealso")==0)
      return ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS_SEE_ALSO;
    else if (conceptMethod.compareTo("MSA_seealso_asso")==0)
      return ENUM_SEMANTIC_METHOD.e_MSA_SEE_ALSO_ASSO;
    else if (conceptMethod.compareTo("MSA_anchors_seealso_asso")==0)
      return ENUM_SEMANTIC_METHOD.e_MSA_ANCHORS_SEE_ALSO_ASSO;    
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
        cachedConceptsInfo = new HashMap<String,CachedConceptInfo>(/*5000000*/);
        String title = "";
	String docno = "";
        String cat1 = "";
        String cat2 = "";
        IndexableField[] multi = null;
        CachedConceptInfo cachedInfo = null;
        for(ScoreDoc d : topDocs.scoreDocs) {
          // retrieve title
          title = indexReader.document(d.doc).getFields("title")[0].stringValue();

          // retrieve docno
          docno = indexReader.document(d.doc).getField("docno").stringValue();
          
          // retrieve length
          int length = Integer.parseInt(indexReader.document(d.doc).getField("length").stringValue());

          // retrieve categories
          multi = indexReader.document(d.doc).getFields("category");
          cat1 = multi.length>0? multi[0].stringValue():"";
          cat2 = multi.length>1? multi[1].stringValue():"";
          
          cachedInfo = new CachedConceptInfo(title, length, docno, cat1, cat2);
          cachedConceptsInfo.put(title,  cachedInfo);
          
          if(cachedConceptsInfo.size()>500000)
            break;
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
      
      titleIntMapping = new HashMap<String,Integer>(/*5000000*/);
      titleStrMapping = new HashMap<Integer,String>(/*5000000*/);
      
      cachedAssociationsInfo = new HashMap<Integer,CachedAssociationInfo>(/*5000000*/);
      
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
            if(association.length!=2)
              System.out.println(line);                        
            else
              associationInfo.addAssociation(idx,Integer.parseInt(association[1]));
          }
        }
        if(associations.length>0)
          cachedAssociationsInfo.put(new Integer(key), associationInfo);
        
        if(cachedAssociationsInfo.size()>500000)
          break;
        line = f.readLine();
      }
      f.close();
      
    } catch (IOException e) {
        throw new RuntimeException();
    }    
  }
}


