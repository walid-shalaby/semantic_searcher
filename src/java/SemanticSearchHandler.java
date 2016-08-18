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

import java.io.FileWriter;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.lang.StringEscapeUtils;
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
import org.apache.solr.response.ResultContext;
import org.apache.solr.response.SolrQueryResponse;
import org.apache.solr.search.DocIterator;
import org.apache.solr.search.DocList;
import org.apache.solr.search.SyntaxError;
import org.apache.solr.util.SolrPluginUtils;

import wiki.toolbox.semantic.SemanticSearchConfigParams;
import wiki.toolbox.semantic.Enums;
import wiki.toolbox.semantic.SemanticsGenerator;

/**
 *
 * Refer SOLR-281
 *
 */

public class SemanticSearchHandler extends SearchHandler
{
  SemanticsGenerator semanticsGenerator = null;
  String wikiUrl = "";
  String associationspath = "";
  String hostPort = "";
  
  /* 
   *
   */
  @Override
  public void init(PluginInfo info) {
    super.init(info);
    
    wikiUrl = System.getProperty("wikiurl", "http://localhost:5678/solr/collection1/");
    hostPort = System.getProperty("jetty.port", "8990");
    associationspath = System.getProperty("associationspath", "./wiki_associations.txt");
    System.out.println(wikiUrl);
    System.out.println(hostPort);
    System.out.println(associationspath);
    
    semanticsGenerator = new SemanticsGenerator();
    
    // cache Wiki see also associations for fast retrieval
    semanticsGenerator.cacheAssociationsInfo(associationspath);

    // cache Wiki titles with some required information for fast retrieval
    semanticsGenerator.cacheConceptsInfo(wikiUrl, true, true);
    //cachedConceptsInfo = new HashMap<String,CachedConceptInfo>();
    
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
     
      SemanticSearchConfigParams params = new SemanticSearchConfigParams(); 
      
      ModifiableSolrParams qparams = new ModifiableSolrParams(req.getParams());
      
      // parse request
      params.wikiUrl = wikiUrl;
      ParseRequestParams(req, params, qparams);
      
      //req.setParams(qparams);
      
      //if(params.hidden_relax_search==false) {
      //  super.handleRequestBody(req, rsp);
      //}

      if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_SEARCH) {
        super.handleRequestBody(req, rsp);
      }
      else if(params.concepts_num>=0 && 
          (params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_TECH_EXPLORE || 
          params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_TECH_LANDSCAPE || 
          params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_PRIOR)) {
        String orgQ = req.getParams().get(CommonParams.Q);
        semanticConceptsInfo = semanticsGenerator.doSemanticSearch(orgQ, params);
        if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_TECH_EXPLORE) {
          
        }
        if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_TECH_LANDSCAPE) {
          // construct new query with concepts added to it
          String newQuery;
          if(params.hidden_include_q==true) {
            String tmpq = orgQ;
            if(params.hidden_boolean==false)
              newQuery = getNewQuery(tmpq.toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and "), semanticConceptsInfo);
            else
              newQuery = getNewQuery(tmpq, semanticConceptsInfo);
          }
          else
            newQuery = getNewQuery("", semanticConceptsInfo);
          
          fillTechLandscapeQParams(newQuery, new String[]{"{!ex=fqs}publication_year,ipc_section"}, new String[]{}, params, qparams);
          
          qparams.set("fl", new String[]{"id","title","abstract","assignee_orgname","assignee_addressbook_orgname","type","publication_doc_number","tags"});          
          req.setParams(qparams);
          super.handleRequestBody(req, rsp);
          qparams.set(CommonParams.Q, orgQ);
          qparams.remove("fq");
        }
        if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_PRIOR) {
          // construct new query with concepts added to it
          String newQuery;
          if(params.hidden_include_q==true) {
            String tmpq = req.getParams().get(CommonParams.Q);
            if(params.hidden_boolean==false)
              tmpq = tmpq.toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and ");
            newQuery = getNewQuery(tmpq, semanticConceptsInfo);
          }
          else
            newQuery = getNewQuery("", semanticConceptsInfo);
          
          qparams.set(CommonParams.Q, newQuery);
          qparams.set(CommonParams.ROWS, params.results_num);
          qparams.set("qf", getQuerySearchFields(params));
          qparams.set("fl", new String[]{"id","title","abstract","assignee_orgname","assignee_addressbook_orgname","type","publication_doc_number","tags"});
          req.setParams(qparams);
          super.handleRequestBody(req, rsp);
          qparams.set(CommonParams.Q, orgQ);
          qparams.remove("fq");
          /*ResultContext results = (org.apache.solr.response.ResultContext)rsp.getValues().get("response");
          if(results!=null && results.docs!=null & results.docs.size()>0) {
            SolrDocumentList docs = SolrPluginUtils.docListToSolrDocumentList(results.docs, req.getSearcher(), new HashSet<String>(Arrays.asList("id")), null);
            for(SolrDocument doc : docs) {
                System.out.println(doc.getFieldValue("id"));
              }
          }*/          
        }
      }
      else if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_CI) {
        String orgQ = req.getParams().get(CommonParams.Q);
        String techQ = "";
        if(params.ci_freetext==false) {
          // construct a new query of given assignee innovations and retrieve them
          HttpSolrServer server = new HttpSolrServer("http://localhost:"+hostPort+"/solr/collection1/");
          ModifiableSolrParams innoQParams = new ModifiableSolrParams();
          innoQParams.set(CommonParams.Q, orgQ.toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and "));
          innoQParams.set("defType","edismax");
          innoQParams.set("qf", new String[]{"assignee"});
          innoQParams.set(CommonParams.ROWS, params.ci_patents_num);
          innoQParams.set("sort", "publication_date desc");
          
          // add target extract field
          String[] extractFields = getQueryExtractFields(params);
          innoQParams.set("fl", extractFields);
          QueryResponse innoQResp = server.query(innoQParams);
        
          // loop on results and construct a new query to explore technologies
          SolrDocumentList results = innoQResp.getResults();
          if(results!=null && results.size()>0) {
            for(int i=0; i<results.size(); i++) {
              SolrDocument doc = results.get(i);
              for(String field : extractFields) {
                techQ += (String)doc.getFieldValue(field) + " ";
              }              
            }
          }
        }
        else
          techQ = orgQ;
        
        if(techQ.length()>0) {
          if(params.ci_freetext==false || params.hidden_boolean==false)
            techQ = techQ.toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and ");
          semanticConceptsInfo = semanticsGenerator.doSemanticSearch(techQ, params);
          System.out.println(techQ);
          
          // construct new query with concepts added to it
          String newQuery;
          if(params.hidden_include_q==true)
            newQuery = getNewQuery(techQ, semanticConceptsInfo);
          else
            newQuery = getNewQuery("", semanticConceptsInfo);
                    
          fillTechLandscapeQParams(newQuery, new String[]{"{!ex=fqs}publication_year,ipc_section"}, 
              new String[]{"{!ex=fqs}assignee","{!ex=fqs}assignee_addressbook_orgname_exact","{!ex=fqs}assignee_exact"}, params, qparams);         
          
          req.setParams(qparams);
          super.handleRequestBody(req, rsp);
          qparams.set(CommonParams.Q, orgQ);
          qparams.remove("fq");
        }
      }
      else if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_RELEVANCY) {
        String orgQ = req.getParams().get(CommonParams.Q);
        String relQ = "";
     
        // construct a new query of given assignee innovations and retrieve them
        HttpSolrServer server = new HttpSolrServer("http://localhost:"+hostPort+"/solr/collection1/");
        ModifiableSolrParams srcQParams = new ModifiableSolrParams();
        srcQParams.set(CommonParams.Q, orgQ.toUpperCase());
        srcQParams.set("defType","edismax");
        srcQParams.set("qf", new String[]{"id"});
        srcQParams.set(CommonParams.ROWS, 1);
        
        // add target extract field
        String[] tmpf = getQueryExtractFields(params);
        String[] extractFields = Arrays.copyOf(tmpf, tmpf.length+2);
        extractFields[tmpf.length] = "publication_date";
        extractFields[tmpf.length+1] = "*";
        
        srcQParams.set("fl", extractFields);
        QueryResponse srcQResp = server.query(srcQParams);
      
        // loop on results and construct a new query to explore technologies
        HashMap<String, String> srcinfo = new HashMap<String, String>();
        String pubdate = null;
        
        SolrDocumentList results = srcQResp.getResults();
        if(results!=null && results.size()==1) {
          SolrDocument doc = results.get(0);
          for(String field : extractFields) {
            if(field.compareTo("publication_date")==0) {
              pubdate = ((Date)doc.getFieldValue(field)).toInstant().toString();
            }
            else if(field.compareTo("*")!=0) {
              String tmpval = (String)doc.getFieldValue(field);
              relQ += tmpval + " ";
            }
          }
          
          // add source info
          srcinfo.put("id", (String)doc.getFieldValue("id"));
          srcinfo.put("title", (String)doc.getFieldValue("title"));
          srcinfo.put("abstract", (String)doc.getFieldValue("abstract"));
          srcinfo.put("publication_date", pubdate);
        }
        else {
          // clear request and send back bad request syntax
          req.setParams(new ModifiableSolrParams());
          throw new SyntaxError("("+orgQ+") does not exist.");
        }
        
        relQ = relQ.toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and ");
        
        if(relQ.length()>0) {
          semanticConceptsInfo = semanticsGenerator.doSemanticSearch(relQ, params);
          System.out.println(relQ);
          
          // construct new query with concepts added to it
          String newQuery;
          if(params.hidden_include_q==true)
            newQuery = getNewQuery(relQ, semanticConceptsInfo);
          else
            newQuery = getNewQuery("", semanticConceptsInfo);
                    
          qparams.set(CommonParams.Q, newQuery);
          qparams.set(CommonParams.ROWS, params.priors_hits);
          qparams.set("qf", getQuerySearchFields(params));          
          qparams.set("fl", new String[]{"id"});
          qparams.add("fq","publication_date:[* TO "+pubdate+"]");
          QueryResponse qResp = server.query(qparams);
       
          // loop on results and calculate relevancy
          results = qResp.getResults();          
          HashMap<String,String> priors = new HashMap<String,String>();
          if(params.relevancy_priors.length>0 && params.relevancy_priors[0].length()>0) {
          for(String s : params.relevancy_priors)
            priors.put(s, "not found");          
            
            if(results!=null && results.size()>1) {
              for(int i=0; i<results.size(); i++) {
                SolrDocument doc = results.get(i);
                String id = ((String)doc.getFieldValue("id"));
                if(priors.containsKey(id))
                  priors.put(id, String.valueOf(i+1));
              }
            }
          }
          else {
            if(results!=null && results.size()>1) {
              for(int i=0; i<results.size(); i++) {
                SolrDocument doc = results.get(i);
                String id = ((String)doc.getFieldValue("id"));                
                priors.put(id, String.valueOf(i+1));
              }
            }
          }
          rsp.add("sourceinfo", srcinfo);
          rsp.add("relevancy", priors);
          //req.setParams(qparams);
          //super.handleRequestBody(req, rsp);
          //qparams.set(CommonParams.Q, orgQ);
          //qparams.remove("fq");
        }
      }
      else if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_RELATEDNESS) {
        semanticConceptsInfo = semanticsGenerator.doSemanticRelatedness(req.getParams().get(CommonParams.Q), 
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
  
  private void fillTechLandscapeQParams(String newQuery, String[] pivots, String[] facets, SemanticSearchConfigParams params,
      ModifiableSolrParams qparams) {
    //qparams.set(CommonParams.Q, newQuery.toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and "));
    qparams.set(CommonParams.Q, newQuery);
    if(params.enable_show_records==false)
      qparams.set(CommonParams.ROWS, 0);
    
    qparams.set("facet", true);
    if(pivots.length>0)
      qparams.set("facet.pivot",  pivots);
    if(facets.length>0)
      qparams.set("facet.field",  facets);
    
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
    
    qparams.set("qf", getQuerySearchFields(params));
  }

  private void fillTechLandscapeQParams1(String newQuery, String[] facets, SemanticSearchConfigParams params,
      ModifiableSolrParams qparams) {
    //qparams.set(CommonParams.Q, newQuery.toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and "));
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
    
    qparams.set("qf", getQuerySearchFields(params));
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

  private String[] getQuerySearchFields(SemanticSearchConfigParams params) {
    ArrayList<String> qf = new ArrayList<String>();
    if(params.enable_search_all)
      qf.add("text");
    else {
      if(params.enable_search_title)
        qf.add("title");
      if(params.enable_search_abstract)
        qf.add("abstract");
      if(params.enable_search_description)
        qf.add("description");
      if(params.enable_search_claims)
        qf.add("claims");
    }
    return qf.toArray(new String[qf.size()]);
  }

  private String[] getQueryExtractFields(SemanticSearchConfigParams params) {
    ArrayList<String> qf = new ArrayList<String>();
    if(params.enable_extract_all)
      qf.add("text");          
    else {
      if(params.enable_extract_title)
        qf.add("title");
      if(params.enable_extract_abstract)
        qf.add("abstract");
      if(params.enable_extract_description)
        qf.add("description");
      if(params.enable_extract_claims)
        qf.add("claims");
    }
    return qf.toArray(new String[qf.size()]);
  }

  private void ParseRequestParams(SolrQueryRequest req, SemanticSearchConfigParams params,
      ModifiableSolrParams qparams) throws SyntaxError {
    
    qparams.set(CommonParams.Q, req.getParams().get(CommonParams.Q).toLowerCase().replace(" not ", " \\not ").replace(" or ", " \\or ").replace(" and ", " \\and "));
    
    // get method of semantic concepts retrievals
    String tmp = req.getParams().get("conceptsmethod");
    if(tmp!=null) {
      params.e_Method = Enums.getSemanticMethod(tmp);
      if(params.e_Method==Enums.ENUM_SEMANTIC_METHOD.e_UNKNOWN) {
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
    
    // get pagerank weighting flag 
    tmp = req.getParams().get("hpagerankweight");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_pagerank_weighting = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hpagerankweight)");
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
      params.ignored_concepts.add(java.net.URLDecoder.decode(s));
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
    
    // get CI freetext flag 
    tmp = req.getParams().get("hfreetext");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.ci_freetext = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hfreetext)");
      }
    }
    else
      params.ci_freetext = false;
    
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
      params.abs_explicit = false;
    
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

    // get boolean flag 
    tmp = req.getParams().get("hboolean");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_boolean = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hboolean)");
      }
    }
    else
      params.hidden_boolean = false;

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
    
    // get relax filtering flag 
    tmp = req.getParams().get("hrelaxfilters");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareTo("on")==0)
        params.hidden_relax_filters = true;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hrelaxfilters)");
      }
    }
    else
      params.hidden_relax_filters = false;
    
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
      params.in_path = StringEscapeUtils.escapeHtml(tmp);
      qparams.set("hexperin", params.in_path);
    }
    // get method of experiment input file
    tmp = req.getParams().get("hexperout");
    if(tmp!=null && tmp.length()>0) {
      params.out_path = StringEscapeUtils.escapeHtml(tmp);        
      qparams.set("hexperout", params.out_path);
    }
    
    // get distance metric method
    tmp = req.getParams().get("hsim");
    if(tmp!=null && tmp.length()>0) {
      params.e_Distance = Enums.getDistanceMethod(tmp);
      if(params.e_Distance==Enums.ENUM_DISTANCE_METRIC.e_UNKNOWN) {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hsim)");
      }
    }
    
    // get analytic type
    tmp = req.getParams().get("analytic");
    if(tmp!=null && tmp.length()>0) {
      if(tmp.compareToIgnoreCase("explore")==0)
        params.e_analytic_type = Enums.ENUM_ANALYTIC_TYPE.e_TECH_EXPLORE;
      else if(tmp.compareToIgnoreCase("landscape")==0)
        params.e_analytic_type = Enums.ENUM_ANALYTIC_TYPE.e_TECH_LANDSCAPE;
      else if(tmp.compareToIgnoreCase("competitive")==0)
        params.e_analytic_type = Enums.ENUM_ANALYTIC_TYPE.e_CI;
      else if(tmp.compareToIgnoreCase("prior")==0)
        params.e_analytic_type = Enums.ENUM_ANALYTIC_TYPE.e_PRIOR;
      else if(tmp.compareToIgnoreCase("relatedness")==0)
        params.e_analytic_type = Enums.ENUM_ANALYTIC_TYPE.e_RELATEDNESS;
      else if(tmp.compareToIgnoreCase("search")==0)
        params.e_analytic_type = Enums.ENUM_ANALYTIC_TYPE.e_SEARCH;
      else if(tmp.compareToIgnoreCase("relevancy")==0)
        params.e_analytic_type = Enums.ENUM_ANALYTIC_TYPE.e_RELEVANCY;
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (analytic)");
      }
    }
    
    if(params.e_analytic_type==Enums.ENUM_ANALYTIC_TYPE.e_RELEVANCY) {
      // get ipc filter
      tmp = req.getParams().get("ipcfilter");
      if(tmp!=null && tmp.length()>0) {
        if(tmp.compareToIgnoreCase("none")==0)
          params.e_ipc_filter = Enums.ENUM_IPC_FILTER.e_NONE;
        else if(tmp.compareToIgnoreCase("section")==0)
          params.e_ipc_filter = Enums.ENUM_IPC_FILTER.e_IPC_SECTION;
        else if(tmp.compareToIgnoreCase("class")==0)
          params.e_ipc_filter = Enums.ENUM_IPC_FILTER.e_IPC_CLASS;
        else if(tmp.compareToIgnoreCase("subclass")==0)
          params.e_ipc_filter = Enums.ENUM_IPC_FILTER.e_IPC_SUBCLASS;
        else if(tmp.compareToIgnoreCase("group")==0)
          params.e_ipc_filter = Enums.ENUM_IPC_FILTER.e_IPC_GROUP;
        else if(tmp.compareToIgnoreCase("subgroup")==0)
          params.e_ipc_filter = Enums.ENUM_IPC_FILTER.e_IPC_SUBGROUP;
        else {
          // clear request and send back bad request syntax
          req.setParams(new ModifiableSolrParams());
          throw new SyntaxError("Invalid value for request parameter (ipc filter)");
        }
      }
      
      // get priors, should be comma separated
      tmp = req.getParams().get("priors");
      if(tmp!=null) {
          params.relevancy_priors = tmp.split(",");          
      }
      else {
          // clear request and send back bad request syntax
          req.setParams(new ModifiableSolrParams());
          throw new SyntaxError("Invalid value for request parameter (priors)");
      }
   // get priors, should be comma separated
      tmp = req.getParams().get("priorshits");
      if(tmp!=null) {
        if (tmp.matches("\\d+")){
          params.priors_hits = Integer.parseInt(tmp);
        }
        else {
          // clear request and send back bad request syntax
          req.setParams(new ModifiableSolrParams());
          throw new SyntaxError("Invalid value for request parameter (priors)");
        }
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
    
    // get maximum levels in concept graph
    tmp = req.getParams().get("hmaxlevels");
    if(tmp!=null) {
      if (tmp.matches("\\d+")){
        params.hidden_max_levels = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (hmaxlevels)");
      }
    }
    else
      params.hidden_max_levels = 2;
    
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
    
    // get number of results to retrieve
    tmp = req.getParams().get("resultsno");
    if(tmp!=null && tmp.length()>0) {
      if (tmp.matches("\\d+")){
        params.results_num = Integer.parseInt(tmp);
      }
      else {
        // clear request and send back bad request syntax
        req.setParams(new ModifiableSolrParams());
        throw new SyntaxError("Invalid value for request parameter (resultsno)");
      }
    }
  }

}



