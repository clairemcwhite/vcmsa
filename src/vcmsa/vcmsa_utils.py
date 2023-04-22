#!/usr/bin/env python3

# To generate embeddings

# To calculate similarity
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Printing for debugging with icecream
try:
    from icecream import ic
    ic.configureOutput(includeContext=True, outputFunction=print) # Prints line number and function
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

# This is combat with patsy requirement removed
from vcmsa.combat_vsmsa_mod import combat
#from transformer_infrastructure.feedback import remove_feedback_edges_old

# For networks
import igraph

# For copying graph objects
import copy

# For parsing fasta
from Bio import SeqIO

# To layout alignment
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# For plotting PCA
import matplotlib.pyplot as plt
import random

import logging
from vcmsa.logger_module import logger


#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.ic_stats)

# General purpose
import numpy as np
from pandas.core.common import flatten
import pandas as pd
from collections import Counter, OrderedDict
import logging
from time import time

###############################
### Classes
class AA:

    def __init__(self):
       self.seqnum = ""
       self.seqindex = ""
       self.seqpos = ""
       self.seqaa = ""
       self.index = ""
       self.clustid = ""
       self.prevaa = ""
       self.nextaa = ""


   #__str__ and __repr__ are for pretty printing

    def __str__(self):
        return("{}-{}-{}".format(self.seqnum, self.seqpos, self.seqaa))


    def __repr__(self):
        return str(self)


class Alignment:
    def __init__(self, alignment, seqnames = []):
        self.alignment = alignment
        if not seqnames:
            self.seqnames = list(range(0, len(self.alignment)))
        else:
            self.seqnames = seqnames
        self.numseqs = len(self.alignment)
        self.width = len(self.alignment[0])
        self.numassigned = len([x for x in flatten(self.alignment) if x != "-"])
        self.numgaps = len([x for x in flatten(self.alignment) if x == "-"])
        self.str_formatted = self.str_format(alignment)


    def str_format(self, alignment):
        str_alignment = []
        for line in alignment:
          row_str = ""
          for aa in line:
                if aa == "-":
                   row_str  = row_str + aa
                else:
                   row_str = row_str + aa.seqaa
          str_alignment.append(row_str)

        return(str_alignment)




    def format_aln(self, style = "clustal"):
        records = []
        for i in range(len(self.str_formatted)):
             #ic(self.str_formatted[i])

             alignment_str = "".join([self.str_formatted[i]])
             records.append(SeqRecord(Seq(alignment_str), id=str(self.seqnames[i]),  description = "", name = ""))
        align = MultipleSeqAlignment(records)
        if style == "clustal":

            formatted = format(align, 'clustal')
        elif style == "fasta":
            formatted = format(align, 'fasta')

        return(formatted)

    def __str__(self):
        return(self.format_aln( "clustal"))

    def __repr__(self):
        return str(self)

#################################
###### Alignment formatting utils



def get_new_clustering_old(G, betweenness_cutoff = 0.10,  apply_walktrap = True, prev_G = None, pos_to_clustid = {}):

    logging.debug(G.vs()['name'])
    if prev_G:
        logging.debug(prev_G.vs()['name'])
        if G.vs()['name'] == prev_G.vs()['name']:
            return([], {})
    logging.info("get_new_clustering")#                 all_alternates_dict = {**all_alternates_dict, **alternates_dict}
    new_clusters = []
    connected_set = G.vs()['name']
   # #ic("after ", sub_connected_set)
    #ic("connected set", connected_set)
    all_alternates_dict = {}
    new_clusters = []

    finished = check_completeness(connected_set)
    # Complete if no duplicates
    if finished == True:
        logging.debug("finished connected set {}".format(connected_set))
        new_clusters = [connected_set]

    else:
        min_dupped =  min_dup(connected_set, 1.2)
        # Only do walktrap is cluster is overlarge
        logging.debug("min_dupped at start {}".format(min_dupped))
        names = [x['name'] for x in G.vs()]
        logging.debug("names at start".format(names))
        if (len(connected_set) > min_dupped) and apply_walktrap and len(G.vs()) >= 5:
            # First remove weakly linked aa's then try again
            # Walktrap is last resort
            hub_scores = G.hub_score()
            names = [x['name'] for x in G.vs()]
            logging.debug(names)
            vx_names = G.vs()
            hub_names = list(zip(names, vx_names, hub_scores))

            high_authority_nodes = [x[0] for x in hub_names if x[2]  > 0.2]
            logging.debug("high authority_nodes".format(high_authority_nodes))
            high_authority_nodes_vx = [x[1] for x in hub_names if x[2]  > 0.2]

            low_authority_nodes = [x[0] for x in hub_names if x[2]  <= 0.2]
            low_authority_node_ids = [x[1] for x in hub_names if x[2]  <= 0.2]

            logging.debug("removing low authority_nodes {}".format(low_authority_nodes))
            #low_authority_nodes = []
            if len(low_authority_nodes) > 0 and len(high_authority_nodes) > 0:
                #logging.debug("before {}".format(G))
                #logging.debug([x for x in G.es()])

                logging.debug([x['name'] for x in G.vs()])
                logging.debug(low_authority_nodes)
                high_authority_edges =  G.es.select(_within = high_authority_nodes_vx)
                #If not edges left after removing low authority nodes
                logging.debug("high authority_edges {}".format(high_authority_edges))
                logging.debug([x for x in high_authority_edges])
                if len(high_authority_edges) > 0:

                    G.delete_vertices(low_authority_node_ids)
                    logging.debug("After delete {}".format(G))
                #G = G.subgraph(high_authority_nodes_vx)
                names = [x['name'] for x in G.vs()]
                logging.debug("names prior to new clusters {}".format(names))

                min_dupped = min_dup(names, 1.2)
                #print("after", G)
                #print([x for x in G.es()])
                #print([x['name'] for x in G.vs()])
            if len(names) <= min_dupped:
                logging.debug("get_new_clustering:new_G {}".format(G))
                processed_cluster, alternates_dict =  process_connected_set(names, G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff, pos_to_clustid = pos_to_clustid)
                logging.debug("processed_cluster {}".format(processed_cluster))
                #ic("alternates_dict", alternates_dict)
                if alternates_dict:
                   all_alternates_dict = {**all_alternates_dict, **alternates_dict}
                new_clusters = new_clusters + processed_cluster
            else:
                logging.debug("applying walktrap")
                # Change these steps to 3??
                # steps = 1 makes clear errors
                logging.debug("len(connected_set {}, min_dupped {}".format(len(connected_set), min_dupped))
                logging.debug(G)
                clustering = G.community_walktrap(steps = 3, weights = 'weight').as_clustering()
                for sub_G in clustering.subgraphs():
                     sub_connected_set =  sub_G.vs()['name']
                     logging.debug("post cluster subgraph {}".format(sub_connected_set))

                     # New clusters may be too large still, try division process w/ betweenness

                     processed_cluster, alternates_dict = process_connected_set(sub_connected_set, sub_G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff, pos_to_clustid = pos_to_clustid)
                     new_clusters = new_clusters + processed_cluster
                     if alternates_dict:
                        all_alternates_dict = {**all_alternates_dict, **alternates_dict}
        else:
            logging.debug("get_new_clustering:connected_set {}".format(connected_set))
            processed_cluster, alternates_dict =  process_connected_set(connected_set, G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff, pos_to_clustid = pos_to_clustid)
            new_clusters = new_clusters + processed_cluster
            if alternates_dict:
                 all_alternates_dict = {**all_alternates_dict, **alternates_dict}

    logging.debug("get_new_clustering:all_alternates_dict {}".format(all_alternates_dict))
    return(new_clusters, all_alternates_dict)

#@profile
def make_alignment(cluster_order, seqnums, clustid_to_clust, seqnames):
    # Set up a bunch of vectors of "-"
    # Replace with matches
    # cluster_order = list in the order that clusters go
    #ic("Alignment clusters")
    #for clustid, clust in clustid_to_clust.items():
        #ic(clustid, clust)

    numseqs = len(seqnums)
    alignment_lol =  [["-"] * len(cluster_order) for i in range(numseqs)]
    for order in range(len(cluster_order)):
       cluster = clustid_to_clust[cluster_order[order]]
       c_dict = {}
       for x in cluster:
           c_dict[x.seqnum]  = x # x.seqaa
       for seqnum_index in range(numseqs):
               try:
                  # convert list index position to actual seqnum
                  seqnum = seqnums[seqnum_index]
                  alignment_lol[seqnum_index][order] = c_dict[seqnum]
               except Exception as E:
                   continue
    alignment_str = ""
    #print("Alignment")

    alignment = Alignment(alignment_lol, seqnames)
    str_alignment = alignment.str_formatted
    for row_str in str_alignment:
       logging.debug("Align: {}".format(row_str[0:170]))

    return(alignment)


#@profile
def clusts_from_alignment(alignment):
   '''
   Takes an alignment object
   Converts it to a cluster order and a dictionary of clustid:cluster members
   '''
   clustid_to_clust = {}

   cluster_order = range(0, alignment.width)
   for i in cluster_order:
       clust = [x[i] for x in alignment.alignment if not x[i] == "-"]

       clustid_to_clust[i] = clust

   return(cluster_order, clustid_to_clust)



###############################
####### Sequence utils
#@profile
def get_seqs_aas(seqs, seqnums):
    '''
    Formats sequences as a lists of lists of AAs
    '''
    seqs_aas = []
    seq_to_length = {}


    for i in range(len(seqs)):

        seq_aas = []
        seqnum = seqnums[i]
        seq_to_length[i] = len(seqs[i])
        for j in range(len(seqs[i])):
           # If first round, start new AA
           # Otherwise, use the next aa as the current aa
           if j == 0:
               aa = AA()
               aa.seqnum = seqnum
               aa.seqpos = j
               aa.seqaa =  seqs[i][j]


           else:
               aa = nextaa
               aa.prevaa = prevaa
           prevaa = aa
           if j < len(seqs[i]) - 1:
              nextaa = AA()
              nextaa.seqnum = seqnum
              nextaa.seqpos = j + 1
              nextaa.seqaa = seqs[i][j + 1]
              aa.nextaa = nextaa


           seq_aas.append(aa)

        seqs_aas.append(seq_aas)
    return(seqs_aas, seq_to_length)


################################
######## Topological sort functions
        

#@profile
def organize_clusters(clusterlist, seqs_aas, gapfilling_attempt,  minclustsize = 1, all_alternates_dict = {}, seqnames = [], args = None):
    
    seqnums = [x[0].seqnum for x in seqs_aas]
    logging.debug("start organize clusters, during gapfilling attempt {}".format(gapfilling_attempt))
    logging.debug("clusters at start of organize {}".format(clusterlist))

    cluster_orders_dict, pos_to_clust, clustid_to_clust, dag_reached = clusters_to_dag(clusterlist, seqs_aas, remove_both = True, gapfilling_attempt = gapfilling_attempt, minclustsize = minclustsize, all_alternates_dict = all_alternates_dict, args = args)

    dag_attempts = 1
    while dag_reached == False:
                    
          clusters_filt = list(clustid_to_clust.values())
          logging.debug("dag_attempts {} {}".format(dag_attempts, clusters_filt))
          if len(clusters_filt) < 2:
               logging.debug("Dag not reached, no edges left to remove. count {}".format(count))
               return(1)
          cluster_orders_dict, pos_to_clust, clustid_to_clust,  dag_reached = clusters_to_dag(clusters_filt, seqs_aas, remove_both = True, gapfilling_attempt = gapfilling_attempt, minclustsize = minclustsize, all_alternates_dict = all_alternates_dict, args = args)
          dag_attempts = dag_attempts + 1

    cluster_order, clustid_to_clust, pos_to_clustid =  dag_to_cluster_order(list(cluster_orders_dict.values()), seqs_aas, pos_to_clust, clustid_to_clust)
    alignment = make_alignment(cluster_order, seqnums, clustid_to_clust, seqnames)

    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment)

 
#@profile
def graph_from_cluster_orders(cluster_orders_lol):
    ''' 
    Take lists of lists of each sequence's path through cluster id
    Convert to edges in a network
    '''
    
    order_edges = []
    for order in cluster_orders_lol:
       for i in range(len(order) - 1):
          edge = (order[i], order[i + 1])
          #if edge not in order_edges:
          order_edges.append(edge)
 
    G_order = igraph.Graph.TupleList(edges=order_edges, directed=True)
    weights = [1] * len(G_order.es)

    # Remove multiedges and self loops
    G_order.es['weight'] = weights
    G_order = G_order.simplify(combine_edges=sum)

    return(G_order, order_edges)

#@profile
def get_topological_sort(cluster_orders_lol):
    cluster_orders_nonempty = [x for x in cluster_orders_lol if len(x) > 0]
    
    dag_or_not = graph_from_cluster_orders(cluster_orders_nonempty)[0].simplify().is_dag()

    G_order = graph_from_cluster_orders(cluster_orders_nonempty)[0]
    G_order = G_order.simplify()

    topo_sort_indices = G_order.topological_sorting()
    cluster_order = []

    # Note: this is in vertex indices. Need to convert to name to get clustid
    for i in topo_sort_indices:
       cluster_order.append(G_order.vs[i]['name'])
    return(cluster_order) 


#@profile
def remove_order_conflicts(cluster_order, seqs_aas, pos_to_clustid):
   '''
   Called by dag_to_cluster_order
   check if ever does anything
   '''
   bad_clustids = []
   for x in seqs_aas:
      prevpos = -1  
      for posid in x:

          try:
              clustid = pos_to_clustid[posid]
          except Exception as E:
              continue

          pos = posid.seqpos   
          if pos < prevpos:
              #print("Order violation", posid, clustid)
              bad_clustids.append(clustid)
   cluster_order =  [x for x in cluster_order if x not in bad_clustids]
   return(cluster_order)



#@profile
def dag_to_cluster_order(cluster_orders, seqs_aas, pos_to_clust_dag, clustid_to_clust_dag):
    '''
    Cluster orders is a list of lists of each sequence's path through the clustids
    Finds a consensus cluster order using a topological sort
    Requires cluster orders to be a DAG
    '''

    cluster_order = get_topological_sort(cluster_orders)
    #ic("For each sequence check that the cluster order doesn't conflict with aa order")
    # Check if this ever does anything
    cluster_order = remove_order_conflicts(cluster_order, seqs_aas, pos_to_clust_dag)


    clustid_to_clust = {key:val for key, val in clustid_to_clust_dag.items() if key  in cluster_order}

    cluster_order_dict = {}
    for i in range(len(cluster_order)):
        cluster_order_dict[cluster_order[i]] = i

    clustid_to_clust_inorder = {}
    pos_to_clust_inorder = {}
    cluster_order_inorder = []
    for i in range(len(cluster_order)):
         clustid_to_clust_inorder[i] = clustid_to_clust[cluster_order[i]]
         cluster_order_inorder.append(i)


    for key in pos_to_clust_dag.keys():
         # Avoid particular situation where a sequence only has one matched amino acid, so  
         # it isn't in the cluster order sequence
         if pos_to_clust_dag[key] in cluster_order_dict.keys():
              pos_to_clust_inorder[key] = cluster_order_dict[pos_to_clust_dag[key]]


    return(cluster_order_inorder, clustid_to_clust_inorder, pos_to_clust_inorder)

#@profile
def clusters_to_dag(clusters_filt, seqs_aas, gapfilling_attempt, remove_both = True, dag_reached = False, alignment_group = 0, attempt = 0, minclustsize = 1, all_alternates_dict = {}, args = None):
    ######################################3
    # Remove feedback loops in paths through clusters
    # For getting consensus cluster order
    dag_reached = False
    #ic("status of remove_both", remove_both)
    numseqs = len(seqs_aas)
    logging.debug("get cluster dict")
    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusters_filt)
    logging.debug("from clusters_to_dag {}".format(clusters_filt))
    cluster_orders_dict = get_cluster_orders(pos_to_clustid, seqs_aas)

    # For each cluster that's removed, try adding it back one at a time an alternate conformation
    # alternates_dict needs to be added to new main clustering function
    #clusters_filt_dag_all = remove_feedback_edges2(cluster_orders_dict, clustid_to_clust,  gapfilling_attempt, remove_both, alignment_group = alignment_group, attempt = attempt, all_alternates_dict= all_alternates_dict, args = args)
    logging.debug("starting feedback edge removal")
    clustid_to_clust_dag = remove_feedback_aas(cluster_orders_dict, clustid_to_clust,  gapfilling_attempt, remove_both, alignment_group = alignment_group, attempt = attempt, all_alternates_dict= all_alternates_dict, args = args)
    # see if dag was reached after aa removal    
    clusters_filt_dag_all = list(clustid_to_clust_dag.values())
    pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag_all)
    cluster_orders_dict = get_cluster_orders(pos_to_clust_dag, seqs_aas)
    dag_or_not_func = graph_from_cluster_orders(list(cluster_orders_dict.values()))[0].simplify().is_dag()
    
    if dag_or_not_func == True:
        dag_reached = True
        logging.debug("dag reached after removing amino acids causing feedback arcs")
        too_small = []
        clusters_filt_dag = []
        for clust in clusters_filt_dag_all:
              if len(clust) >= minclustsize:
                    clusters_filt_dag.append(clust)
              else:
                    if len(clust) > 2:
                       too_small.append(clust)
    else:

        clusters_filt_dag, clusters_to_add_back = remove_feedback_edges(cluster_orders_dict, clustid_to_clust,  gapfilling_attempt, remove_both, alignment_group = alignment_group, attempt = attempt, all_alternates_dict= all_alternates_dict, args = args)


    #for x in clusters_filt_dag:
       #ic("clusters_filt_dag", x)
  
    #for x in too_small:
       #ic("removed, too small", x)

    #print("Feedback edges removed")
    #print("Get cluster order after feedback removeal")

        pos_to_clust_dag, clustid_to_clust_dag = get_cluster_dict(clusters_filt_dag)
        cluster_orders_dict = get_cluster_orders(pos_to_clust_dag, seqs_aas)

        dag_or_not_func = graph_from_cluster_orders(list(cluster_orders_dict.values()))[0].simplify().is_dag()
        #ic(dag_or_not_func)

        if dag_or_not_func == True:
              dag_reached = True
              logging.debug("dag reached after removing clusters")

    return(cluster_orders_dict, pos_to_clust_dag, clustid_to_clust_dag, dag_reached)


#@profile
def get_cluster_orders(cluster_dict, seqs_aas):
    # This is getting path of each sequence through clusters 
    cluster_orders_dict = {}

    for i in range(len(seqs_aas)):
        seqnum = seqs_aas[i][0].seqnum # Get the setnum of the set of aas
        cluster_order = []

        for j in range(len(seqs_aas[i])):
           key = seqs_aas[i][j]
           #ic("key", key)
           try:
              clust = cluster_dict[key]
              #ic("clust", clust)
              cluster_order.append(clust)
           except Exception as E:
              #ic(E)
              # Not every aa is sorted into a cluster
              continue
        cluster_orders_dict[seqnum] = cluster_order
    return(cluster_orders_dict)


#@profile
def save_ordernet(outnet, G_order, fas, clustid_to_clust):
       with open(outnet, "w") as outfile:
          outfile.write("c1,c2,aas1,aas2,gidx1,gidx2,weight,feedback\n")
          for edge in G_order.es():
             feedback = "no"
             if edge.index in fas:
                feedback = "yes"
             source_name = G_order.vs[edge.source]["name"]
             target_name = G_order.vs[edge.target]["name"]
             source_aas = "_".join([str(x) for x in clustid_to_clust[source_name]])
             target_aas = "_".join([str(x) for x in clustid_to_clust[target_name]])
             outstring = "{},{},{},{},{},{},{},{}\n".format(source_name, target_name, source_aas, target_aas , edge.source, edge.target, edge['weight'], feedback)
             outfile.write(outstring)

#@profile
def remove_feedback_edges2(cluster_orders_dict, clustid_to_clust, gapfilling_attempt, remove_both = True, alignment_group = 0, attempt = 0, all_alternates_dict = {}, args = None):
    """
    Remove both improves quality of initial alignment by remove both aas that are found out of order
    Then attempt to add stronger node back

    """
    #ic(args)
    record_dir = args.record_dir
    outfile_name = args.outfile_name

    logging.debug("before feedback_edges")
    logging.debug(clustid_to_clust)

    G_order, order_edges = graph_from_cluster_orders(list(cluster_orders_dict.values()))
    
    dag_or_not = G_order.is_dag()
    #print(dag_or_not)
    cluster_order_lols = list(cluster_orders_dict.values())
    # The edges to remove to make a directed acyclical graph
    # Corresponds to "look backs"
    # With weight, fas, with try to less well connected nodes
    # Feedback arc sets are edges that point backward in directed graph
    #start = time()
    # Very fast
    fas = G_order.feedback_arc_set(weights = 'weight')
    #end = time()
    #ic("FAS TIME", end - start)

    #ic("feedback arc set")
    for x in fas:
        logging.debug("arc {}".format(x))

    write_ordernet = True
    logging.debug("gapfilling_attempt {}".format(gapfilling_attempt))
    if write_ordernet == True: 
       
       outnet = "{}/{}.ordernet_{}_attempt-{}_gapfilling-{:04}.csv".format(record_dir, outfile_name, alignment_group, attempt, gapfilling_attempt)
       save_ordernet(outnet, G_order, fas, clustid_to_clust)

    to_remove_clustids, to_remove_edges = preserve_stronger_node(G_order, fas)
    


    cluster_order_lols = [[x for x in sublist if x not in to_remove_clustids] for sublist in cluster_order_lols]   
    G_order, order_edges = graph_from_cluster_orders(cluster_order_lols)
    
    dag_or_not = G_order.is_dag()
    #print(dag_or_not) 

    logging.debug("removed clusters {}".format(to_remove_clustids))

    reduced_clusters = []
    for clustid, clust in clustid_to_clust.items():
          #new_clust = []
          #if clustid in removed_clustids:
                #ic("Can remove", clustid, clust)
                #for aa in clust:
                #    if aa in all_alternates_dict.keys():
                #        #ic("Alternate found for AA ",aa, all_alternates_dict[aa])
                #         # ADD list of lists of both clustids in edge to try with alternates
          if not clustid in to_remove_clustids:
              reduced_clusters.append(clust)
          else:
              logging.debug("cluster removed {}".format(clust))
          #else:
          #    too_small_clusters.append(new_clust)
    for f in reduced_clusters:
        logging.debug("reduced cluster {}".format(f))
    return(reduced_clusters)


#@profile
def preserve_stronger_node(G_order, fas): 
    to_remove = []
    removed_edges = []
    to_remove_clustids = []

    for feedback_arc in fas:
       edge = G_order.es()[feedback_arc]
       source_name = G_order.vs[edge.source]["name"]
       target_name = G_order.vs[edge.target]["name"]

       #ic("Feedback edge {}, index {}, edge.source {} edge.target {} source_name {}, target_name {}" .format(edge, edge.index, edge.source, edge.target, source_name, target_name))
       # If one node in the feedback edges is significantly stronger (i.e. more incumbent edges"
       strength_source =  G_order.strength(edge.source, weights = "weight")
       strength_target =  G_order.strength(edge.target, weights = "weight")
       #print("STRENGTH source",  source_name, strength_source)
       #print("STRENGTH target", target_name, strength_target)
       if strength_source >= 1.5 * strength_target:
             #print("keeping ", source_name)
             to_remove_clustids.append(target_name)
             removed_edges.append((None, edge.target, None, target_name))
       elif strength_target >= 1.5 * strength_source:
             #print("keeping ", target_name)
             to_remove_clustids.append(source_name)
             removed_edges.append((edge.source, None , source_name, None))

       else:
           removed_edges.append((edge.source, edge.target, source_name, target_name))
           to_remove_clustids.append(source_name)
           to_remove_clustids.append(target_name)
       
    return(to_remove_clustids, removed_edges)

def contains(subseq, inseq):
   #https://stackoverflow.com/a/24634740/15223329
   return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))


def get_consensus_clust_path(clust, cluster_orders_dict, clustid):
       paths = {}
       new_clust = []
       for aa in clust:
                   
          path = cluster_orders_dict[aa.seqnum]
          for i in range(len(path) - 1):
               if path[i] == clustid:
                    edge = "{},{}".format(path[i], path[i + 1])
                    logging.debug("edge {}".format(edge))
                    paths[aa] = edge
       if len(paths.values()) > 0:             
           most_common_path, count = Counter(paths.values()).most_common(1)[0]
           if count >= 0.5 * len(clust):
               for aa, path in paths.items():
                   if path == most_common_path:
                       logging.debug("keeping path {}".format(path))
                       new_clust.append(aa) 
                   else:
                       logging.debug("Removing aa {}".format(aa))

       return(new_clust)

# Can detect feedback edges
# Then remove amino acids that cause cycles
# Check if dag
# If not, detect feedback edges again. 
# Then remove all clusters in feedback edges

def remove_feedback_aas(cluster_orders_dict, clustid_to_clust, gapfilling_attempt, remove_both = True, alignment_group = 0, attempt = 0, all_alternates_dict = {}, args = None):

    G_order, order_edges = graph_from_cluster_orders(list(cluster_orders_dict.values()))

    #ic(G_order)
    weights = [1] * len(G_order.es)

    # Remove multiedges and self loops
    #ic(G_order)
    G_order.es['weight'] = weights
    G_order = G_order.simplify(combine_edges=sum)

    #ic("after combine")
    #ic(G_order)
    dag_or_not = G_order.is_dag()

    start = time()
    fas = G_order.feedback_arc_set(weights = 'weight')
    end = time()
    #updated = []
    for feedback_arc in fas:
       #print("scan feedback_arc")
       edge = G_order.es()[feedback_arc]
       source_name = G_order.vs[edge.source]["name"]
       target_name = G_order.vs[edge.target]["name"]
       source_clust = clustid_to_clust[source_name]
       target_clust = clustid_to_clust[target_name]
       #print("source_name", "target_name", source_name, target_name)
       #print("scan source")
       for aa in source_clust:
          path = cluster_orders_dict[aa.seqnum]
          #print("arc aa", aa, path, source_name, target_name)
          if contains([source_name, target_name], path):
              logging.debug("{} is part of the feedback arc".format(aa))

       new_source_clust = get_consensus_clust_path(source_clust, cluster_orders_dict, source_name)
       #print("pre_source_clust", source_clust)
       #print("new_source_clust", new_source_clust)
       clustid_to_clust[source_name] = new_source_clust  
       #print("scan target")
       for aa in target_clust:
          path = cluster_orders_dict[aa.seqnum]
          #print("arc aa", aa, path, source_name, target_name)
          if contains([source_name, target_name], path):
              logging.debug("{} is part of the feedback arc".format(aa))
       new_target_clust = get_consensus_clust_path(target_clust, cluster_orders_dict, target_name)
       #print("pre_target_clust", target_clust)
       #print("new_target_clust", new_target_clust)
       clustid_to_clust[target_name] = new_target_clust
 
    return(clustid_to_clust)

#@profile
def remove_feedback_edges(cluster_orders_dict, clustid_to_clust, gapfilling_attempt, remove_both = True, alignment_group = 0, attempt = 0, all_alternates_dict = {}, args = None):

    """
    Remove both improves quality of initial alignment by remove both aas that are found out of order
    Then attempt to add stronger node back
    Function much too long

    """
    #ic("argssss", args)
    record_dir = args.record_dir
    outfile_name = args.outfile_name

    logging.debug("before feedback_edges")
    logging.debug(clustid_to_clust)
    G_order, order_edges = graph_from_cluster_orders(list(cluster_orders_dict.values()))

    #ic(G_order)
    weights = [1] * len(G_order.es)

    # Remove multiedges and self loops
    #ic(G_order)
    G_order.es['weight'] = weights
    G_order = G_order.simplify(combine_edges=sum)

    #ic("after combine")
    #ic(G_order)
    dag_or_not = G_order.is_dag()

    # The edges to remove to make a directed acyclical graph
    # Corresponds to "look backs"
    # With weight, fas, with try to remove lighter edges
    # Feedback arc sets are edges that point backward in directed graph
    start = time()
    fas = G_order.feedback_arc_set(weights = 'weight')
    end = time()
    #ic("FAS TIME", end - start)

    #fas = G_order.feedback_arc_set(weights = 'weight')


    #ic("feedback arc set")
    for x in fas:
        logging.debug("arc {}".format(x))
    logging.debug("We're in gapfilling attempt {}".format(gapfilling_attempt))
    write_ordernet = True
    if write_ordernet == True:

       outnet = "{}/{}.ordernet_{}_attempt-{}_gapfilling-{:04}.csv".format(record_dir, outfile_name, alignment_group, attempt, gapfilling_attempt)
       #ic("outnet", outnet, gapfilling_attempt)
       with open(outnet, "w") as outfile:
          outfile.write("c1,c2,aas1,aas2,gidx1,gidx2,weight,feedback\n")
          # If do reverse first, don't have to do second resort
          for edge in G_order.es():
             feedback = "no"
             if edge.index in fas:
                feedback = "yes"
             source_name = G_order.vs[edge.source]["name"]
             target_name = G_order.vs[edge.target]["name"]
             source_aas = "_".join([str(x) for x in clustid_to_clust[source_name]])
             target_aas = "_".join([str(x) for x in clustid_to_clust[target_name]])
             outstring = "{},{},{},{},{},{},{},{}\n".format(source_name, target_name, source_aas, target_aas , edge.source, edge.target, edge['weight'], feedback)
             outfile.write(outstring)

    #i = 0
    to_remove = []
    removed_edges = []
    to_remove_clustids = []




    for feedback_arc in fas:
       edge = G_order.es()[feedback_arc]
       source_name = G_order.vs[edge.source]["name"]
       target_name = G_order.vs[edge.target]["name"]
       
          

       logging.debug("Feedback edge {}, index {}, edge.source {} edge.target {} source_name {}, target_name {}" .format(edge, edge.index, edge.source, edge.target, source_name, target_name))
       # If one node in the feedback edges is significantly stronger (i.e. more incumbent edges"
       strength_source =  G_order.strength(edge.source, weights = "weight")
       strength_target =  G_order.strength(edge.target, weights = "weight")
       logging.debug("STRENGTH source {} {}".format(source_name, strength_source))
       logging.debug("STRENGTH target {} {}".format(target_name, strength_target))
       if strength_source >= 2* strength_target:
             logging.debug("keeping {}".format(source_name))
             to_remove_clustids.append(target_name)
             #removed_edges.append((None, edge.target, None, target_name))
       elif strength_target >= 2* strength_source:
             logging.debug("keeping {}".format(target_name))
             to_remove_clustids.append(source_name)
             #removed_edges.append((edge.source, None , source_name, None))
       #elif len(clustid_to_clust[source_name]) < len(clustid_to_clust[target_name]):
       #      removed_clustids.append(source_name)

       #elif len(clustid_to_clust[target_name]) < len(clustid_to_clust[source_name]):
       #      removed_clustids.append(target_name)

       # If one of the clusters has alternates, remove the one without alternates, and add back the cluster with an alternate.  
       # Ideas, is it possible to see if the amino acids introducing cycles have alternates?.  
       # Take cycle of 5->2. 
       # Check which path each sequence takes, find ones that are 5->2. If those have alternates, remove them.    
 
       


       else:
           removed_edges.append((edge.source, edge.target, source_name, target_name))
    #return(0)
    # Delete feed back arc edges
    G_order.delete_edges(fas)

    # Check if graph is still dag if edge is added back.
    # If so, keep it
    # Operations on edges are confusing
    for removed_edge in removed_edges:
         #ic("try to return", removed_edge[2:])
         G_order.add_edges(  [removed_edge[0:2]]) # vertex id pairs 
         #G_order.add_edges([removed_edge[2:]]) # vertex id pairs 
         #ic(G_order.is_dag())
         # Retain edges that aren't actually feedback loops
         # Some edges identified by feedback_arc aren't actually cycles (???)

         if not G_order.is_dag():
             G_order.delete_edges([removed_edge[0:2]])
             to_remove.append(removed_edge[2:4]) # list of clustid pairs 

    logging.debug("to_remove {}".format(to_remove))
    for x in to_remove:
        for clustid in x:
           logging.debug("removing after feedback {} {}".format(clustid, clustid_to_clust[clustid]))
           to_remove_clustids.append(clustid)
    remove_dict = {}

    #if remove_both == True:
    #    to_remove_flat = list(flatten(to_remove))
    #else:
    #    to_remove_flat = [x[0] for x in to_remove]
    #
    # 
    #
    #clusters_to_add_back = {} # Dictionary of list of lists containing pairs of clusters to add back with modifications 
    # 
    ##group_count = 0
    
    #for seqnum, clustorder in cluster_orders_dict.items():
    #  remove_dict[seqnum] = []
    #  remove = []
    #  if len(clustorder) == 1:
    #      if clustorder[0] in to_remove_flat:
    #          remove_dict[seqnum] = [clustorder[0]]
    #  logging.debug("state 1 of remove_dict", remove_dict)
    #
    #  for j in range(len(clustorder) - 1):
    #       new_clusts_i = []
    #       new_clusts_j = []
    #       if (clustorder[j], clustorder[j +1]) in to_remove:
    #           clust_i = clustid_to_clust[clustorder[j]]
    #           clust_j = clustid_to_clust[clustorder[j + 1]]
    #           clusters_to_add_back_list = []
    #           for aa in clust_i:
    #                if aa in all_alternates_dict.keys():
    #                    for alternate in all_alternates_dict[aa]:
    #                        #ic("replacing {} with {}".format(aa, alternate))
    #                        new_clust_i = [x for x in clust_i if x != aa] + [alternate]
    #                        new_clusts_i.append(new_clust_i)
    #                           #clusters_to_add_back.append([new_clust_i, clust_j])
    #           for aa in clust_j:
    #                if aa in all_alternates_dict.keys():
    #                    for alternate in all_alternates_dict[aa]:
    #                        #ic("replacing {} with {}".format(aa, alternate))
    #                        new_clust_j = [x for x in clust_j if x != aa] + [alternate]
    #                        new_clusts_j.append(new_clust_j)
    #
    #           for new_clust_i in new_clusts_i:
    #                clusters_to_add_back_list.append([new_clust_i, clust_j])
    #                for new_clust_j in new_clusts_j:
    #                       clusters_to_add_back_list.append([clust_i, new_clust_j])
    #                       clusters_to_add_back_list.append([new_clust_i, new_clust_j])
    #           clusters_to_add_back[frozenset([j, j + 1])] = clusters_to_add_back_list
    #
    #           if remove_both == True:
    #               remove.append(clustorder[j])
    #           remove.append(clustorder[j + 1])
    #       remove_dict[seqnum] = list(set(remove))
    #       logging.debug("state of remove_dict 2", remove_dict)
    #
    #print("final remove_dict", remove_dict)
    #ic("Doing remove")
    reduced_clusters = []
    logging.debug("removed_clustids from before remove dict {}".format(to_remove_clustids))
    #removed_clustids = removed_clustids + list(flatten(list(remove_dict.values())))
    logging.debug("final removed_clustids {}".format(to_remove_clustids))
    for clustid, clust in clustid_to_clust.items():
          new_clust = []
          #if clustid in to_remove: #removed_clustids: CHANGED here
          #      logging.debug("Can remove", clustid, clust)
          #      for aa in clust:
          #          if aa in all_alternates_dict.keys():
          #              ic("Alternate found for AA ",aa, all_alternates_dict[aa])
                         # ADD list of lists of both clustids in edge to try with alternates
          if clustid not in to_remove_clustids:
              reduced_clusters.append(clust)
          #else:
          #    too_small_clusters.append(new_clust)
    clusters_to_add_back = []
    #ic("minclustsize", minclustsize)
    #ic("All alternates_dict", all_alternates_dict)
    #ic("reduced clusters", reduced_clusters)
    #ic("too small clusters" too_small_clusters)

    return(reduced_clusters, clusters_to_add_back)



########################################
######## Amino acid similarity functions

# Can be switched to numba
#@profile
def reshape_flat(hstates_list):
    # Go from (numseqs, seqlen, emb) to (numseqs * seqlen, emb)
    hidden_states = np.reshape(hstates_list, (hstates_list.shape[0]*hstates_list.shape[1], hstates_list.shape[2]))
    return(hidden_states)

#@profile
def do_batch_correct(hidden_states, levels, batch_list):
    hidden_states_pd = pd.DataFrame(hidden_states.T) # So that each aa in a column
    #ic(hidden_states_pd)
    
    batch_series = pd.Series(batch_list)
    #levels = list(range(len(seqs_aas)))
    design_list = [(batch_series == level) * 1 for level in levels]
    design = pd.concat(design_list, axis = 1)
    hidden_states_batch = combat(hidden_states_pd, batch_list, design)
    #ic(hidden_states_batch)
    hidden_states_corrected = np.array(hidden_states_batch).T.astype(np.float32)
    return(hidden_states_corrected)




#@profile
def remove_maxlen_padding(hidden_states, seqs_aas, padded_seqlen):
    # Can be 25s for 30 ~600 long sequences
    # See timings on this step
    # Initial index to remove maxlen padding from input embeddings
    #print("Start remove_maxlen_padding")
    index_to_aa = {}
    aa_indices = []
    logging.debug(hidden_states.shape)
    logging.debug(padded_seqlen)
    seqlens = [len(x) for x in seqs_aas]
    for i in range(len(seqs_aas)):
        for j in range(padded_seqlen):
           if j >= seqlens[i]:
             continue
           #print(i, j)
           aa = seqs_aas[i][j]
           index_num = i * padded_seqlen + j
           index_to_aa[index_num] = aa
           aa_indices.append(index_num)

    
    # Remove maxlen padding from aa embeddings
    #ic(hidden_states.shape)

    hidden_states = np.take(hidden_states, list(index_to_aa.keys()), 0)
    #ic(hidden_states.shape)

    index_to_aa = {}
    count_index = 0
    batch_list = []
    seqnum_to_index = {}
    
    for i in range(len(seqs_aas)):
       seqnum_to_index[i] = []
       for j in range(0, seqlens[i]):
           batch_list.append(i)
           aa = seqs_aas[i][j]
           aa.index = count_index
           aa.seqindex = i
           seqnum_to_index[i].append(count_index)
           index_to_aa[count_index] = aa
           count_index = count_index + 1

    #print("End maxlen padding")
    return(index_to_aa, hidden_states, seqnum_to_index, batch_list)



#@profile
def split_distances_to_sequence(D, I, index_to_aa, numseqs, seqlens):
   # Formarly, was 1: [[aa1:1.0, aa2:0.9]
   # reconfiguring to be {aa1: 1 : OrderedDict{aa1:1.0, aa2:0.9}

   query_aa_dict = {}
   for i in range(len(I)):
      query_aa = index_to_aa[i]
      # Make dictionary, one per sequence
      #target_dict = {} 
      target_dict = {}
      for k in range(numseqs): # Check that this indexing is right (by numseq instead of seqnums)
          #target_dict[k] = []
          target_dict[k] = OrderedDict()
      for j in range(len(I[i])):
           try:
              target_aa = index_to_aa[I[i][j]]
           except Exception as E:
               continue
           seqindex = target_aa.seqindex
           #target_dict[seqindex].append([target_aa, D[i][j]])
           target_dict[seqindex][target_aa] = D[i][j]
      #print("dict1", target_dict)
      query_aa_dict[query_aa] = target_dict
   return(query_aa_dict)


#@profile
def get_besthits2(I, minscore = 0.1):
   hitlist = []
   for k, v in I.items():
       for k1, v1 in v.items(): 
          if len(v1) > 0:
            if v1[0][1] >= minscore:
                hitlist.append([k, v1[0][0], v1[0][1]])
 
   #hitlist = [x for x in hitlist if x[2] >= minscore]
   return(hitlist)



#@profile
def get_rbhs2(hitlist):
   '''
   get_rhbs using igraph function is faster than this
   '''
   rbh_list = []  
   hitlist_edges_only = [[x[0], x[1]] for x in hitlist]

   for edge in hitlist: 
       inverse_edge = [edge[1], edge[0]]
       if inverse_edge in hitlist_edges_only:
           rbh_list.append(edge) 
   return(rbh_list)


#@profile
def get_besthits(I,  minscore = 0.1 ): 

   hitlist = []
   for aa in I.keys():
      for targetseq in I[aa].keys():
          if len(I[aa][targetseq]) > 0 :
              # Top score is always first
              #besthit = I[aa][targetseq][0]
              # Get the first item from an OrderedDict
              besthit = next(iter(I[aa][targetseq].items()))  
              besthit_aa = besthit[0] # AA
              besthit_score = besthit[1] #score to query aa

              if besthit_score >= minscore:
                  hitlist.append([aa, besthit_aa, besthit_score])

   return(hitlist)


#@profile
def get_rbhs(hitlist_top, min_edges = 0):
    '''
    [aa1, aa2, score (higher = better]
    '''

    logger.info("Get reciprocal best hits")

    G_hitlist = igraph.Graph.TupleList(edges=hitlist_top, directed=True)
    weights = [x[2] for x in hitlist_top]

    rbh_bool = G_hitlist.is_mutual()

    hitlist = []
    G_hitlist.es['weight'] = weights

    G_hitlist.es.select(_is_mutual=False).delete()
    G_hitlist.vs.select(_degree=0).delete()


    sources = [G_hitlist.vs[x.source]['name'] for x in G_hitlist.es()]
    targets = [G_hitlist.vs[x.target]['name'] for x in G_hitlist.es()]
    weights = G_hitlist.es()['weight']

    logger.debug("len check {} {} {}".format(len(sources), len(targets),len(weights)))
    hitlist = list(zip(sources,targets, weights))

    return(hitlist)

#@profile
def graph_from_rbh(rbh_list, directed = False):

    weights = [x[2] for x in rbh_list]
    G = igraph.Graph.TupleList(edges=rbh_list, directed = directed)
    G.es['weight'] = weights
    G = G.simplify(combine_edges = "first")
    return(G)


###################################
#### Amino acid clustering functions

#@profile
def get_cluster_dict(clusters):
    ''' in use'''
    pos_to_clustid = {}
    clustid_to_clust = {}
    for i in range(len(clusters)):
       clust = clusters[i]
       clustid_to_clust[i] = clust
       for seq in clust:
              pos_to_clustid[seq] = i

    return(pos_to_clustid, clustid_to_clust)


#@profile
def remove_highbetweenness(G, betweenness_cutoff = 0.10):
            n = len(G.vs())
            if n <= 5:
               return(G)
            bet = G.betweenness(directed=True) # experiment with cutoff based on n for speed
            bet_norm = []

           #get_new_clustering(G, betweenness_cutoff = 0.15,  apply_walktrap = True)
            correction = ((n - 1) * (n - 2)) / 2
            for x in bet:
                x_norm = x / correction
                #if x_norm > 0.45:
                bet_norm.append(x_norm)

                #bet_dict[sub_G.vs["name"]] = norm
            G.vs()['bet_norm'] = bet_norm
           # #ic("before", sub_G.vs()['name'])

            bet_names = list(zip(G.vs()['name'], bet_norm))
            # A node with bet_norm 0.5 is perfectly split between two clusters
            # Only select nodes with normalized betweenness before 0.45
            pruned_vs = G.vs.select([v for v, b in enumerate(bet_norm) if b < betweenness_cutoff])

            new_G = G.subgraph(pruned_vs)
            return(new_G)

#@profile
def check_completeness(cluster):

            seqnums = [x.seqnum for x in cluster]
            clustcounts = Counter(seqnums)
            # If any sequence found more than once
            for value in clustcounts.values():
                if value > 1:
                   return(False)
            return(True)



#@profile
def get_represented_seqs(connected_set):
    represented_seqs = list(set([x.seqnum for x in connected_set]))
    return(represented_seqs)



#@profile
def min_dup(connected_set, dup_thresh):

    represented_seqs = get_represented_seqs(connected_set)
    return(dup_thresh *  len(represented_seqs))




################################
#### Sequence clustering functions
#@profile
def candidate_to_remove(G, v_names,z = -5):


    weights = {}
    num_prots = len(G.vs())
    #ic("num_prots")
    if num_prots <=3:
        return([])

    for i in v_names:
        g_new = G.copy()
        vs = g_new.vs.find(name = i)
        weight = sum(g_new.es.select(_source=vs)['weight'])
        weights[i] = weight
    questionable_z = []
    for i in v_names:

        others = []
        for key,value in weights.items():
            if key == i:
                own_value = value
            else:
                others.append(value)

        #ic(own_value, others)
        seq_z = (own_value - np.mean(others))/np.std(others)
        #ic("sequence ", i, " zscore ", seq_z)


        if seq_z < z:
            questionable_z.append(i)

    #ic("questionable_z", questionable_z)
    return(questionable_z)




#################################
### Sequence search functions
#@profile
def graph_from_distindex(index, dist):
    '''
    Convert indicis and similarities to graph
    '''
    edges = []
    weights = []
    complete = []
    for i in range(len(index)):
       for j in range(len(index[i])):
          weight = dist[i,j]
          if weight < 0:
             weight = 0.001
          edge = (i, index[i, j])
          edges.append(edge)
          weights.append(weight)

    G = igraph.Graph.TupleList(edges=edges, directed=True) # Prevent target from being placed first in edges
    G.es['weight'] = weights

    return(G)

#@profile
def seq_index_search(sentence_array, k_select, s_index = None):
    '''
    Called by get_seqsims
    '''
    if not s_index:
        s_index = build_index_flat(sentence_array, scoretype = "cosinesim")

    s_distance, s_index2 = s_index.search(sentence_array, k = k_select)
    return(s_distance, s_index2)

#@profile
def get_seqsims(sentence_array, k = None, sentence_index = None):
    '''
    Take numpy array (float32) [[],[]] and calculate k-nearest neighbors either among array or from precomputed index of arrays. 
    
    If k is not provided, returned sequences will be the length of the sentence array if no precomputed index provided, or number of vectors in the index if that is provided.
    
    Time to return additional k is negligable
    
    '''
    if not k:
       k = sentence_array.shape[0]

    start_time = time()

    
    if not sentence_index:
        sentence_index = build_index_flat(sentence_array)

    #print("Searching index")
    distances, indices = seq_index_search(sentence_array, k, sentence_index)


    end_time = time()
    #print("Index searched in {} seconds".format( end_time - start_time))

    start_time = time()
    G = graph_from_distindex(indices, distances)
    end_time = time()
    #print("Index converted to edges in {} seconds".format(end_time - start_time))
    return(G, sentence_index)




################################
####### Limited search functions

#@profile
def get_unassigned_aas(seqs_aas, pos_to_clustid, too_small = []):
    ''' 
    Get amino acids that aren't in a sequence
    '''
    too_small_list = list(flatten(too_small))
    unassigned = []
    for i in range(len(seqs_aas)):
        prevclust = []
        nextclust = []
        unsorted = []
        last_unsorted = -1
        for j in range(len(seqs_aas[i])):
           if j <= last_unsorted:
               continue
           key = seqs_aas[i][j]

           if key in pos_to_clustid.keys():
              # Read to first cluster hit
              clust = pos_to_clustid[key]
              prevclust = clust
           # If it's not in a clust, it's unsorted
           else:
              unsorted = []
              unsorted.append(key)
              for k in range(j + 1, len(seqs_aas[i])):
                  key = seqs_aas[i][k]
                  if key in pos_to_clustid.keys():
                     nextclust = pos_to_clustid[key]
                     #ic(nextclust)
                     break
                  # Go until you hit next clust or end of seq
                  else:
                     unsorted.append(key)
                     last_unsorted = k

              unsorted = [x for x in unsorted if x not in too_small_list]
              unassigned.append([prevclust, unsorted, nextclust, i])
              nextclust = []
              prevclust = []
    return(unassigned)



#@profile
def get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid):
    # If unassigned sequence goes to the end of the sequence
    if not ending_clustid and ending_clustid != 0:
       ending_clustid = np.inf
    # If unassigned sequence extends before the sequence
    if not starting_clustid and starting_clustid != 0:
       starting_clustid = -np.inf

    # cluster_order must be zero:n
    pos_lists = []
    for x in seqs_aas:
            #ic('get_ranges:x:', x)
            pos_list = []
            startfound = False

            # If no starting clustid, add sequence until hit ending_clustid
            if starting_clustid == -np.inf:
                 startfound = True

            prevclust = ""
            for pos in x:
                if pos in pos_to_clustid.keys():
                    pos_clust = pos_to_clustid[pos]
                    prevclust = pos_clust

                    # Stop looking if clustid after ending clustid
                    if pos_clust >= ending_clustid:
                         break
                    # If the clustid is between start and end, append the position
                    elif pos_clust > starting_clustid and pos_clust < ending_clustid:
                        pos_list.append(pos)
                        startfound = True

                    # If no overlap (total gap) make sure next gap sequence added
                    elif pos_clust == starting_clustid:
                         startfound = True
                else:
                        if startfound == True or prevclust == cluster_order[-1]:
                           if prevclust:
                               if prevclust >= starting_clustid and prevclust <= ending_clustid:
                                   pos_list.append(pos)
                           else:
                              pos_list.append(pos)
            pos_lists.append(pos_list)
    return(pos_lists)

#@profile
def format_gaps(unassigned, highest_clustnum):
    # Do this when getting ranges in the first place. Add more to list
    # what is the different between this and np.inf and -inf as used before?

    output_unassigned = []
    for gap in unassigned:
        starting_clustid =  gap[0]
        ending_clustid = gap[2]

        if not ending_clustid and ending_clustid != 0:
            ending_clustid = highest_clustnum + 1 # capture the last cluster
         # If unassigned sequence extends before the sequence
        if not starting_clustid and starting_clustid != 0:
           starting_clustid = -1
        output_unassigned.append([starting_clustid, gap[1], ending_clustid])
    return(output_unassigned)


#@profile
def get_looser_scores(aa, index, hidden_states):
     '''Get all scores with a particular amino acid'''
     start  = time()
     hidden_state_aa = np.take(hidden_states, [aa.index], axis = 0)
     logging.debug("np.take time {}".format(time() - start))
     # Search the total number of amino acids
     n_aa = hidden_states.shape[0]

     # Faiss k limited to 2048
     if n_aa > 2048:
        n_aa = 2048
     D_aa, I_aa =  index.search(hidden_state_aa, k = n_aa)
     return(list(zip(D_aa.tolist()[0], I_aa.tolist()[0])))


#@profile
def get_particular_score(D, I, aa1, aa2):
        ''' Use with squish, replace with get_looser_scores '''

        scores = D[aa1.index][aa1.seqpos][aa2.index]
        ids = I[aa1.index][aa1.seqpos][aa2.index]
        for i in range(len(ids)):
           if ids[i] == aa2:
              return(scores[i])
        else:
           return(0)

#@profile
def get_set_of_scores(gap_aa, index, hidden_states, index_to_aa):

    start = time()
    candidates = get_looser_scores(gap_aa, index, hidden_states)
    end = time()
    logging.debug("search time {}".format(end-start))
    candidates_aa = []
    for score in candidates:
        try:
           target_aa = index_to_aa[score[1]]
        except Exception as E:
           # Not all indices correspond to an aa.
           continue
        candidates_aa.append([target_aa, score[0]])
    
    return(candidates_aa)


#@profile
def remove_overlap_with_old_clusters(new_clusters, prior_clusters):
    '''
    Discard any new clusters that contain elements of old clusters
    Only modify new clusters in best match-to-cluster process
    '''

    aas_in_prior_clusters = list(flatten(prior_clusters))
    #ic("aas in prior", aas_in_prior_clusters)

    final_new_clusters = []
    for n in new_clusters:
        #for p in prior_clusters:
        overlap =  list(set(aas_in_prior_clusters).intersection(set(n)))
        if len(overlap) > 0:
             #ic("prior", p)
             #ic("new with overlap of old ", n)
             continue
        elif n in final_new_clusters:
             continue
        else:
             final_new_clusters.append(n)

    return(final_new_clusters)


##################################
###### Address stranded columns of amino acids
#@profile
def address_stranded(alignment):
    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)
    to_remove =[]
    new_cluster_order = []
    new_clustid_to_clust = {}
    for i in range(0, len(cluster_order)):
         logging.debug(i)
         # If it's the first cluster
         if i == 0:
             prevclust = []

         else:
             prevclustid =cluster_order[i - 1]
             prevclust = clustid_to_clust[prevclustid]

         # If it's the last cluster 
         if i == len(cluster_order) - 1:
             nextclust = []
         else:
             nextclustid =cluster_order[i + 1]
             nextclust = clustid_to_clust[nextclustid]

         currclustid =cluster_order[i]
         currclust = clustid_to_clust[currclustid]
         removeclust = False
         #print(currclust)
         # Don't do stranding before bestmatch
         # Because a good column can be sorted into gaps
         
         for aa in currclust:
             if aa.prevaa not in prevclust and aa.nextaa not in nextclust:
                  logging.debug("remove_stranding: cluster {} {} {} {} {} {}".format(i, aa,  aa.prevaa, aa.nextaa,prevclust,nextclust))
                 
                  logging.debug("cluster {} {} {} {} {}".format(i,  aa.prevaa, aa.nextaa,prevclust,nextclust))
                  logging.debug("{} in clust {} is stranded".format(aa, currclust))
                  logging.debug("removing")
                  removeclust = True

         if removeclust == False:
             new_cluster_order.append(currclustid)
             new_clustid_to_clust[currclustid] = currclust
         else:
             logging.debug("Found stranding, Removing stranded clust {}".format(currclust))
    return(new_cluster_order, new_clustid_to_clust)




########################################
######## Main clustering functions


#@profile
def first_clustering(G,  betweenness_cutoff = .10, ignore_betweenness = False, apply_walktrap = True, pos_to_clustid = {}):
    '''
    Get betweenness centrality
    Each node's betweenness is normalized by dividing by the number of edges that exclude that node. 
    n = number of nodes in disconnected subgraph
    correction = = ((n - 1) * (n - 2)) / 2 
    norm_betweenness = betweenness / correction 
    '''

    logging.debug("Start first clustering")
    logging.debug("pos_to_clustid {}".format(pos_to_clustid))
    # Islands in the RBH graph
    logging.debug("{} {} {}".format(betweenness_cutoff, ignore_betweenness, apply_walktrap))
    islands = G.connected_components(mode = "weak")
    new_subgraphs = []
    cluster_list = []
    hb_list = []
    all_alternates_dict = {}

    # For each island, evaluate what to do

    # Natural clusters
    for sub_G in islands.subgraphs():
        logging.debug("sub_G".format(sub_G))
        # Don't remove edges if G = size 2
        if len(sub_G.vs()) < 4:
               betweenness_cutoff = 1
        else:
           betweenness_cutoff = betweenness_cutoff

        # First start with only remove very HB nodes
        new_G = remove_highbetweenness(sub_G, betweenness_cutoff = betweenness_cutoff)
        # Natural clusters after removing high betweeneess nodes
        sub_islands = new_G.connected_components(mode = "weak")

        for sub_sub_G in sub_islands.subgraphs():

            #ic("first_clustering: betweenness cutoff", betweenness_cutoff, "apply_walktrap", apply_walktrap)
            new_clusters, alternates_dict = get_new_clustering_old(sub_sub_G, betweenness_cutoff = betweenness_cutoff,  apply_walktrap = apply_walktrap, pos_to_clustid = pos_to_clustid)
            if alternates_dict:
                 all_alternates_dict = {**all_alternates_dict, **alternates_dict}
            logging.debug("get_new_clustering".format(new_clusters))
            #ic("first_clustering:adding", new_clusters, "to cluster_list")
            cluster_list = cluster_list + new_clusters

    #ic("cluster_list at end of first_clustering", cluster_list)
    #ic("Alternates at end of first clustering", all_alternates_dict)
    return(cluster_list, all_alternates_dict)


# Absolutely too much going on here with the clustering. 


# Take network
# For each island in network:
#   

#@profile
def remove_low_authority(G):

    hub_scores = G.hub_score()
    names = [x['name'] for x in G.vs()]
    vx_names = G.vs()
    hub_names = list(zip(names, vx_names, hub_scores))

    high_authority_nodes = [x[0] for x in hub_names if x[2]  > 0.2]
    high_authority_nodes_vx = [x[1] for x in hub_names if x[2]  > 0.2]

    low_authority_nodes = [x[0] for x in hub_names if x[2]  <= 0.2]
    low_authority_node_ids = [x[1] for x in hub_names if x[2]  <= 0.2]

    #ic("removing low authority_nodes", low_authority_nodes)
    if len(low_authority_nodes) > 0 and len(high_authority_nodes) > 0:
        high_authority_edges =  G.es.select(_within = high_authority_nodes_vx)
        #If not edges left after removing low authority nodes
        if len(high_authority_edges) > 0:

            G.delete_vertices(low_authority_node_ids)
        names = [x['name'] for x in G.vs()]
    return(G)



#@profile
def process_island(island, betweenness_cutoff = 0.1, apply_walktrap = True, prev_G = None, clusters = []):
    if len(island.vs()) < 4:
        betweenness_cutoff = 1

    island_names = island.vs()['name']


    if check_completeness(island_names) == True:
         clusters = clusters + island_names
  
    elif len(island_names) <= min_dup(island_names, 1.2) or len(island_names) < 5:
        island_names, alternates_dict = remove_doubles_by_graph(island_names, island)
        if check_completeness(island_names):
               #print("new cluster after remove_by_graph", island_names)
               #return(island_names) 
               clusters = clusters + island_names
    else:
        ################# 
        # Betweenness
        new_island = remove_highbetweenness(island, betweenness_cutoff = betweenness_cutoff)
        new_islands = new_island.clusters(mode = "weak")
        if len(new_islands.subgraphs()) > 1:
            clusters = process_network(new_islands, clusters)

        else:
            new_island_names = new_island.vs()['name']
            if check_completeness(new_island_names) == True:
                 clusters = clusters + new_island_names
            elif len(new_island_names) <= min_dup(new_island_names, 1.2) or len(new_island_names) < 5:
                 new_island_names, alternates_dict = remove_doubles_by_graph(new_island_names, new_island)
                 if check_completeness(new_island_names):
                     
                     clusters = clusters + new_island_names
            else:
               ######################3
               # Low authority
               new_new_island = remove_low_authority(new_island)
               new_new_islands = new_new_islands.clusters(mode = "weak")
               if len(new_new_islands.subgraphs()) > 1:
                  new_clusters = process_network(new_new_islands, betweenness_cutoff = betweenness_cutoff)
                  clusters.append(new_clusters)
               else:
                  new_new_island_names = new_new_island.vs()['name']
                  if check_completeness(new_new_island_names) == True:
                      clusters = clusters + new_new_island_names
                  elif len(new_new_island_names) <= min_dup(new_new_island_names, 1.2) or len(new_new_island_names) < 5:
                     new_new_island_names, alternates_dict = remove_doubles_by_graph(new_new_island_names, new_new_island)
                     if check_completeness(new_new_island_names):
                          clusters + clusters + new_island_names
                  ##########################
                  # Walktrap   
                  else:
                      nnn_islands = new_new_islands.community_walktrap(steps = 3, weights = 'weight').as_clustering()
                      if len(nnn_islands.subgraphs()) > 1:
                          clusters = process_network(nnn_islands, clusters, betweenness_cutoff = betweenness_cutoff)
    return(clusters)


# Work-in-progress new main clustering functions to replace first_clustering function
#@profile
def process_network(G, betweenness_cutoff = 0.1, apply_walktrap = True, prev_G = None, clusters = []):
    
    islands = G.connected_components(mode = "weak")
    for island in islands:
         clusters = process_island(island, clusters, betweenness_cutoff = betweenness_cutoff, apply_walktrap = apply_walktrap)

    #for x in clusters:
          #print(x)
    return(clusters)

def delete_low_authority(G):
      hub_scores = G.hub_score()
      names = [x['name'] for x in G.vs()]
      indices = [x.index for x in G.vs()]

      vx_names = G.vs()
      hub_names = list(zip(names, vx_names, hub_scores))

      logging.debug(hub_names)
      high_authority =  [x[0:2] for x in hub_names if x[2]  > 0.2]
      high_authority_nodes = [x[0] for x in high_authority]
      high_authority_nodes_vx = [x[1] for x in high_authority]
      

      logging.debug("high authority".format(high_authority_nodes))
      low_authority_nodes = [x for x in names if x not in high_authority_nodes]   #[x[0] for x in hub_names if x[2]  <= 0.2]
      low_authority_node_ids = [x for x in indices if x not in high_authority_nodes_vx]  # [x[1] for x in hub_names if x[2]  <= 0.2]

      logging.debug("low authority".format(low_authority_nodes))
      
      if len(low_authority_nodes) > 0 and len(high_authority_nodes) > 0:
          high_authority_edges =  G.es.select(_within = high_authority_nodes_vx)
          #If not edges left after removing low authority nodes
          if len(high_authority_edges) > 0:
              G.delete_vertices(low_authority_node_ids)
      return(G)


def first_clustering2(G, natural = False, remove_low_authority = False):
    # Different/worse than first_clustering function. Find out in what ways they aren't equivalent

    #logging.info("Doing clusterings, number of vertices {}, number of edges {}".format(G.vcount(), G.ecount()))

    if G.vcount() == 0:
        return([], G)
    new_clusters = []

    # Is not working
    #if remove_low_authority == True:
    #   G = delete_low_authority(G)

    if natural == True:
       clustering = G.connected_components(mode = "weak")

    else:
       clustering = G.community_walktrap(steps = 3, weights = 'weight').as_clustering()

    clusters = [x.vs()["name"] for x in clustering.subgraphs()]
    #print("clusters", clusters)

    reduced_clusters = []
    for clust in clusters:
         #print("first clusters before filtering", clust)
         to_remove = get_doubled_seqnums(clust)
         if check_completeness(clust) == True:
            reduced_clusters.append(clust)
   
         else:
             min_dupped =  min_dup(clust, 1.2) 
             if len(clust) <= min_dupped:
                 clust, alternates_dict = remove_doubles_by_graph(clust, G)
                 #print("clust after remove_doubles", clust) 
                 if check_completeness(clust) == True:
                      #print("doubles removed by graph")
                      reduced_clusters.append(clust)
 
                 # logging.debug("pre-removal by consistency")
                 # clust = remove_doubles_by_consistency(clust, pos_to_clustid)
                 # to_remove = get_doubled_seqnums(clust)

             #if len(to_remove) > 0:
             #     clust, alternates_dict = remove_doubles_by_scores(clust, index, hidden_states, index_to_aa)
                  #if alternates_dict:
             #     #     all_alternates_dict = {**all_alternates_dict, **alternates_dict}
              #    to_remove = get_doubled_seqnums(clust)

     

    new_clusters = [x for x in reduced_clusters if check_completeness(x) == True]

  
    # Remove doubles



    # Remove amino acids that have been clustered from the graph for future rounds
    clustered_names = list(flatten(new_clusters)) 
    to_delete = [v.index for v in G.vs.select(name_in=clustered_names)]
    G.delete_vertices(to_delete)   
 
    
    # The idea here is to look for natural clusters, 
    # Remove those vertices from the graph
    # Then do a walktrap, which will break up the network. (Find graph from clustering function?)
    # Remove any complete clusters from the graph. (Including too small clusters). 
    # Then that walktraped graph will be the new input graph (controlled when function is called).
    return(new_clusters, G)
    


#@profile
def get_new_clustering(G, betweenness_cutoff = 0.10,  apply_walktrap = True, prev_G = None, pos_to_clustid = {}):
    '''
    Main clustering function, goes back and forth with process_connected_set
    '''
    #ic(G.vs()['name'])
    # Prevent getting stuck 
    if prev_G:
        #ic(prev_G.vs()['name'])
        if G.vs()['name'] == prev_G.vs()['name']:
            return([], {})
    #ic("get_new_clustering")
    #all_alternates_dict = {**all_alternates_dict, **alternates_dict}
    new_clusters = []
    connected_set = G.vs()['name']
    all_alternates_dict = {}
    new_clusters = []

    finished = check_completeness(connected_set)
    # Complete if no duplicates
    if finished == True:
        #ic("finished connected set", connected_set)
        new_clusters = [connected_set]

    else:
        min_dupped =  min_dup(connected_set, 1.2)
        # Only do walktrap is cluster is overlarge
        #ic("min_dupped at start", min_dupped)
        names = [x['name'] for x in G.vs()]
        #ic("names at start", names)

        if (len(connected_set) > min_dupped) and apply_walktrap and len(G.vs()) >= 5:
            # First remove weakly linked aa's then try again
            # Walktrap is last resort
            #ic("similarity_jaccard, authority_score")
            G = delete_low_authority(G)
            #hub_scores = G.hub_score()
            #names = [x['name'] for x in G.vs()]
            #ic(names)
            #vx_names = G.vs()
            #hub_names = list(zip(names, vx_names, hub_scores))

            #high_authority_nodes = [x[0] for x in hub_names if x[2]  > 0.2]
            #high_authority_nodes_vx = [x[1] for x in hub_names if x[2]  > 0.2]

            #low_authority_nodes = [x[0] for x in hub_names if x[2]  <= 0.2]
            #low_authority_node_ids = [x[1] for x in hub_names if x[2]  <= 0.2]

            #ic("removing low authority_nodes", low_authority_nodes)
            #if len(low_authority_nodes) > 0 and len(high_authority_nodes) > 0:
            #    high_authority_edges =  G.es.select(_within = high_authority_nodes_vx)
            #    #If not edges left after removing low authority nodes
            #    if len(high_authority_edges) > 0:
#
            #       G.delete_vertices(low_authority_node_ids)
            names = [x['name'] for x in G.vs()]

            min_dupped = min_dup(names, 1.2)
            # If removing low authority nodes made the cluster small enough, continue to process connected set
            if len(names) <= min_dupped:
                #ic("get_new_clustering:new_G", G)
                processed_cluster, alternates_dict =  process_connected_set(names, G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff, pos_to_clustid = pos_to_clustid)
                #ic("processed_cluster", processed_cluster)
                if alternates_dict:
                   all_alternates_dict = {**all_alternates_dict, **alternates_dict}
                new_clusters = new_clusters + processed_cluster

            # Otherwise, try applying walktrap 
            else:
                #ic("applying walktrap")
                #ic("len(connected_set, min_dupped", len(connected_set), min_dupped)
                clustering = G.community_walktrap(steps = 3, weights = 'weight').as_clustering()
                for sub_G in clustering.subgraphs():
                     sub_connected_set =  sub_G.vs()['name']
                     #ic("post cluster subgraph", sub_connected_set)

                     # New clusters may be too large still, try division process w/ betweenness

                     processed_cluster, alternates_dict = process_connected_set(sub_connected_set, sub_G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff, pos_to_clustid = pos_to_clustid)
                     new_clusters = new_clusters + processed_cluster
                     if alternates_dict:
                        all_alternates_dict = {**all_alternates_dict, **alternates_dict}
        else:
            #ic("get_new_clustering:connected_set", connected_set)
            processed_cluster, alternates_dict =  process_connected_set(connected_set, G, dup_thresh = 1.2, betweenness_cutoff = betweenness_cutoff, pos_to_clustid = pos_to_clustid)
            new_clusters = new_clusters + processed_cluster
            if alternates_dict:
                 all_alternates_dict = {**all_alternates_dict, **alternates_dict}

    return(new_clusters, all_alternates_dict)


def process_connected_set(connected_set, G, dup_thresh = 1.2,  betweenness_cutoff = 0.10, prev_G = None, pos_to_clustid = {}):
    ''' 
    This will take a connected sets and 
    1) Check if it's very duplicated
    2) If so, remove high betweenness nodes, and check for completeness again
    3) If it's not very duplicated, removed any duplicates by best score
    '''
    logging.debug("process_connected_set: betweenness {} dup_thresh {}".format(betweenness_cutoff, dup_thresh))
    new_clusters = []
    all_alternates_dict = {}
    min_dupped =  min_dup(connected_set, dup_thresh)
    if len(connected_set) > min_dupped:
        logging.debug("cluster too big {}".format(connected_set))
        # TRY removing high betweenness and evaluating completeness
        new_G = remove_highbetweenness(G, betweenness_cutoff = 0.10)
        #ic("prebet", G.vs()['name'])
        #ic("postbet", new_G.vs()['name'])
        #if len(new_Gs) > 1:
        new_islands = new_G.connected_components(mode = "weak")
        for sub_G in new_islands.subgraphs():
                alternates_dict = {}
                sub_connected_set = sub_G.vs()['name']
                #ic("postbet_island", sub_connected_set)
                sub_min_dupped =  min_dup(sub_connected_set, dup_thresh)
                #ic("sub_min_dupped", sub_min_dupped)
                # Actually going to keep all clusters below min dup thresh
                if (len(sub_connected_set) <= sub_min_dupped) or (len(sub_connected_set) <= 5):
                    if pos_to_clustid:
                        logging.debug("trimming by consistency")
                        trimmed_sub_connected_set = remove_doubles_by_consistency(sub_connected_set, pos_to_clustid)
                    else:
                        logging.debug("trimming by graph")
                        trimmed_sub_connected_set, alternates_dict = remove_doubles_by_graph(sub_connected_set, sub_G
)
                else:
                    new_walktrap_clusters, alternates_dict = get_new_clustering(sub_G, betweenness_cutoff = betweenness_cutoff,  apply_walktrap = True, prev_G = G, pos_to_clustid = pos_to_clustid)
                    #print("CONNECTED_SET", connected_set)
                    #print("SUB_CONNECTECT_SET", sub_connected_set)
                    #print("NEW_WALKTRAP_CLUSTERS", new_walktrap_clusters)

                    #if new_walktrap_clusters == connected_set
                    for cluster in new_walktrap_clusters:
                            new_clusters.append(cluster)
                    #ic(new_clusters)
                    if alternates_dict:
                        all_alternates_dict = {**all_alternates_dict, **alternates_dict}
        #return(new_clusters)
    else:
        if pos_to_clustid:
            logging.debug("trimming by consistency")
            trimmed_connected_set = remove_doubles_by_consistency(connected_set, pos_to_clustid)
   
        else:
            logging.debug("trimming by graph")
            trimmed_connected_set, all_alternates_dict = remove_doubles_by_graph(connected_set, G)
        logging.debug("after trimming by removing doubles".format(trimmed_connected_set))
        new_clusters = [trimmed_connected_set]
    # If no new clusters, returns []
    return(new_clusters, all_alternates_dict)


#@profile
def process_connected_set_new(connected_set, G, dup_thresh = 1.2,  betweenness_cutoff = 0.10, prev_G = None):
    ''' 
    # This is worse than the old process connected set
    This will take a connected sets and 
    1) Check if it's very duplicated
    2) If so, remove high betweenness nodes, and check for completeness again
    3) If it's not very duplicated, removed any duplicates by best score
    '''

    #ic("process_connected_set: betweenness", betweenness_cutoff, "dup_thresh", dup_thresh)
    new_clusters = []
    all_alternates_dict = {}
    min_dupped =  min_dup(connected_set, dup_thresh)
    if len(connected_set) > min_dupped:

        # Try removing high betweenness and evaluating completeness
        #print("Check for betweenness")

        new_G = remove_highbetweenness(G, betweenness_cutoff = 0.10)
        new_islands = new_G.connected_components(mode = "weak")
        for sub_G in new_islands.subgraphs():
                alternates_dict = {}
                sub_connected_set = sub_G.vs()['name']
                #ic("postbet_island", sub_connected_set)
                sub_min_dupped =  min_dup(sub_connected_set, dup_thresh)

                # Actually going to keep all clusters below min dup thresh
                if (len(sub_connected_set) <= sub_min_dupped) or (len(sub_connected_set) <= 5):
                    # Why not bu consistency at this point?
                    trimmed_sub_connected_set, alternates_dict = remove_doubles_by_graph(sub_connected_set, sub_G)

                else:
                    #ic("still about min_dupped applying walktrap")
                    new_walktrap_clusters, alternates_dict = get_new_clustering(sub_G, betweenness_cutoff = betweenness_cutoff,  apply_walktrap = True, prev_G = G, pos_to_clustid = pos_to_clustid)
                    for cluster in new_walktrap_clusters:
                            new_clusters.append(cluster)
                    if alternates_dict:
                        all_alternates_dict = {**all_alternates_dict, **alternates_dict}

    else:
        # Why not remove by consistency
        # Check the dedup function

        trimmed_connected_set, all_alternates_dict = remove_doubles_by_graph(connected_set, G)
        #ic("after trimming by removing doubles", trimmed_connected_set)
        new_clusters = [trimmed_connected_set]
    # If no new clusters, returns []
    return(new_clusters, all_alternates_dict)








########################################
######### Limited search: clustering phase

#@profile
def fill_in_unassigned_w_clustering(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, I2, gapfilling_attempt,  minscore = 0.1, minclustsize = 2, ignore_betweenness = False, betweenness_cutoff = 0.3, apply_walktrap = True, rbh_dict = {}, seqnames = [], args = None ):
    '''
    Run the same original clustering, ??? allows overwritting of previous clusters
    
    If a new member of an rbh cluster has a large unassigned range, check if has higher rbh t o sequence?
    '''
    start_clust = time()
    all_alternates_dict = {}
    clusters_filt = list(clustid_to_clust.values())
    #ic("fill_in_unassigned_w_clustering: TESTING OUT CLUSTERS_FILT")
    #for x in clusters_filt:
        #ic("fill_in_unassigned_w_clustering: preassignment clusters_filt", x)
    # extra arguments?
    edgelist = []
    #ic("fill_in_unassigned_w_clustering:unassigned", unassigned)
    get_targets_time = time()

    for gap in unassigned:
        t1_time = time()
        #print("fill_in_unassigned_w_clustering", gap)
        gap_edgelist = get_targets(gap, seqs_aas, cluster_order, pos_to_clustid)
        edgelist = edgelist + gap_edgelist
        #print("TESTER: individual time", time() - t1_time)
    #print("TESTER: get_targets_time", time() - get_targets_time)

    subgraph_time = time()
    edgelist_G = igraph.Graph.TupleList(edges=edgelist, directed=False)
    edgelist_G = edgelist_G.simplify() # Remove duplicate edges
    islands = edgelist_G.connected_components(mode = "weak")

    # Instead of full edgelist, just do a  minimum spanning tree to allow finding islands

    new_clusters_from_rbh = []
    all_new_rbh  = []
    total_address = time()

    # This is parallelizable here
    for sub_G in islands.subgraphs():
       neighbors = {}
     
       for vertex in sub_G.vs():
           vertex_neighbors = sub_G.neighbors(vertex)
           logging.debug("vertex, neighbors {} {}".format(vertex, vertex_neighbors))
           neighbors[vertex['name']] = sub_G.vs[vertex_neighbors]["name"] + [vertex['name']]
       logging.debug("name     {}".format(sub_G.vs()['name']))
       logging.debug("neighbors {}".format(neighbors))

       address_time = time()

       newer_clusters, newer_rbh, rbh_dict, alternates_dict = address_unassigned_aas(sub_G.vs()['name'], I2, minscore = 0.5, ignore_betweenness = False,  betweenness_cutoff = 0.3, minsclustsize = 2, apply_walktrap = apply_walktrap, rbh_dict = rbh_dict, pos_to_clustid = pos_to_clustid)

       #print("TESTER: address time", time()  - address_time)
       #print("newer clusters", newer_clusters)
       #ic(newer_rbh[0:5])
       all_alternates_dict = {**all_alternates_dict, **alternates_dict}
       new_clusters_from_rbh  = new_clusters_from_rbh + newer_clusters
       all_new_rbh = all_new_rbh + newer_rbh

    #print("TESTER, subgraph time", time() - subgraph_time)
    #print("TESTER total_addressing time", time() - total_address)
    new_clusters = []
    too_small = []
    #print("new_clusters_from_rbh", new_clusters_from_rbh)


    for clust in new_clusters_from_rbh:
          if len(clust) >= minclustsize:
                new_clusters.append(clust)
          else:
             # This is never happening?
             if len(clust) > 1:
                too_small.append(clust)
    dedup_time = time()
    # It was at one point very important to removeSublist here
    # Is it anymore?
    new_clusters = removeSublist(new_clusters)
    aa_counter = {}
    new_clusters_flat  = flatten(new_clusters)
    aa_counts = Counter(new_clusters_flat)
    dupped_aas = {key for key, val in aa_counts.items() if val != 1}
    #ic("dupped aas", dupped_aas)

    # From doubled aas from clusters list of lists
    new_clusters = [[aa for aa in clust if aa not in dupped_aas] for clust in new_clusters]


    # If this makes clusters too small remove them 
    new_clusters = [clust for clust in new_clusters if len(clust) >= minclustsize]

    #print("TESTER dedup time", time() - dedup_time)
    overlap_time = time()
    # In this section, remove any members of a new cluster that would bridge between previous clusters and cause over collapse
    new_clusters_filt = []
    for clust in new_clusters:
         clustids = []
         posids = []
         new_additions = []
         for pos in clust:
            #ic(pos)
            if pos in pos_to_clustid.keys():
               clustid = pos_to_clustid[pos]
               clustids.append(clustid)
               posids.append(pos)
               #ic(pos, clustid)
            else:
                # Position wasn't previously clustered
                #ic("new_additions", clust,pos)
                new_additions.append(pos)
         #ic("new_additions", new_additions)
         #ic("posids", posids)                  
         if len(list(set(clustids))) > 1:
            #ic("new cluster contains component of multiple previous clusters. Keeping largest matched cluster")
            clustcounts = Counter(clustids)
            largest_clust = max(clustcounts, key=clustcounts.get)
            sel_pos = [posids[x] for x in range(len(posids)) if clustids[x] == largest_clust]
            #ic("Split cluster catch", clustcounts, largest_clust, posids, clustids, sel_pos)
            new_clust = sel_pos + new_additions

         else:
            new_clusters_filt.append(clust)


    new_clusters_filt = removeSublist(new_clusters_filt)

    clusters_new = remove_overlap_with_old_clusters(new_clusters_filt, clusters_filt)
    clusters_merged = clusters_new + clusters_filt
    #print("TESTER overlap time", time() - overlap_time)
    #for x in clusters_merged:
    #   logging.debug("clusters_merged", x)
    #print("TESTER:clustering_time", time() - start_clust)
    start = time() 
    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(clusters_merged, seqs_aas, gapfilling_attempt, minclustsize, all_alternates_dict = all_alternates_dict, seqnames = seqnames, args = args)
    #print("TESTER:organizing_time", time() - start)

    #print("clustid_to_clust after feedback arc removal", clustid_to_clust)
    #print(too_small)
    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment, too_small, rbh_dict, all_new_rbh)




#@profile

def get_targets_old(gap, seqs_aas, cluster_order, pos_to_clustid):

        starting_clustid = gap[0]
        ending_clustid = gap[2]
        gap_seqaas = gap[1]
        #gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]
        target_aas_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)

        edgelist = []
        target_aas = list(flatten(target_aas_list))
        for query in gap_seqaas:
              for target in target_aas:
                  edgelist.append([query, target])

        return(edgelist)

def get_targets(gap, seqs_aas, cluster_order, pos_to_clustid):

        starting_clustid = gap[0]
        ending_clustid = gap[2]
        gap_seqaas = gap[1]
        #gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]
        get_range_time = time()
        # This is fast enough
        target_aas_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)
        #print("TESTER get_ranges time", time() - get_range_time)

        print_time = time()
        #print("target_aas_list", gap, target_aas_list)
        #print("TESTER print time", time() - print_time)
        edgelist = []

        flatten_time = time()
        target_aas = list(flatten(target_aas_list))
        #print("TESTER flatten_time", time() - flatten_time)
        last_time = time()
        # THIS LOOP IS TAKING WAYY TOO LONG
        #print("TESTER targets", target_aas)
        #print("TESTER gap_seqaas", gap_seqaas)
        #print("TESTER listlens", len(gap_seqaas), len(target_aas))
        

        all_aas = target_aas + gap_seqaas
        #print("TESTER, all_aas", all_aas)
        # Make scope aas into minially connected network. 
        for i in range(len(all_aas) - 1):
            edgelist.append([all_aas[i], all_aas[i + 1]])

        #for edge in edgelist:
        #       logging.debug("e1", edge)
        #edgelist = []
        #for query in gap_seqaas:
        #      for target in target_aas:
        #          edgelist.append([query, target])
        #print("TESTER edgelist", len(edgelist))
        #for edge in edgelist2:
        #    logging.debug("e2", edge)        
        #print("TESTER: last time", time() - last_time)
        return(edgelist)


#@profile
def address_unassigned_aas(scope_aas, I2, minscore = 0.5, ignore_betweenness = False,  betweenness_cutoff = 0.3, minsclustsize = 2, apply_walktrap = True, rbh_dict = {}, pos_to_clustid = {}):

        # Avoid repeats of same rbh calculation
        if True: #not frozenset(scope_aas) in rbh_dict.keys():
            limited_I2 = {}
            # Suspect that this is slow

            copy_time = time()
            for key in I2.keys():
               if key in scope_aas:
                   limited_I2[key] = I2[key].copy()
            #print("TESTER copytime", time() - copy_time)
            #ic("limited")
            #ic("address_unassigned_aas:new_rbh_minscore", minscore)

            # The larger the number of k selected at the knn stage, the slower this will be          
            limited_I2_start = time()
            # This conversion to a set is very important
            scope_aas_set = set(scope_aas)
            # REVISIT HERE, what is the neighbors object in hfa2
            #print("score_aas_set", scope_aas_set)
            for query_aa in limited_I2.keys():
                 # These keys 
                 for seq in limited_I2[query_aa].keys():
                       #limited_I2[query_aa][seq] = [x for x in limited_I2[query_aa][seq] if x[0] in neighbors[query_aa]]
                       limited_I2[query_aa][seq] =  OrderedDict({k:v for k,v in limited_I2[query_aa][seq].items() if k in scope_aas_set}) 

            logging.debug(limited_I2)
            # This basically scales with k from KNN
            #print("TESTER limited_I2_select", time() - limited_I2_start)
            # Get reciprocal best hits in a limited range
            new_hitlist = get_besthits(limited_I2, minscore)
            
            logging.debug(len(new_hitlist))


            new_rbh = get_rbhs(new_hitlist)
            logger.debug(len(new_rbh))
            new_rbh = maximum_increasing(new_rbh)
            logger.debug(len(new_rbh))
            #ic("new_rbh from get_rbh", new_rbh[0:5])


            rbh_dict[frozenset(scope_aas)] = new_rbh
        else:
            #ic("address_unassigned_aas RBH pulled from cache")
            new_rbh = rbh_dict[frozenset(scope_aas)]
        for x in new_rbh:
             logger.debug("address_unassigned_aas:new_rbh {}".format(x))
       
        G = graph_from_rbh(new_rbh)
        all_alternates_dict = {}
        #cluster_list1 = []
        #cluster_list2 = []
        #cluster_list3 = []
        #cluster_list4 = []

        #cluster_list1, G = first_clustering2(G, natural = True)
        #print("cluster_list1", cluster_list1)
 
        #cluster_list2, G = first_clustering2(G, natural = True, remove_low_authority = True)
        #print("cluster_list2", cluster_list2)
        #print("after 2", G)
        #cluster_list3, G = first_clustering2(G, natural = False, remove_low_authority = True)

        #print("after 3", G)
        #print("cluster_list3", cluster_list3)
        #cluster_list4, G = first_clustering2(G, natural = False, remove_low_authority = True)
        #print("cluster_list4", cluster_list4)
        #new_clusters = cluster_list1 + cluster_list2 + cluster_list3 + cluster_list4

        # TEST CHANGE
        new_clusters, all_alternates_dict  = first_clustering(G, betweenness_cutoff = betweenness_cutoff,  ignore_betweenness = ignore_betweenness, apply_walktrap = apply_walktrap, pos_to_clustid = pos_to_clustid )
        logger.debug("new_clusters_old_function {}".format(new_clusters))
        for x in new_clusters:
            logger.debug("address_unassigned_aas:new_clusters {}".format(x))
       
        clustered_aas = list(flatten(new_clusters))

        return(new_clusters, new_rbh, rbh_dict, all_alternates_dict)




#@profile
def removeSublist(lst):
    #https://www.geeksforgeeks.org/python-remove-sublists-that-are-present-in-another-sublist/
    curr_res = []
    result = []
    for ele in sorted(map(set, lst), key = len, reverse = True):
        if not any(ele <= req for req in curr_res):
            curr_res.append(ele)
            result.append(list(ele))

    return result


########################################
######### Limited search: best match phase


#@profile
def fill_in_unassigned_w_search(unassigned, seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, index, hidden_states,  index_to_aa, gapfilling_attempt, minclustsize = 1, remove_both = True, match_dict = {}, seqnames = [], args = None, I2 = {}):

    '''
    Try group based assignment, this time using new search for each unassigned
    Decision between old cluster and new cluster?
    '''
    clusters = list(clustid_to_clust.values())
    matches = []

    match_scores = []
    
    unassigned = format_gaps(unassigned, max(clustid_to_clust.keys()))

    for gap in unassigned:
        logging.debug("gap {}".format(gap))
        starting_clustid =  gap[0]
        ending_clustid = gap[2]
        #print(gap)
        if starting_clustid in clustid_to_clust.keys():
              # If before start of clusts, will be -1
              starting_clust =  clustid_to_clust[starting_clustid]
        else:
              starting_clust = []
        if ending_clustid  in clustid_to_clust.keys():
              ending_clust =  clustid_to_clust[ending_clustid]
        else:
              ending_clust = []

        already_searched = frozenset(gap[1] + starting_clust + ending_clust)
        if already_searched in match_dict.keys():
            #ic("Matches pulled from cache")
            matches = matches + match_dict[already_searched]
        else:
            gap_matches = []

            for gap_aa in gap[1]:
                #start = time()
                #candidates_aa2 = I2[gap_aa]
                #print("pull from dict", time() - start)

                #start = time()
                
                #candidates_aa = get_set_of_scores(gap_aa, index, hidden_states, index_to_aa)
                #print("get_set_of_scores", time() - start)
                # Slower
                #print('candidates_aa', candidates_aa)
                start = time()
                # For each clustid_to_clust, it should be checked for consistency. 
                #output = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, candidates_aa, I2)
                output = get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust,  I2)
                # Fast
                #print("get_best_matches", time() - start)
                if output:
                   logging.debug("output {}".format(output))
                   gap_matches.append(output)
            matches = matches + gap_matches
            match_dict[already_searched] = gap_matches

    #for x in matches:
        #ic("match", x)

    clustid_to_clust = get_best_of_matches(clustid_to_clust, matches)

    clusterlist = list(clustid_to_clust.values())

    new_clusterlist = []
    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusterlist)
    all_alternates_dict  =  {}
    for clustnum, clust in clustid_to_clust.items():
         to_remove = get_doubled_seqnums(clust)
         if len(to_remove) > 0:
              clust = remove_doubles_by_consistency(clust, pos_to_clustid)
              to_remove = get_doubled_seqnums(clust)
         if len(to_remove) > 0:
              clust, alternates_dict = remove_doubles_by_scores(clust, index, hidden_states, index_to_aa)
              if alternates_dict:
                   all_alternates_dict = {**all_alternates_dict, **alternates_dict}
              to_remove = get_doubled_seqnums(clust)
         if len(to_remove) == 0:
            new_clusterlist.append(clust)

    #for x in new_clusterlist:
        #ic("Clusters from best match", x)

    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(new_clusterlist, seqs_aas, gapfilling_attempt, minclustsize, all_alternates_dict = all_alternates_dict, seqnames = seqnames, args = args)
    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment, match_dict)



#@profile
def get_best_of_matches(clustid_to_clust, matches):
    for clustid in clustid_to_clust.keys():
         potential_matches = [x for x in matches if x[1] == clustid]

         if potential_matches :
             match_seqnums = [x[0].seqnum for x in potential_matches]
             match_seqnums = list(set(match_seqnums))
             for seqnum in match_seqnums:
                 potential_matches_seqnum = [x for x in potential_matches if x[0].seqnum == seqnum]
                 #ic("seqnum: {}, matches {}".format(seqnum, potential_matches_seqnum)) 

                 current_bestscore = 0
                 current_bestmatch = ""
                 for match in potential_matches_seqnum:
                     if match[2] > current_bestscore:
                           current_bestscore = match[2]
                           current_bestmatch = match[0]


                 newclust = clustid_to_clust[clustid] + [current_bestmatch]
                 #ic("Updating {} from {} to {}".format(clustid, clustid_to_clust[clustid], newclust))
                 clustid_to_clust[clustid] = newclust
    #ic(clustid_to_clust)
    return(clustid_to_clust)

#@profile
#def get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, candidates_w_score):
#     # Don't update the cluster here, send it back. 
#     # candidate_w_score = zipped tuple [aa, score]
#    scores = []
#    clustid_to_clust[clustid] = newclust
#    #ic(clustid_to_clust)
#    return(clustid_to_clust)

#@profile
def get_best_matches(starting_clustid, ending_clustid, gap_aa, clustid_to_clust, I2): #, candidates_w_score, I2):
     # Don't update the cluster here, send it back. 
     # candidate_w_score = zipped tuple [aa, score]
    scores = []
    current_best_score = 0
    current_best_match = ""
    match_found = False
    #ic(starting_clustid, ending_clustid)
    #ic(gap_aa)
    #ic(clustid_to_clust)
    #print("gap_aa", gap_aa)
    #print(I2[gap_aa])
    for cand in range(starting_clustid + 1, ending_clustid):
         #print("candidate", gap_aa, cand,  clustid_to_clust[cand], "bestscore", current_best_score, current_best_match)

         candidate_aas =  clustid_to_clust[cand]
         
         start = time()
         incluster_scores = []
     
         for x in candidate_aas:
            try:
                incluster_scores.append([x, I2[gap_aa][x.seqindex][x]])
            except Exception as E:
                #print(x, "not found")                
                # Not all clustered aa's will by in the I2 for a particular query aa
                continue
         
         #print("incluster scores2", incluster_scores2)
         #print("ic scores2", time() - start)
         #for target in candidates_aas:
         #incluster_scores2 = [x for x in I2[aa] if x
               
         
         #incluster_scores = [x for x in candidates_w_score if x[0] in candidate_aas]
         #print("incluster scores", incluster_scores)
         if incluster_scores:
             total_incluster_score = sum([x[1] for x in incluster_scores]) / len(incluster_scores) # Take the mean score within the cluster. Or median?
             #print("total_inclucster", total_incluster_score)
             if total_incluster_score > current_best_score:
                if total_incluster_score > 0.5: # Bad matches being added (ex. 0.3), note, this isn't happening anymore because scores like this don't make it into the index
                  current_best_score = total_incluster_score
                  current_best_match = cand
                  match_found = True


    if match_found:
        #print("Match found!", current_best_score, current_best_match, clustid_to_clust[current_best_match])
        #old = clustid_to_clust[current_best_match]
        #new = old + [gap_aa]
        #clustid_to_clust[current_best_match] = new
        output = [gap_aa, current_best_match, current_best_score]
        #ic("Updating cluster {} from \n{}\nto\n{}".format(current_best_match, old, new)) 
        #match_score = [gap_aa, current_best_score, current_best_match]

    else:

         #ic("no match found in (existing clusters")    
         output = []
    return(output)

################################
###### Limited search: Amino acids with not matches phase
#@profile
def fill_in_hopeless(unassigned,  seqs_aas, cluster_order, clustid_to_clust, pos_to_clustid, index, hidden_states, gapfilling_attempt, args = None):
    '''
    For amino acids with no matches at all, add in as singleton clusters
    '''
    seqnums = [x[0].seqnum for x in seqs_aas]
    clusters_filt = list(clustid_to_clust.values())
    for gap in unassigned:
        #ic("GAP", gap)
        starting_clustid = gap[0]
        ending_clustid = gap[2]

        gap_seqaas = gap[1]
        gap_seqnum = list(set([x.seqnum for x in gap_seqaas]))[0]

        target_seqs_list = get_ranges(seqs_aas, cluster_order, starting_clustid, ending_clustid, pos_to_clustid)


        target_seqs = list(flatten(target_seqs_list))
        target_seqs = [x for x in target_seqs if not x.seqnum == gap_seqnum]

        for aa in gap_seqaas:
               clusters_filt.append([aa])

    cluster_order, clustid_to_clust, pos_to_clustid, alignment = organize_clusters(clusters_filt, seqs_aas, gapfilling_attempt, minclustsize = 1, args = args)
    return(cluster_order, clustid_to_clust, pos_to_clustid, alignment)

#################################
######## Limited search: Combine adjacent clusters
#@profile
def squish_clusters(alignment, index, hidden_states, index_to_aa):

    '''
    There are cases where adjacent clusters should be one cluster. 
    If any quality scores, squish them together(tetris style)
    XA-X  ->  XAX
    X-AX  ->  XAX
    XA-X  ->  XAX
    Start with doing this at the end
    With checks for unassigned aa's could do earlier
    Get total score between adjacent clusters
    Only record if no conflicts
    Set up network
    Merge highest score out edge fom each cluster
    Repeat a few times

    '''
    #ic("attempt squish")
    cluster_order, clustid_to_clust = clusts_from_alignment(alignment)




    candidate_merge_list = []
    for i in range(len(cluster_order)-1):

       c1 = clustid_to_clust[cluster_order[i]]
       # skip cluster that was 2nd half of previous squish
       if len(c1) == 0:
         continue
       c2 = clustid_to_clust[cluster_order[i + 1]]
       c1_seqnums = [x.seqnum for x in c1]
       c2_seqnums = [x.seqnum for x in c2]
       seqnum_overlap = set(c1_seqnums).intersection(set(c2_seqnums))

    # Can't merge if two clusters already have same sequences represented
       if len(seqnum_overlap) > 0:
          continue
       else:
          intra_clust_hits= []
          for aa1 in c1:
            candidates = get_looser_scores(aa1, index, hidden_states)
            for candidate in candidates:
                #try:
                   score = candidate[0]
                   candidate_index = candidate[1]
                   if candidate_index == -1:
                      continue
                   target_aa = index_to_aa[candidate_index]
                   #ic("target_aa", target_aa)
                   if target_aa in c2:
                       if score > 0:
                           intra_clust_hits.append(score )
                           #print(aa1, target_aa, score)
                #except Exception as E:
                #   # Not all indices correspond to an aa, yes they do
                #   continue
          #ic("intra clust hits", intra_clust_hits)
          #ic("c1", c1)
          #ic("c2", c2)
          combo = c1 + c2
          #scores = [x[2] for x in intra_clust_hits if x is not None]
          candidate_merge_list.append([cluster_order[i], cluster_order[i + 1], sum(intra_clust_hits)])
          #print("candidate merge list", candidate_merge_list)

    removed_clustids = []
    edges = []
    weights = []
    for x in candidate_merge_list:
         edges.append((x[0], x[1]))
         weights.append(x[2])
    to_merge = []

    for squish in [1,2, 3, 4, 5, 6, 7, 8, 9, 10]:

        # Start with scores between adjacent clusters
        # Want to merge the higher score when there's a choice
        #ic(edges)        
        G = igraph.Graph.TupleList(edges=edges, directed=False)
        G.es['weight'] = weights
        islands = G.clusters(mode = "weak")
        edges = []
        weights = []
        for sub_G in islands.subgraphs():
            n = len(sub_G.vs())

            #ic(sub_G)
            #ic(n)

            node_highest = {}
            # If isolated pair, no choice needed
            if n == 2:
                 to_merge.append([x['name'] for x in sub_G.vs()])

            for vertex in sub_G.vs():
               node_highest[vertex['name']] = 0
               if vertex.degree() == 1:
                  continue
               vertex_id= vertex.index
               sub_edges = sub_G.es.select(_source = vertex_id)
               max_weight = max(sub_edges['weight'])
               #ic(max_weight)

               maybe = sub_edges.select(weight_eq = max_weight)

               #ic(vertex)
               for e in maybe:
                  highest_edge = [x['name'] for x in sub_G.vs() if x.index  in e.tuple]
                  #ic(highest_edge, max_weight)
                  #if max_weight > node_highest[highest_edge[0]]:
                  #      node_highest[highest_edge[0]] = max_weight
                  if highest_edge not in edges:
                      edges.append(highest_edge)
                      weights.append(max_weight)


    #ic("to_merge", to_merge)

    for c in to_merge:
              #c = [cluster1, cluster2]
              removed_clustids.append(c[1])
              clustid_to_clust[c[0]] =   clustid_to_clust[c[0]] + clustid_to_clust[c[1]]
              clustid_to_clust[c[1]] = []

    #ic("Old cluster order", cluster_order)
    cluster_order = [x for x in cluster_order if x not in removed_clustids]
        #ifor vs in sub_G.vs():


    return(cluster_order, clustid_to_clust)







###############################
###### Remove doubles utils
##i.e. when a cluster contains two amino acids from the same sequence

#@profile
def get_doubled_seqnums(cluster):
      seqnums = [x.seqnum for x in cluster]


      clustcounts = Counter(seqnums)
      to_remove = []
      for key, value in clustcounts.items():
           if value > 1:
               to_remove.append(key)

      return(to_remove)


#@profile
def remove_doubles_by_consistency_old(cluster, pos_to_clustid, add_back = True):
    '''
    Keep any doubled amino acids that pass the consistency check  based on previous and next cluster
    Option to keep both if neither are consistent, to send on to further remove_doubles attempts
    '''

    to_remove = get_doubled_seqnums(cluster)
    if len(to_remove) == 0:
          return(cluster)

    cluster_minus_targets = [x for x in cluster if x.seqnum not in to_remove]
    # Can be more than one doubled seqnum per cluster
    # Remove all doubles, then add back in any that are consistent
    to_add_back = []
    
    for seqnum in to_remove:
        target_aas = [x for x in cluster if x.seqnum == seqnum]
        consistent_ones = []
        for aa in target_aas:
            logging.debug("{} {}".format(aa, cluster_minus_targets))
            if consistency_check ( [aa] + cluster_minus_targets, pos_to_clustid ) == True:
                consistent_ones.append(aa)
        # Add back in consistent to cluster     
        if len(consistent_ones) == 1:
            cluster_minus_targets = cluster_minus_targets + [consistent_ones[0]]
        else:
           if add_back == True:
              to_add_back = to_add_back + target_aas

    if add_back == True:
         #ic("Adding back", to_add_back, "to", cluster_minus_targets)
         cluster_minus_targets = cluster_minus_targets + to_add_back

    return(cluster_minus_targets)

def remove_doubles_by_consistency(cluster, pos_to_clustid, add_back = True):
    '''
    Keep any doubled amino acids that pass the consistency check  based on previous and next cluster
    Option to keep both if neither are consistent, to send on to further remove_doubles attempts
    '''

    to_remove = get_doubled_seqnums(cluster)
    if len(to_remove) == 0:
          return(cluster)

    cluster_minus_targets = [x for x in cluster if x.seqnum not in to_remove]
    # Can be more than one doubled seqnum per cluster
    # Remove all doubles, then add back in any that are consistent
    to_add_back = []
    #print(prevset, nextset)
    for seqnum in to_remove:
        target_aas = [x for x in cluster if x.seqnum == seqnum]
        consistent_ones = []
        # start consistency check a consistency level 1, then continue if necessary
 
        found_true = False
        for i in [1,2,3,4,5]:
           max_num_consistent = 0
           if found_true == False:
               for aa in target_aas:
                 logging.debug("{} {}".format(aa, cluster_minus_targets))
                 prevlist, nextlist = get_prepost(cluster_minus_targets, pos_to_clustid)    
                 logging.debug("{} {}".format(prevlist, nextlist))
                 logging.debug("prevlist status {}".format(prevlist))
                 check_results, num_consistent =  consistency_check2 (aa, prevlist, nextlist, diversity = i, pos_to_clustid = pos_to_clustid)
                 if check_results == True and num_consistent >= max_num_consistent :
                    logging.debug("{} {}".format(aa, num_consistent))
                    consistent_ones.append(aa)
                    max_num_consistent = num_consistent
                    found_true = True
                    add_back = False

        # Add back in consistent to cluster     
        if len(consistent_ones) == 1:
            cluster_minus_targets = cluster_minus_targets + [consistent_ones[0]]
        else:
           if add_back == True:
              to_add_back = to_add_back + target_aas

    if add_back == True:
         #ic("Adding back", to_add_back, "to", cluster_minus_targets)
         cluster_minus_targets = cluster_minus_targets + to_add_back

    return(cluster_minus_targets)

#@profile

def consistency_check2(aa, prevlist, nextlist, diversity= 1, pos_to_clustid = {}):
        logging.debug("starting {}".format(prevlist))
        logging.debug("starting {}".format(nextlist))

        prevset = set(prevlist)
        nextset = set(nextlist) 
 
        if aa.prevaa in pos_to_clustid.keys():
             if pos_to_clustid[aa.prevaa] in prevset:           
                 prevlist.append(pos_to_clustid[aa.prevaa])   
             prevset.add(pos_to_clustid[aa.prevaa])

        if aa.nextaa in pos_to_clustid.keys():           
             if pos_to_clustid[aa.nextaa] in nextset:           
                 nextlist.append(pos_to_clustid[aa.nextaa])   
             nextset.add(pos_to_clustid[aa.nextaa])

        logging.debug("ending {}".format(prevlist))
        logging.debug("ending {}".format(nextlist))

        num_consistent = 0
        logging.debug("consistency_check {} {} {} {} {} {}".format(aa, aa.prevaa, aa.nextaa,  prevset, nextset, diversity))
        results = False
        if len(prevset) <= diversity:
           results = True
           num_consistent = num_consistent + len(prevlist)
        if len(nextset) <= diversity:
           results = True
           num_consistent = num_consistent + len(nextlist)
        return(results, num_consistent)


def get_prepost(cluster_minus_aa, pos_to_clustid):
    prev_list = []
    next_list = []
    #print(cluster_minus_aa) 
    #print(pos_to_clustid)
    for aa in cluster_minus_aa:
        #print(aa, aa.prevaa)
        if aa.prevaa in pos_to_clustid.keys():
            #print(aa, aa.prevaa, prev_list)
            prev_list.append(pos_to_clustid[aa.prevaa])
          
        if aa.nextaa in pos_to_clustid.keys():
            next_list.append(pos_to_clustid[aa.nextaa])
        
    return(prev_list, next_list)           




def consistency_check(cluster, pos_to_clustid):
    '''
    For a cluster, see if previous or next amino acids are also all part os same cluster
    '''


    prev_list = []
    next_list = []
    for aa in cluster:
        if aa.prevaa in pos_to_clustid.keys():
            prev_list.append(pos_to_clustid[aa.prevaa])
        if aa.nextaa in pos_to_clustid.keys():
            next_list.append(pos_to_clustid[aa.nextaa])
    logging.debug(prev_list)
    logging.debug(next_list)
    prevset = list(set(prev_list))
    nextset = list(set(next_list))
  
    total_prevset = 0
    total_nextset = 0
    if len(prevset) == 1:
         total_prevset = len(prev_list)
    if len(nextset) == 1: 
         total_nextset = len(next_list)
    total = sum(total_prevset + total_nextset)

    if len(prevset) == 1 or len(nextset) == 1:
        return(True)

    else:
        return(False)




#@profile
def remove_doubles_by_scores(clust, index, hidden_states, index_to_aa):

    alternates_dict = {}
    doubled_seqnums = get_doubled_seqnums(clust)
    if doubled_seqnums:
         clust_minus_dub_seqs = [x for x in clust if x.seqnum not in doubled_seqnums]
         #ic("sequence {} in {}, {} is doubled".format(doubled_seqnums, clustnum, clust))
         for seqnum in doubled_seqnums:
             saved = None
             bestscore = 0
             double_aas = [x for x in clust if x.seqnum == seqnum]
             #ic(double_aas)
             for aa in double_aas:
                 candidates_w_score = get_set_of_scores(aa, index, hidden_states, index_to_aa)
                 incluster_scores = [x for x in candidates_w_score if x[0] in clust_minus_dub_seqs ]
                 total_incluster_score = sum([x[1] for x in incluster_scores])
                 #ic(total_incluster_score)
                  #ic(incluster_scores)
                 if total_incluster_score > bestscore:
                     saved = aa
                     bestscore = total_incluster_score
             #ic("Adding back {} to {}".format(keeper, clust_minus_dub_seqs))
             if saved:
                 alts = [x for x in double_aas if x != saved]
                 clust_minus_dub_seqs = clust_minus_dub_seqs + [saved]
                 alternates_dict[saved] = alts
#
         return(clust_minus_dub_seqs, alternates_dict)
    else:
         return(clust, alternates_dict)


#@profile
def remove_doubles_by_graph(cluster, G,  minclustsize = 0, keep_higher_degree = False, keep_higher_score = True, remove_both = False):
            ''' 
            If a cluster contains more 1 amino acid from the same sequence, remove that sequence from cluster
 
            '''
            alternates_dict = {}
            #ic("remove_doubles_by_graph")
            #ic(cluster)
            to_remove = get_doubled_seqnums(cluster)
            #ic(keep_higher_degree, keep_higher_score, to_remove)
            # If there's anything in to_remove, keep the one with highest degree


            if len(to_remove) > 0 and keep_higher_score == True:

                 G = G.vs.select(name_in=cluster).subgraph()
                 for seqnum in to_remove:
                     cluster, saved, alts = remove_lower_score(cluster, seqnum, G)
                     alternates_dict[saved] = alts
                 to_remove = get_doubled_seqnums(cluster)
            if len(to_remove) > 0 and keep_higher_degree == True:
                 
                 G = G.vs.select(name_in=cluster).subgraph()
                 for seqnum in to_remove:
                    cluster, saved, alts = remove_lower_degree(cluster, seqnum, G)
                    alternates_dict[saved] = alts

                 to_remove = get_doubled_seqnums(cluster)
            # Otherwise, remove any aa's from to_remove sequence
            if len(to_remove) > 0 and remove_both == True:
                #for x in to_remove:
                   #ic("Removing sequence {} from cluster".format(x))
                cluster = [x for x in cluster if x.seqnum not in to_remove]
            if len(cluster) < minclustsize:
               return([], {})
            else:
                return(cluster, alternates_dict) 

#@profile
def remove_lower_degree(cluster, seqnum, G): 
    '''
    This is never called, remove it
    '''
    target_aas = [x for x in cluster if x.seqnum == seqnum]
            #ic(aas)       
    degrees = []
    for aa in target_aas:

         degrees.append(G.vs.find(name  = aa).degree())
    #ic("dupped aas", target_aas)

    highest_degree = target_aas[np.argmax(degrees)]
    #ic("high degree", highest_degree)
    to_remove = [x for x in target_aas if x != highest_degree]
    cluster_filt = [x for x in cluster if x not in to_remove]
    #ic("cluster", cluster)
    #ic("cluster_filt", cluster_filt)
    return(cluster_filt)


#@profile
def remove_lower_score(cluster, seqnum, G):

    target_aas = [x for x in cluster if x.seqnum == seqnum]
            #ic(aas)       
    degrees = []
    edge_sums = {}
    #ic(target_aas)
    #ic(G)
    #ic(G.vs()['name'])

    aa_idxs = [G.vs.find(name =x) for x in target_aas]
    for aa in target_aas:
         g_new = G.copy()
         query_vs = g_new.vs.find(name = aa)
         target_vs = [x for x in g_new.vs() if x not in aa_idxs]
         #ic("aa_idxs", aa_idxs) 
         #ic("target_vs", target_vs) 
         #ic("query_vs", query_vs) 
         edges = g_new.es.select(_source=query_vs)#   ['weight'])
         edge_sums[aa] = sum(edges['weight'])
    #ic(edge_sums)
    #ic("dupped aas", target_aas)

    highest_score = max(edge_sums, key=edge_sums.get)
    saved = highest_score
    #ic("high score", highest_score)
    to_remove = [x for x in target_aas if x != highest_score]
    cluster_filt = [x for x in cluster if x not in to_remove]
    #ic("cluster", cluster)
    #ic("cluster_filt", cluster_filt)
    alts= to_remove  # Save these for later (if the saved on causes a feedback arc, try the alts)
    return(cluster_filt, saved, alts)



#@profile
def dedup_clusters(clusters_list, G, minclustsize):
    new_clusters_list = []

    pos_to_clustid, clustid_to_clust = get_cluster_dict(clusters_list)
    for clust in clusters_list:
        if len(clust) > len(get_represented_seqs(clust)):

             resolved = False
             #ic("has dups after very first clustering", clust)
             #for pos in clust:
             for otherclust in clusters_list:
               if clusters_list == otherclust:
                   continue
               # Check if removing a smaller cluster resolved duplicates
               if len(set(clust).intersection(set(otherclust))) >= 2:
                    trimmed_clust = [x for x in clust if x not in otherclust]
                    complete = check_completeness(trimmed_clust)

                    if complete:
                         if trimmed_clust not in new_clusters_list:
                            #ic("trimmed cluster", trimmed_clust)
                            new_clusters_list.append(trimmed_clust)
                            resolved = True
             if resolved == False:
                  # Start by trying to resolve with consistency check
                  reduced_clust =  remove_doubles_by_consistency(clust, pos_to_clustid, add_back = True)
                  complete = check_completeness(reduced_clust)
                  if complete:
                      #ic("resolved after consistency removal", reduced_clust)
                      new_clusters_list.append(reduced_clust)
                  # Then try by higher score in the original rbh
                  # Potentially replace this with new search "removed_doubles_w_search"
                  else:
                      reduced_clust, alternates_dict =  remove_doubles_by_graph(reduced_clust, G, keep_higher_score = True, remove_both = False)
                      #ic(reduced_clust, alternates_dict)
                      complete = check_completeness(reduced_clust)
                      if complete:
                          #ic("resolved after graph removal", reduced_clust)
                          new_clusters_list.append(reduced_clust)
        else:
             if clust not in new_clusters_list:
                  new_clusters_list.append(clust)
    return(new_clusters_list)

### AA class
#class AA:
#   def __init__(self):
#       self.seqnum = ""
#       self.seqname = ""
#       self.seqindex = ""
#       self.seqpos = ""
#       self.seqaa = ""
#       self.index = ""
#       self.clustid = ""
#   #__str__ and __repr__ are for pretty #printing
#   def __str__(self):
#        return("{}-{}-{}".format(self.seqnum, self.seqpos, self.seqaa))
#   def __repr__(self):
#    return str(self)


### Sequence formatting

#def format_sequence(sequence, add_spaces = True):
#
#   if add_spaces == False:
#       seq_spaced = sequence
#
#   else:
#       seq_spaced = " ".join(list(sequence))
#
#   return seq_spaced

#def parse_fasta(fasta_path, sequence_out = "", add_spaces = True, maxlength=None):
#
#   sequences = []
#   with open(sequence_out, "w") as outfile:
#
#       for record in SeqIO.parse(fasta_path, "fasta"):
#            if maxlength:
#                sequence = record.seq[0:maxlength]
#            else:
#                sequence = record.seq
#            seq_spaced = format_sequence(sequence, add_spaces)
#            
#            if sequence_out:
#                outstring = "{},{}\n".format(record.id, seq_spaced)
#                outfile.write(outstring)
#
#            sequences.append([record.id, sequence, seq_spaced])
#   
#   return(sequences)



### Similarity
#@profile
def build_index_flat(hidden_states, scoretype = "cosinesim", index = None,  normalize_l2 = True):


    if not index:
        d = hidden_states.shape[1]
        if scoretype == "euclidean":
           index = faiss.index_factory(d, "Flat")
        else:
            index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)

    if normalize_l2 == True:
        faiss.normalize_L2(hidden_states)

    index.add(hidden_states)

    return(index)


################################
##### Visualization utils
#@profile
def  do_pca_plot(hidden_states, index_to_aa, outfile, clustid_to_clust = None, seq_to_length = None):

        seqnums = list(set([x.seqnum for x in index_to_aa.values()]))
        #ic("seqnums", seqnums)
        if clustid_to_clust:
            aa_to_clustid = {}


            #ic(clustid_to_clust)
            for clustid, aas in clustid_to_clust.items():
               for aa in aas:
                  aa_to_clustid[aa] = clustid

            clustid_to_color = {}
            for key in clustid_to_clust.keys():
               #ic(key)
               clustid_to_color[key] = (random.random(), random.random(),random.random())

        elif seq_to_length:
            seqnum_to_color = {}

            for key in seqnums:
                seqnum_to_color[key] = (random.random(), random.random(),random.random())

        indexes = list(index_to_aa.keys())
        hidden_states_aas = hidden_states[indexes, :]

        d1 = hidden_states.shape[1]
        target = 128

        pca = faiss.PCAMatrix(d1, target)

        pca.train(np.array(hidden_states))



        bias = faiss.vector_to_array(pca.b)
        pcamatrix = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)

        reduced = np.array(hidden_states_aas) @ pcamatrix.T + bias
        #ic(reduced.shape)

        colorlist = []

        labellist  = []
        aalist = []
        seqlist = []
        poslist = []
        for i in range(len(hidden_states)):
            if i in index_to_aa.keys():
              aa = index_to_aa[i]
              aalist.append(aa.seqaa)
              seqlist.append(aa.seqnum)
              poslist.append(aa.seqpos)

              if clustid_to_clust:
                  clustid = aa_to_clustid[aa]
                  #ic(clustid)


                  color = clustid_to_color[clustid]
                  labellist.append(clustid)
              else:
                  color = seqnum_to_color[aa.seqnum]
                  labellist.append(aa.seqnum)

              colorlist.append(color)
        label_arr = np.array(labellist)
        color_arr = np.array(colorlist)


        for dim1 in [1,2,3]:
          for dim2 in [1,2,3]:
            if dim1 == dim2:
                continue

            plt.figure()
            if clustid_to_clust:
                for iclust in clustid_to_clust.keys():
                   plt.scatter(reduced[:,dim1-1][label_arr == iclust], reduced[:,dim2-1][label_arr == iclust], c = color_arr[label_arr == iclust], alpha = 0.8, label = iclust)
            if seq_to_length:
                for iclust in seq_to_length.keys():
                   plt.scatter(reduced[:,dim1-1][label_arr == iclust], reduced[:,dim2-1][label_arr == iclust], c = color_arr[label_arr == iclust], alpha = 0.8, label = iclust)
            plt.legend()
            plt.xlabel('component {}'.format(dim1))
            plt.ylabel('component {}'.format(dim2))


            plt.savefig("{}.pca{}{}.png".format(outfile,dim1,dim2))
            plt.clf()
        pcasave= pd.DataFrame(reduced[:,[0,1,2,3,4,5,6]])
        pcasave['clustid'] = labellist
        pcasave['color'] = colorlist
        pcasave['seq'] = seqlist
        pcasave['pos'] = poslist
        pcasave['aa'] = aalist

        #ic(pcasave)
        pcasave.to_csv("{}.pca.csv".format(outfile), index = False)





###################
##### Unused utils
def LongestIncreasingSubsequence(X):
    """
    Source: https://stackoverflow.com/a/40732886, chiwangc
    Find and return longest increasing subsequence of S.
    If multiple increasing subsequences exist, the one that ends
    with the smallest value is preferred, and if multiple
    occurrences of that value can end the sequence, then the
    earliest occurrence is preferred.
    # Check multiple, change to that maximum subsequence values are preferred
    """

    n = len(X)
    X = [None] + X  # Pad sequence so that it starts at X[1]
    M = [None]*(n+1)  # Allocate arrays for M and P
    P = [None]*(n+1)
    L = 0
    for i in range(1,n+1):
        if L == 0 or X[M[1]] >= X[i]:
            # there is no j s.t. X[M[j]] < X[i]]
            j = 0
        else:
            # binary search for the largest j s.t. X[M[j]] < X[i]]
            lo = 1      # largest value known to be <= j
            hi = L+1    # smallest value known to be > j
            while lo < hi - 1:
                mid = (lo + hi)//2
                if X[M[mid]] < X[i]:
                    lo = mid
                else:
                    hi = mid
            j = lo
    
        P[i] = M[j]
        if j == L or X[i] < X[M[j+1]]:
            M[j+1] = i
            L = max(L,j+1)

    # Backtrack to find the optimal sequence in reverse order
    output = []
    #print("M", M)
    #print("P", P)
    pos = M[L]
    while L > 0:
        output.append(X[pos])
        pos = P[pos]
        L -= 1

    output.reverse()
    return output




def maximum_increasing(hitlist):
    # Remove initial RBHs that cross a streak of matches
    # Simplify network for feedback search
    seqnums = [x[0].seqnum for x in hitlist]
    seqnums = list(set(seqnums))

    #seqnums = [x[0].seqnum for x in seqs_aas]
    #seqnums = list(set(seqnums))
    #print(seqnums)
    filtered_hitlist = []
    for seqnum_i in seqnums:
       #seqnum_i = seqnums[i]
       query_prot = [x for x in hitlist if x[0].seqnum == seqnum_i]
       for seqnum_j in seqnums:
          #print("query,target",seqnum_i, seqnum_j)
          target_prot = [x for x in query_prot if x[1].seqnum == seqnum_j]
          #print(target_prot)

          target_indices = [x[1].seqpos for x in target_prot]
          #print(target_indices)
          #print("original  ", seqnum_i, seqnum_j, target_indices)
            
          longest_increasing = LongestIncreasingSubsequence(target_indices)
          #print("inc_subseq", seqnum_i, seqnum_j ,longest_increasing)
          #print(len(longest_increasing))
          filtered_hitlist = filtered_hitlist +  [x for x in target_prot if x[1].seqpos in longest_increasing]  
          #print([x for x in target_prot if x[1].seqpos in longest_increasing]) 
          #print(len(filtered_hitlist))
    return(filtered_hitlist)



#@profile
def remove_streakbreakers(hitlist, seqs_aas, seqlens, streakmin = 3):
    # Not in use
    # Remove initial RBHs that cross a streak of matches
    # Simplify network for feedback search
    seqnums = [x[0].seqnum for x in seqs_aas]
    seqnums = list(set(seqnums))
    filtered_hitlist = []
    for i in range(len(seqs_aas)):
       #print(i)
       seqnum_i = seqnums[i]
       query_prot = [x for x in hitlist if x[0].seqnum == seqnum_i]
       for j in range(len(seqs_aas)):
          #print(j)
          seqnum_j = seqnums[j]
          target_prot = [x for x in query_prot if x[1].seqnum == seqnum_j]

          # check shy this is happening extra at ends of sequence
          #ic("remove lookbehinds")
          prevmatch = 0
          seq_start = -1
          streak = 0

          no_lookbehinds = []
          for match_state in target_prot:
               #ic(match_state)
               if match_state[1].seqpos <= seq_start:
                     #ic("lookbehind prevented")
                     streak = 0
                     continue
               no_lookbehinds.append(match_state)

               if match_state[1].seqpos - prevmatch == 1:
                  streak = streak + 1
                  if streak >= streakmin:
                     seq_start = match_state[1].seqpos
               else:
                  streak = 0
               prevmatch  = match_state[1].seqpos

          #ic("remove lookaheads")
          prevmatch = seqlens[j]
          seq_end = seqlens[j]
          streak = 0
          filtered_target_prot = []
          for match_state in no_lookbehinds[::-1]:
               #ic(match_state, streak, prevmatch)
               if match_state[1].seqpos >= seq_end:
                    #ic("lookahead prevented")
                    streak = 0
                    continue
               filtered_target_prot.append(match_state)
               if prevmatch - match_state[1].seqpos == 1:
                  streak = streak + 1
                  if streak >= streakmin:
                     seq_end = match_state[1].seqpos
               else:
                  streak = 0
               prevmatch = match_state[1].seqpos

          filtered_hitlist = filtered_hitlist + filtered_target_prot
    return(filtered_hitlist)

#@profile
def consistency_clustering(G, minclustsize = 0, dup_thresh = 1):
    '''
    First, naturally consistent
    Second, cluster members prev or next aas fall in same cluster. 

    '''
    # Get naturally disconnected sets
    islands = G.clusters(mode = "weak")
    natural_cluster_list = []
    cluster_list = []
    for sub_G in islands.subgraphs():
        natural_cluster = sub_G.vs()['name']
        #ic("Natural connected set", sub_G.vs()['name'])
        min_dupped =  min_dup(natural_cluster, dup_thresh)
        #ic(min_dupped, minclustsize)
        if(len(natural_cluster) <= min_dupped):
            if(len(natural_cluster) >= minclustsize):
                 natural_cluster_list.append(natural_cluster)

    pos_to_clustid, clustid_to_clust = get_cluster_dict(natural_cluster_list)
    for natural_cluster in natural_cluster_list:

          # Need to check if duplicated here first
          #ic("Checking", natural_cluster)
          if consistency_check(natural_cluster, pos_to_clustid) == True:
              finished = check_completeness(natural_cluster)
              if finished == True:
                  cluster_list.append(natural_cluster)
          else:
             ic("Check if duplicated")
             ic("If duplicated, see if removing one of the aas makes consistent")
             seqnums = [x.seqnum for x in natural_cluster]
             if len(seqnums) < len(natural_cluster):
                  new_cluster = remove_doubles_by_consistency(natural_cluster, pos_to_clustid, add_back = True)
                  finished = check_completeness(new_cluster)
                  if finished == True:
                     cluster_list.append(new_cluster)
    #for x in cluster_list:
          #ic("natural_cluster", x)

    return(cluster_list)

#@profile
def remove_low_match_prots(numseqs, seqlens, clusters, threshold_min = 0.5):
    ############## No badly aligning sequences check
    # Remove sequences that have a low proportion of matches from cluster
    # Do another one of these after ordering criteria
    matched_count =  [0] * numseqs
    for pos in flatten(clusters):
        seqnum = get_seqnum(pos)
        matched_count[seqnum] = matched_count[seqnum] + 1


    matched_prop = [matched_count[x]/seqlens[x] for x in range(0, numseqs)]
    poor_seqs = []
    for i in range(0, numseqs):
        if matched_prop[i] < threshold_min:
            #ic("Seq {} is poorly matching, fraction positions matched {}, removing until later".format(i, matched_prop[i]))
            poor_seqs.append(i)

    clusters_tmp = []
    for clust in clusters:
       clust_tmp = []
       for pos in clust:
            if not get_seqnum(pos) in poor_seqs:
               clust_tmp.append(pos)
       clusters_tmp.append(clust_tmp)

    clusters  = clusters_tmp
    return(clusters)

#@profile
def address_isolated_aas(unassigned_aa, cohort_aas, D, I, minscore):
    '''
    Maybe overwrite score? 
    Or match to cluster with higher degree
    '''
    #ic("Address isolated aas")
    connections = []
    for cohort_aa in cohort_aas:
        score = get_particular_score(unassigned_aa, cohort_aa, D, I)
        #ic(unassigned_aa, cohort_aa, score)

    return(cluster)



#def clustering_to_clusterlist(G, clustering):
#    """
#    Go from an igraph clustering to a list of clusters [[a,b,c], [d,e]]
#    """
#    cluster_ids = clustering.membership
#    vertices = G.vs()["name"]
#    clusters = list(zip(cluster_ids, vertices))
#
#    clusters_list = []
#    for i in range(len(clustering)):
#         clusters_list.append([vertices[x] for x in clustering[i]])
#    return(clusters_list)















    #ic("clusters_to_dag", all_alternates_dict)
    #ic("clusters to add back", clusters_to_add_back)

    #already_placed = []
    #for key, cluster_group_to_add_back in clusters_to_add_back.items():
    #   skip_first = False
    #   skip_second = False
    #   #ic("looking at cluster group", key, cluster_group_to_add_back)
    #     # Meaning that this cluster has already been placed after error correction
    #   if list(key)[0] in already_placed:
    #         skip_first = True
    #         #ic(list(key)[0], "already placed")
    #   if list(key)[1] in already_placed:
    #         skip_second = True
    #         #ic(list(key)[1], "already placed")
    #
    #   for cluster_pair_to_add_back in cluster_group_to_add_back:
    #     if skip_first == True:
    #         cluster_pair_to_add_back = [cluster_pair_to_add_back[1]]
    #     if skip_second == True:
    #         cluster_pair_to_add_back = [cluster_pair_to_add_back[0]]

 
    #     trial =  clusters_filt_dag_all + cluster_pair_to_add_back

    #     pos_to_clustid, clustid_to_clust = get_cluster_dict(trial)
    #     cluster_orders_dict = get_cluster_orders(pos_to_clustid, seqs_aas)
    #     clusters_filt_dag_trial, clusters_to_add_back_trial = remove_feedback_edges2(cluster_orders_dict, clustid_to_clust,  gapfilling_attempt, remove_both, alignment_group = alignment_group, attempt = attempt, all_alternates_dict= all_alternates_dict, args = args)
    #     if len(clusters_filt_dag_trial) > len(clusters_filt_dag_all):
    #          clusters_filt_dag_all = clusters_filt_dag_trial
    #          #ic("Alternate worked better")
    #          #ic("Added back", cluster_pair_to_add_back)
    #          already_placed = already_placed + list(flatten(key))
    #          #ic("Already placed", already_placed)
    #          break

