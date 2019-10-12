# This is a demo script for paper "Towards Explaining Block Models of Graphs" submitted to SIAM Data Mining SDM2020
# Paper authors: Zilong Bai, S.S. Ravi, Ian Davidson
# Code author: Zilong Bai
import numpy as np
from gurobipy import *

def unpack_gurobi_single_graph_descriptors(m,T):
	single_graph_description = np.zeros(T)
	i = 0
	print len(m.getVars())
	try:
		for v in m.getVars():
			if v.varname.startswith('Graph_Level_Edge_Explanation'):
				single_graph_description[i] = v.x
				i = i + 1
	except:
		single_graph_description = []
		print 'Unpack model descriptors Infeasible!'
	return single_graph_description
		
def unpack_gurobi_model_descriptors(m,k,T):
        multigraph_descriptions = np.zeros((k, T))
        multigraph_forgotten_tags = np.zeros((k,T))
        tl_d = np.zeros((k*T))
#        tl_ft = np.zeros((k*T))
        i = 0
        j = 0
#	for v in m.getVars():
#		print v
        try:
                for v in m.getVars():
                        if v.varname.startswith('Graph_Level_Edge_Explanation'):
                                tl_d[i] = v.x
                                i = i + 1
 #                       if v.varname.startswith('Tags_by_Disconnections_Forgot'):
  #                              tl_ft[j] = v.x
   #                             j = j + 1

                multigraph_descriptions = tl_d.reshape((k, T))
    #            multigraph_forgotten_tags = tl_ft.reshape((k, T))
        except:
                multigraph_descriptions = []
     #           multigraph_forgotten_tags = []
                print 'Unpack model descriptors Infeasible!'
	return multigraph_descriptions
      #  return multigraph_descriptions, multigraph_forgotten_tags 

def unpack_gurobi_model_descriptors_BM(m,Ng,k,T):
        multigraph_descriptions = np.zeros((Ng, T))
	Within_block_descriptions = np.zeros((k, T))
	Between_block_descriptions = np.zeros((Ng-k, T))
        multigraph_forgotten_tags = np.zeros((k,T))
        tl_d = np.zeros((Ng*T))
#        tl_ft = np.zeros((k*T))
        i = 0
        j = 0
#       for v in m.getVars():
#               print v
        try:
                for v in m.getVars():
                        if v.varname.startswith('Graph_Level_Edge_Explanation'):
                                tl_d[i] = v.x
                                i = i + 1
 #                       if v.varname.startswith('Tags_by_Disconnections_Forgot'):
 #                               tl_ft[j] = v.x
 #                               j = j + 1

                multigraph_descriptions = tl_d.reshape((Ng, T))
		Within_block_descriptions = multigraph_descriptions[:k,:]
		Between_block_descriptions = multigraph_descriptions[k:,:]
#                multigraph_forgotten_tags = tl_ft.reshape((k, T))
        except:
                multigraph_descriptions = []
#                multigraph_forgotten_tags = []
                print 'Unpack model descriptors Infeasible!'
	return multigraph_descriptions, Within_block_descriptions, Between_block_descriptions
        #return multigraph_descriptions, multigraph_forgotten_tags, Within_block_descriptions, Between_block_descriptions
	

def unpack_gurobi_model_descriptors_OvsN(m,T):
	k = 2 # Only two descriptions: edge-occurring vs non-edge-occurring
        subgraph_descriptions = np.zeros((k, T))
        edge_descriptions = np.zeros((1, T))
        disconnection_descriptions = np.zeros((1, T))
        tl_d = np.zeros((k*T))
        i = 0
        try:
                for v in m.getVars():
                        if v.varname.startswith('Graph_Level_Edge_Explanation'):
                                tl_d[i] = v.x
                                i = i + 1

                subgraph_descriptions = tl_d.reshape((k, T))
                print 'WAAAAT'
		edge_descriptions = subgraph_descriptions[0,:]
                disconnection_descriptions = subgraph_descriptions[1,:]
        except:
                subgraph_descriptions = []
                print 'Unpack model descriptors Infeasible!'
        return subgraph_descriptions, edge_descriptions, disconnection_descriptions

def unpack_gurobi_removed_tags(m,Ng,T):
	removed_tags = np.zeros((Ng, T))
	tl_rt = np.zeros((Ng*T))
	i = 0
	try:
		for v in m.getVars():
			if v.varname.startswith('Remove_from_each_label_universe'):
				tl_rt[i] = v.x
				i = i+1
		removed_tags = tl_rt.reshape((Ng, T))
	except:
		print 'Unpack removed tags Infeasible!'
	return removed_tags
