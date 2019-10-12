# This is file of methods for paper "Towards Explaining Block Models of Graphs" submitted to SIAM Data Mining SDM2020
# Paper authors: Zilong Bai, S.S. Ravi, Ian Davidson
# Code author: Zilong Bai
# This file contains functions to implement ILP formulations for discovering explanations as set covering problem or its variant.
import numpy as np
from gurobipy import *

def baseline_edges_real(Els, Ng, T):
        print("This is the edge cover version based on baseline approach in NeurIPS 2018")
        m = Model('Orthogonal Descriptions')
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        ub = m.addVars(T, vtype=GRB.INTEGER, name='Upper_bound_over')

        obj1 = quicksum(D[i, t] for t in range(T)
                                for i in range(Ng))

        obj2 = quicksum(ub[t] for t in range(T))

        m.setObjective(obj1+obj2, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                                                                           for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((quicksum(D[i,t] for i in range(Ng)) <= ub[t] for t in range(T)), name='Upper_bound_overlap')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def baseline_real(Els, Ng, T):
        print("This is the edge cover version based on baseline approach in NeurIPS 2018")
        m = Model('Orthogonal Descriptions')
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        ub = m.addVars(T, vtype=GRB.INTEGER, name='Upper_bound_over')

        obj1 = quicksum(D[i, t] for t in range(T)
                                for i in range(Ng))

	obj2 = quicksum(ub[t] for t in range(T))

        m.setObjective(obj1+obj2, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs((quicksum(D[i,t]*Els[i][e,t] for t in range(T)) >= 1 for i in range(Ng)
                                                                          for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((quicksum(D[i,t] for i in range(Ng)) <= ub[t] for t in range(T)), name='Upper_bound_overlap')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def vtae_all_min_ol(Els, Ng, T):
        print("This is the edge cover version based on baseline approach in NeurIPS 2018")
        m = Model('Orthogonal Descriptions')
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        ub = m.addVar(vtype=GRB.INTEGER, name='Upper_bound_over')

        obj1 = quicksum(D[i, t] for t in range(T)
                                for i in range(Ng))

        obj2 = ub 

        m.setObjective(obj1+obj2, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                                                                          for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((quicksum(D[i,t] for i in range(Ng)) <= ub for t in range(T)), name='Upper_bound_overlap')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def baseline_cof_ubf(Els, Ng, T, ub):
        print("Cover or forget DTDM NIPS 2018")
        m = Model('DTDM cover or forget')
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanation')
        cof = []
        for i in range(Ng):
                cof.append(m.addVars(len(Els[i]), vtype=GRB.BINARY, name='Forget_Edges'))

        obj = quicksum(D[i, t] for t in range(T)
                                for i in range(Ng))

        m.setObjective(obj, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs((cof[i][e]+quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                                                                                    for e in range(len(Els[i]))), name='cover_or_forget_edges')

        m.addConstrs((quicksum(D[i,t] for i in range(Ng)) <= 1 for t in range(T)), name='non_overlap')

	m.addConstr((quicksum(cof[i][e] for i in range(Ng)
					for e in range(len(Els[i]))) <= ub), name='upper_bound_forgotten')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def baseline_cof(Els, Ng, T):
	print("Cover or forget DTDM NIPS 2018")
	m = Model('DTDM cover or forget')
	D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanation')
	cof = []
	for i in range(Ng):
		cof.append(m.addVars(len(Els[i]), vtype=GRB.BINARY, name='Forget_Edges'))

	obj1 = quicksum(D[i, t] for t in range(T)
                                for i in range(Ng))

        obj2 = quicksum(cof[i][e] for i in range(Ng)
				  for e in range(len(Els[i])))

	m.setObjective(obj1+obj2, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs((cof[i][e]+quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                           	                                                    for e in range(len(Els[i]))), name='cover_or_forget_edges')

	m.addConstrs((quicksum(D[i,t] for i in range(Ng)) <= 1 for t in range(T)), name='non_overlap')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def baseline_cof_v(Els, Ng, T):
        print("Cover or forget DTDM NIPS 2018")
        m = Model('DTDM cover or forget')
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanation')
        cof = []
        for i in range(Ng):
                cof.append(m.addVars(Els[i].shape[0], vtype=GRB.BINARY, name='Forget_Edges'))

        obj1 = quicksum(D[i, t] for t in range(T)
                                for i in range(Ng))

        obj2 = quicksum(cof[i][e] for i in range(Ng)
                                  for e in range(len(Els[i])))

        m.setObjective(obj1+obj2, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs((cof[i][e]+quicksum(D[i,t]*Els[i][e,t] for t in range(T)) >= 1 for i in range(Ng)
                                                                                    for e in range(len(Els[i]))), name='cover_or_forget_edges')

        m.addConstrs((quicksum(D[i,t] for i in range(Ng)) <= 1 for t in range(T)), name='non_overlap')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def baseline(Els, Ng, T):
	print("This is the edge cover version based on baseline approach in NeurIPS 2018")
	m = Model('Orthogonal Descriptions')
	D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	ub = m.addVar(vtype=GRB.INTEGER, name='Upper_bound_over')

	obj1 = quicksum(D[i, t] for t in range(T) 
			        for i in range(Ng))

	m.setObjective(obj1+ub, GRB.MINIMIZE)

	print("Constructing constraints...")
        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                                                                           for e in range(len(Els[i]))), name='cover_edges')

	m.addConstrs((quicksum(D[i,t]*D[j,t] for t in range(T)) <= ub for i in range(Ng)
								      for j in range(Ng)
								      if i != j), name='Upper_bound_overlap')

	print("model update")
	m.update()
	print("model optimize")
	m.optimize()
	
	return m

def baseline_advanced(Els, Mc, Ng, T):
        print("This is the edge cover version based on baseline approach in NeurIPS 2018 advanced with partial orthogonality")
        m = Model('Orthogonal Descriptions')
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
#        tf = m.addVars(T, vtype=GRB.BINARY, name="Tags_by_Disconnections_Forgotten")

#       upper = m.addVar(vtype=GRB.INTEGER, name="Overlap_Upperbound")

        obj1 = quicksum(D[i, t] for t in range(T) 
                                for i in range(Ng))

#       obj = upper

        obj2 = quicksum(D[i,t]*D[j,t] for t in range(T)
                                      for i in range(Ng)
                                      for j in range(Ng)
                                      if i < j
         			      if Mc[i,j] == 0) # Overall overlappings

	obj = obj1 + obj2
        m.setObjective(obj, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                                                                           for e in range(len(Els[i]))), name='cover_edges')


#       m.addConstrs((quicksum(D[i,t]*D[j,t] for t in range(T)) <= upper
#                                            for i in range(Ng)
#                                            for j in range(Ng)
#                                            if i < j
#					     if Mc[i,j]== 0), name='overlap-upper-bound')

#        m.addConstrs((D[i,t]+D[j,t] <= 1 for i in range(Ng)
#                                         for j in range(Ng)
#                                         if i < j
#					  if Mc[i,j] == 0
#                                         for t in range(T)), name='between-graph-orthogonality') # This is for ALL-pair orthogonality  
        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def baseline_cover_or_forget(Els, Mc, Ng, T):
        print("This is the edge cover version based on baseline approach in NeurIPS 2018 advanced with partial orthogonality")
        m = Model('Orthogonal Descriptions')
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	z = [] 
	for i in range(Ng):
		z.append(m.addVars(len(Els[i]),vtype=GRB.BINARY, name='Ignored'))
#        tf = m.addVars(T, vtype=GRB.BINARY, name="Tags_by_Disconnections_Forgotten")

#       upper = m.addVar(vtype=GRB.INTEGER, name="Overlap_Upperbound")

        obj1 = quicksum(D[i, t] for t in range(T)
                                for i in range(Ng))

        obj2 = quicksum(z[i][e] for i in range(Ng)
				for e in range(len(Els[i]))) # Overall ignored

        obj = obj1 + obj2
        m.setObjective(obj, GRB.MINIMIZE)

        print("Constructing constraints...")
        m.addConstrs(( z[i][e] + quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                          	                                                      for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((D[i,t]+D[j,t] <= 1 for i in range(Ng)
                                         for j in range(Ng)
                                         if i < j
                                         if Mc[i,j] == 0
                                         for t in range(T)), name='between-graph-orthogonality') # This is for ALL-pair orthogonality  

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def meta_info_single_graph(El, IEl, C, gs, gu, gl, T):
	m = Model("Meta-information single graph")
	Ng = Mc.shape[0]
	P = C.shape[1]
	D = m.addVars(T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        obj = quicksum(D[t] for t in range(T))

        m.setObjective(obj, GRB.MINIMIZE)    

        m.addConstrs((quicksum(D[t]*El[e][t] for t in range(T)) >= 1 for e in range(len(El))), name='cover_edges')

        m.addConstrs((IEl[t]*D[t]+(1-D[t])*gs*len(El) >= gs*len(El) for t in range(T)), name='guarantee_significance')

        m.addConstrs((quicksum(D[t]*C[t,p]) >= gl for p in range(P)), name='facilitate_diversity_lower')
        
        m.addConstrs((quicksum(D[t]*C[t,p]) <= gu for p in range(P)), name='facilitate_diversity_upper')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

#TODO: combine max significance and min description length: they are mutually compatible, at least on the Twitter data.

def meta_info_single_graph_cons_dl_max_sig(El, IEl, C, cons_dl, gu, gl, T):
        m = Model("Meta-information single graph: max sig!!!")
        P = C.shape[1]
        D = m.addVars(T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')

        obj = quicksum(IEl[t]*D[t] for t in range(T)) # maximize overall significance

        m.setObjective(obj, GRB.MAXIMIZE)

        m.addConstrs((quicksum(D[t]*El[e][t] for t in range(T)) >= 1 for e in range(len(El))), name='cover_edges')

        m.addConstr((quicksum(D[t] for t in range(T)) <= cons_dl), name='bound_description_length')

        m.addConstrs((quicksum(D[t]*C[t,p] for t in range(T)) >= gl for p in range(P)), name='facilitate_diversity_lower')

        m.addConstrs((quicksum(D[t]*C[t,p] for t in range(T)) <= gu for p in range(P)), name='facilitate_diversity_upper')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def single_graph_dl_ub_max_sig(El, IEl, cons_dl, T):
	m = Model("Single grahp max sig bound dl length")
	D = m.addVars(T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	
	obj = quicksum(IEl[t]*D[t] for t in range(T)) # Maximize overall significance
	m.setObjective(obj, GRB.MAXIMIZE)
	m.addConstrs((quicksum(D[t]*El[e][t] for t in range(T)) >= 1 for e in range(len(El))), name='cover_edges')

        m.addConstr((quicksum(D[t] for t in range(T)) <= cons_dl), name='bound_description_length')

	print("model update")
	m.update()
	print("model optimize")
	m.optimize()

	return m

def meta_info_single_graph_max_sig(El, IEl, C, gu, gl, T):
        m = Model("Meta-information single graph: max sig!!!")
        P = C.shape[1]
        D = m.addVars(T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	gs = m.addVar(vtype=GRB.CONTINUOUS, name='Lower_bounding_significance')

#        obj = quicksum(D[t] for t in range(T))

        m.setObjective(gs, GRB.MAXIMIZE)

        m.addConstrs((quicksum(D[t]*El[e][t] for t in range(T)) >= 1 for e in range(len(El))), name='cover_edges')

        m.addConstrs((IEl[t]*D[t]+(1-D[t])*gs*len(El) >= gs*len(El) for t in range(T)), name='guarantee_significance')

        m.addConstrs((quicksum(D[t]*C[t,p] for t in range(T)) >= gl for p in range(P)), name='facilitate_diversity_lower')

        m.addConstrs((quicksum(D[t]*C[t,p] for t in range(T)) <= gu for p in range(P)), name='facilitate_diversity_upper')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def meta_info_single_graph_min_dl(El, IEl, C, gu, gl, T):
        m = Model("Meta-information single graph: max sig!!!")
        P = C.shape[1]
        D = m.addVars(T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')

        obj = quicksum(D[t] for t in range(T))

        m.setObjective(obj, GRB.MINIMIZE)

        m.addConstrs((quicksum(D[t]*El[e][t] for t in range(T)) >= 1 for e in range(len(El))), name='cover_edges')

        m.addConstrs((quicksum(D[t]*C[t,p] for t in range(T)) >= gl for p in range(P)), name='facilitate_diversity_lower')

        m.addConstrs((quicksum(D[t]*C[t,p] for t in range(T)) <= gu for p in range(P)), name='facilitate_diversity_upper')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def single_graph_min_dl(El, IEl, T):
	m = Model("Min descriptor length.")
	D = m.addVars(T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	obj = quicksum(D[t] for t in range(T))
	m.setObjective(obj, GRB.MINIMIZE)

	m.addConstrs((quicksum(D[t]*El[e][t] for t in range(T)) >= 1 for e in range(len(El))), name='cover_edges')

	print("model update")
	m.update()
	print("model optimize")
	m.optimize()

	return m

def vtae_min_ol_min_dl_real(Els, IEls, Mc, k, T):
        print("VTAE orthogonality relaxed")
        m = Model("VTAE orthogonality relaxation")
        Ng = Mc.shape[0]
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        olu = m.addVar(vtype=GRB.INTEGER, name='Overlap_upper_bound')
        obj = quicksum(D[i,t] for t in range(T)
                              for i in range(Ng))

        m.setObjective(obj+olu, GRB.MINIMIZE)
        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((quicksum(D[i,t]+D[j,t] for j in range(Ng)
					     if i < j
					     if Mc[i,j] == 0) <= olu for t in range(T)
								     for i in range(T)), name='upper_bound_overlap')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def vtae_min_ol_min_dl(Els, IEls, Mc, k, T):
	print("VTAE orthogonality relaxed")
	m = Model("VTAE orthogonality relaxation")
	Ng = Mc.shape[0]
	D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	olu = m.addVar(vtype=GRB.INTEGER, name='Overlap_upper_bound')
        obj = quicksum(D[i,t] for t in range(T)
                              for i in range(Ng))

        m.setObjective(obj+olu, GRB.MINIMIZE)    
	m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

	m.addConstrs((quicksum(D[i,t]*D[j,t] for t in range(T)) <= olu for i in range(Ng)
								       for j in range(Ng)
								       if i < j
								       if Mc[i,j] == 0), name='upper_bound_overlap')

	print("model update")
	m.update()
	print("model optimize")
	m.optimize()

	return m

def meta_info_sufficient_cover_orth_relaxed(Els, IEls, C, gs, gu, gl, Mc, k, T):
	print("Total amount of edges in the first graph:", len(Els[0]))
	print("Meta-information with orthogonality relaxation")
	m = Model("Meta-information with Orthogonality Relaxation")
	Ng = Mc.shape[0]
	P = C.shape[1]
	print C.shape
	print P 
	D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	obj = quicksum(D[i,t] for t in range(T)
			      for i in range(Ng))

	orth_relax = quicksum(D[i,t]*D[j,t] for t in range(T)
                  	                    for i in range(Ng)
                               	     	    for j in range(Ng)
                               		    if i < j
                               		    if Mc[i,j] == 0)

	m.setObjective(obj+orth_relax, GRB.MINIMIZE)	

        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((IEls[i][t]*D[i,t]+(1-D[i,t])*gs*len(Els[i]) >= gs*len(Els[i]) for i in range(Ng)
                                                    				    for t in range(T)), name='guarantee_significance')

	
	m.addConstrs((quicksum(D[i,t]*C[t,p] for t in range(T)) >= gl  for i in range(Ng)
									for p in range(P)), name='facilitate_diversity_lower')
	
	m.addConstrs((quicksum(D[i,t]*C[t,p] for t in range(T)) <= gu for i in range(Ng)
								      for p in range(P)), name='facilitate_diversity_upper')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def vtae_max_sig_cons_ol_cons_dl_real(Els, IEls, Mc, k, T, cons_ol, cons_dl):
        print("Maximize influence constrained by overlap and description length")
        m = Model("Maximize influence constrained by overlap and description length")
        Ng = Mc.shape[0]
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        obj = quicksum(D[i,t]*IEls[i][t] for t in range(T)
                                         for i in range(Ng))
        m.setObjective(obj, GRB.MAXIMIZE)

	m.addConstrs((quicksum(D[i,t]+D[j,t] for j in range(Ng)
                                             if i < j
                                             if Mc[i,j] == 0) <= cons_ol for t in range(T)
									 for i in range(Ng)), name='cons_overlap')

        m.addConstr((quicksum(D[i,t] for t in range(T)
                                     for i in range(Ng)) <= cons_dl),name='cons_description_length')

        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def vtae_max_sig_cons_ol_cons_dl(Els, IEls, Mc, k, T, cons_ol, cons_dl):
	print("Maximize influence constrained by overlap and description length")
	m = Model("Maximize influence constrained by overlap and description length")
	Ng = Mc.shape[0]
	D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	obj = quicksum(D[i,t]*IEls[i][t] for t in range(T)
					 for i in range(Ng))
	m.setObjective(obj, GRB.MAXIMIZE)

	m.addConstrs((quicksum(D[i,t]*D[j,t] for t in range(T)) <= cons_ol for i in range(Ng)
					    				  for j in range(Ng)
					    				  if i < j
					    				  if Mc[i,j] == 0), name='cons_overlap')

	m.addConstr((quicksum(D[i,t] for t in range(T)
				     for i in range(Ng)) <= cons_dl),name='cons_description_length')

        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

	print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def meta_info_sufficient_cover_orth_relaxed_max_sig(Els, IEls, C, gsalpha, gu, gl, Mc, k, T):
        print("Total amount of edges in the first graph:", len(Els[0]))
        print("Meta-information with orthogonality relaxation")
        m = Model("Meta-information with Orthogonality Relaxation")
        Ng = Mc.shape[0]
        P = C.shape[1]
        print C.shape
        print P
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	gs = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name='Lower_bound_influence')
        obj = quicksum(D[i,t] for t in range(T)
                              for i in range(Ng))

        orth_relax = quicksum(D[i,t]*D[j,t] for t in range(T)
                                            for i in range(Ng)
                                            for j in range(Ng)
                                            if i < j
                                            if Mc[i,j] == 0)

        m.setObjective(obj+orth_relax-gsalpha*gs, GRB.MINIMIZE)

        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

#	m.addConstr(gs < 1, name='up_ratio')
#	m.addConstr(gs >= 0, name='low_ratio')	

        m.addConstrs((IEls[i][t]*D[i,t]+(1-D[i,t])*gs*len(Els[i]) >= gs*len(Els[i]) for i in range(Ng)
                                                                                    for t in range(T)), name='guarantee_significance')


        m.addConstrs((quicksum(D[i,t]*C[t,p] for t in range(T)) >= gl  for i in range(Ng)
                                                                        for p in range(P)), name='facilitate_diversity_lower')

        m.addConstrs((quicksum(D[i,t]*C[t,p] for t in range(T)) <= gu for i in range(Ng)
                                                                      for p in range(P)), name='facilitate_diversity_upper')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m



def meta_info_sufficient_cover_orth_rigorous(Els, IEls, C, gs, gu, gl, Mc, k, T):
        print("Total amount of edges in the first graph:", len(Els[0]))
        print("Meta-information with orthogonality relaxation")
        m = Model("Meta-information with Orthogonality Relaxation")
        Ng = Mc.shape[0]
        P = C.shape[1]
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        obj = quicksum(D[i,t] for t in range(T)
                              for i in range(Ng))

        m.setObjective(obj+orth_relax, GRB.MINIMIZE)

        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((D[i,t]+D[j,t] <= 1 for i in range(Ng)
                                         for j in range(Ng)
                                         if i < j
                                         if Mc[i,j] == 0
                                         for t in range(T)), name='between-graph-orthogonality') # This is for ALL-pair orthogonality

        m.addConstrs((IEls[i][t]*D[i,t]+(1-D[i,t])*gs*len(Els[i]) >= gs*len(Els[i]) for i in range(Ng)
                                                                                    for t in range(T)), name='guarantee_significance')


        m.addConstrs((quicksum(D[i,t]*C[t,p] for t in range(T)) >= gl for p in range(P)
                                                     		       for i in range(Ng)), name='facilitate_diversity_lower')

        m.addConstrs((quicksum(D[i,t]*C[t,p] for t in range(T)) <= gu for p in range(P)
                                                     		       for i in range(Ng)), name='facilitate_diversity_upper')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def meta_info_sufficient_cover_type_orthogonality(Els, IEls, Cats, Ng, k, T):
	print("Total amount of edges in the first graph:", len(Els[0]))
        print("Meta-information with orthogonality relaxation")
        m = Model("Meta-information with Orthogonality Relaxation")
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
        obj = quicksum(D[i,t] for t in range(T)
                              for i in range(Ng))

        m.setObjective(obj, GRB.MINIMIZE)

        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

	m.addConstrs((D[i,t]*Cats[t,1] == 0 for i in range(k)
					    for t in range(T)),name='intra-covering')

	m.addConstrs((D[i,t]*Cats[t,0] == 0 for i in range(k,Ng)
					    for t in range(T)),name='inter-covering')

#        m.addConstrs((D[i,t]*Cats[t,c]+D[j,t]*Cats[t,c] <= 1 for i in range(k)
 #                                        		     for j in range(k,Ng)
  #                                       		     for t in range(T)
#							     for c in range(2)), name='type-oriented-orthogonality') # This is for ALL-pair orthogonality

#        m.addConstrs((IEls[i][t]*D[i,t]+(1-D[i,t])*gs*len(Els[i]) >= gs*len(Els[i]) for i in range(Ng)
 #                                                                                   for t in range(T)), name='guarantee_significance')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def meta_info_sufficient_cover_type_learning(Els, IEls, Ng, k, T):
        print("Total amount of edges in the first graph:", len(Els[0]))
        print("Meta-information with orthogonality relaxation")
        m = Model("Meta-information with Orthogonality Relaxation")
        D = m.addVars(Ng, T, vtype=GRB.BINARY, name='Graph_Level_Edge_Explanations')
	Cats = m.addVars(T,3,vtype=GRB.BINARY, name='Type_Clustering')

        obj1 = quicksum(Cats[t,2] for t in range(T))
#	obj = quicksum(Cats[t,0]+Cats[t,1] for t in range(T))
	obj2 = quicksum(D[i,t] for t in range(T)
			       for i in range(Ng))
	obj = obj1 + obj2
        m.setObjective(obj, GRB.MINIMIZE)

        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng) # Necessarily covering each graph/subgraph.
                                                                           for e in range(len(Els[i]))), name='cover_edges')

        m.addConstrs((D[i,t]*Cats[t,1] == 0 for i in range(k)
                                            for t in range(T)),name='intra-covering') # Covering intra-block edges must NOT use intermediate

        m.addConstrs((D[i,t]*Cats[t,0] == 0 for i in range(k,Ng)
                                            for t in range(T)),name='inter-covering') # Covering inter-block edges must NOT use community-speak

	m.addConstrs((quicksum(Cats[t,c] for c in range(3)) == 1 for t in range(T)), name='Orthogonality_in_Type')

#        m.addConstrs((D[i,t]*Cats[t,c]+D[j,t]*Cats[t,c] <= 1 for i in range(k)
 #                                                           for j in range(k,Ng)
  #                                                          for t in range(T)
#                                                            for c in range(2)), name='type-oriented-orthogonality') # This is for ALL-pair orthogonality

#        m.addConstrs((IEls[i][t]*D[i,t]+(1-D[i,t])*gs*len(Els[i]) >= gs*len(Els[i]) for i in range(Ng)
 #                                                                                   for t in range(T)), name='guarantee_significance')

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

def minimize_cross_type_covering(Els, IEls, Mc, Ng, T): # TOP promising model - for paper NO.3, completely avoiding orthogonality constraints
# Benefits: clear interpretation of quasi-sufficient conditions
# Always feasible
# Considers disconnections and cross-graph edges simultaneously
# The objective function now is much more intuitive and useful than just attempting to find the "most succinct descriptions".
# Remaining issue: what is the complexity result for this model?
        m = Model('Quasi-sufficient and Complete Necessary Multi-graph Regularize Edge Set Cover')
        D = m.addVars(Ng, T, vtype=GRB.BINARY,name='Graph_Level_Edge_Explanations')
	dl = quicksum(D[i,t] for t in range(T)
			     for i in range(Ng))
        obj_quasi_sufficient_edges_across = quicksum( IEls[j][t]*D[i,t] for t in range(T)
                                                                        for i in range(Ng)
                                                                        for j in range(i+1,Ng)
                                                                        if Mc[i,j] == 0) # Quasi-sufficient for minimizing covered edges across graphs


        obj = obj_quasi_sufficient_edges_across
# + dl
        m.setObjective(obj, GRB.MINIMIZE)

        # Adding Constraints
        print("Constructing constraints ...")
        m.addConstrs((quicksum(D[i,t]*Els[i][e][t] for t in range(T)) >= 1 for i in range(Ng)
                                                                           for e in range(len(Els[i]))), name='cover_edges') # Necessary condition for covering edges in each graa

        print("model update")
        m.update()
        print("model optimize")
        m.optimize()

        return m

	
