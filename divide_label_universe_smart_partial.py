# This is a demo script for paper "Towards Explaining Block Models of Graphs" submitted to SIAM Data Mining SDM2020
# Paper authors: Zilong Bai, S.S. Ravi, Ian Davidson
# Code author: Zilong Bai
import numpy as np
from gurobipy import *

def label_universe_orthogonalize(Ils, Uls, Mc, Ng, T): # This step by itself fails!!!
# There are k label universes in Uls and k influence stats in Ils
# The input tag set collections are grouped according to C (Ng x p) into p non-overlapping groups/cohorts.

        m = Model('Label Universe Orthogonalization with Minimum Effect on Edges')

        # Adding Variables
        ## Main Variable: block level attribute cover set
        RU = m.addVars(Ng,T,vtype=GRB.BINARY,name='Remove_from_each_label_universe')

        ## Objective function
	obj = quicksum(Ils[i][t]*RU[i,t] for t in range(T)
	                                 for i in range(Ng)) # overall affected tags that are on the edges within blocks

        m.setObjective(obj, GRB.MINIMIZE) # Find a way to wipe out tags to achieve orthogonality while minimizing the affected by minimizing the affected tags on edges.

        # Adding Constraints
        ## Define Intermediate Variables
	print Mc.shape
	print Ng
        m.addConstrs(((Uls[i][t]*(1-RU[i,t]) + Uls[j][t]*(1-RU[j,t])) <= 1 for t in range(T)
									   for i in range(Ng)
                                                        		   for j in range(Ng)
                                                        		   if i < j
                                                        		   if Mc[i][j] == 0
										), name='universe-orthogonality')

        # model update
        m.update()
        # model optimize
        m.optimize()

        return m

def luo_min_fl(Uls, Mc, Ng, T):
        m = Model('Label Universe Orthogonalization with Minimal Forgotten Labels')

        # Adding Variables
        ## Main Variable: block level attribute cover set
        RU = m.addVars(Ng,T,vtype=GRB.BINARY,name='Remove_from_each_label_universe')
	C = m.addVar(vtype=GRB.INTEGER, name='upper_bound')

        m.setObjective(C, GRB.MINIMIZE) # Find a way to wipe out tags to achieve orthogonality while minimizing the affected by minimizing the affected tags on edges.

        # Adding Constraints
        ## Define Intermediate Variables
        print Mc.shape
        print Ng
        m.addConstrs(((Uls[i][t]*(1-RU[i,t]) + Uls[j][t]*(1-RU[j,t])) <= 1 for t in range(T)
                                                                           for i in range(Ng)
                                                                           for j in range(Ng)
                                                                           if i < j
                                                                           if Mc[i][j] == 0
                                                                                ), name='universe-orthogonality')
	m.addConstrs((quicksum(RU[i,t] for t in range(T)) <= C for i in range(Ng)),name='bounding_removed')

        # model update
        m.update()
        # model optimize
        m.optimize()

        return m

def luo_min_influence_cons_fl(Ils, Uls, Mc, Ng, T, cons_fl): # This step by itself fails!!!
# There are k label universes in Uls and k influence stats in Ils
# The input tag set collections are grouped according to C (Ng x p) into p non-overlapping groups/cohorts.

        m = Model('Label Universe Orthogonalization with Minimum Effect on Edges')

        # Adding Variables
        ## Main Variable: block level attribute cover set
        RU = m.addVars(Ng,T,vtype=GRB.BINARY,name='Remove_from_each_label_universe')

        ## Objective function
        obj = quicksum(Ils[i][t]*RU[i,t] for t in range(T)
                                         for i in range(Ng)) # overall affected tags that are on the edges within blocks

        m.setObjective(obj, GRB.MINIMIZE) # Find a way to wipe out tags to achieve orthogonality while minimizing the affected by minimizing the affected tags on edges.

        # Adding Constraints
        ## Define Intermediate Variables
        print Mc.shape
        print Ng
        m.addConstrs(((Uls[i][t]*(1-RU[i,t]) + Uls[j][t]*(1-RU[j,t])) <= 1 for t in range(T)
                                                                           for i in range(Ng)
                                                                           for j in range(Ng)
                                                                           if i < j
                                                                           if Mc[i][j] == 0
                                                                                ), name='universe-orthogonality')

	m.addConstrs((quicksum(RU[i,t] for t in range(T)) <= cons_fl for i in range(Ng)), name='cons_fl')
        # model update
        m.update()
        # model optimize
        m.optimize()

        return m

