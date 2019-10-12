# This is a demo script for paper "Towards Explaining Block Models of Graphs" submitted to SIAM Data Mining SDM2020
# Paper authors: Zilong Bai, S.S. Ravi, Ian Davidson
# Code author: Zilong Bai
# This code demonstrates how Algorithm 1 works to solve for a relaxed version of VTAE with partial disjointness
from gurobipy import *
import numpy as np
import sys
import descriptor_discover
import divide_label_universe_smart_partial	
import edge_set_operations_smart
import util_smart

X = np.genfromtxt('StructuralAdjIUR.csv',delimiter=',');
X[np.where(X>0)] = 1;
X[np.where(X<1)] = 0;
# Make the graph undirected if it is not.
X = X + np.transpose(X);
X[np.where(X>0)] = 1;
X[np.where(X<1)] = 0;

[n,n1] = X.shape

for i in range(n):
        X[i,i] = 0

L = np.genfromtxt('StructuralUserHashtagIUR.csv',delimiter=',');
L[np.where(L>0)] = 1;
L[np.where(L<1)] = 0;

[n,T] = X.shape
print("Number of tags:", T)

ll = np.matmul(L,np.transpose(L))
ll[np.where(ll>0)] = 1;
ll[np.where(ll<1)] = 0;

Behavioral_Graph = X # Retweet/following graph

Tags = L # We use hashtags as node tags/labels

T = Tags.shape[1]

mydata = {}

ktmp = 4; rep = 4;

kstr = str(ktmp)
repstr = str(rep)
print("Find descriptions for the intra-block edge collections:")
# Now, only consider different graphs or multiple graphs/subgraphs. No partial orthogonality concerned.
Els = []; IEls = []; 
UEls = []

C = []; # Cluster indicator indicating the graphs' membership to facilitate partial orthogonality

F = np.genfromtxt(('nmtf_'+kstr+'_'+repstr+'_F.csv'),delimiter=',')

for j in range(ktmp):
	Xtmp = Behavioral_Graph[np.where(F[:,j]>0)[0],:]
	Xtmp = Xtmp[:,np.where(F[:,j]>0)[0]]
	Ltmp = Tags[np.where(F[:,j]>0)[0],:]
	T = Ltmp.shape[1]
	El = edge_set_operations_smart.label_set_collection_creation(Xtmp, Ltmp, T)
	IEl, UEl = edge_set_operations_smart.label_stats(El, T)
	Els.append(El); UEls.append(UEl)
	IEls.append(IEl)
	C.append(np.array([1,0]))
	# Now we build edge set collections and disconnection set collections as bipartite graphs
for i in range(ktmp):
	for j in range(i+1,ktmp): # j > i, undirected graph.
		Xtmp = Behavioral_Graph[np.where(F[:,i]>0)[0],:]
		Xtmp = Xtmp[:,np.where(F[:,j]>0)[0]]
		Li = Tags[np.where(F[:,i]>0)[0],:]
		Lj = Tags[np.where(F[:,j]>0)[0],:]
		T = Li.shape[1]
		El = edge_set_operations_smart.label_set_collection_creation_bipartite(Xtmp, Li, Lj, T)
		IEl, UEl = edge_set_operations_smart.label_stats(El, T)
		Els.append(El); UEls.append(UEl)
		IEls.append(IEl)
		C.append(np.array([0,1]))

C = np.array(C) # Convert it into a [k + k*(k-1)/2] x p numpy matrix.

Mc = np.dot(C, np.transpose(C))
Mc = np.array(Mc)
print Mc.shape

print C.shape

orth_gt = []
for i in range(C.shape[0]):
	for j in range(i+1, C.shape[0]):
		orth_gt.append(np.dot(C[i,:], C[j,:]))

k = ktmp; Ng = len(Els)
print Ng
print len(IEls)

original_edges = 0
for i in range(Ng):
	original_edges = original_edges + len(Els[i])

D_min_dl = np.zeros((Ng, T))
D_max_sig = np.zeros((Ng, T))

m = divide_label_universe_smart_partial.luo_min_fl(UEls, Mc, Ng, T)
luo_step1_time = m.runTime
cons_fl = m.objVal
m = divide_label_universe_smart_partial.luo_min_influence_cons_fl(IEls, UEls, Mc, Ng, T, cons_fl)
luo_step2_time = m.runTime
RU = util_smart.unpack_gurobi_removed_tags(m, Ng, T)

Els = edge_set_operations_smart.batch_label_set_clean_up(Els, RU)

cover_time_min_dl = np.zeros(Ng)
cover_time_max_sig = np.zeros(Ng)

survived_edges = 0
for i in range(Ng):
	survived_edges = survived_edges + len(Els[i])

for i in range(Ng):
	if len(Els[i])>0:
		print len(Els[i])
		m = descriptor_discover.single_graph_min_dl(Els[i],IEls[i],T)
		cover_time_min_dl[i] = m.runTime		
		ub = m.objVal
		D_min_dl[i,:] = util_smart.unpack_gurobi_single_graph_descriptors(m,T)
		m = descriptor_discover.single_graph_dl_ub_max_sig(Els[i], IEls[i], ub, T)
		cover_time_max_sig[i] = m.runTime
		D_max_sig[i,:] = util_smart.unpack_gurobi_single_graph_descriptors(m,T)
	else:
		print Ng
		print i
		print len(Els[i])
		print "No edges in this set!!!"
mydata[kstr+'_'+repstr] = [D_min_dl, D_max_sig, luo_step1_time, luo_step2_time, cover_time_min_dl, cover_time_max_sig, original_edges, survived_edges]
print original_edges
print survived_edges

np.savez('VTAE_partial_T880.npz',data=mydata)
