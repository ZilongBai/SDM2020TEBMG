# This is a demo script for paper "Towards Explaining Block Models of Graphs" submitted to SIAM Data Mining SDM2020
# Paper authors: Zilong Bai, S.S. Ravi, Ian Davidson
# Code author: Zilong Bai
# Minimize overlap for VTAE with global disjointness
from gurobipy import *
import numpy as np
import sys
import descriptor_discover
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

m = descriptor_discover.vtae_all_min_ol(Els, Ng, T)

D, Dw, Db = util_smart.unpack_gurobi_model_descriptors_BM(m,Ng,k,T)

mydata[kstr+'_'+repstr] = [D,m.runTime]

np.savez('VTAE_mo_g_T880.npz',data=mydata)
