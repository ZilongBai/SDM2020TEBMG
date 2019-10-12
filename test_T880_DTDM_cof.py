# This is a demo script to implement related work for baseline to compare in the paper "Towards Explaining Block Models of Graphs" submitted to SIAM Data Mining SDM2020
# Paper authors: Zilong Bai, S.S. Ravi, Ian Davidson
# Code author: Zilong Bai
# This code serve to reproduce the DTDM with cover or forget relaxation from paper "The Cluster Description Problem - Complexity Results, Formulations and Approximations"
# authored by Ian Davidson, Antoine Gourru, and S Ravi, published at Advances in Neural Information Processing Systems 31 (NIPS 2018)
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

#kmax = 7; repmax = 10;

mydata = {}

#initk = 2; initrep = 1;

ktmp = 4;
rep = 4;
kstr = str(ktmp)
repstr = str(rep)
print("Find descriptions for the intra-block edge collections:")
# Now, only consider different graphs or multiple graphs/subgraphs. No partial orthogonality concerned.
Vls = []; 

C = []; # Cluster indicator indicating the graphs' membership to facilitate partial orthogonality

F = np.genfromtxt(('nmtf_'+kstr+'_'+repstr+'_F.csv'),delimiter=',')

for j in range(ktmp):
	Ltmp = Tags[np.where(F[:,j]>0)[0],:]
	llength = np.sum(Ltmp,axis=1)
	Ljtmp = Ltmp[np.where(llength>0)[0],:]
	Vls.append(Ljtmp); 

m = descriptor_discover.baseline_cof_v(Vls, ktmp, T);

D, Dw, Db = util_smart.unpack_gurobi_model_descriptors_BM(m,ktmp,ktmp,T)

mydata[kstr+'_'+repstr] = [D,m.runTime]


np.savez('The_DTDM_cof_T880_V.npz',data=mydata)
