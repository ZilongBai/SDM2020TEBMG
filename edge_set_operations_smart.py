# This is a demo script for paper "Towards Explaining Block Models of Graphs" submitted to SIAM Data Mining SDM2020
# Paper authors: Zilong Bai, S.S. Ravi, Ian Davidson
# Code author: Zilong Bai
import numpy as np

def label_set_collection_creation(X,L,T):
	print("Generate label set collection")
	[n,n1] = X.shape

	El = []

	for i in range(n):
		for j in range(n):
			if i < j: # We consider undirected graphs for now
				lij = np.multiply(L[i,:], L[j,:])
				if np.sum(lij) > 0:
					if X[i,j] == 1:
						El.append(lij) 

	return El

def label_set_collection_creation_graph_based(Xb,Xs,L,T):
	[n,n1] = Xb.shape
	El = []

	print Xb.shape
	print Xs.shape
	print L.shape
	for i in range(n):
		for j in range(n):
			if i < j:
				lij = np.multiply(L[i,:], L[j,:]) 
				if np.sum(lij) > 0:
					if (Xb[i,j] == 1 and Xs[i,j] == 0):
						El.append(lij)
	return El

def label_set_collection_creation_bipartite(X,L1,L2,T):
        print("Generate label set collection: for bipartite graphs")
# Generate two collections of label set from each input graph G, which is bipartite:
# Return:
## El: the collection of label sets for edges.
        [n1,n2] = X.shape

        El = []

        for i in range(n1):
                for j in range(n2): # i and j range in different sets of nodes as the input graph is bipartite
			lij = np.multiply(L1[i,:], L2[j,:])
			if np.sum(lij) > 0:
				if X[i,j] == 1:
					El.append(lij)

        return El

def label_stats(El, T): 
# Form label universe and compute label influence
# Return:
## Ul: label universe
## Il: label influence, i.e., how many edges are covered by each label
	print("Compute label set related stats: label universe and label influence")
	Il = np.zeros(T)
	for el in El:
		Il = Il + el

	Ul = np.zeros(T)
	Ul[np.where(Il > 0)] = 1

	return Il, Ul

def vertex_stats(Vl, T):
	print("Compute Vertex Set related stats")
	Il = np.zeros(T)
	for i in range(Vl.shape[0]):
		Il = Il + Vl[i,:]
	Ul = np.zeros(T)
	Ul[np.where(Il > 0)] = 1
	return Il, Ul

def label_set_clean_up(El, ru):
# Remove labels indicated by ru from all the label sets in the edge label set collection El
	print("Clean up label sets in each collection ...")
	su = np.ones(ru.shape) - ru
	Elnew = []
	for e in range(len(El)):
		Eltmp = np.multiply(El[e], su)
		if np.sum(Eltmp>0):
			Elnew.append(Eltmp)
	return Elnew

def batch_label_set_collection_creation(Xs, Ls, T, N):
# Given N graphs G's (i.e., len(Xs) == N), generate El for each graph
	Els = []
	for s in range(N):
		X = Xs[s]
		L = Ls[s]
		El = label_set_collection_creation(X,L,T)
		Els.append(El)

def batch_label_stats(Els, T):
	Ils = []
	Uls = []
	for El in Els:
		Il, Ul = label_stats(El, T)
		Ils.append(Il)
		Uls.append(Ul)
	return Ils, Uls

def batch_label_set_clean_up(Els, RU):
	for s in range(len(Els)):
		Els[s] = label_set_clean_up(Els[s], RU[s,:])

	return Els
