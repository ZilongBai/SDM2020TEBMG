# SDM2020TEBMG

This repository contains files for our paper "Towards Explaining Block Models of Graphs" submitted to SDM 2020.

Authors:

Zilong Bai, S.S. Ravi, Ian Davidson

NOTES: 

>Filenames with *test_* are demo programs for implementation or our methods and the baseline:

- Filenames with *VTAE* demonstrate how our formulations and algorithm work
-- "mo" stands for minimizing overlap
-- filenames without "mo" but have "VTAE" demonstrate how our algorithm 1 works.

- Filename with *DTDM-cof* stands for our implementation of the cover-or-forget relaxation of DTDM from paper ["The Cluster Description Problem - Complexity Results, Formulations and Approximations" by I. Davidson et al.](https://papers.nips.cc/paper/7857-the-cluster-description-problem-complexity-results-formulations-and-approximations). This serves as our baseline in our experiments.

>Files for python methods:
- descriptor_discover.py : This file contains functions to implement ILP formulations for discovering explanations as set covering problem or its variant.
- divide_label_universe_smart_global.py : This file contains functions to achieve global disjointness for tag universes. Part of Algorithm 1.
- divide_label_universe_smart_partial.py : This file contains functions to achieve partial disjointness for tag universes. Part of Algorithm 1.
- edge_set_operations_smart.py : This file contains methods to create edge sets and edge set collection based on graph structure and the given block model. This file also contains methods to process the tag allocation matrices after excluding tags from different tag universes.
- util_smart.py: Methods in this file serve to post-process the result of Gurobi solvers, in particular to extract descriptors.
