# Divisive hierarchical clustering of network partitions

## Author:

Martin Rosvall

For contact information, see http://www.mapequation.org/about


## Getting started:

In a terminal with the GNU Compiler Collection installed,
just run 'make' in the current directory to compile the
code with the included Makefile.


Call: ./hpc [-s \<seed\>]  [-N \<number of attempts\>] [-k \<number of clusters\>] [-d \<number of clusters in each division (>= 2)\>] input_partitions.txt output_clustering_txt  
seed: Any positive integer.  
number of clusters: The preferred number of clusters. Default is 100.  
nunmber of attempts: The number of attempts to optimize the cluster assignments. Default is 1.  
number of clusters in each division (>= 2): The number of clusters the cluster with highest divergence will be divided into. Default is 2.  
input_partitions.txt: Each column corresponds to a partition and each row corresponds to a node id.  
output_clustering.txt: partitionID clusterID    