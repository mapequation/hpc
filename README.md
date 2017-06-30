# Divisive hierarchical clustering of network partitions

Finds clusters of partitions such that the maximum distance, measured by the complement of the weighted Jaccard similarity, does not exceed a user specified threshold.

## Author:

Martin Rosvall

For contact information, see http://www.mapequation.org/about


## Getting started:

In a terminal with the GNU Compiler Collection installed,
just run 'make' in the current directory to compile the
code with the included Makefile.


Call: ./hpc [-h] [-s \<seed\>] [-N \<number of attempts\>] [-n \<max distance iterations\>] [-t \<distance threshold\>] [-k \<max number of clusters\>] [-d \<number of clusters in each division (>= 2)\>] [--skiplines N] input_partitions.txt output_clustering_txt  
seed: Any positive integer.  
number of attempts: The number of clustering attempts. The best will be printed.  
max distance attempts: The number of iterations to estimate the maximum distance in a cluster. Default is 1.
max number of clusters: The max number of clusters after divisive clustering. Default is 100.  
nunmber of attempts: The number of attempts to optimize the cluster assignments. Default is 1.  
number of clusters in each division (>= 2): The number of clusters the cluster with highest divergence will be divided into. Default is 2.  
--skiplines N: Skip N lines in input_partitions.txt before reading data    
input_partitions.txt: Each column corresponds to a partition and each row corresponds to a node id.  
output_clustering.txt: clusterID partitionID  

Example:

./hpc -N 10 -s 123 -t 0.5 input_partitions.txt output_clustering.txt    

input_partitions.txt  
1 1 1 1 1 1 6 1 1 1  
1 1 1 1 1 1 1 1 1 1  
1 2 1 1 1 2 2 6 2 2  
2 2 2 2 1 2 2 2 2 2  
2 2 2 2 2 3 3 3 6 3  
2 2 2 3 2 3 3 3 3 3  
3 3 2 3 3 4 4 4 4 6  
3 3 3 3 3 4 4 4 4 4  
3 3 3 3 3 5 5 5 5 5  

output_clustering.txt   
\# Clustered 10 partitions into 2 clusters with maximum distance 0.466666666666667, average maximum distance  0.344444444444444,  and maximum cluster size 5  
\# ClusterId PartitionId  
\# Cluster 1: 5 nodes with max distance 0.466666666666667  
1 1  
1 2  
1 3  
1 4  
1 5  
\# Cluster 2: 5 nodes with max distance 0.222222222222222  
2 6  
2 7  
2 8  
2 9  
2 10   