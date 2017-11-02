# Divisive hierarchical clustering of network partitions

Finds clusters of partitions such that the maximum distance, measured by the complement of the weighted Jaccard similarity, does not exceed a user specified threshold.

## Author:

Martin Rosvall

For contact information, see http://www.mapequation.org/about


## Getting started:

In a terminal with the GNU Compiler Collection installed,
just run 'make' in the current directory to compile the
code with the included Makefile.


Call: ./hpc [-h] [-s \<seed\>] [-N \<number of attempts\>] [-n \<max distance iterations\>] [-t \<distance threshold\>] [-dt \<divisive distance threshold\>] [-d \<number of clusters in each division (>= 2)\>] [--skiplines N] [--validate N] input_partitions.txt output_clustering_txt  

seed: Any positive integer.  
number of attempts: The number of clustering attempts. The best will be printed.   
max distance attempts: The number of iterations to estimate the maximum distance in a cluster. Default is 1.   
distance threshold: The max distance between two partitions in any cluster. Default is 0.2.  
divisive distance threshold: The max distance between two partitions in any cluster when the divisive clustering stops. Default is distance threshold.    
number of clusters in each division (>= 2): The number of clusters the cluster with highest divergence will be divided into. Default is 2.  
number of attempts: The number of attempts to optimize the cluster assignments. Default is 1.  
--skiplines N: Skip N lines in input_partitions.txt before reading data.  
--validate N: The number of partitions N at the end that will be used for validation. The first partitions will be used to find clusters. Default is 0 validtion partitions.   
input_partitions.txt: Each column corresponds to a partition and each row corresponds to a node id.  
output_clustering.txt: clusterID partitionID.  
-h: This help.  

Example:

./hpc -N 10 -s 123 -t 0.5 -dt 0.5 input_partitions.txt output_clustering.txt    

input_multilevel_partitions.txt  
1:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1 1:1
1:1 1:1 1:1 1:1 1:2 1:1 1:1 1:1 1:2 1:1
1:2 1:2 1:2 1:2 1:3 1:2 1:1 1:2 1:2 1:2
1:2 1:3 1:3 1:2 1:4 1:2 1:2 1:2 1:2 1:2
2:1 2:1 2:1 2:1 2:1 1:3 1:2 1:2 1:3 1:3
2:2 2:2 2:2 2:2 2:1 2:1 2:1 2:1 2:1 2:1
3:1 3:1 3:1 3:1 3:1 2:1 2:1 2:1 2:1 2:1
3:1 3:1 3:1 3:2 3:2 2:2 2:1 2:2 2:1 2:1
3:2 3:2 3:2 3:2 3:3 2:2 2:2 2:2 2:2 2:2

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