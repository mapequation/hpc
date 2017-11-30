#include "hpc.h"

using namespace std;
using std::cout;
using std::cin;
using std::endl;

unsigned stou(char *s){
  return strtoul(s,(char **)NULL,10);
}

  // Call: trade <seed> <Ntries>
int main(int argc,char *argv[]){

  cout << "Version: June 28, 2017." << endl;
  cout << "Command: ";
  cout << argv[0];
  for(int i=1;i<argc; i++)
    cout << " " << argv[i];
  cout << endl;

  // Parse command input
  const string CALL_SYNTAX = "Call: ./hpc [-h] [-s <seed>] [-N <number of attempts>] [-n <max distance attempts>] [-t <distance threshold>] [-dt <divisive distance threshold>] [-d <number of clusters in each division (>= 2)>] [--skiplines N] [--validate N] [--k-fold-crossvalidate k] input_partitions.txt output_clustering_txt\n";
  if( argc == 1 ){
    cout << CALL_SYNTAX;
    exit(-1);
  }
  unsigned int seed = 1234;

  string inFileName = "noname";
  string outFileName = "noname";
  int Nskiplines = 0;

  int argNr = 1;
  unsigned int NsplitClu = 2;
  double distThreshold = 0.2;
  double splitDistThreshold = distThreshold;
  bool readSplitDistThreshold = false;
  int Nattempts = 1;
  int NdistAttempts = 1;
  int NvalidationPartitions = 0;
  int crossvalidate_k = 0;
  while(argNr < argc){
    if(to_string(argv[argNr]) == "-h"){
      cout << CALL_SYNTAX;
      cout << "seed: Any positive integer." << endl;  
      cout << "number of attempts: The number of clustering attempts. The best will be printed." << endl; 
      cout << "max distance attempts: The number of iterations to estimate the maximum distance in a cluster. Default is 1." << endl;  
      cout << "distance threshold: The max distance between two partitions in any cluster. Default is 0.2." << endl;
      cout << "divisive distance threshold: The max distance between two partitions in any cluster when the divisive clustering stops. Default is distance threshold." << endl;  
      cout << "number of clusters in each division (>= 2): The number of clusters the cluster with highest divergence will be divided into. Default is 2." << endl;
      cout << "number of attempts: The number of attempts to optimize the cluster assignments. Default is 1." << endl;  
      cout << "--skiplines N: Skip N lines in input_partitions.txt before reading data." << endl;
      cout << "--validate N: The number of partitions N at the end that will be used for validation. The first partitions will be used to find clusters. Default is 0 validation partitions." << endl; 
      cout << "--k-fold-crossvalidate k: Perform k-fold cross-validation of all partitions. The training partitions will be used to find clusters and the other ones for validation. Default is 0 folds for no cross-validation." << endl;
      cout << "input_partitions.txt: Each column corresponds to a partition and each row corresponds to a node id." << endl;  
      cout << "output_clustering.txt: clusterID partitionID" << endl;  
      cout << "-h: This help" << endl;
      exit(-1);
    }
    else if(to_string(argv[argNr]) == "-s"){
      argNr++;
      seed = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "--skiplines"){
      argNr++;
      Nskiplines = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "-N"){
      argNr++;
      Nattempts = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "-n"){
      argNr++;
      NdistAttempts = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "-dt"){
      readSplitDistThreshold = true;
      argNr++;
      splitDistThreshold = atof(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "-t"){
      argNr++;
      distThreshold = atof(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "--validate"){
      argNr++;
      NvalidationPartitions = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "--k-fold-crossvalidate"){
      argNr++;
      crossvalidate_k = atoi(argv[argNr]);
      argNr++;
    }  
    else if(to_string(argv[argNr]) == "-d"){
      argNr++;
      NsplitClu = atoi(argv[argNr]);
      if(NsplitClu < 2){
        cout << "Command error: -d must be integer larger or equal to 2." << endl;
        cout << CALL_SYNTAX;
        exit(-1);
      }
      argNr++;
    }
    else{

      if(argv[argNr][0] == '-'){
        cout << "Unknown command: " << to_string(argv[argNr]) << endl;
        cout << CALL_SYNTAX;
        exit(-1);
      }

      inFileName = string(argv[argNr]);
      argNr++;
      outFileName = string(argv[argNr]);
      argNr++;
    }

  }

  if(!readSplitDistThreshold)
    splitDistThreshold = distThreshold;

  if(inFileName == "noname"){
    cout << "Missing infile" << endl;
    cout << CALL_SYNTAX;
    exit(-1);
  }
  if(outFileName == "noname"){
    cout << "Missing outfile" << endl;
    cout << CALL_SYNTAX;
    exit(-1);
  }
  
  cout << "Setup:" << endl;
  cout << "-->Using seed: " << seed << endl;
  cout << "-->Will cluster partitions such that no cluster contains two partitions with distance larger than: " << distThreshold << endl;
  cout << "-->Will first divide clusters until no cluster contains two partitions with distance larger than: " << splitDistThreshold << endl;
  cout << "-->Will iteratively divide worst cluster into number of clusters: " << NsplitClu << endl;
  cout << "-->Will make number of attempts: " << Nattempts << endl;
  cout << "-->Will estimate max distance in cluster with number of attempts: " << NdistAttempts << endl;
  cout << "-->Will read partitions from file: " << inFileName << endl;
  if(Nskiplines > 0)
    cout << "-->skipping " << Nskiplines << " lines" << endl;
  if(crossvalidate_k > 0)
    cout << "-->performing " << crossvalidate_k << "-fold cross-valiation." << endl;
  else if(NvalidationPartitions > 0)
    cout << "-->using the last " << NvalidationPartitions << " partitions for validation." << endl;
  cout << "-->Will write clusters to file: " << outFileName << endl;
  cout << "-->Will use number of threads: " <<  omp_get_max_threads() << endl;

  Partitions partitions(inFileName,outFileName,Nskiplines,distThreshold,splitDistThreshold,NsplitClu,Nattempts,NdistAttempts,NvalidationPartitions,crossvalidate_k,seed);

  if(crossvalidate_k == 0){
    partitions.clusterPartitions(0);
    if(NvalidationPartitions > 0)
      partitions.validatePartitions(0);
    partitions.printClusters();
  }
  else{
    for(int fold = 0;fold<crossvalidate_k;fold++){
      cout << endl << "Fold " << fold+1 << "/" << crossvalidate_k << endl;
      partitions.clusterPartitions(fold);
      partitions.validatePartitions(fold);
      cout << "Fraction of validation partitions that fits in a cluster after " << fold+1 << " folds: " << 1.0*partitions.NtotValidated/partitions.NtotTested << endl;
    }
  }

}