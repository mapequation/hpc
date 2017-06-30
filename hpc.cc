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
  const string CALL_SYNTAX = "Call: ./hpc [-h] [-s <seed>] [-N <number of attempts>] [-n <max distance iterations>] [-t <distance threshold>] [-k <number of clusters>] [-d <number of clusters in each division (>= 2)>] [--skiplines N] input_partitions.txt output_clustering_txt\n";
  if( argc == 1 ){
    cout << CALL_SYNTAX;
    exit(-1);
  }
  unsigned int seed = 1234;

  string inFileName = "noname";
  string outFileName = "noname";
  int Nskiplines = 0;

  int argNr = 1;
  unsigned int NfinalClu = 100;
  unsigned int NsplitClu = 2;
  double distThreshold = 0.0;
  int Nattempts = 1;
  int NdistAttempts = 1;
  while(argNr < argc){
    if(to_string(argv[argNr]) == "-h"){
      cout << CALL_SYNTAX;
      cout << "seed: Any positive integer." << endl;  
      cout << "number of attempts: The number of clustering attempts. The best will be printed." << endl; 
      cout << "max distance attempts: The number of iterations to estimate the maximum distance in a cluster. Default is 1." << endl;  
      cout << "max number of clusters: The max number of clusters after divisive clustering. Default is 100." << endl;  
      cout << "nunmber of attempts: The number of attempts to optimize the cluster assignments. Default is 1." << endl;  
      cout << "number of clusters in each division (>= 2): The number of clusters the cluster with highest divergence will be divided into. Default is 2." << endl;
      cout << "--skiplines N: Skip N lines in input_partitions.txt before reading data  " << endl;  
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
    else if(to_string(argv[argNr]) == "-k"){
      argNr++;
      NfinalClu = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "-t"){
      argNr++;
      distThreshold = atof(argv[argNr]);
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
  cout << "-->[NOT IMPLEMENTED YET] Will let number of clusters reach: " << NfinalClu << endl;
  cout << "-->Will iteratively divide worst cluster into number of clusters: " << NsplitClu << endl;
  cout << "-->Will make number of attempts: " << Nattempts << endl;
  cout << "-->Will estimate max distance in cluster with number of attempts: " << NdistAttempts << endl;
  cout << "-->Will read partitions from file: " << inFileName << endl;
  if(Nskiplines > 0)
    cout << "-->skipping " << Nskiplines << " lines";
  cout << "-->Will write clusters to file: " << outFileName << endl;

  Partitions partitions(inFileName,outFileName,Nskiplines,distThreshold,NfinalClu,NsplitClu,Nattempts,NdistAttempts,seed);

  partitions.clusterPartitions();
  partitions.printClusters();

}