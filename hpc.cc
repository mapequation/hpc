#include "wjs-kmedoids++-lumping.h"

using namespace std;
using std::cout;
using std::cin;
using std::endl;

unsigned stou(char *s){
  return strtoul(s,(char **)NULL,10);
}

  // Call: trade <seed> <Ntries>
int main(int argc,char *argv[]){

  cout << "Version: July 17, 2016." << endl;
  cout << "Command: ";
  cout << argv[0];
  for(int i=1;i<argc; i++)
    cout << " " << argv[i];
  cout << endl;

  // Parse command input
  const string CALL_SYNTAX = "Call: ./dangling-lumping [-s <seed>] [-N <number of attempts>] [-k <number of clusters>] [-d <number of clusters in each division (>= 2)>] [--fast] [--batchoutput] input_state_network.net output_state_network.net\n";
  if( argc == 1 ){
    cout << CALL_SYNTAX;
    exit(-1);
  }
  unsigned int seed = 1234;

  string inFileName;
  string outFileName;

  int argNr = 1;
  unsigned int NfinalClu = 100;
  unsigned int NsplitClu = 2;
  int Nattempts = 1;
  bool batchOutput = false;
  bool fast = false;
  while(argNr < argc){
    if(to_string(argv[argNr]) == "-h"){
      cout << CALL_SYNTAX;
      exit(-1);
    }
    else if(to_string(argv[argNr]) == "-s"){
      argNr++;
      seed = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "--batchoutput"){
      batchOutput = true;
      argNr++;
    }
    else if(to_string(argv[argNr]) == "--fast"){
      fast = true;
      argNr++;
    }
    else if(to_string(argv[argNr]) == "-N"){
      argNr++;
      Nattempts = atoi(argv[argNr]);
      argNr++;
    }
    else if(to_string(argv[argNr]) == "-k"){
      argNr++;
      NfinalClu = atoi(argv[argNr]);
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
  
  cout << "Setup:" << endl;
  cout << "-->Using seed: " << seed << endl;
  cout << "-->Will lump state nodes into number of clusters per physical node: " << NfinalClu << endl;
  cout << "-->Will iteratively divide worst cluster into number of clusters: " << NsplitClu << endl;
  cout << "-->Will make number of attempts: " << Nattempts << endl;
  if(fast)
    cout << "-->Will use medoid center to approximate cluster entropy rate during assignment." << endl;
  else
    cout << "-->Will use aggregate cluster to calculate entropy rate during assignment." << endl;
  // if(tune)
  //   cout << "-->Will tune medoids for bestter accuracy." << endl;
  // else
  //   cout << "-->Will not tune medoids for better accuracy." << endl;
  cout << "-->Will read state network from file: " << inFileName << endl;
  cout << "-->Will write processed state network to file: " << outFileName << endl;

  StateNetwork statenetwork(inFileName,outFileName,NfinalClu,NsplitClu,Nattempts,fast,batchOutput,seed);

  int NprocessedBatches = 0;
  while(statenetwork.loadStateNetworkBatch()){ // NprocessedBatches < 5 &&
    statenetwork.lumpStateNodes();
    NprocessedBatches++;
    if(statenetwork.keepReading || statenetwork.Nbatches > 1){
      statenetwork.printStateNetworkBatch();
      statenetwork.concludeBatch();
    }
    else{
      statenetwork.printStateNetwork();
      break;
    }
  }

  if(statenetwork.Nbatches > 1)
    statenetwork.compileBatches();

}