#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#include <stdio.h>
#else
  #define omp_get_thread_num() 0
  #define omp_get_max_threads() 1
#endif
using namespace std;
#include <limits>
const double epsilon = 1.0e-15;
const double bignum = 1.0;
const double threshold = 1.0e-10;

// ofstream with higher precision to avoid truncation errors
struct my_ofstream : ofstream {
  explicit my_ofstream(streamsize prec = 15)
  {
    this->precision(prec);
  }
};

enum WriteMode { STATENODES, LINKS, CONTEXTS };

template <class T>
inline string to_string (const T& t){
	stringstream ss;
	ss << t;
	return ss.str();
}

struct pairhash {
public:
  template <typename T, typename U>
  size_t operator()(const pair<T, U> &x) const
  {
    return x.first*31 + x.second;
  }
};

// // Identical hashes for T,U and U,T, but that will never happen since T,U are ordered
// struct pairhash {
// public:
//   template <typename T, typename U>
//   size_t operator()(const pair<T, U> &x) const
//   {
//     return hash<T>()(x.first) ^ hash<U>()(x.second);
//   }
// };

class StateNode{
public:
	StateNode();
	StateNode(int stateid, int physid, double outweight);
	int stateId;
	int updatedStateId;
	int physId;
	double outWeight;
	bool active = true;
	map<int,double> links;
	map<int,double> physLinks;
	vector<string> contexts;
};

StateNode::StateNode(){
};

StateNode::StateNode(int stateid, int physid, double outweight){
	stateId = stateid;
	physId = physid;
	outWeight = outweight;
}

class LocalStateNode{
public:
	LocalStateNode();
	LocalStateNode(int stateid);
	int stateId;
	double minDiv = bignum;
	double minDiv2 = bignum;
	StateNode *minCenterStateNode;
	StateNode *stateNode;
};

LocalStateNode::LocalStateNode(){
};

LocalStateNode::LocalStateNode(int stateid){
	stateId = stateid;
}

typedef multimap< double, vector<LocalStateNode>, greater<double> > SortedMedoids;

class Medoids{
public:

	Medoids();
	unsigned int maxNstatesInMedoid = 0;
	double sumMinDiv = 0.0;
	SortedMedoids sortedMedoids;
};

Medoids::Medoids(){
};

// struct CompareMedoids {
//   bool operator ()(const Medoid& m1, const Medoid& m2) const { 
//     return m1.first < m2.first;
//   }
// };

// bool cmpMedoids(const Medoid& m1, const Medoid& m2){ 
// 	return m1.first < m2.first;
// };

class PhysNode{
public:
	PhysNode();
	vector<int> stateNodeIndices;
	vector<int> stateNodeDanglingIndices;
};

PhysNode::PhysNode(){
};


class StateNetwork{
private:
	double calcEntropyRate();
	double calcEntropyRate(PhysNode &physNode);
	double calcEntropyRate(Medoids &medoids);
	double calcEntropyRate(vector<LocalStateNode> &medoid);
	double wJSdiv(int stateId1, int stateId2);
	double wJSdiv(StateNode &stateNode1, StateNode &stateNode2);
	void findCenters(Medoids &medoids);
	void findClusters(Medoids &medoids);
	double updateCenters(unordered_map<int,pair<double,vector<LocalStateNode> > > &newMedoids);
	void performLumping(Medoids &medoids);
	bool readLines(string &line,vector<string> &lines);
	void writeLines(ifstream &ifs_tmp, ofstream &ofs, WriteMode &writeMode, string &line,int &batchNr);
	void writeLines(ifstream &ifs_tmp, ofstream &ofs, WriteMode &writeMode, string &line);
	int randInt(int from, int to);
	double randDouble(double to);

	// For all batches
	string inFileName;
	string outFileName;
	string tmpOutFileName;
	string tmpOutFileNameStates;
	string tmpOutFileNameLinks;
	string tmpOutFileNameContexts;
	bool batchOutput = false;
	int Nattempts = 1;
	bool fast = false;
	vector<mt19937> mtRands;
	ifstream ifs;
  string line;
  double totWeight = 0.0;
  double accumWeight = 0.0;
  int updatedStateId = 0;
	double entropyRate = 0.0;
	unordered_map<int,int> completeStateNodeIdMapping;
  int totNphysNodes = 0;
	int totNstateNodes = 0;
	int totNlinks = 0;
	int totNdanglings = 0;
	int totNcontexts = 0;
	int totNphysDanglings = 0;

	// For each batch
	double weight = 0.0;
	int NphysNodes = 0;
	int NstateNodes = 0;
	int Nlinks = 0;
	int Ndanglings = 0;
	int Ncontexts = 0;
	int NphysDanglings = 0;
	unsigned int NfinalClu;
	unsigned int NsplitClu;
	// unordered_map<pair<int,int>,double,pairhash> cachedWJSdiv;
	unordered_map<int,int> stateNodeIdMapping;
	unordered_map<int,int> stateToPhysNodeMapping;
	unordered_map<int,PhysNode> physNodes;
	unordered_map<int,StateNode> stateNodes;

public:
	StateNetwork(string inFileName,string outFileName,unsigned int NfinalClu,unsigned int NsplitClu,int Nattempts,bool fast,bool batchOutput,int seed); 
	void lumpStateNodes();
	void loadNodeMapping();
	bool loadStateNetworkBatch();
	void printStateNetworkBatch();
	void printStateNetwork();
	void concludeBatch();
	void compileBatches();

	bool keepReading = true;
  int Nbatches = 0;

};

StateNetwork::StateNetwork(string inFileName,string outFileName,unsigned int NfinalClu,unsigned int NsplitClu,int Nattempts,bool fast,bool batchOutput,int seed){
	this->NfinalClu = NfinalClu;
	this->NsplitClu = NsplitClu;
	this->Nattempts = Nattempts;
	this->fast = fast;
	this->batchOutput = batchOutput;
	this->inFileName = inFileName;
	this->outFileName = outFileName;
	this->tmpOutFileName = string(outFileName).append("_tmp");
	this->tmpOutFileNameStates = string(outFileName).append("_tmpstates");
	this->tmpOutFileNameLinks = string(outFileName).append("_tmplinks");
	this->tmpOutFileNameContexts = string(outFileName).append("_tmpcontexts");

	int threads = max(1, omp_get_max_threads());

	for(int i = 0; i < threads; i++){
    mtRands.push_back(mt19937(seed+1));
  }

	// Open state network for building state node to physical node mapping
	ifs.open(inFileName.c_str());
	if(!ifs){
		cout << "failed to open \"" << inFileName << "\" exiting..." << endl;
		exit(-1);
	}
	loadNodeMapping();
	ifs.close();

	// Open state network to read batches one by one
	line = "First line";
	ifs.open(inFileName.c_str());

}

int StateNetwork::randInt(int from, int to){

	uniform_int_distribution<int> rInt(from,to);
	return rInt(mtRands[omp_get_thread_num()]);

}

double StateNetwork::randDouble(double to){

	uniform_real_distribution<double> rDouble(0.0,to);
	return rDouble(mtRands[omp_get_thread_num()]);

}

double StateNetwork::wJSdiv(StateNode &stateNode1, StateNode &stateNode2){

	if(stateNode1.stateId == stateNode2.stateId){
		return 0.0;
	}

	double h1 = 0.0; // The entropy rate of the first state node
	double h2 = 0.0; // The entropy rate of the second state node
	double h12 = 0.0; // The entropy rate of the lumped state node


	// else if(stateIndex1 > stateIndex2){ // Swap to make stateIndex1 lowest 
	// 	swap(stateIndex1,stateIndex2);
	// }

	// Cached values
	// unordered_map<pair<int,int>,double,pairhash>::iterator wJSdiv_it = cachedWJSdiv.find(make_pair(stateIndex1,stateIndex2));
	// if(wJSdiv_it != cachedWJSdiv.end())
	// 	return wJSdiv_it->second;

	// StateNode &stateNode1 = stateNodes[stateIndex1];
	// StateNode &stateNode2 = stateNodes[stateIndex2];

	// The out-link weights of the state nodes
	double ow1 = stateNode1.outWeight;
	double ow2 = stateNode2.outWeight;
	// Normalized weights over entire network
	double w1 = ow1/totWeight;
	double w2 = ow2/totWeight;
	// Normalized weights over state nodes 1 and 2
	double pi1 = w1 / (w1 + w2);
	double pi2 = w2 / (w1 + w2);

	if(ow1 < epsilon || ow2 < epsilon){ // If one or both state nodes are dangling
		return 0.0;
	}

	map<int,double>::iterator links1 = stateNode1.physLinks.begin();
	map<int,double>::iterator links2 = stateNode2.physLinks.begin();
	map<int,double>::iterator links1end = stateNode1.physLinks.end();
	map<int,double>::iterator links2end = stateNode2.physLinks.end();
	
	while(links1 != links1end && links2 != links2end){

		int diff = links1->first - links2->first;

		if(diff < 0){
		// If the first state node has a link that the second has not

			double p1 = links1->second/ow1;
			h1 -= p1*log2(p1);
			double p12 = pi1*links1->second/ow1;
			h12 -= p12*log2(p12);
			links1++;

		}
		else if(diff > 0){
		// If the second state node has a link that the second has not

			double p2 = links2->second/ow2;
			h2 -= p2*log2(p2);
			double p12 = pi2*links2->second/ow2;
			h12 -= p12*log2(p12);
			links2++;

		}
		else{ // If both state nodes have the link

			double p1 = links1->second/ow1;
			h1 -= p1*log2(p1);
			double p2 = links2->second/ow2;
			h2 -= p2*log2(p2);
			double p12 = pi1*links1->second/ow1 + pi2*links2->second/ow2;
			h12 -= p12*log2(p12);
			links1++;
			links2++;

		}
	}

	while(links1 != links1end){
		// If the first state node has a link that the second has not

		double p1 = links1->second/ow1;
		h1 -= p1*log2(p1);
		double p12 = pi1*links1->second/ow1;
		h12 -= p12*log2(p12);
		links1++;

	}

	while(links2 != links2end){
		// If the second state node has a link that the second has not

		double p2 = links2->second/ow2;
		h2 -= p2*log2(p2);
		double p12 = pi2*links2->second/ow2;
		h12 -= p12*log2(p12);
		links2++;

	}



	double div = (w1+w2)*h12 - w1*h1 - w2*h2;

	if(div < epsilon)
		div = epsilon;

	// Cached values
	// cachedWJSdiv[make_pair(stateIndex1,stateIndex2)] = div;


	return div;
}

double StateNetwork::wJSdiv(int stateIndex1, int stateIndex2){

	double h1 = 0.0; // The entropy rate of the first state node
	double h2 = 0.0; // The entropy rate of the second state node
	double h12 = 0.0; // The entropy rate of the lumped state node

	if(stateIndex1 == stateIndex2){
		return 0.0;
	}
	else if(stateIndex1 > stateIndex2){ // Swap to make stateIndex1 lowest 
		swap(stateIndex1,stateIndex2);
	}

	// Cached values
	// unordered_map<pair<int,int>,double,pairhash>::iterator wJSdiv_it = cachedWJSdiv.find(make_pair(stateIndex1,stateIndex2));
	// if(wJSdiv_it != cachedWJSdiv.end())
	// 	return wJSdiv_it->second;

	StateNode &stateNode1 = stateNodes[stateIndex1];
	StateNode &stateNode2 = stateNodes[stateIndex2];

	// The out-link weights of the state nodes
	double ow1 = stateNode1.outWeight;
	double ow2 = stateNode2.outWeight;
	// Normalized weights over entire network
	double w1 = ow1/totWeight;
	double w2 = ow2/totWeight;
	// Normalized weights over state nodes 1 and 2
	double pi1 = w1 / (w1 + w2);
	double pi2 = w2 / (w1 + w2);

	if(ow1 < epsilon || ow2 < epsilon){ // If one or both state nodes are dangling
		return 0.0;
	}

	map<int,double>::iterator links1 = stateNode1.physLinks.begin();
	map<int,double>::iterator links2 = stateNode2.physLinks.begin();
	map<int,double>::iterator links1end = stateNode1.physLinks.end();
	map<int,double>::iterator links2end = stateNode2.physLinks.end();
	
	while(links1 != links1end && links2 != links2end){

		int diff = links1->first - links2->first;

		if(diff < 0){
		// If the first state node has a link that the second has not

			double p1 = links1->second/ow1;
			h1 -= p1*log2(p1);
			double p12 = pi1*links1->second/ow1;
			h12 -= p12*log2(p12);
			links1++;

		}
		else if(diff > 0){
		// If the second state node has a link that the second has not

			double p2 = links2->second/ow2;
			h2 -= p2*log2(p2);
			double p12 = pi2*links2->second/ow2;
			h12 -= p12*log2(p12);
			links2++;

		}
		else{ // If both state nodes have the link

			double p1 = links1->second/ow1;
			h1 -= p1*log2(p1);
			double p2 = links2->second/ow2;
			h2 -= p2*log2(p2);
			double p12 = pi1*links1->second/ow1 + pi2*links2->second/ow2;
			h12 -= p12*log2(p12);
			links1++;
			links2++;

		}
	}

	while(links1 != links1end){
		// If the first state node has a link that the second has not

		double p1 = links1->second/ow1;
		h1 -= p1*log2(p1);
		double p12 = pi1*links1->second/ow1;
		h12 -= p12*log2(p12);
		links1++;

	}

	while(links2 != links2end){
		// If the second state node has a link that the second has not

		double p2 = links2->second/ow2;
		h2 -= p2*log2(p2);
		double p12 = pi2*links2->second/ow2;
		h12 -= p12*log2(p12);
		links2++;

	}



	double div = (w1+w2)*h12 - w1*h1 - w2*h2;

	if(div < epsilon)
		div = epsilon;

	// Cached values
	// cachedWJSdiv[make_pair(stateIndex1,stateIndex2)] = div;


	return div;
}


double StateNetwork::calcEntropyRate(){

	double h = 0.0;

	for(unordered_map<int,StateNode>::iterator it = stateNodes.begin(); it != stateNodes.end(); it++){
		StateNode &stateNode = it->second;
		// if(stateNode.active){
		// 	double H = 0.0;
		// 	double weight = 0.0;
		// 	unordered_map<int,double> physLinks;
		// 	for(map<int,double>::iterator it_link = stateNode.links.begin(); it_link != stateNode.links.end(); it_link++){
		// 		physLinks[stateToPhysNodeMapping[it_link->first]] += it_link->second;
		// 		weight += it_link->second;
		// 	}
		// 	for(unordered_map<int,double>::iterator it_link = physLinks.begin(); it_link != physLinks.end(); it_link++){
		// 		double p = it_link->second/weight;
		// 		H -= p*log2(p);	
		// 	}
			
		// 	h += weight*H/totWeight;
		// }
		if(stateNode.active){
			double H = 0.0;

			for(map<int,double>::iterator it_link = stateNode.physLinks.begin(); it_link != stateNode.physLinks.end(); it_link++){
				double p = it_link->second/stateNode.outWeight;
				H -= p*log2(p);
			}
			h += stateNode.outWeight*H/totWeight;
		}
	}

	return h;

}

double StateNetwork::calcEntropyRate(PhysNode &physNode){

	double h = 0.0;

	for(vector<int>::iterator stateNodeId_it = physNode.stateNodeIndices.begin(); stateNodeId_it != physNode.stateNodeIndices.end(); stateNodeId_it++){

		StateNode &stateNode = stateNodes[*stateNodeId_it];
		if(stateNode.active){
			double H = 0.0;

			for(map<int,double>::iterator it_link = stateNode.physLinks.begin(); it_link != stateNode.physLinks.end(); it_link++){
				double p = it_link->second/stateNode.outWeight;
				H -= p*log2(p);
			}
			h += stateNode.outWeight*H/totWeight;
		}
	}

	return h;

}

double StateNetwork::calcEntropyRate(Medoids &medoids){

	double h = 0.0;

	for(SortedMedoids::iterator medoid_it = medoids.sortedMedoids.begin(); medoid_it != medoids.sortedMedoids.end(); medoid_it++){
		// Inner loop over each set of medoids
		vector<LocalStateNode> &medoid = medoid_it->second;
		int NstatesInMedoid = medoid.size();
		// Create aggregated physical links
		unordered_map<int,double> medoidPhysLinks;
		double medoidOutWeight = 0.0;
		for(int i=0;i<NstatesInMedoid;i++){
			
			StateNode &stateNode = *medoid[i].stateNode;
			
			// Aggregate physical links
			for(map<int,double>::iterator link_it = stateNode.physLinks.begin(); link_it != stateNode.physLinks.end(); link_it++){
				medoidPhysLinks[link_it->first] += link_it->second;
			}
	
			medoidOutWeight += stateNode.outWeight;
		}
	
		double H = 0.0;

		for(unordered_map<int,double>::iterator it_link = medoidPhysLinks.begin(); it_link != medoidPhysLinks.end(); it_link++){
			double p = it_link->second/medoidOutWeight;
			H -= p*log2(p);
		}
		
		h += medoidOutWeight*H/totWeight;
	
	}

	return h;

}

double StateNetwork::calcEntropyRate(vector<LocalStateNode> &medoid){

	double h = 0.0;
	
	int NstatesInMedoid = medoid.size();
	// Create aggregated physical links
	unordered_map<int,double> medoidPhysLinks;
	double medoidOutWeight = 0.0;
	for(int i=0;i<NstatesInMedoid;i++){
		
		StateNode &stateNode = *medoid[i].stateNode;
		
		// Aggregate physical links
		for(map<int,double>::iterator link_it = stateNode.physLinks.begin(); link_it != stateNode.physLinks.end(); link_it++){
			medoidPhysLinks[link_it->first] += link_it->second;
		}

		medoidOutWeight += stateNode.outWeight;
	}

	double H = 0.0;
	for(unordered_map<int,double>::iterator it_link = medoidPhysLinks.begin(); it_link != medoidPhysLinks.end(); it_link++){
		double p = it_link->second/medoidOutWeight;
		H -= p*log2(p);
	}
	
	h += medoidOutWeight*H/totWeight;

	return h;

}

double StateNetwork::updateCenters(unordered_map<int,pair<double,vector<LocalStateNode> > > &newMedoids){
	
	double sumMinDiv = 0.0;

	// // Loop over all medoids to find new centers
	// for(vector<unordered_map<int,vector<LocalStateNode> > >::iterator medoids_it = medoidsTree.begin(); medoids_it != medoidsTree.end(); medoids_it++){
	// 	// Outer loop over sets of medoids
	// 	int NmedoidsInSet = medoids_it->size();
	// 	int NPstateNodesInSet = 0;
	// 	for(unordered_map<int,vector<LocalStateNode> >::iterator medoid_it = medoids_it->begin(); medoid_it != medoids_it->end(); medoid_it++){
	// 		// Inner loop over each set of medoids
	// 		vector<LocalStateNode> &medoid = medoid_it->second;
	// 		int NstatesInMedoid = medoid.size();
	// 		NPstateNodesInSet += NstatesInMedoid;
	// 		double minDivSumInMedoid = bignum*NstatesInMedoid;
	// 		int minDivSumInMedoidIndex = 0;
			
	
	// 		// Find new center with pivot method
	// 		if(NstatesInMedoid > 50){
	
	// 			// Find random state node to identify remote state node S1
	// 			uniform_int_distribution<int> randInt(0,NstatesInMedoid-1);
	// 			int randomStateInMedoidIndex = randInt(mtRands[omp_get_thread_num()]);
	// 			double maxDiv = 0.0;
	// 			int S1MedoidIndex = randomStateInMedoidIndex;
	// 			for(int j=0;j<NstatesInMedoid;j++){
	// 				double div = wJSdiv(*medoid[randomStateInMedoidIndex].stateNode,*medoid[j].stateNode);
	// 				medoid[j].minDiv = div;
	// 				if(div > maxDiv){
	// 					maxDiv = div;
	// 					S1MedoidIndex = j;
	// 				}
	// 			}
	// 			// Calculate distances to S1 and find S2
	// 			int S2MedoidIndex = S1MedoidIndex;
	// 			maxDiv = 0.0;
	// 			for(int j=0;j<NstatesInMedoid;j++){
	// 				double div = wJSdiv(*medoid[S1MedoidIndex].stateNode,*medoid[j].stateNode);
	// 				medoid[j].minDiv = div;
	// 				if(div > maxDiv){
	// 					maxDiv = div;
	// 					S2MedoidIndex = j;
	// 				}
	// 			}
	// 			// Calculate distances to S2 and projected distances to S1
	// 			double divS1toS2squared = pow(maxDiv,2.0);
	// 			vector<double> projDivToS1;
	// 			projDivToS1.reserve(NstatesInMedoid);
	
	// 			for(int j=0;j<NstatesInMedoid;j++){
	// 				double div = wJSdiv(*medoid[S2MedoidIndex].stateNode,*medoid[j].stateNode);
	// 				medoid[j].minDiv2 = div;
	// 				double divStoS2squared = pow(div,2.0);
	// 				double divStoS1squared = pow(medoid[j].minDiv,2.0);
	
	// 				projDivToS1.push_back(0.5*(divStoS1squared + divS1toS2squared - divStoS2squared)/divS1toS2squared);
	// 			}
	
	// 			// Find median projected distance to S1 on the line from S1 to S2
	// 			int n = projDivToS1.size()/2;
	// 			nth_element(projDivToS1.begin(), projDivToS1.begin()+n, projDivToS1.end());
	// 			double projMedianDiv = projDivToS1[n];
	
	// 			// Find state node clostest to median projected distance to S1 on the line from S1 to S2
	// 			for(int j=0;j<NstatesInMedoid;j++){
	// 				double DivSumInMedoid = fabs(medoid[j].minDiv-projMedianDiv) + fabs(medoid[j].minDiv2-(medoid[S1MedoidIndex].minDiv2-projMedianDiv));
	// 				if(DivSumInMedoid < minDivSumInMedoid){
	// 					minDivSumInMedoid = DivSumInMedoid;
	// 					minDivSumInMedoidIndex = j;
	// 				}
	// 			}
	
	// 		}
	// 		else{ 
	
	// 			// Find total divergence for each state node in medoid
	// 			for(int j=0;j<NstatesInMedoid;j++)
	// 				medoid[j].minDiv = 0.0;
	// 			for(int j=0;j<NstatesInMedoid;j++){
	// 				for(int k=j+1;k<NstatesInMedoid;k++){
	// 					double div = wJSdiv(*medoid[j].stateNode,*medoid[k].stateNode);
	// 					medoid[j].minDiv += div;
	// 					medoid[k].minDiv += div;
	// 				}
	// 			}
	// 			for(int j=0;j<NstatesInMedoid;j++){		
	// 				if(medoid[j].minDiv < minDivSumInMedoid){
	// 					minDivSumInMedoid = medoid[j].minDiv;
	// 					minDivSumInMedoidIndex = j;
	// 				}
	// 			}
	
	// 		}
	
	
	// 		// // Find total divergence for random subset of state nodes in medoid
	// 		// if(NrandStates > 0 && NstatesInMedoid > NrandStates){
	// 		// 	int NrandStatesGenerated = 0;
	// 		// 	vector<int> randStates = vector<int>(NrandStates);
	// 		// 	vector<int> randStateSpace = vector<int>(NstatesInMedoid);
	// 		// 	for(int i=0;i<NstatesInMedoid;i++)
	// 		// 		randStateSpace[i] = i;
	// 		// 	for(int i=0;i<NrandStates;i++){
	// 		// 		uniform_int_distribution<int> randInt(0,NstatesInMedoid-NrandStatesGenerated-1);
	// 		// 		int randElement = randInt(mtRands[omp_get_thread_num()]);
	// 		// 		randStates[NrandStatesGenerated] = randStateSpace[randElement];
	// 		// 		NrandStatesGenerated++;
	// 		// 		swap(randStateSpace[randElement],randStateSpace[NstatesInMedoid-NrandStatesGenerated-1]);
	// 		// 	}
	// 		// 	for(int jre=0;jre<NrandStates;jre++){
	// 		// 		medoid[randStates[jre]].minDiv = 0.0;
	// 		// 	}
	// 		// 	for(int jre=0;jre<NrandStates;jre++){
	// 		// 		int j = randStates[jre];
	// 		// 		for(int kre=jre+1;kre<NrandStates;kre++){
	// 		// 			int k = randStates[kre];
	// 		// 			double div = wJSdiv(*medoid[j].stateNode,*medoid[k].stateNode);
	// 		// 			medoid[j].minDiv += div;
	// 		// 			medoid[k].minDiv += div;
	// 		// 		}
	// 		// 	}
	// 		// 	for(int jre=0;jre<NrandStates;jre++){	
	// 		// 		int j = randStates[jre];
	// 		// 		if(medoid[j].minDiv < minDivSumInMedoid){
	// 		// 			minDivSumInMedoid = medoid[j].minDiv;
	// 		// 			minDivSumInMedoidIndex = j;
	// 		// 		}
	// 		// 	}
	// 		// }
	// 		// else{
	// 		// 	// Find total divergence for each state node in medoid
	// 		// 	for(int j=0;j<NstatesInMedoid;j++)
	// 		// 		medoid[j].minDiv = 0.0;
	// 		// 	for(int j=0;j<NstatesInMedoid;j++){
	// 		// 		for(int k=j+1;k<NstatesInMedoid;k++){
	// 		// 			double div = wJSdiv(*medoid[j].stateNode,*medoid[k].stateNode);
	// 		// 			medoid[j].minDiv += div;
	// 		// 			medoid[k].minDiv += div;
	// 		// 		}
	// 		// 	}
	// 		// 	for(int j=0;j<NstatesInMedoid;j++){		
	// 		// 		if(medoid[j].minDiv < minDivSumInMedoid){
	// 		// 			minDivSumInMedoid = medoid[j].minDiv;
	// 		// 			minDivSumInMedoidIndex = j;
	// 		// 		}
	// 		// 	}
	// 		// }
	
	// 		// Update localStateNodes to have minCenterStateNode point to the global StateNodes and place the center first in medoid.
	// 		StateNode *newCenterStateNode = medoid[minDivSumInMedoidIndex].stateNode;
	// 		for(int j=0;j<NstatesInMedoid;j++){
	// 			medoid[j].minCenterStateNode = newCenterStateNode;
	// 		}
	// 		swap(medoid[0],medoid[minDivSumInMedoidIndex]); // Swap such that the first index in medoid is center
	// 		medoid[0].minDiv = bignum; // Reset for next round
	// 	}

	// 	// Move to medoid associated with closest center
	// 	double sumMinDivInSet = bignum*(NPstateNodesInSet-NmedoidsInSet);
		
	// 	// Updated centers first in updated medoids
	// 	unordered_map<int,vector<LocalStateNode> > updatedMedoids;
	// 	for(unordered_map<int,vector<LocalStateNode> >::iterator medoid_it = medoids_it->begin(); medoid_it != medoids_it->end(); medoid_it++){
	// 		updatedMedoids[medoid_it->second[0].stateId].push_back(medoid_it->second[0]);
	// 	}
	
	// 	// Find closest center for all state nodes in physical node
	// 	for(unordered_map<int,vector<LocalStateNode> >::iterator medoid_it = medoids_it->begin(); medoid_it != medoids_it->end(); medoid_it++){
	// 		vector<LocalStateNode> &medoid = medoid_it->second;
	// 		int NstatesInMedoid = medoid.size();
	// 		for(int j=1;j<NstatesInMedoid;j++){ // Start at 1 because first index in medoid is center and already taken care of
	// 			medoid[j].minDiv = bignum;
	// 			for(unordered_map<int,vector<LocalStateNode> >::iterator updatedMedoid_it = updatedMedoids.begin(); updatedMedoid_it != updatedMedoids.end(); updatedMedoid_it++){
	// 				double div = wJSdiv(*medoid[j].stateNode,*updatedMedoid_it->second[0].stateNode);
	// 				if(div < medoid[j].minDiv){
	// 					sumMinDivInSet -= medoid[j].minDiv;
	// 					medoid[j].minDiv = div;
	// 					sumMinDivInSet += div;
	// 					medoid[j].minCenterStateNode = updatedMedoid_it->second[0].stateNode;
	// 				}
	// 			}	
	// 			// Add to updated medoids in medoid with closest center
	// 			medoid[j].minDiv = bignum; // Reset for next run
	// 			updatedMedoids[medoid[j].minCenterStateNode->stateId].push_back(medoid[j]);
	// 		}
	// 	}
	// 	sumMinDiv += sumMinDivInSet;
	// 	swap((*medoids_it),updatedMedoids);

	// }

	return sumMinDiv;

}

void StateNetwork::findCenters(Medoids &medoids){
	// Modifies the order of medoid(s) such that the fist NsplitClu will be the centers.
	// Also, all elements will contain the stateId it is closest to.

	vector<LocalStateNode> &medoid = medoids.sortedMedoids.begin()->second;
	unsigned int NstatesInMedoid = medoid.size();
	if(NstatesInMedoid <= NsplitClu){
		// All state nodes in medoid form their own medoids in the updated medoids
		for(unsigned int i=0;i<NstatesInMedoid;i++){
			medoid[i].minDiv = 0.0;
			medoid[i].minCenterStateNode = medoid[i].stateNode;
		}
	}
	else{
		// Find NsplitClu < NstatesInMedoid new centers in updated medoids
		double minDivSumInMedoid = bignum*NstatesInMedoid;
		unsigned int Ncenters = 0;
		// // Find NsplitClu random centers
		// while(Ncenters < NsplitClu)
		// 	// Find random state node in physical node as first center
		// 	uniform_int_distribution<int> randInt(Ncenters,NstatesInMedoid-1);
		// 	int newCenterIndex = randInt(mtRands[omp_get_thread_num()]);
		// 	minDivSumInMedoid -= medoid[newCenterIndex].minDiv;
		// 	medoid[newCenterIndex].minDiv = 0.0;
		// 	medoid[newCenterIndex].minCenterStateNode = medoid[newCenterIndex].stateNode;

		// 	// Put the center in first non-center position (Ncenters = 0) by swapping elements
		// 	swap(medoid[Ncenters],medoid[newCenterIndex]);
		// 	Ncenters++
		// 	StateNode *lastClusterStateNode = medoid[Ncenters-1].stateNode;
		// 	for(int i=Ncenters;i<NstatesInMedoid;i++){
		// 		double div = wJSdiv(*medoid[i].stateNode,*lastClusterStateNode);
		// 		if(div < medoid[i].minDiv){
		// 			// Found new minimum divergence to center
		// 			minDivSumInMedoid -= medoid[i].minDiv;
		// 			minDivSumInMedoid += div;
		// 			medoid[i].minDiv = div;
		// 			medoid[i].minCenterStateNode = lastClusterStateNode;
		// 		}
		// 	}				
		// 
		// Find random state node in physical node as first center
		// uniform_int_distribution<int> randInt(0,NstatesInMedoid-1);
		// int firstCenterIndex = randInt(mtRands[omp_get_thread_num()]);
		int firstCenterIndex = randInt(0,NstatesInMedoid-1);
		
		// ************ Begin find state node proportional to distance from random node
		StateNode *firstClusterStateNode = medoid[firstCenterIndex].stateNode;
		vector<double> firstMinDiv(NstatesInMedoid);
		double sumFirstMinDiv = 0.0;
		for(unsigned int i=0;i<NstatesInMedoid;i++){
			double div = wJSdiv(*medoid[i].stateNode,*firstClusterStateNode);
			firstMinDiv[i] = div;
			sumFirstMinDiv += div;
		}
		// Pick new center proportional to minimum divergence
		// uniform_real_distribution<double> randDouble(0.0,sumFirstMinDiv);
		// double randMinDivSum = randDouble(mtRands[omp_get_thread_num()]);
		double randMinDivSum = randDouble(sumFirstMinDiv);
		double minDivSum = 0.0;
		for(unsigned int i=0;i<NstatesInMedoid;i++){
			minDivSum += firstMinDiv[i];
			if(minDivSum > randMinDivSum){
				firstCenterIndex = i;
				break;
			}
		}
		firstMinDiv = vector<double>(0);
		// ************* End find state node proportional to distance from random node

		medoid[firstCenterIndex].minCenterStateNode = medoid[firstCenterIndex].stateNode;
		minDivSumInMedoid -= medoid[firstCenterIndex].minDiv;
		medoid[firstCenterIndex].minDiv = 0.0;
		// Put the center in first non-center position (Ncenters = 0) by swapping elements
		swap(medoid[Ncenters],medoid[firstCenterIndex]);
		Ncenters++;
	
	
		// Find NsplitClu-1 more centers based on the k++ algorithm
		while(Ncenters < NsplitClu){
			StateNode *lastClusterStateNode = medoid[Ncenters-1].stateNode;
			for(unsigned int i=Ncenters;i<NstatesInMedoid;i++){
				double div = wJSdiv(*medoid[i].stateNode,*lastClusterStateNode);
				if(div < medoid[i].minDiv){
					// Found new minimum divergence to center
					minDivSumInMedoid -= medoid[i].minDiv;
					medoid[i].minDiv = div;
					minDivSumInMedoid += medoid[i].minDiv;
					medoid[i].minCenterStateNode = lastClusterStateNode;
				}
			}
			// Pick new center proportional to minimum divergence
			// uniform_real_distribution<double> randDouble(0.0,minDivSumInMedoid);
			// double randMinDivSum = randDouble(mtRands[omp_get_thread_num()]);
			double randMinDivSum = randDouble(minDivSumInMedoid);
			double minDivSum = 0.0;
			unsigned int newCenterIndex = Ncenters;
			for(unsigned int i=Ncenters;i<NstatesInMedoid;i++){
				minDivSum += medoid[i].minDiv;
				if(minDivSum > randMinDivSum){
					newCenterIndex = i;
					break;
				}
			}
			minDivSumInMedoid -= medoid[newCenterIndex].minDiv;
			medoid[newCenterIndex].minDiv = 0.0;
			medoid[newCenterIndex].minCenterStateNode = medoid[newCenterIndex].stateNode;
			// Put the center in first non-center position by swapping elements
			swap(medoid[Ncenters],medoid[newCenterIndex]);
			Ncenters++;
		}
	
		// Check if last center gives minimum divergence for some state nodes
		StateNode *lastClusterStateNode = medoid[Ncenters-1].stateNode;
		for(unsigned int i=Ncenters;i<NstatesInMedoid;i++){
			double div = wJSdiv(*medoid[i].stateNode,*lastClusterStateNode);
			if(div < medoid[i].minDiv){
				// Found new minimum divergence to center
				minDivSumInMedoid -= medoid[i].minDiv;
				medoid[i].minDiv = div;
				minDivSumInMedoid += medoid[i].minDiv;
				medoid[i].minCenterStateNode = lastClusterStateNode;
			}
			
		}

	}
			
  // Identify new medoids
  double localSumMinDiv = 0.0;
	unordered_map<int,pair<double,vector<LocalStateNode> > > newMedoids;
	unordered_map<int,pair<double,vector<LocalStateNode> > >::iterator newMedoids_it;

	for(unsigned int i=0;i<NstatesInMedoid;i++){

		int centerId = medoid[i].minCenterStateNode->stateId;
		double minDiv = medoid[i].minDiv;
		localSumMinDiv += minDiv;
		medoid[i].minDiv = bignum; // Reset for next iteration
		newMedoids_it = newMedoids.find(centerId);

		if(newMedoids_it == newMedoids.end()){
			pair<double,vector<LocalStateNode> > newMedoid;
			newMedoid.first = minDiv;
			newMedoid.second.push_back(medoid[i]);
			newMedoids.emplace(make_pair(centerId,newMedoid));
		}
		else{
			newMedoids_it->second.first += minDiv;
			newMedoids_it->second.second.push_back(medoid[i]);
		} 

	}



	// Remove the split medoid
	medoids.sumMinDiv -= medoids.sortedMedoids.begin()->first;
	medoids.maxNstatesInMedoid = min(medoids.maxNstatesInMedoid,static_cast<unsigned int>(medoids.sortedMedoids.begin()->second.size()));
	medoids.sortedMedoids.erase(medoids.sortedMedoids.begin());
	
	// Add the new medoids
	for(newMedoids_it = newMedoids.begin(); newMedoids_it != newMedoids.end(); newMedoids_it++){
		// double h = calcEntropyRate(newMedoids_it->second.second);
		medoids.sumMinDiv += newMedoids_it->second.first;
		// newMedoids_it->second.first = h;

		medoids.maxNstatesInMedoid = max(medoids.maxNstatesInMedoid,static_cast<unsigned int>(newMedoids_it->second.second.size()));
		medoids.sortedMedoids.insert(move(newMedoids_it->second));
	} 

	if(medoids.sortedMedoids.size() < NfinalClu)
		findCenters(medoids);

}

void StateNetwork::findClusters(Medoids &medoids){
	// Modifies the order of medoid(s) such that the fist NsplitClu will be the centers.
	// Also, all elements will contain the stateId it is closest to.

	// Find medoid with highest entropy rate that can be split
	SortedMedoids::iterator sortedMedoidsIt = medoids.sortedMedoids.begin();
	while(sortedMedoidsIt != medoids.sortedMedoids.end() && sortedMedoidsIt->second.size() == 1)
		sortedMedoidsIt++;

	if(sortedMedoidsIt == medoids.sortedMedoids.end())
		return;

	vector<LocalStateNode> &medoid = sortedMedoidsIt->second;
	unsigned int NstatesInMedoid = medoid.size();
	if(NstatesInMedoid <= NsplitClu){
		// All state nodes in medoid form their own medoids in the updated medoids
		for(unsigned int i=0;i<NstatesInMedoid;i++){
			medoid[i].minDiv = 0.0;
			medoid[i].minCenterStateNode = medoid[i].stateNode;
		}
	}
	else{
		// Find NsplitClu < NstatesInMedoid new centers in updated medoids
		double minDivSumInMedoid = bignum*NstatesInMedoid;
		unsigned int Ncenters = 0;
		int seedCenterIndex = randInt(0,NstatesInMedoid-1);
		swap(medoid[Ncenters],medoid[seedCenterIndex]);

		vector<double> seedMinDiv(NstatesInMedoid);
		while(Ncenters < NsplitClu){
		
			// ************ Begin find state node proportional to distance from random node
			int lastCenter = Ncenters-1;
			if(lastCenter < 0)
			 	lastCenter = 0;
			StateNode *lastClusterStateNode = medoid[lastCenter].stateNode;
			
			double sumSeedMinDiv = 0.0;
			for(unsigned int i=lastCenter;i<NstatesInMedoid;i++){
				double div = wJSdiv(*medoid[i].stateNode,*lastClusterStateNode);
				seedMinDiv[i] = div;
				sumSeedMinDiv += div;
			}
			
			// Pick new center proportional to minimum divergence
			// uniform_real_distribution<double> randDouble(0.0,sumFirstMinDiv);
			// double randMinDivSum = randDouble(mtRands[omp_get_thread_num()]);
			double randMinDivSum = randDouble(sumSeedMinDiv);
			double minDivSum = 0.0;
			for(unsigned int i=lastCenter;i<NstatesInMedoid;i++){
				minDivSum += seedMinDiv[i];
				if(minDivSum >= randMinDivSum){
					seedCenterIndex = i;
					break;
				}
			}

			// cout << lastCenter << " " << sumSeedMinDiv << " " << minDivSum << endl;

			// ************* End find state node proportional to distance from random node
	
			medoid[seedCenterIndex].minCenterStateNode = medoid[seedCenterIndex].stateNode;
			minDivSumInMedoid -= medoid[seedCenterIndex].minDiv;
			medoid[seedCenterIndex].minDiv = 0.0;
			// Put the center in first non-center position (Ncenters = 0) by swapping elements
			swap(medoid[Ncenters],medoid[seedCenterIndex]);
			Ncenters++;

		}
		seedMinDiv = vector<double>(0);
		// cout << endl;
		// Iterate in random order
		int NrandStatesGenerated = Ncenters;
		vector<int> randStateOrder(NstatesInMedoid);
		for(unsigned int i=0;i<NstatesInMedoid;i++)
			randStateOrder[i] = i;
		for(unsigned int i=Ncenters;i<NstatesInMedoid;i++){
			int randElement = randInt(NrandStatesGenerated,NstatesInMedoid-1);
			swap(randStateOrder[NrandStatesGenerated],randStateOrder[randElement]);
			NrandStatesGenerated++;
		}

		// Create clusters
		vector<StateNode> aggregatedClusters(Ncenters);
		for(unsigned int i=0;i<Ncenters;i++){
			aggregatedClusters[i].outWeight = medoid[i].stateNode->outWeight;
			aggregatedClusters[i].physLinks = medoid[i].stateNode->physLinks;
		}

		// Cluster remaining states, lump in each step
		for(unsigned int i=Ncenters;i<NstatesInMedoid;i++){
			int randStateId = randStateOrder[i];
			StateNode &randStateNode = *medoid[randStateId].stateNode;
			int bestCluster = 0;
			double minDiv = bignum;
			for(unsigned int j=0;j<Ncenters;j++){
				double div = wJSdiv(randStateNode,aggregatedClusters[j]);
				if(div < minDiv){
					// Found new minimum divergence to center
					minDiv = div;
					bestCluster = j;
				}
			}
			// Perform best lumping
			aggregatedClusters[bestCluster].outWeight += medoid[randStateId].stateNode->outWeight;
			for(map<int,double>::iterator linkIt = randStateNode.physLinks.begin(); linkIt != randStateNode.physLinks.end(); linkIt++){
				aggregatedClusters[bestCluster].physLinks[linkIt->first] += linkIt->second;
			}
			medoid[randStateId].minCenterStateNode = medoid[bestCluster].stateNode;
		}

	}
			
  // Identify new medoids
  // double localSumMinDiv = 0.0;
	unordered_map<int,pair<double,vector<LocalStateNode> > > newMedoids;
	unordered_map<int,pair<double,vector<LocalStateNode> > >::iterator newMedoids_it;

	for(unsigned int i=0;i<NstatesInMedoid;i++){

		int centerId = medoid[i].minCenterStateNode->stateId;
		// double minDiv = medoid[i].minDiv;
		// localSumMinDiv += minDiv;
		// medoid[i].minDiv = bignum; // Reset for next iteration
		newMedoids_it = newMedoids.find(centerId);

		if(newMedoids_it == newMedoids.end()){
			pair<double,vector<LocalStateNode> > newMedoid;
			// newMedoid.first = minDiv;
			newMedoid.second.push_back(medoid[i]);
			newMedoids.emplace(make_pair(centerId,newMedoid));
		}
		else{
			// newMedoids_it->second.first += minDiv;
			newMedoids_it->second.second.push_back(medoid[i]);
		} 

	}

	// Remove the split medoid
	medoids.sumMinDiv -= sortedMedoidsIt->first;
	medoids.maxNstatesInMedoid = min(medoids.maxNstatesInMedoid,static_cast<unsigned int>(sortedMedoidsIt->second.size()));
	medoids.sortedMedoids.erase(sortedMedoidsIt);
	
	// Add the new medoids
	for(newMedoids_it = newMedoids.begin(); newMedoids_it != newMedoids.end(); newMedoids_it++){
		double h = calcEntropyRate(newMedoids_it->second.second);
		newMedoids_it->second.first = h;
		medoids.sumMinDiv += h;
		// cout << newMedoids_it->first << " " << h << " " << newMedoids_it->second.second.size() << endl;
		
		medoids.maxNstatesInMedoid = max(medoids.maxNstatesInMedoid,static_cast<unsigned int>(newMedoids_it->second.second.size()));
		medoids.sortedMedoids.insert(move(newMedoids_it->second));
	} 

	if(medoids.sortedMedoids.size() < NfinalClu)
		findClusters(medoids);

}

void StateNetwork::performLumping(Medoids &medoids){

// int NPstateNodes = localStateNodes[0].size();
	// Update stateNodes to reflect the lumping

	// // Validation
	// unordered_set<int> c;
	// for(int i=0;i<NsplitClu;i++)
	// 	c.insert(localStateNodes[i].stateId);
	// for(int i=NsplitClu;i<NPstateNodes;i++){
	// 	unordered_set<int>::iterator it = c.find(localStateNodes[i].minCenterStateId);
	// 	if(it == c.end())
	// 		cout << ":::::::::+++++++ ERROR for pos " << i << " " << localStateNodes[i].minDiv << " " << localStateNodes[i].minCenterStateId << endl;
	// }

	for(SortedMedoids::iterator medoid_it = medoids.sortedMedoids.begin(); medoid_it != medoids.sortedMedoids.end(); medoid_it++){
		// Inner loop over each set of medoids
		vector<LocalStateNode> &medoid = medoid_it->second;
		int NstatesInMedoid = medoid.size();

		// Only lump non-centers; first element is a center.
		for(int i=1;i<NstatesInMedoid;i++){

			// Only lump non-centers; first NsplitClu elements contain centers.
			StateNode &lumpedStateNode = *medoid[i].minCenterStateNode;
			StateNode &lumpingStateNode = *medoid[i].stateNode;
			// Add context to lumped state node
			lumpedStateNode.contexts.insert(lumpedStateNode.contexts.begin(),lumpingStateNode.contexts.begin(),lumpingStateNode.contexts.end());
			// Add links to lumped state node
			for(map<int,double>::iterator link_it = lumpingStateNode.links.begin(); link_it != lumpingStateNode.links.end(); link_it++){
				lumpedStateNode.links[link_it->first] += link_it->second;
			}
			// Add physical links to lumped state node
			for(map<int,double>::iterator link_it = lumpingStateNode.physLinks.begin(); link_it != lumpingStateNode.physLinks.end(); link_it++){
				lumpedStateNode.physLinks[link_it->first] += link_it->second;
			}
	
			lumpedStateNode.outWeight += lumpingStateNode.outWeight;
	
			// Update state id of lumping state node to point to lumped state node and make it inactive
			lumpingStateNode.updatedStateId = lumpedStateNode.stateId;
			lumpingStateNode.active = false;
		}
	}

}

void StateNetwork::lumpStateNodes(){

	cout << "Lumping state nodes in each physical node, using " << omp_get_max_threads() << " threads:" << endl;
	#ifdef _OPENMP
	// Initiate locks to keep track of best solutions
	omp_lock_t lock[NphysNodes];
	for (int i=0; i<NphysNodes; i++)
    omp_init_lock(&(lock[i]));
	#endif

	// To keept track of best solutions
	vector<int> attemptsLeftVec(NphysNodes,0);
	// vector<vector<pair<int,double> > > runDetails(NphysNodes);
	vector<double> bestEntropyRate(NphysNodes);
	vector<Medoids> bestMedoids(NphysNodes);

	// To be able to parallelize loop over physical nodes
	int NtotAttempts = 0;
	int physNodeNr = 0;
	unordered_map<int,int> attemptToPhysNodeNrMap;
	vector<PhysNode*> physNodeVec;
	for(unordered_map<int,PhysNode>::iterator phys_it = physNodes.begin(); phys_it != physNodes.end(); phys_it++){
		unsigned int NPstateNodes = phys_it->second.stateNodeIndices.size();
		if(NPstateNodes > NfinalClu){
			// Non-trivial problem with need for multiple attempts
			bestEntropyRate[physNodeNr] = 100.0;
			for(int i=0;i<Nattempts;i++){
				physNodeVec.push_back(&phys_it->second);
				attemptToPhysNodeNrMap[NtotAttempts] = physNodeNr;
				attemptsLeftVec[physNodeNr]++;
				NtotAttempts++;
			}
		}
		else{
			// Trivial problem with no need for multiple attempts
			physNodeVec.push_back(&phys_it->second);
			attemptToPhysNodeNrMap[NtotAttempts] = physNodeNr;
			attemptsLeftVec[physNodeNr]++;
			NtotAttempts++;
		}
		physNodeNr++;
	}

	// #pragma omp parallel 
  {
  	// #pragma omp single nowait
    {
			// for(unordered_map<int,PhysNode>::iterator phys_it = physNodes.begin(); phys_it != physNodes.end(); phys_it++){
			// for(vector<PhysNode*>::iterator phys_it = physNodeVec.begin(); phys_it < physNodeVec.end(); phys_it++){
    	#pragma omp parallel for schedule(dynamic,1) // default(none) shared(attemptsLeftVec,bestEntropyRate,bestMedoidsTree,physNodeVec,lock)
    	for(int attempt=0;attempt<NtotAttempts;attempt++){

				// #pragma omp task
        {
        	int physNodeNr = attemptToPhysNodeNrMap[attempt];
					PhysNode &physNode = *physNodeVec[attempt];
					// PhysNode &physNode = phys_it->second;
					unsigned int NPstateNodes = physNode.stateNodeIndices.size();
					double preLumpingEntropyRate = calcEntropyRate(physNode);

					if(NPstateNodes > NfinalClu){

						// Initialize vector of vectors with state nodes in physical node with minimum necessary information
						// The first NsplitClu elements will be centers
						vector<LocalStateNode> medoid(NPstateNodes);
						for(unsigned int i=0;i<NPstateNodes;i++){
							medoid[i].stateId = physNode.stateNodeIndices[i];
							medoid[i].stateNode = &stateNodes[physNode.stateNodeIndices[i]];
						}

						Medoids medoids;
						medoids.maxNstatesInMedoid = NphysNodes;
						medoids.sortedMedoids.emplace(make_pair(0.0,move(medoid)));

						// unordered_map<int,vector<LocalStateNode> > medoids;
						// medoids[0] = move(medoid);

						double attemptEntropyRate;

						if(fast){
							findCenters(medoids);
							attemptEntropyRate = calcEntropyRate(medoids);
						}
						else{
							findClusters(medoids);
							attemptEntropyRate = medoids.sumMinDiv;
						}

						// cout << medoids.sumMinDiv << " " << attemptEntropyRate << endl;
						// for(SortedMedoids::iterator it = medoids.sortedMedoids.begin(); it != medoids.sortedMedoids.end(); it++){
						// 	cout << it->first << " " << calcEntropyRate(it->second) << endl;
						// }


						// Update best solution
						#ifdef _OPENMP
						omp_set_lock(&(lock[physNodeNr]));
						#endif
						// runDetails[physNodeNr].push_back(make_pair(omp_get_thread_num(),medoids.sumMinDiv));
						if(attemptEntropyRate < bestEntropyRate[physNodeNr]){
							bestEntropyRate[physNodeNr] = attemptEntropyRate;
							bestMedoids[physNodeNr] = move(medoids);
						}
						attemptsLeftVec[physNodeNr]--;
						if(attemptsLeftVec[physNodeNr] == 0){

							// Perform the lumping and update stateNodes
							performLumping(bestMedoids[physNodeNr]);

							double postLumpingEntropyRate = bestEntropyRate[physNodeNr];

							string output = "\n-->Lumped " + to_string(NPstateNodes) + " states to " + to_string(bestMedoids[physNodeNr].sortedMedoids.size()) + " states with max " + to_string(bestMedoids[physNodeNr].maxNstatesInMedoid) + " lumped states and total divergence " + to_string(bestMedoids[physNodeNr].sumMinDiv) + " and " +  to_string(100.0*(postLumpingEntropyRate-preLumpingEntropyRate)/preLumpingEntropyRate) + "\% entropy increase after " + to_string(NtotAttempts) + " updates in physical node " + to_string(physNodeNr+1) + "/" + to_string(NphysNodes) + ".               ";
							// for(int i=0;i<runDetails[physNodeNr].size();i++)
							// 	output += " " + to_string(runDetails[physNodeNr][i].first) + " " + to_string(runDetails[physNodeNr][i].second) + "/";

							cout << output;

						}
						#ifdef _OPENMP
						omp_unset_lock(&(lock[physNodeNr]));
						#endif

			
						// Free cached divergences
						// cachedWJSdiv = unordered_map<pair<int,int>,double,pairhash>();

			
					}
					else{

						if(NPstateNodes == 0)
							NphysDanglings++;

						string output = "\n-->Did not touch " + to_string(NPstateNodes) + " states in physical node " + to_string(physNodeNr+1) + "/" + to_string(NphysNodes) + ".               ";
						cout << output;
					}
					
				} // end of #pragma omp task
			} // end of for loop
		} // end of #pragma omp single nowait
	} // end of #pragma omp parallel
	cout << endl << "-->Updating state node ids" << endl;

	// Update stateIds
	// First all active state nodes that other state nodes have lumped to
	Nlinks = 0; // Update number of links
	NstateNodes = 0; // Update number of links
	for(unordered_map<int,StateNode>::iterator it = stateNodes.begin(); it != stateNodes.end(); it++){
		StateNode &stateNode = it->second;
		if(stateNode.active){
			Nlinks += stateNode.links.size(); // Update number of links
			NstateNodes++;
			stateNodeIdMapping[stateNode.stateId] = updatedStateId;
			stateNode.updatedStateId = updatedStateId;
			updatedStateId++;
		}
	}
	// Then all inactive state nodes that have lumped to other state nodes
	for(unordered_map<int,StateNode>::iterator it = stateNodes.begin(); it != stateNodes.end(); it++){
		StateNode &stateNode = it->second;
		if(!stateNode.active){
			stateNodeIdMapping[stateNode.stateId] = stateNodeIdMapping[stateNode.updatedStateId];
		}
	}

	#ifdef _OPENMP
	for (int i=0; i<NphysNodes; i++)
    omp_destroy_lock(&(lock[i]));
  #endif
}

bool StateNetwork::readLines(string &line,vector<string> &lines){
	
	while(getline(ifs,line)){
		if(line[0] == '*'){
			return true;
		}
		else if(line[0] != '=' && line[0] != '#'){
			lines.push_back(line);
		}
	}

	return false; // Reached end of file
}

void StateNetwork::loadNodeMapping(){

	string buf;
	istringstream ss;
	bool isStateNode = false;
	cout << "Loading state network to generate state node to physical node mapping:" << endl;
	cout << "-->Reading states..." << flush;
	while(getline(ifs,line)){
		if(line[0] == '*'){
			ss.clear();
			ss.str(line);
			ss >> buf;
			if(buf == "*States")
				isStateNode = true;
			else
				isStateNode = false;
		}
		else if(isStateNode && line[0] != '=' && line[0] != '#'){
			ss.clear();
			ss.str(line);
			ss >> buf;
			int stateId = atoi(buf.c_str());
			ss >> buf;
			int physId = atoi(buf.c_str());
			stateToPhysNodeMapping[stateId] = physId;
		}
	}
	cout << "found " << stateToPhysNodeMapping.size() << " states." << endl; 


}

bool StateNetwork::loadStateNetworkBatch(){

	vector<string> stateLines;
	vector<string> linkLines;
	vector<string> contextLines;
	bool readStates = false;
	bool readLinks = false;
	bool readContexts = false;
	string buf;
	istringstream ss;

	// ************************* Read statenetwork batch ************************* //
	
	// Read until next data label. Return false if no more data labels
	if(keepReading){
		cout << "Reading statenetwork, batch " << Nbatches+1 << ":" << endl;
		if(line[0] != '*'){
			while(getline(ifs,line)){
				if(line[0] == '*')
					break;
				size_t foundTotSize = line.find("# Total weight: ");
				if(foundTotSize != string::npos)
					totWeight = atof(line.substr(foundTotSize+16).c_str());
			}
		}
	}
	else{
		cout << "-->No more statenetwork batches to read." << endl;
		return false;
	}


	while(!readStates || !readLinks || !readContexts){

		ss.clear();
		ss.str(line);
		ss >> buf;
		if(!readStates && buf == "*States"){
			cout << "-->Reading states..." << flush;
			readStates = true;
			keepReading = readLines(line,stateLines);
			NstateNodes = stateLines.size();
			cout << "found " << NstateNodes << " states." << endl;
		}
		else if(!readLinks && buf == "*Links"){
			cout << "-->Reading links..." << flush;
			readLinks = true;
			keepReading = readLines(line,linkLines);
			Nlinks = linkLines.size();
			cout << "found " << Nlinks << " links." << endl;
		}
		else if(!readContexts && buf == "*Contexts"){
			cout << "-->Reading contexts..." << flush;
			readContexts = true;
			keepReading = readLines(line,contextLines);
			Ncontexts = contextLines.size();
			cout << "found " << Ncontexts << " contexts." << endl;
		}
		else{
			cout << "Expected *States, *Links, or *Contexts, but found " << buf << " exiting..." << endl;
			exit(-1);
		}
	}

	// ************************* Process statenetwork batch ************************* //
	Nbatches++;
	cout << "Processing statenetwork, batch " << Nbatches << ":" << endl;

	//Process states
	cout << "-->Processing " << NstateNodes  << " state nodes..." << flush;
	for(int i=0;i<NstateNodes;i++){
		ss.clear();
		ss.str(stateLines[i]);
		ss >> buf;
		int stateId = atoi(buf.c_str());
		ss >> buf;
		int physId = atoi(buf.c_str());
	  ss >> buf;
	  double outWeight = atof(buf.c_str());
	  weight += outWeight;
		if(outWeight > epsilon)
			physNodes[physId].stateNodeIndices.push_back(stateId);
		else{
			physNodes[physId].stateNodeDanglingIndices.push_back(stateId);
			Ndanglings++;
		}
		stateNodes[stateId] = StateNode(stateId,physId,outWeight);
	}
	NphysNodes = physNodes.size();
	cout << "found " << Ndanglings << " dangling state nodes in " << NphysNodes << " physical nodes, done!" << endl;

	// Process links 
	cout << "-->Processing " << Nlinks  << " links..." << flush;
	for(int i=0;i<Nlinks;i++){
		ss.clear();
		ss.str(linkLines[i]);
		ss >> buf;
		int source = atoi(buf.c_str());
		ss >> buf;
		int target = atoi(buf.c_str());
		ss >> buf;
		double linkWeight = atof(buf.c_str());
		stateNodes[source].links[target] += linkWeight;
		stateNodes[source].physLinks[stateToPhysNodeMapping[target]] += linkWeight;
	}
 	cout << "done!" << endl;

	// Process contexts
	cout << "-->Processing " << Ncontexts  << " contexts..." << flush;
	for(int i=0;i<Ncontexts;i++){
		ss.clear();
		ss.str(contextLines[i]);
		ss >> buf;
		int stateNodeId = atoi(buf.c_str());
		string context = contextLines[i].substr(buf.length()+1);
		stateNodes[stateNodeId].contexts.push_back(context);
	}
	cout << "done!" << endl;

	// // Validate out-weights
 // 	for(unordered_map<int,StateNode>::iterator it = stateNodes.begin(); it != stateNodes.end(); it++){
	// StateNode &stateNode = it->second;
	// 	double w = 0.0;
	// 	for(map<int,double>::iterator it_link = stateNode.links.begin(); it_link != stateNode.links.end(); it_link++){
	// 		w += it_link->second;
	// 	}
	// 	if((w < (stateNode.outWeight-epsilon)) || (w > (stateNode.outWeight+epsilon))){
	// 		cout << setprecision(15) << "::::::::::: Warning: out-weight does not match link weights for state node " << stateNode.stateId << ": " << stateNode.outWeight << " vs " << w << " " << stateNode.links.size() << ", updating. :::::::::::" << endl;
	// 		stateNode.outWeight = w;
	// 	}
	// }

 	return true;

}


void StateNetwork::printStateNetworkBatch(){

	cout << "Writing temporary results:" << endl;

  my_ofstream ofs;
  if(batchOutput){
  	if(Nbatches == 1){ // Start with empty file for first batch
			ofs.open(tmpOutFileName.c_str());
		}
		else{ // Append to existing file
			ofs.open(tmpOutFileName.c_str(),ofstream::app);
		}
		ofs << "===== " << Nbatches << " =====\n";
		ofs << "*States\n";
		ofs << "#stateId ==> (physicalId, outWeight)\n";
  }
  else{
  	 if(Nbatches == 1){ // Start with empty file for first batch
			ofs.open(tmpOutFileNameStates.c_str());
		}
		else{ // Append to existing file
			ofs.open(tmpOutFileNameStates.c_str(),ofstream::app);
		}
  }

	cout << "-->Writing " << NstateNodes << " state nodes..." << flush;
	// To order state nodes by id
	map<int,int> orderedStateNodeIds;
	for(unordered_map<int,StateNode>::iterator it = stateNodes.begin(); it != stateNodes.end(); it++){
		StateNode &stateNode = it->second;
		if(stateNode.active)
			orderedStateNodeIds[stateNode.updatedStateId] = stateNode.stateId;
	}

	for(map<int,int>::iterator it = orderedStateNodeIds.begin(); it != orderedStateNodeIds.end(); it++){
		StateNode &stateNode = stateNodes[it->second];
		ofs << stateNode.stateId << " " << stateNode.physId << " " << stateNode.outWeight << "\n";
	}
	cout << "done!" << endl;

	if(batchOutput){
		ofs << "*Links\n";
		ofs << "#(source target) ==> weight\n";
	}
	else{
		ofs.close();
		if(Nbatches == 1){ // Start with empty file for first batch
			ofs.open(tmpOutFileNameLinks.c_str());
		}
		else{ // Append to existing file
			ofs.open(tmpOutFileNameLinks.c_str(),ofstream::app);
		}
	}
	cout << "-->Writing " << Nlinks << " links..." << flush;
	for(map<int,int>::iterator it = orderedStateNodeIds.begin(); it != orderedStateNodeIds.end(); it++){	
		StateNode &stateNode = stateNodes[it->second];
		for(map<int,double>::iterator it_link = stateNode.links.begin(); it_link != stateNode.links.end(); it_link++){
				ofs << stateNode.stateId << " " << it_link->first << " " << it_link->second << "\n";
		}
	}
	cout << "done!" << endl;

	if(batchOutput){
		ofs << "*Contexts \n";
		ofs << "#stateId <== (physicalId priorId [history...])\n";
	}
	else{
		ofs.close();
		if(Nbatches == 1){ // Start with empty file for first batch
			ofs.open(tmpOutFileNameContexts.c_str());
		}
		else{ // Append to existing file
			ofs.open(tmpOutFileNameContexts.c_str(),ofstream::app);
		}
	}
	cout << "-->Writing " << Ncontexts << " contexts..." << flush;
	for(map<int,int>::iterator it = orderedStateNodeIds.begin(); it != orderedStateNodeIds.end(); it++){	
		StateNode &stateNode = stateNodes[it->second];
		for(vector<string>::iterator it_context = stateNode.contexts.begin(); it_context != stateNode.contexts.end(); it_context++){
			ofs << stateNode.stateId << " " << (*it_context) << "\n";
		}
	}
	cout << "done!" << endl;

}

void StateNetwork::printStateNetwork(){

	entropyRate = calcEntropyRate();

  my_ofstream ofs;
  ofs.open(outFileName.c_str());

	cout << "No more batches, entropy rate is " << entropyRate << ", writing results to " << outFileName << ":" << endl;
	cout << "-->Writing header comments..." << flush;
  ofs << "# Number of physical nodes: " << NphysNodes << "\n";
  ofs << "# Number of state nodes: " << NstateNodes << "\n";
  ofs << "# Number of dangling physical (and state) nodes: " << NphysDanglings << "\n";
  ofs << "# Number of links: " << Nlinks << "\n";
  ofs << "# Number of contexts: " << Ncontexts << "\n";
  ofs << "# Total weight: " << weight << "\n";
  ofs << "# Entropy rate: " << entropyRate << "\n";
	cout << "done!" << endl;

	cout << "-->Writing " << NstateNodes << " state nodes..." << flush;
	// To order state nodes by id
	map<int,int> orderedStateNodeIds;
	for(unordered_map<int,StateNode>::iterator it = stateNodes.begin(); it != stateNodes.end(); it++){
		StateNode &stateNode = it->second;
		if(stateNode.active)
			orderedStateNodeIds[stateNode.updatedStateId] = stateNode.stateId;
	}

	ofs << "*States\n";
	ofs << "#stateId ==> (physicalId, outWeight)\n";
	for(map<int,int>::iterator it = orderedStateNodeIds.begin(); it != orderedStateNodeIds.end(); it++){
		StateNode &stateNode = stateNodes[it->second];
		ofs << stateNodeIdMapping[stateNode.stateId] << " " << stateNode.physId << " " << stateNode.outWeight << "\n";
	}
	cout << "done!" << endl;

	cout << "-->Writing " << Nlinks << " links..." << flush;
	ofs << "*Links\n";
	ofs << "#(source target) ==> weight\n";
	for(map<int,int>::iterator it = orderedStateNodeIds.begin(); it != orderedStateNodeIds.end(); it++){	
		StateNode &stateNode = stateNodes[it->second];
		int source = stateNodeIdMapping[stateNode.stateId];
		// Remove link redundance from lumped targets
		unordered_map<int,double> lumpedLinks;
		for(map<int,double>::iterator it_link = stateNode.links.begin(); it_link != stateNode.links.end(); it_link++)
			lumpedLinks[stateNodeIdMapping[it_link->first]] += it_link->second;
		for(unordered_map<int,double>::iterator it_link = lumpedLinks.begin(); it_link != lumpedLinks.end(); it_link++){
			ofs << source << " " << it_link->first << " " << it_link->second << "\n";
		}
	}
	cout << "done!" << endl;

	cout << "-->Writing " << Ncontexts << " contexts..." << flush;
	ofs << "*Contexts \n";
	ofs << "#stateId <== (physicalId priorId [history...])\n";
	for(map<int,int>::iterator it = orderedStateNodeIds.begin(); it != orderedStateNodeIds.end(); it++){	
		StateNode &stateNode = stateNodes[it->second];
		for(vector<string>::iterator it_context = stateNode.contexts.begin(); it_context != stateNode.contexts.end(); it_context++){
			ofs << stateNodeIdMapping[stateNode.stateId] << " " << (*it_context) << "\n";
		}
	}
	cout << "done!" << endl;

}

void StateNetwork::concludeBatch(){

	cout << "Concluding batch:" << endl;

	entropyRate += calcEntropyRate();
	accumWeight += weight;
	totNphysNodes += NphysNodes;
	totNstateNodes += NstateNodes;
	totNlinks += Nlinks;
	totNdanglings += Ndanglings;
	totNcontexts += Ncontexts;
	totNphysDanglings += NphysDanglings;
	weight = 0.0;
	NphysNodes = 0;
	NstateNodes = 0;
	Nlinks = 0;
	Ndanglings = 0;
	Ncontexts = 0;
	NphysDanglings = 0;

	cout << "-->Current estimate of the entropy rate: " << entropyRate*totWeight/accumWeight << endl;

	completeStateNodeIdMapping.insert(stateNodeIdMapping.begin(),stateNodeIdMapping.end());
	stateNodeIdMapping.clear();
	physNodes.clear();
	stateNodes.clear();
	// cachedWJSdiv.clear();

}

void StateNetwork::compileBatches(){


  my_ofstream ofs;
  ofs.open(outFileName);
  string buf;
	istringstream ss;
	bool writeStates = false;
	bool writeLinks = false;
	bool writeContexts = false;
	int batchNr = 1;

	cout << "Writing final results to " << outFileName << ":" << endl;
  
  cout << "-->Writing header comments..." << flush;
  ofs << "# Number of physical nodes: " << totNphysNodes << "\n";
  ofs << "# Number of state nodes: " << totNstateNodes << "\n";
  ofs << "# Number of dangling physical (and state) nodes: " << totNphysDanglings << "\n";
  ofs << "# Number of links: " << totNlinks << "\n";
  ofs << "# Number of contexts: " << totNcontexts << "\n";
  ofs << "# Total weight: " << totWeight << "\n";
  ofs << "# Entropy rate: " << entropyRate << "\n";
	cout << "done!" << endl;

	cout << "-->Relabeling and writing " << totNstateNodes << " state nodes, " << totNlinks << " links, and " << totNcontexts << " contexts:" << endl;

	if(batchOutput){

		ifstream ifs_tmp(tmpOutFileName.c_str());
	
		// Copy lines directly until data format
		while(getline(ifs_tmp,line)){
			if(line[0] == '*'){
				break;	
			}
			else if(line[0] == '='){
				ofs << "=== " << batchNr << "/" << Nbatches << " ===\n";
			}
			else{
				ofs << line << "\n";
			}
		}
		while(!ifs_tmp.eof()){
	
			if(!writeStates && !writeLinks && !writeContexts){
				cout << "-->Batch " << batchNr << "/" << Nbatches << endl;
			}
			ofs << line << "\n";
			ss.clear();
			ss.str(line);
			ss >> buf;
			if(buf == "*States"){
				cout << "-->Writing state nodes..." << flush;
				writeStates = true;
				WriteMode writeMode = STATENODES;
				writeLines(ifs_tmp,ofs,writeMode,line,batchNr);
			}
			else if(buf == "*Links"){
				cout << "-->Writing links..." << flush;
				writeLinks = true;
				WriteMode writeMode = LINKS;
				writeLines(ifs_tmp,ofs,writeMode,line,batchNr);
			}
			else if(buf == "*Contexts"){
				cout << "-->Writing contexts..." << flush;
				writeContexts = true;
				WriteMode writeMode = CONTEXTS;
				writeLines(ifs_tmp,ofs,writeMode,line,batchNr);
			}
			else{
				cout << "Failed on line: " << line << endl;
			}
			cout << "done!" << endl;
			if(writeStates && writeLinks && writeContexts){
				writeStates = false;
				writeLinks = false;
				writeContexts = false;
				batchNr++;
			}
		}
		remove( tmpOutFileName.c_str() );
	}
	else{

		cout << "-->Writing " << totNstateNodes << " state nodes..." << flush;
		ofs << "*States\n";
		ofs << "#stateId ==> (physicalId, outWeight)\n";
		ifstream ifs_tmp(tmpOutFileNameStates.c_str());
		WriteMode writeMode = STATENODES;
		writeLines(ifs_tmp,ofs,writeMode,line);	
		ifs_tmp.close();
		cout << "done!" << endl;


		cout << "-->Writing " << totNlinks << " links..." << flush;
		ofs << "*Links\n";
		ofs << "#(source target) ==> weight\n";
		ifs_tmp.open(tmpOutFileNameLinks.c_str());
		writeMode = LINKS;
		writeLines(ifs_tmp,ofs,writeMode,line);	
		ifs_tmp.close();
		cout << "done!" << endl;

		cout << "-->Writing " << totNcontexts << " contexts..." << flush;
		ofs << "*Contexts \n";
		ofs << "#stateId <== (physicalId priorId [history...])\n";		
		ifs_tmp.open(tmpOutFileNameContexts.c_str());		
		writeMode = CONTEXTS;
		writeLines(ifs_tmp,ofs,writeMode,line);	
		ifs_tmp.close();
		cout << "done!" << endl;

		remove( tmpOutFileNameStates.c_str() );
		remove( tmpOutFileNameLinks.c_str() );
		remove( tmpOutFileNameContexts.c_str() );
	}

}

void StateNetwork::writeLines(ifstream &ifs_tmp, ofstream &ofs, WriteMode &writeMode, string &line,int &batchNr){

	string buf;
	istringstream ss;

	unordered_map<pair<int,int>,double,pairhash> aggregatedLinks;

	while(getline(ifs_tmp,line)){
		if(line[0] != '*'){
			if(line[0] != '=' && line[0] != '#'){
				ss.clear();
				ss.str(line);
				ss >> buf;
				if(writeMode == STATENODES){
					int stateId = atoi(buf.c_str());
					ss >> buf;
					int physId = atoi(buf.c_str());
	 				ss >> buf;
	 				double outWeight = atof(buf.c_str());
					ofs << completeStateNodeIdMapping[stateId] << " " << physId << " " << outWeight << "\n";
				}
				else if(writeMode == LINKS){
					int source = atoi(buf.c_str());
					ss >> buf;
					int target = atoi(buf.c_str());
					ss >> buf;
					double linkWeight = atof(buf.c_str());
					ofs << completeStateNodeIdMapping[source] << " " << completeStateNodeIdMapping[target] << " " << linkWeight << "\n";
					aggregatedLinks[make_pair(source,target)] += linkWeight;				
				}
				else if(writeMode == CONTEXTS){
					int stateNodeId = atoi(buf.c_str());
					string context = line.substr(buf.length()+1);
					ofs << completeStateNodeIdMapping[stateNodeId] << " " << context << "\n";
				}
			}
			else if(line[0] == '='){
				ofs << "=== " << batchNr+1 << "/" << Nbatches << " ===\n";
			}
			else{
				ofs << line << "\n";
			}
		}
		else{
			break;
		}
	}

	if(writeMode == LINKS ){
		for(unordered_map<pair<int,int>,double,pairhash>::iterator link_it = aggregatedLinks.begin(); link_it != aggregatedLinks.end(); link_it++){
			ofs << link_it->first.first << " " << link_it->first.second << " " << link_it->second << "\n";
		}
	}
}

void StateNetwork::writeLines(ifstream &ifs_tmp, ofstream &ofs, WriteMode &writeMode, string &line){

	string buf;
	istringstream ss;

	while(getline(ifs_tmp,line)){
		ss.clear();
		ss.str(line);
		ss >> buf;
		if(writeMode == STATENODES){
			int stateId = atoi(buf.c_str());
			ss >> buf;
			int physId = atoi(buf.c_str());
	 		ss >> buf;
	 		double outWeight = atof(buf.c_str());
			ofs << completeStateNodeIdMapping[stateId] << " " << physId << " " << outWeight << "\n";
		}
		else if(writeMode == LINKS){
			int source = atoi(buf.c_str());
			ss >> buf;
			int target = atoi(buf.c_str());
			ss >> buf;
			double linkWeight = atof(buf.c_str());
			ofs << completeStateNodeIdMapping[source] << " " << completeStateNodeIdMapping[target] << " " << linkWeight << "\n";					
		}
		else if(writeMode == CONTEXTS){
			int stateNodeId = atoi(buf.c_str());
			string context = line.substr(buf.length()+1);
			ofs << completeStateNodeIdMapping[stateNodeId] << " " << context << "\n";
		}
	}
}
