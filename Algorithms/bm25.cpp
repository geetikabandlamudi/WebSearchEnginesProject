#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
//#include <bits/stdc++.h>
#include <limits>
#include <set>
#include <sstream>


using namespace std;


long long findWordOffset(const string& index_file, const string& word) {
    ifstream file(index_file, ios::binary);
    if (!file.is_open()) {
        cerr << "Unable to open index file" << endl;
        return -1;
    }

    while (!file.eof()) {
        unsigned int wordLength;
        file.read(reinterpret_cast<char*>(&wordLength), sizeof(wordLength));
        if (file.eof()) break;

        string currentWord(wordLength, '\0');
        file.read(&currentWord[0], wordLength);

        if (currentWord == word) {
            long long offset;
            file.read(reinterpret_cast<char*>(&offset), sizeof(offset));
            return offset;
        } else {
            // Skip the offset
            file.seekg(sizeof(long long), ios::cur);
        }
    }

    return -1; // Word not found
}

map<string, vector<tuple<unsigned int, unsigned int, unsigned int>>> readWordData(const string& main_file, long long offset) {
    map<string, vector<tuple<unsigned int, unsigned int, unsigned int>>> wordData;
    ifstream file(main_file, ios::binary);

    if (!file.is_open()) {
        cerr << "Unable to open main binary file" << endl;
        return wordData;
    }

    // Go to the word's location in the main file
    file.seekg(offset, ios::beg);

    unsigned int wordLength;
    file.read(reinterpret_cast<char*>(&wordLength), sizeof(wordLength));

    string word(wordLength, '\0');
    file.read(&word[0], wordLength);

    unsigned int dataBlockLength;
    file.read(reinterpret_cast<char*>(&dataBlockLength), sizeof(dataBlockLength));

    vector<tuple<unsigned int, unsigned int, unsigned int>> docInfo;

    for (unsigned int i = 0; i < dataBlockLength / 12; ++i) { // Each block is 12 bytes
        unsigned int pid, freq, passageLength;
        file.read(reinterpret_cast<char*>(&pid), sizeof(pid));
        file.read(reinterpret_cast<char*>(&freq), sizeof(freq));
        file.read(reinterpret_cast<char*>(&passageLength), sizeof(passageLength));

        docInfo.push_back(make_tuple(pid, freq, passageLength));
    }

    wordData[word] = docInfo;

    file.close();
    return wordData;
}

map<string,string> readLastTermMap()
{
    string finalMapFile = "finalMap.bin"; // Set the path to your final map file
    ifstream file(finalMapFile, ios::binary);
    
    map<string, string> lastTermMap;

    if (!file.is_open()) {
        cerr << "Unable to open file: " << finalMapFile << endl;
    }

    while (!file.eof()) {
        unsigned int wordLength, filenameLength;

        // Read the length of the last word
        file.read(reinterpret_cast<char*>(&wordLength), sizeof(wordLength));
        if (file.eof()) break;

        // Read the last word
        string lastWord(wordLength, '\0');
        file.read(&lastWord[0], wordLength);

        // Read the length of the file name
        file.read(reinterpret_cast<char*>(&filenameLength), sizeof(filenameLength));
        if (file.eof()) break;

        // Read the file name
        string filename(filenameLength, '\0');
        file.read(&filename[0], filenameLength);

        // Add to map
        lastTermMap[lastWord] = filename;
    }

    file.close();

    return lastTermMap;
}

// Helper function to get a set of docIDs for a given word
set<unsigned int> getDocIDs(const map<string, vector<tuple<unsigned int, unsigned int, unsigned int>>>& wordMap, const string& word) {
    set<unsigned int> docIDs;
    auto it = wordMap.find(word);
    if (it != wordMap.end()) {
        for (const auto& docInfo : it->second) {
            docIDs.insert(get<0>(docInfo)); // Insert docID
        }
    }
    return docIDs;
}

// Function to find common docIDs among all words in the vector of maps
set<unsigned int> findCommonDocIDs(const vector<map<string, vector<tuple<unsigned int, unsigned int, unsigned int>>>>& words) {
    set<unsigned int> commonDocIDs;

    for (size_t i = 0; i < words.size(); ++i) {
        for (const auto& wordPair : words[i]) {
            set<unsigned int> currentDocIDs = getDocIDs(words[i], wordPair.first);

            if (i == 0) {
                // For the first word, just assign its docIDs to commonDocIDs
                commonDocIDs = currentDocIDs;
            } else {
                // For subsequent words, find the intersection
                set<unsigned int> intersection;
                set_intersection(commonDocIDs.begin(), commonDocIDs.end(),
                                 currentDocIDs.begin(), currentDocIDs.end(),
                                 inserter(intersection, intersection.begin()));
                commonDocIDs = intersection;
            }

            // If at any point the commonDocIDs become empty, break early
            if (commonDocIDs.empty()) {
                break;
            }
        }

        // Break the outer loop if no common docIDs are found
        if (commonDocIDs.empty()) {
            break;
        }
    }

    return commonDocIDs;
}


double IDF(int N, int n) {
    return log((N - n + 0.5) / (n + 0.5));
}

bool comparePairs(const pair<string, double>& a, const pair<string, double>& b) {
    return a.second > b.second; // Sort in descending order
}

void bm25(const vector<map<string, vector<tuple<unsigned int, unsigned int, unsigned int>>>>& words, const set<unsigned int>& commonIDs, unsigned int N) {
    double k1 = 1.5;  // Tuning parameter
    double b = 0.75;
    double avgdl = 52; // Average document length
    vector<pair<string, double>> scores;

    for (unsigned int docID : commonIDs) {
        double score = 0;
        int docLength = 0;
        bool docLengthSet = false;

        for (const auto& wordMap : words) {
            for (const auto& wordData : wordMap) {
                const auto& tuples = wordData.second;

                for (const auto& tuple : tuples) {
                    if (get<0>(tuple) == docID) {
                        int termFrequency = get<1>(tuple);
                        if (!docLengthSet) {
                            docLength = get<2>(tuple);
                            docLengthSet = true;
                        }
                        double idf = IDF(N, tuples.size());
                        score += idf * (termFrequency * (k1 + 1)) / (termFrequency + (k1 * (1 - b + (b * (docLength / avgdl)))));
                    }
                }
            }
        }

        scores.push_back(make_pair(to_string(docID), score)); // Using docID as the key in the score map
    }

    sort(scores.begin(), scores.end(), comparePairs);

    // Print the sorted scores
    for (size_t i = 0; i < scores.size() && i < 25; ++i) {
        string docID = scores[i].first;
        double score = scores[i].second;
        cout <<docID<<endl;
    }
}


int main(int argc, char* argv[])
{

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <query_string> <num_docIDs> <docID1> <docID2> ..." << endl;
        return 1;
    }

    string query = argv[1];

    //cout<<"Loading Search Engine...."<<endl;
    map<string, string> lastTermMap;

    lastTermMap = readLastTermMap();

    unsigned int numberOfDocs = 8841823;

    //cout<<query<<endl;

    istringstream iss(query);

    string map_path;
    int i=0;
        
    string word;

    vector<map<string, vector<tuple<unsigned int, unsigned int, unsigned int>>>> words;

    while (iss >> word) {
        i=0;
        for (const auto& entry : lastTermMap) {
            i++;
            if(word<=entry.first)
            {
                map_path = entry.second;
                break;
            }
        }
        
        std::replace(map_path.begin(), map_path.end(), '\\', '/');
        if (!map_path.empty() && map_path[0] == '.') {
            map_path.erase(map_path.begin());
            }
//        map_path.erase(std::remove(map_path.begin(), map_path.end(), '.'), map_path.end());
        if (!map_path.empty() && map_path[0] == '/') {
            map_path.erase(map_path.begin());
            }
//        return 0;
        string index_file = "binIndices/binIndex"+to_string(i)+".bin";
        long long offset = findWordOffset(map_path, word);    
        if (offset != -1) {
            words.push_back(readWordData(index_file, offset));
        }
//        else {
//            cout << "Word not found in index map." << endl;
//        }
    }

    set<unsigned int> commonIDs;
    if(words.size()>1)
    {
        commonIDs = findCommonDocIDs(words);
    }
    else
    {
        commonIDs = getDocIDs(words[0], query);
    }
    

    bm25(words,commonIDs,numberOfDocs);

    
}
