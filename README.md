# WebSearchEnginesProject
Project work created as a part of Web Search Engines course at New York University.

## Problem Statement

Problem Statement

Sparse lexical retrieval methods such as BM25 miss a lot of good results due to vocabulary mismatch and related issues. One approach to overcome this issue uses query expansion. More recently, it has been proposed to instead map queries and documents into a high-dimensional space using say a BERT transformer, and to then retrieve the documents that are closest to the query in this high-dimensional space, Finding these nearest neighbors could then be done via optimized Approximate Nearest Neighbor (ANN) methods such as HNSW, DiskANN, or NSG. This project has several different parts that are needed: (1) Transforming a collection such as MSMarco and the queries into the high-dimensional space using (say) BERT, (2) a fast ANN algorithm such as HNSW, (3) also combining such a dense retrieval with traditional methods such as BM25 into a hybrid method. 


The repository structure is as follows:

    - EmbeddingGeneration
        - data
            Contains the collection dataset. This is obtained from 
            https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020 

        - embeddings
            Empty initially. Will be filled up by generateEmbeddings and embeddingConcatenator. 
            It will host the individual embeddings for multiple chunks of passages. 
            It will host the combined_corpus_embeddings file
    
        - generateEmbeddings.ipynb
            Dependencies: pip install SentenceTransformer

            Generates the embeddings in chunks
        
        - embedding Concatenator.py
            Dependencies: already taken care of by previous notebook

            This file combines the corpus_embeddings files from previoud file into a single file.
            If you want to skip these embedding generation phase, download the embeddings from our drive repo:
            combined_corpus_embeddings: https://drive.google.com/file/d/1wD-RjVjj7PeiENHKEeLtKMJWBI7_YtDe/view?usp=sharing
            corpus: https://drive.google.com/file/d/1ux72i05nou1BmKi2LfPebXshfm3bxIIS/view?usp=sharing


    - Algorithms
        
        Dependencies: 
            1. Make sure to unzip binIndices.zip in the same directory. 
                Download this from https://drive.google.com/file/d/1swYu8i8ohV_zGEIMZ1sdMMZd_ukZSiAy/view?usp=sharing 
            2. pip install nltk and related dependencies.
            3. pip install hnswlib
            4. Run g++ bm25.cpp -o bm25 -std=c++14 and generate the bm25 executable
        
        - maps, finalMap.bin, binIndices.zip
            Contain the inverted indices, and the dependent files for bm25
        
        - bm25.cpp
            Hosts the code that exposes the bm25 functionality as an executable. 
        
        - cosineSimilaritySearch.py
            Algorithm for traditional cosine similarity search using all-MiniLM-L6-v2 
            transformed passage embeddings
            Provide the query in the main function and run the file using 
            python3 cosineSimilaritySearch.py

        - bm25+algo.ipynb
            Algorithm that uses BM25 followed by Semantic Search Algorithm
            Algorithm that uses BM25 followed by Nearest Neighbors Algorithm
            Query expansion logic using KeyBERT
        
        - bm25+algo_vocabularyfeedback.ipynb
            Algorithm that uses KeyBERT for Query Expansion and Keyword Generation
            Performs Semantic Search and Nearest Neighbour search over expanded query
        
        - bm25+hnsw.ipynb
            Ladder algorithm that uses BM25 to find the most relevant documents 
            followed by HNSW algorithm to refine the results

        - hnswConstruction.ipynb
            Contains the logic to generate the HNSW index file for all the embeddings 
            generated previously. 
            Make sure to perform Embedding Generation before this step.
            If you want to skip this step, use this link to download the hnsw index bin file:
            https://drive.google.com/file/d/1zqZfbKm-tqDH7zdoHIuXX8oQiJfbFd4E/view?usp=sharing

        - hnsw+bm25+reverse-ladder.ipynb
            Inverse Ladder method that uses HNSW to get the documents first for a 
            query and performs a keyword search on the returned documents to formulate 
            a new query. This is passed to BM25 to get the most relevant documents
        
        - hnsw.ipynb
            Houses the code to test just the hnsw results for a given query
        
        - docContentManager.py
            Reads the corpus from EmbeddingGeneration/embeddings/corpus
            For a given pid, it returns the passage content
        
        - diversityScore.py
            Hosts the importable functions to calculate the diversity between 
            passages returned by the various algorithms and the similarity score 
            between a passage and the query.
        


 ## ABSTRACT        
This project addresses the limitations of sparse lexical retrieval methods, such as BM25, by leveraging dense retrieval techniques based on embeddings generated from BERT (Bidirectional Encoder Representations from Transformers). The focus is on finding the nearest passages in a given set efficiently. The proposed system incorporates a multi-faceted approach, involving the transformation of the MS MARCO passage dataset into a high-dimensional space using BERT, fast approximate nearest neighbor algorithms (HNSW), proper evaluation methodologies, and a potential hybridization with traditional methods like BM25. The system integrates a ladder model that combines BM25 for keyword-based relevance and semantic similarity through BERT embeddings. This conjunctive approach aims to enhance precision by selecting passages enhanced by both methods. We have added examples to share our view and take over the performance and results of these methods. 



