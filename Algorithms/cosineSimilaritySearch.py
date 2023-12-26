from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import time
from docContentManager import DocContentManager


class CosineSimilaritySearch():

    corpus_embeddings = []
    sentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = {}
    docContentManager = DocContentManager()
    
    def load_corpus_embeddings(self):
        with open('./embeddings/corpus_embeddings_1', 'rb') as f:
            self.corpus_embeddings = pickle.load(f)
        
    def search(self, query, top_k=100):
        query_embedding = self.sentenceTransformer.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        indices = [idx for idx in top_results[1]]
        scores = top_results[0]

        return {'docIds': indices, 'similarityScore': scores}
    
    def printDocumentContent(self, results):
        
        for i in range(len(results['docIds'])):
            content = self.docContentManager.fetchPassageContent(results['docIds'][i])
            print("PassageID: ", results['docIds'][i])
            print("Similarity Score: ", results['similarityScore'][i])
            print(content)
            print("\n\n")
        

def main():
    query = "what is dopamine"
    cosineSimilaritySearch = CosineSimilaritySearch()
    cosineSimilaritySearch.load_corpus_embeddings()
    start_time = time.time()
    results = cosineSimilaritySearch.search(query, 10)
    end_time = time.time()
    print("Query:: ", query)
    print("Took ", end_time-start_time, "s to fetch the results.")
    cosineSimilaritySearch.printDocumentContent(results)

main()