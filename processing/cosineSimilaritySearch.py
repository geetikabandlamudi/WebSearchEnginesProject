from sentence_transformers import SentenceTransformer, util
import torch
import pickle
from docContentManager import DocContentManager


class CosineSimilaritySearch():

    corpus_embeddings = []
    sentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = {}
    docContentManager = DocContentManager()
    
    def load_corpus_embeddings(self):
        with open('./embeddings/combined_corpus_embeddings_1', 'rb') as f:
            self.corpus_embeddings = pickle.load(f)
        
    def search(self, query, top_k=100):
        query_embedding = self.sentenceTransformer.encode(query, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        indices = [idx for idx in top_results[1]]
        scores = top_results[0]

        return {'docIds': indices, 'similarityScore': scores}
    
    def printDocumentContent(self, results):
        for docId in results['docIds']:
            content = self.docContentManager.fetchPassageContent(docId)
            print("DOCID: ", docId)
            print(content)
            print("\n\n")
        

def main():
    query = "cats and dogs"
    cosineSimilaritySearch = CosineSimilaritySearch()
    cosineSimilaritySearch.load_corpus_embeddings()
    results = cosineSimilaritySearch.search(query)
    print(results)
    cosineSimilaritySearch.printDocumentContent(results)

main()