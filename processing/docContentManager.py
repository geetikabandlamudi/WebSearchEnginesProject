import pickle

class DocContentManager:
    corpus = {}
    def __init__(self):
        with open('./embeddings/corpus', 'rb') as f:
            self.corpus = pickle.load(f)

    def fetchPassageContent(self, passageID):
        return self.corpus[passageID]
