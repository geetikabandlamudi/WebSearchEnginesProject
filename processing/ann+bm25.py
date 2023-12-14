import tensorflow as tf
import pickle
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle
from collections import namedtuple
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
import subprocess
import json


sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')


with open('corpus_embeddings1', 'rb') as f:
    corpus_embeddings = pickle.load(f)



for i in range(2,30):
    with open('corpus_embeddings'+str(i), 'rb') as f:
        corpus_embeddings2 = pickle.load(f)
    corpus_embeddings = torch.cat((corpus_embeddings, corpus_embeddings2), dim=0)

print(corpus_embeddings.size())

nn_model = NearestNeighbors(n_neighbors=1000, metric='euclidean')
nn_model.fit(corpus_embeddings)

user_query = "how long did abraham lincoln serve"

user_query_embedding = sentence_transformer_model.encode([user_query])[0]
distances, indices = nn_model.kneighbors([user_query_embedding])
#most_similar_documents = [corpus_texts[i] for i in indices[0]]
indice = indices[0]
print(indice)

'''

def search(inp_question):
    indices = []
    start_time = time.time()
    question_embedding = sentence_transformer_model.encode(inp_question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings)
    end_time = time.time()
    hits = hits[0]

    #print("Input question:", inp_question)
    #print("Results (after {:.3f} seconds):".format(end_time-start_time))
    for hit in hits[0:100]:
        #print("\t{:.3f}\t{}".format(hit['score'], corpus_texts[hit['corpus_id']]))
        indices.append(hit['corpus_id'])
    return indices

user_query = "cat"
indice = search(user_query)

'''


cpp_executable_path = '../bm25_5.exe'
input_parameters=[]
for i in indice:
    input_parameters.append(str(i))

#print(input_parameters)
command = [cpp_executable_path, user_query, str(len(indice))]+input_parameters
process = subprocess.run(command, capture_output=True, text=True)

if process.returncode == 0:
    output_lines = process.stdout.strip().split('\n')

    # Process each line of output
    for line in output_lines:
        print(line)
    print(process.stderr)
else:
    print("Error:", process.stderr)
