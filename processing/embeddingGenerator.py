from sentence_transformers import SentenceTransformer, util
import torch
import sys
import pickle

embedder = SentenceTransformer('all-MiniLM-L6-v2')
k=1


for j in range(1,643,4):
    corpus = []
    for i in range(j,j+4):
        with open('./bert_docs/intermediate_postings'+str(i)+'.txt', 'r', encoding='utf-8') as file:
            content = file.read()

        sections = content.split('\n\n\n')



        # Process each section
        for section in sections:
            lines = section.split('\n')

            # Check if there are at least two lines (document ID and content)
            if len(lines) >= 2:
                doc_content = ' '.join(lines[1:]).strip()  
                #document length

                corpus.append(doc_content)
        print('File '+str(i)+' read')
            
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)



    with open('./embeddings/corpus_embeddings'+str(k), 'wb') as file:
        pickle.dump(corpus_embeddings, file)

    del corpus_embeddings
    del corpus
    k=k+1

    print(f"Array saved")
