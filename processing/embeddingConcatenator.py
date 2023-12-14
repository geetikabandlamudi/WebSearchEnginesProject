import pickle
import torch
combined_corpus_embeddings = None
for k in range(1, 128):
    file_name = f'embeddings/corpus_embeddings{k}'
    with open(file_name, 'rb') as file:
        corpus_embeddings = pickle.load(file)
        corpus_embeddings = torch.tensor(corpus_embeddings)
        if combined_corpus_embeddings is None:
            combined_corpus_embeddings = corpus_embeddings
        else:
            combined_corpus_embeddings = torch.cat((combined_corpus_embeddings, corpus_embeddings), dim=0)

output_file = 'corpus_embeddings'
with open(output_file, 'wb') as file:
    pickle.dump(combined_corpus_embeddings, file)

print(f"Combined corpus embeddings saved to {output_file}")
