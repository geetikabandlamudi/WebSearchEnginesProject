def diversity_at_k(top_k_documents):
    num_documents = len(top_k_documents)
    diversity_scores = []

    for i in range(num_documents):
        for j in range(i + 1, num_documents):
            embedding_i = corpus_embeddings[top_k_documents[i]]
            embedding_j = corpus_embeddings[top_k_documents[j]]
            similarity = cosine_similarity(embedding_i.reshape(1, -1), embedding_j.reshape(1, -1))[0, 0]
            diversity_scores.append(1 - similarity)

    diversity_at_k = sum(diversity_scores) / len(diversity_scores)
    return diversity_at_k



def query_passage_similarity(query, pid):
    embedding_i = get_embedding(query)
    embedding_j = corpus_embeddings[pid]
    similarity = cosine_similarity(embedding_i.reshape(1, -1), embedding_j.reshape(1, -1))[0, 0]
    return similarity


diversity_sum = 0
diversity_data = []
for index, query in queries.iterrows():
    search_result = semantic_search(query['query'])
    diversity_score = diversity_at_k(search_result)
    query_similarity = query_passage_similarity(query['query'], search_result[0])
    diversity_data.append({"qid": query['qid'], "query": query['query'],
                            "diversity_score": diversity_score, "query_similarity": query_similarity})
    diversity_sum+=diversity_score
    print(query, diversity_score, query_similarity)
