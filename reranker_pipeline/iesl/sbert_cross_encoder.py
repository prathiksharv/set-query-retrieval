import numpy as np
from scipy.special import expit  # Sigmoid function

def rerank_documents(queries, retrieved_docs, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    from sentence_transformers import CrossEncoder
    
    cross_encoder = CrossEncoder(model_name)
    query_doc_pairs = [(query, doc) for query, docs in zip(queries, retrieved_docs) for doc in docs]

    # Compute raw relevance scores (logits)
    raw_scores = cross_encoder.predict(query_doc_pairs)
    
    # Apply sigmoid to normalize scores between 0 and 1
    scores = expit(raw_scores)  # Sigmoid function
    
    # Rerank documents
    reranked_results = []
    idx = 0
    for query, docs in zip(queries, retrieved_docs):
        num_docs = len(docs)
        sorted_docs = sorted(zip(scores[idx:idx+num_docs], docs), key=lambda x: x[0], reverse=True)
        reranked_results.append((query, sorted_docs))
        idx += num_docs
    
    return reranked_results

# Main Execution
queries = ["Wizards and Magic Anime"]
retrieved_docs = [
    ["Fairy Tail: A classic with a large ensemble cast of wizards with diverse magical abilities, known for its action-packed battles and camaraderie.", 
    "Black Clover: Focuses on a young boy with no magical power who strives to become the strongest wizard in his world.", 
    "Attack on Titan: This is a story about the horrors of war, the cycle of hatred, and the need to avoid bigotry. It also explores the consequences of intergenerational trauma and the importance of not giving up."]
]

reranked_docs = rerank_documents(queries, retrieved_docs)

# Print Results with Normalized Scores
for query, docs_with_scores in reranked_docs:
    print(f"🔍 Query: {query}")
    for rank, (score, doc) in enumerate(docs_with_scores, 1):
        print(f"{rank}. {query} - {doc} - Score: {score:.4f}")
