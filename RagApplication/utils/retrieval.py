import numpy as np

def dense_search(query, faiss_index, embed_model, doc_map, top_k):
    q_emb = embed_model.encode([query])
    D, I = faiss_index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx in doc_map:
            results.append({"text": doc_map[idx], "score": float(1 / (score + 1e-6))})
    return results

def bm25_search(query, es, index_name, top_k):
    res = es.search(index=index_name, query={"match": {"content": query}}, size=top_k)
    return [{"text": hit["_source"]["content"], "score": hit["_score"]} for hit in res["hits"]["hits"]]

def combine_scores(dense_res, sparse_res, w_dense=0.5, w_sparse=0.5):
    combined = {}
    for r in dense_res:
        combined[r["text"]] = combined.get(r["text"], 0) + r["score"] * w_dense
    for r in sparse_res:
        combined[r["text"]] = combined.get(r["text"], 0) + r["score"] * w_sparse
    return [{"text": t, "score": s} for t, s in sorted(combined.items(), key=lambda x: x[1], reverse=True)]
