import os, yaml, faiss, uuid
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import redis
import numpy as np
from utils.ingestion import extract_text, chunk_text
from utils.retrieval import bm25_search, dense_search, combine_scores
from utils.generation import call_llm

class RAGOrchestrator:
    def __init__(self, config_path="config.yaml"):
        self.config = yaml.safe_load(open(config_path))
        
        # Embedding model
        self.embed_model = SentenceTransformer(self.config["embedding_model"])
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        
        # FAISS
        if os.path.exists(self.config["dense_store"]["index_file"]):
            self.faiss_index = faiss.read_index(self.config["dense_store"]["index_file"])
            self.doc_map = np.load(self.config["dense_store"]["index_file"] + ".meta.npy", allow_pickle=True).item()
        else:
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            self.doc_map = {}
        
        # Elasticsearch
        self.es = Elasticsearch(self.config["sparse_store"]["host"])
        self.es_index = self.config["sparse_store"]["index"]
        if not self.es.indices.exists(index=self.es_index):
            self.es.indices.create(index=self.es_index)
        
        # Redis cache
        self.rdb = redis.Redis(
            host=self.config["cache_store"]["host"],
            port=self.config["cache_store"]["port"],
            db=self.config["cache_store"]["db"]
        )

    def ingest(self, file_path):
        text = extract_text(file_path)
        chunks = chunk_text(text)
        
        # Index in ES
        for chunk in chunks:
            self.es.index(index=self.es_index, document={"content": chunk})
        
        # Index in FAISS
        start_id = len(self.doc_map)
        embeddings = self.embed_model.encode(chunks)
        for i, chunk in enumerate(chunks):
            self.doc_map[start_id + i] = chunk
        self.faiss_index.add(embeddings)
        
        # Persist FAISS
        faiss.write_index(self.faiss_index, self.config["dense_store"]["index_file"])
        np.save(self.config["dense_store"]["index_file"] + ".meta.npy", self.doc_map)

        return {"chunks_indexed": len(chunks)}

    def retrieve(self, query):
        dense_res = dense_search(query, self.faiss_index, self.embed_model, self.doc_map, self.config["retrieval"]["top_k_dense"])
        sparse_res = bm25_search(query, self.es, self.es_index, self.config["retrieval"]["top_k_sparse"])
        
        hybrid_res = combine_scores(
            dense_res, sparse_res,
            w_dense=self.config["retrieval"]["hybrid_weights"]["dense"],
            w_sparse=self.config["retrieval"]["hybrid_weights"]["sparse"]
        )
        return hybrid_res

    def answer(self, query):
        if (cached := self.rdb.get(query)):
            return {"answer": cached.decode()}
        
        results = self.retrieve(query)
        context = "\n".join([r["text"] for r in results])
        prompt = f"Context:\n{context}\n\nQ: {query}\nA:"
        answer = call_llm(prompt, self.config["llm"])
        
        self.rdb.set(query, answer)
        return {"answer": answer}
