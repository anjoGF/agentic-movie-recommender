import os, json
import numpy as np
import hnswlib
from openai import OpenAI
from config import POCConfig

class SemanticSearchTool:
    def __init__(self, cfg: POCConfig):
        self.cfg = cfg
        self.client = OpenAI()
        self.index = None

    def _embed(self, texts):
        r = self.client.embeddings.create(
            model=self.cfg.embedding_model_id,
            input=texts
        )
        X = np.array([d.embedding for d in r.data], dtype=np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X

    def build_or_load(self, movies):
        os.makedirs("embeddings", exist_ok=True)

        if os.path.exists(self.cfg.embedding_path):
            print("Loading cached embeddings")
            X = np.load(self.cfg.embedding_path)
            self.movie_ids = json.load(open(self.cfg.embedding_ids_path))
        else:
            print("Building embeddings (one-time)")
            texts = (movies.title + " | " + movies.genres).fillna("").tolist()
            X = np.vstack([self._embed(texts[i:i+100]) for i in range(0, len(texts), 100)])
            np.save(self.cfg.embedding_path, X)
            self.movie_ids = movies.movieId.astype(int).tolist()
            json.dump(self.movie_ids, open(self.cfg.embedding_ids_path, "w"))

        self.index = hnswlib.Index(space="cosine", dim=X.shape[1])
        self.index.init_index(len(X), ef_construction=200, M=16)
        self.index.add_items(X, np.arange(len(X)))
        self.index.set_ef(64)

    def search(self, query, k):
        qv = self._embed([query])
        labels, dist = self.index.knn_query(qv, k=k)
        return [
            {"movieId": self.movie_ids[i], "semantic_score": 1.0 - d}
            for i, d in zip(labels[0], dist[0])
        ]
