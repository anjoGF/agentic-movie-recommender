import numpy as np
import pandas as pd
from typing import Dict, Optional


class SimpleCFRecommender:
    """
    Deterministic item-item collaborative filtering recommender.

    Design goals:
    - Stable across runs
    - Fully explainable
    - No stochastic components
    - Safe fallbacks for unknown users
    """

    def __init__(self):
        self.users = None
        self.items = None
        self.user_idx: Dict[int, int] = {}
        self.item_idx: Dict[int, int] = {}
        self.idx_item: Dict[int, int] = {}
        self.sim: Optional[np.ndarray] = None
        self.mat: Optional[np.ndarray] = None

    def fit(self, ratings: pd.DataFrame) -> None:
        """
        Build an implicit user–item interaction matrix and
        compute item–item cosine similarity.
        """
        self.users = ratings["userId"].unique()
        self.items = ratings["movieId"].unique()

        self.user_idx = {int(u): i for i, u in enumerate(self.users)}
        self.item_idx = {int(m): i for i, m in enumerate(self.items)}
        self.idx_item = {i: m for m, i in self.item_idx.items()}

        interaction_matrix = np.zeros(
            (len(self.users), len(self.items)),
            dtype=np.float32
        )

        for row in ratings.itertuples(index=False):
            u = int(row.userId)
            m = int(row.movieId)
            interaction_matrix[self.user_idx[u], self.item_idx[m]] = 1.0

        item_matrix = interaction_matrix.T
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True) + 1e-8
        item_matrix = item_matrix / norms

        self.sim = item_matrix @ item_matrix.T
        self.mat = interaction_matrix

    def recommend(self, user_id: int, k: int = 50) -> pd.DataFrame:
        """
        Recommend items for a user based on item–item similarity.

        Always returns a valid DataFrame.
        """
        if self.mat is None or self.sim is None:
            raise RuntimeError("CF model not fitted. Call fit() first.")

        if user_id not in self.user_idx:
            return pd.DataFrame(columns=["movieId", "cf_score"])

        user_vector = self.mat[self.user_idx[user_id]]
        scores = self.sim @ user_vector

        top_indices = np.argsort(-scores)[:k]

        return pd.DataFrame({
            "movieId": [int(self.idx_item[i]) for i in top_indices],
            "cf_score": scores[top_indices].astype(float)
        })
