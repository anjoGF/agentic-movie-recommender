from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class ItemStats:
    popularity_norm: Dict[int, float]
    avg_rating_norm: Dict[int, float]
    rating_count: Dict[int, int]

    @staticmethod
    def build(ratings: pd.DataFrame) -> "ItemStats":
        # counts
        counts = ratings.groupby("movieId")["rating"].count().astype(int)
        max_c = int(counts.max()) if len(counts) else 1
        popularity_norm = {int(mid): float(c / max_c) for mid, c in counts.items()}

        # avg rating
        avgs = ratings.groupby("movieId")["rating"].mean()
        # Normalize ratings from [0.5..5] -> [0..1] (MovieLens uses 0.5 increments)
        avg_rating_norm = {int(mid): float((r - 0.5) / (5.0 - 0.5)) for mid, r in avgs.items()}

        rating_count = {int(mid): int(c) for mid, c in counts.items()}
        return ItemStats(popularity_norm=popularity_norm, avg_rating_norm=avg_rating_norm, rating_count=rating_count)

    def get_popularity(self, movie_id: int) -> float:
        return float(self.popularity_norm.get(int(movie_id), 0.0))

    def get_avg_rating(self, movie_id: int) -> float:
        return float(self.avg_rating_norm.get(int(movie_id), 0.5))

    def as_debug_dict(self, movie_id: int) -> Dict[str, Any]:
        mid = int(movie_id)
        return {
            "movieId": mid,
            "popularity_norm": self.get_popularity(mid),
            "avg_rating_norm": self.get_avg_rating(mid),
            "rating_count": int(self.rating_count.get(mid, 0)),
        }
