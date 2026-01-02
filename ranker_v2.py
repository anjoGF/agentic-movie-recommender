from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from tools.item_stats import ItemStats


class RankerV2:
    """
    Ranker v2: Advantage-Weighted Collaborative Retrieval (POC approximation)

    Steps:
    1) Build candidate pool from CF + semantic (already in state)
    2) Normalize signals
    3) Compute utility = w_cf*cf + w_sem*semantic
    4) baseline = popularity_norm (proxy for "expected utility")
    5) advantage = utility - advantage_alpha * baseline
    6) novelty_boost = novelty_lambda * (1 - popularity_norm)
    7) final = advantage + novelty_boost
    """

    def __init__(self, cfg, movie_map: Dict[int, Dict[str, str]], item_stats: ItemStats):
        self.cfg = cfg
        self.movie_map = movie_map
        self.item_stats = item_stats

    def __call__(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        cf = state.get("cf", pd.DataFrame())
        sem = pd.DataFrame(state.get("sem", []))

        if cf is None or isinstance(cf, list):
            cf = pd.DataFrame(cf)
        if sem is None:
            sem = pd.DataFrame()

        if cf.empty and sem.empty:
            return []

        if not cf.empty and not sem.empty:
            df = cf.merge(sem, on="movieId", how="outer").fillna(0.0)
        elif not cf.empty:
            df = cf.assign(semantic_score=0.0)
        else:
            df = sem.assign(cf_score=0.0)

        # normalize signals
        for col in ["cf_score", "semantic_score"]:
            mx = float(df[col].max()) if len(df) else 0.0
            if mx > 0:
                df[col] = df[col] / mx

        plan = state.get("plan", {}) or {}
        w_cf = float(plan.get("weight_cf", 0.4))
        w_sem = float(plan.get("weight_semantic", 0.6))

        # utility = hybrid relevance / preference signal
        df["utility"] = w_cf * df["cf_score"] + w_sem * df["semantic_score"]

        # baseline = popularity_norm proxy (expected utility)
        df["baseline"] = df["movieId"].apply(lambda mid: self.item_stats.get_popularity(int(mid)))

        # advantage + novelty
        alpha = float(self.cfg.advantage_alpha)
        novelty_lambda = float(self.cfg.novelty_lambda)

        df["advantage"] = df["utility"] - alpha * df["baseline"]
        df["novelty_boost"] = novelty_lambda * (1.0 - df["baseline"])
        df["score"] = df["advantage"] + df["novelty_boost"]

        df = df.sort_values("score", ascending=False).head(self.cfg.final_k)

        recs: List[Dict[str, Any]] = []
        for r in df.itertuples(index=False):
            mid = int(r.movieId)
            meta = self.movie_map.get(mid, {"title": "Unknown", "genres": ""})
            recs.append(
                {
                    "movieId": mid,
                    "title": meta.get("title", "Unknown"),
                    "genres": meta.get("genres", ""),
                    "score": float(r.score),
                    "signals": {
                        "cf": float(getattr(r, "cf_score", 0.0)),
                        "semantic": float(getattr(r, "semantic_score", 0.0)),
                        "utility": float(getattr(r, "utility", 0.0)),
                        "baseline_popularity": float(getattr(r, "baseline", 0.0)),
                        "advantage": float(getattr(r, "advantage", 0.0)),
                        "novelty_boost": float(getattr(r, "novelty_boost", 0.0)),
                    },
                }
            )
        return recs
