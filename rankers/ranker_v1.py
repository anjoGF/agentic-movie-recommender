from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd


class RankerV1:
    """
    Baseline v1 ranker:
    - linear fusion of normalized CF and semantic scores
    - no advantage weighting
    """

    def __init__(self, cfg, movie_map: Dict[int, Dict[str, str]]):
        self.cfg = cfg
        self.movie_map = movie_map

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

        for col in ["cf_score", "semantic_score"]:
            mx = float(df[col].max()) if len(df) else 0.0
            if mx > 0:
                df[col] = df[col] / mx

        w_cf = float(state["plan"].get("weight_cf", 0.5))
        w_sem = float(state["plan"].get("weight_semantic", 0.5))
        df["score"] = w_cf * df["cf_score"] + w_sem * df["semantic_score"]

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
                    },
                }
            )
        return recs
