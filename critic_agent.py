from __future__ import annotations
from typing import Any, Dict, List
from config import POCConfig
from agents.openai_client import OpenAIJSONClient


class CriticAgent:
    """
    Critic v2:
    - Enforces hard safety / suitability constraints
    - Evaluates novelty & popularity
    - Checks genre diversity
    - Can request advantage-weight + novelty adjustments
    """

    def __init__(self, cfg: POCConfig):
        self.cfg = cfg
        self.client = OpenAIJSONClient(
            model="gpt-4o-mini",
            temperature=0.2,
            verbose=bool(cfg.verbose_openai),
        )
        self.verbose = bool(cfg.verbose)

    def _log(self, msg: str):
        if self.verbose:
            print(f"[CriticAgent] {msg}")

    def _genre_diversity(self, recs: List[Dict[str, Any]], topn: int) -> int:
        genres = set()
        for r in recs[:topn]:
            g = (r.get("genres") or "")
            for token in g.split("|"):
                t = token.strip()
                if t:
                    genres.add(t)
        return len(genres)

    def _mean_popularity(self, recs: List[Dict[str, Any]], topn: int) -> float:
        vals = []
        for r in recs[:topn]:
            pop = ((r.get("signals") or {}).get("baseline_popularity"))
            if isinstance(pop, (int, float)):
                vals.append(float(pop))
        return float(sum(vals) / max(len(vals), 1))

    def run(
        self,
        intent: str,
        context: Dict[str, Any],
        recs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        intent = (intent or "explore").strip().lower()
        topn = int(self.cfg.critic_topn)

        needs_rerank = False
        adjustments: Dict[str, Any] = {}
        trace: List[str] = []

        # 1) DETERMINISTIC GUARDRAILS 

        mean_pop = self._mean_popularity(recs, topn)
        if mean_pop > float(self.cfg.popularity_mean_threshold):
            needs_rerank = True
            adjustments["novelty_lambda"] = min(
                0.35, float(self.cfg.novelty_lambda) + 0.10
            )
            trace.append(f"Top-{topn} mean popularity too high: {mean_pop:.2f}")

        gdiv = self._genre_diversity(recs, topn)
        if gdiv < int(self.cfg.genre_diversity_min_unique):
            needs_rerank = True
            adjustments["diversity_boost"] = 0.10
            trace.append(f"Genre diversity too low: unique_genres={gdiv}")

        # ======================================================
        # 2) HARD MATURITY / SUITABILITY VETO  (OPTION B)
        # ======================================================

        genres: List[str] = []
        for r in recs[:topn]:
            g = r.get("genres", "")
            if isinstance(g, str):
                genres.extend([x.strip() for x in g.split("|")])

        children_ratio = (
            sum(1 for g in genres if g.lower() == "children")
            / max(1, len(genres))
        )

        user_query = (context.get("query") or "").lower()
        children_allowed = any(
            kw in user_query
            for kw in ["kid", "kids", "family", "children", "child"]
        )

        if children_ratio > 0.30 and not children_allowed:
            needs_rerank = True
            adjustments["exclude_genres"] = ["Children"]
            trace.append("hard_veto_children_content")

        # 3) LLM CRITIQUE 

        system_prompt = (
            "You are a recommendation critic for a movie recommender.\n"
            "Return ONLY valid JSON.\n"
            "Do NOT include commentary.\n"
            "Do NOT reveal hidden reasoning.\n"
        )

        user_prompt = (
            f"Intent: {intent}\n"
            f"User context: {context}\n\n"
            "You will be given the top recommendations with optional signals.\n"
            "Assess:\n"
            "- Does the list satisfy the query and constraints?\n"
            "- Is novelty too low (too popular)?\n"
            "- Is the list too narrow in genres?\n"
            "- Are advantage signals reflected in top-ranked items?\n\n"
            "Return JSON with exactly these keys:\n"
            "{\n"
            '  "needs_rerank": boolean,\n'
            '  "adjustments": object,\n'
            '  "trace": array of strings\n'
            "}\n\n"
            f"Top recommendations:\n{recs[:10]}\n"
        )

        schema_hint = {
            "needs_rerank": needs_rerank,
            "adjustments": adjustments,
            "trace": trace,
        }

        res = self.client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_json=True,
            schema_hint=schema_hint,
            max_retries=1,
        )

        data = res.get("data") or {}
        llm_needs = bool(data.get("needs_rerank", False))
        llm_adj = data.get("adjustments", {})
        llm_trace = data.get("trace", [])

        # 4) MERGE DECISIONS (HARD RULES WIN)

        needs_rerank = bool(needs_rerank or llm_needs)
        if isinstance(llm_adj, dict):
            adjustments.update(llm_adj)

        if isinstance(llm_trace, list):
            trace.extend(llm_trace)

        trace.extend(res.get("trace", []))
        trace.append("openai_critic_v2")

        out = {
            "needs_rerank": needs_rerank,
            "adjustments": adjustments,
            "trace": trace,
        }

        self._log(
            f"Critic: needs_rerank={needs_rerank}, adjustments={adjustments}"
        )
        return out
