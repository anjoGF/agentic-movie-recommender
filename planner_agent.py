from __future__ import annotations
from typing import Any, Dict
from config import POCConfig
from agents.openai_client import OpenAIJSONClient


class PlannerAgent:
    """
    Planner v2 (research-aligned):
    - For search intent, do NOT disable CF if cfg.enforce_hybrid_for_search
    - Enforce weight_cf >= cfg.min_cf_weight when hybrid is enabled
    - Maintain strict schema and stable defaults
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
            print(f"[PlannerAgent] {msg}")

    def run(self, intent: str, context: Dict[str, Any]) -> Dict[str, Any]:
        intent = (intent or "explore").strip().lower()
        query = (context.get("query") or "").strip()

        system_prompt = (
            "You are a Netflix-style Strategy Planner for a movie recommender.\n"
            "You must output ONLY valid JSON.\n"
            "Do NOT include commentary.\n"
            "Do NOT reveal hidden reasoning.\n"
        )

        # Research rules encoded as explicit constraints:
        # When a query exists, semantic must be used.
        # When hybrid enforcement is enabled, CF must be non-zero even for search
        user_prompt = (
            f"Intent: {intent}\n"
            f"Query: {query}\n"
            f"Context JSON:\n{context}\n\n"
            "Return JSON with exactly these keys:\n"
            "{\n"
            '  "use_cf": boolean,\n'
            '  "use_semantic": boolean,\n'
            '  "weight_cf": number (0..1),\n'
            '  "weight_semantic": number (0..1),\n'
            '  "trace": array of strings\n'
            "}\n\n"
            "Rules:\n"
            "- If Query is non-empty, use_semantic MUST be true.\n"
            "- If intent is 'search' and hybrid enforcement is enabled, use_cf MUST be true.\n"
            "- If both tools are used, weights MUST sum to 1.\n"
            "- Avoid setting weight_cf to 0 when hybrid enforcement is enabled.\n"
        )

        schema_hint = {
            "use_cf": True,
            "use_semantic": True,
            "weight_cf": 0.4,
            "weight_semantic": 0.6,
            "trace": [],
        }

        res = self.client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_json=True,
            schema_hint=schema_hint,
            max_retries=1,
        )

        obj = res["data"] or {}
        trace = list(obj.get("trace", [])) if isinstance(obj.get("trace"), list) else []

        # Defaults
        use_semantic = bool(obj.get("use_semantic", bool(query)))
        use_cf = bool(obj.get("use_cf", True))

        # Enforce query -> semantic
        if query:
            use_semantic = True

        # Hybrid enforcement for search
        if self.cfg.enforce_hybrid_for_search and intent == "search":
            use_cf = True
            trace.append("Hybrid enforcement: search requires CF + semantic")

        # Weights
        w_cf = float(obj.get("weight_cf", 0.4 if use_cf else 0.0))
        w_sem = float(obj.get("weight_semantic", 0.6 if use_semantic else 0.0))

        if use_cf and use_semantic:
            # Enforce minimum CF share if enabled
            if self.cfg.enforce_hybrid_for_search and intent == "search":
                w_cf = max(w_cf, float(self.cfg.min_cf_weight))
                w_sem = max(0.0, 1.0 - w_cf)

            # Normalize
            s = max(w_cf + w_sem, 1e-6)
            w_cf, w_sem = w_cf / s, w_sem / s

        elif use_cf and not use_semantic:
            w_cf, w_sem = 1.0, 0.0

        elif use_semantic and not use_cf:
            # If hybrid enforcement is on and intent is search, this should not happen
            if self.cfg.enforce_hybrid_for_search and intent == "search":
                use_cf = True
                w_cf = float(self.cfg.min_cf_weight)
                w_sem = 1.0 - w_cf
                trace.append("Corrected: CF re-enabled for hybrid search")
            else:
                w_cf, w_sem = 0.0, 1.0

        else:
            # Never allow neither
            use_cf, use_semantic = True, bool(query)
            w_cf, w_sem = (1.0, 0.0) if not query else (0.4, 0.6)
            trace.append("Corrected: at least one tool required")

        trace.extend(res.get("trace", []))
        trace.append("openai_planner_v2")

        out = {
            "use_cf": use_cf,
            "use_semantic": use_semantic,
            "weight_cf": float(w_cf),
            "weight_semantic": float(w_sem),
            "trace": trace,
        }

        self._log(f"Plan: use_cf={use_cf}, use_semantic={use_semantic}, w_cf={out['weight_cf']:.2f}, w_sem={out['weight_semantic']:.2f}")
        return out
