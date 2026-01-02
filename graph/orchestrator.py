from langgraph.graph import StateGraph, START, END
from typing import Dict, Any


class AgenticGraph:
    """
    LangGraph orchestrator (v1/v2 compatible).

    Key points:
    - uses intent_obj key
    - trace_log is appended at each node
    - supports critic-driven rerank adjustments
    """

    def __init__(self, agents, tools, ranker, cfg):
        self.agents = agents
        self.tools = tools
        self.ranker = ranker
        self.cfg = cfg

    def build(self):
        graph = StateGraph(dict)

        def _append_trace(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
            trace = list(state.get("trace_log", []))
            trace.append(event)
            return {**state, "trace_log": trace}

        def intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            intent_obj = self.agents["intent"].run(state["context"])
            state2 = {**state, "intent_obj": intent_obj}
            return _append_trace(
                state2,
                {
                    "node": "intent",
                    "intent": intent_obj.get("intent"),
                    "confidence": intent_obj.get("confidence"),
                    "trace": intent_obj.get("trace", []),
                },
            )

        def plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
            intent = (state.get("intent_obj") or {}).get("intent", "explore")
            plan = self.agents["planner"].run(intent, state["context"])
            state2 = {**state, "plan": plan}
            return _append_trace(
                state2,
                {
                    "node": "plan",
                    "use_cf": plan.get("use_cf"),
                    "use_semantic": plan.get("use_semantic"),
                    "weights": {"cf": plan.get("weight_cf"), "semantic": plan.get("weight_semantic")},
                },
            )

        def retrieve_node(state: Dict[str, Any]) -> Dict[str, Any]:
            updates = {}
            plan = state.get("plan") or {}
            use_cf = bool(plan.get("use_cf", True))
            use_sem = bool(plan.get("use_semantic", False))

            if use_cf:
                updates["cf"] = self.tools["cf"].recommend(state["user_id"], self.cfg.cf_k)

            if use_sem:
                query = (state["context"].get("query") or "").strip()
                if query:
                    updates["sem"] = self.tools["semantic"].search(query, self.cfg.semantic_k)
                else:
                    updates["sem"] = []

            state2 = {**state, **updates}
            return _append_trace(
                state2,
                {
                    "node": "retrieve",
                    "use_cf": use_cf,
                    "use_semantic": use_sem,
                    "cf_rows": int(len(state2.get("cf"))) if state2.get("cf") is not None else 0,
                    "sem_rows": int(len(state2.get("sem"))) if state2.get("sem") is not None else 0,
                },
            )

        def rank_node(state: Dict[str, Any]) -> Dict[str, Any]:
            recs = self.ranker(state)
            state2 = {**state, "recs": recs}
            top = recs[0]["title"] if recs else None
            return _append_trace(state2, {"node": "rank", "top1": top, "recs_count": len(recs)})

        def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
            intent = (state.get("intent_obj") or {}).get("intent", "explore")
            critic = self.agents["critic"].run(intent, state["context"], state.get("recs", []))
            state2 = {**state, "critic": critic}
            return _append_trace(
                state2,
                {
                    "node": "critic",
                    "needs_rerank": critic.get("needs_rerank"),
                    "adjustments": critic.get("adjustments", {}),
                },
            )

        def should_rerank(state: Dict[str, Any]) -> str:
            critic = state.get("critic") or {}
            return "rerank" if bool(critic.get("needs_rerank", False)) else "explain"

        def rerank_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Apply critic adjustments in-state and re-rank once.
            We do NOT mutate global cfg; we only tweak state knobs.
            """
            critic = state.get("critic") or {}
            adj = critic.get("adjustments") or {}

            # Store knobs in state for ranker_v2 to read if you want to extend;
            # For now, rankers use cfg, so we adjust plan weights only (safe).
            plan = dict(state.get("plan") or {})
            w_cf = float(plan.get("weight_cf", 0.4))
            w_sem = float(plan.get("weight_semantic", 0.6))

            w_cf_delta = float(adj.get("weight_cf_delta", 0.0))
            w_sem_delta = float(adj.get("weight_semantic_delta", 0.0))

            w_cf = max(0.0, min(1.0, w_cf + w_cf_delta))
            w_sem = max(0.0, min(1.0, w_sem + w_sem_delta))

            if w_cf > 0 and w_sem > 0:
                s = max(w_cf + w_sem, 1e-6)
                w_cf, w_sem = w_cf / s, w_sem / s
            elif w_cf > 0:
                w_cf, w_sem = 1.0, 0.0
            else:
                w_cf, w_sem = 0.0, 1.0

            plan["weight_cf"] = float(w_cf)
            plan["weight_semantic"] = float(w_sem)
            plan_trace = list(plan.get("trace", [])) if isinstance(plan.get("trace", []), list) else []
            plan_trace.append("critic_rerank_applied")
            plan["trace"] = plan_trace

            state2 = {**state, "plan": plan}

            # Re-rank once
            recs = self.ranker(state2)
            state3 = {**state2, "recs": recs}

            return _append_trace(
                state3,
                {
                    "node": "rerank",
                    "applied_adjustments": adj,
                    "new_weights": {"cf": plan["weight_cf"], "semantic": plan["weight_semantic"]},
                    "top1": recs[0]["title"] if recs else None,
                },
            )

        def explain_node(state: Dict[str, Any]) -> Dict[str, Any]:
            intent = (state.get("intent_obj") or {}).get("intent", "explore")
            explanation = self.agents["explainer"].run(intent, state.get("plan", {}), state.get("recs", []))
            state2 = {**state, "explanation": explanation}
            return _append_trace(state2, {"node": "explain", "one_liner": explanation.get("one_liner", "")})

        graph.add_node("intent", intent_node)
        graph.add_node("plan", plan_node)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("rank", rank_node)
        graph.add_node("critic", critic_node)
        graph.add_node("rerank", rerank_node)
        graph.add_node("explain", explain_node)

        graph.add_edge(START, "intent")
        graph.add_edge("intent", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "rank")
        graph.add_edge("rank", "critic")
        graph.add_conditional_edges("critic", should_rerank, {"rerank": "rerank", "explain": "explain"})
        graph.add_edge("rerank", "explain")
        graph.add_edge("explain", END)

        return graph.compile()
