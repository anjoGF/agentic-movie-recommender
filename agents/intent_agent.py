import json
from typing import Any, Dict

from agents.openai_client import OpenAIJSONClient


class IntentAgent:
    """
    OpenAI-backed intent and clarification agent.

    IMPORTANT:
    - Does NOT accept an external llm object (prevents JSON serialization issues).
    - Always returns JSON-safe primitives.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = OpenAIJSONClient(model=model, temperature=0.2)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
You are an intent classification agent for a movie recommender.

You must follow these rules:
- Think privately; do NOT reveal chain-of-thought.
- Output ONLY valid JSON (no markdown, no commentary).
- Use only the provided context. Do not invent user preferences.

Your job:
1) classify the user’s intent
2) decide if a clarification question is needed
3) produce a short trace list with high-level reasons (no hidden reasoning)
"""

        user_prompt = f"""
Context (JSON):
{json.dumps(context, indent=2)}

Allowed intents (choose exactly one):
- "search": user has a concrete query phrase or specific attributes
- "explore": browsing / discovery
- "comfort": safer, familiar picks; low novelty tolerance
- "quick_watch": time-constrained; prefers shorter / low-commitment

Hard rules:
- If "query" is a non-empty string -> intent MUST be "search"
- If novelty_tolerance exists and < 0.3 -> intent MUST be "comfort" (unless query is present)
- If available_minutes exists and <= 45 -> intent SHOULD be "quick_watch" (unless query is present)

Clarification rules:
- Set needs_clarification=true ONLY if information is insufficient to satisfy the intent.
- If needs_clarification=true, provide exactly ONE question.

Return ONLY this JSON schema:
{{
  "intent": "search|explore|comfort|quick_watch",
  "confidence": 0.0,
  "needs_clarification": false,
  "clarification_question": "",
  "trace": ["reason_a", "reason_b"]
}}
"""

        schema_hint = {
            "intent": "search",
            "confidence": 0.8,
            "needs_clarification": False,
            "clarification_question": "",
            "trace": ["..."],
        }

        resp = self.llm.generate_json(system_prompt, user_prompt, force_json=True, schema_hint=schema_hint, max_retries=1)

        data = resp["data"] if resp["ok"] else {}
        trace = list(data.get("trace", [])) if isinstance(data.get("trace", []), list) else []
        trace += resp.get("trace", [])
        trace.append("intent_agent_openai")

        # --- HARD GUARDS (deterministic enforcement) ---
        query = (context.get("query") or "").strip()
        novelty = context.get("novelty_tolerance", None)
        minutes = context.get("available_minutes", None)

        intent = str(data.get("intent", "explore"))
        if query:
            intent = "search"
        else:
            if isinstance(novelty, (int, float)) and novelty < 0.3:
                intent = "comfort"
            elif isinstance(minutes, (int, float)) and minutes <= 45:
                intent = "quick_watch"

        try:
            confidence = float(data.get("confidence", 0.6))
        except Exception:
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))

        needs_clarification = bool(data.get("needs_clarification", False))
        clarification_question = str(data.get("clarification_question", "") or "")

        # If low confidence, ask a single deterministic clarification question
        if confidence < 0.5 and not needs_clarification:
            needs_clarification = True

        if needs_clarification and not clarification_question:
            # One question that helps both semantic + taste alignment
            clarification_question = (
                "Do you want the sci-fi to lean more cerebral and philosophical, "
                "or more emotional and character-driven?"
            )
            trace.append("default_clarification_applied")

        return {
            "intent": intent,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "trace": trace,
        }
