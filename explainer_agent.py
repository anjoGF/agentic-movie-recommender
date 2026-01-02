import json
from typing import Any, Dict, List, Optional

from agents.openai_client import OpenAIJSONClient


class ExplainerAgent:
    """
    OpenAI-backed explanation agent.

    Design goals:
    - Conversational and user-facing
    - Explicitly tied to the user's stated constraints
    - Honest about strength of fit (no forced justification)
    - Grounded in titles/genres only (no invented plot facts)
    - No algorithm, model, or scoring language
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = OpenAIJSONClient(model=model, temperature=0.55)

    def run(
        self,
        intent: str,
        strategy: Dict[str, Any],
        recs: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}

        system_prompt = """
You are the Netflix Explainer.

Your job is to explain WHY these recommendations fit the user's request —
without exaggeration and without forcing relevance.

CORE PRINCIPLES:
- Be honest about fit quality. Not every recommendation is a perfect match.
- Emphasize the strongest matches first.
- If a title is only a partial or tonal match, say so gently.
- Never invent plot details or specific scenes.

RULES:
- Speak directly TO the user ("you"), never AS the user.
- Do NOT mention algorithms, models, embeddings, rankings, or scores.
- Do NOT claim deep romance, realism, or emotional depth unless the title clearly supports it.
- Use careful language when appropriate:
  "leans more toward", "has elements of", "is less about romance but shares a grounded tone".
- Explicitly reference the user's constraints (e.g., "feels real", "not cheesy", "grounded", "emotionally honest").
- Avoid generic genre-only explanations.
- Output ONLY valid JSON (no markdown, no commentary).

TONE:
- Calm, editorial, and human.
- Similar to Netflix's in-app editorial blurbs.
"""

        # Keep explanation grounded: title + genres only
        top3 = [
            {
                "title": r.get("title", ""),
                "genres": r.get("genres", ""),
            }
            for r in (recs or [])[:3]
        ]

        user_prompt = f"""
User intent:
{intent}

User context (JSON):
{json.dumps(context, indent=2)}

Top recommendations (title + genres only):
{json.dumps(top3, indent=2)}

WRITE:
- one_liner:
  ONE sentence summarizing why these picks were chosen,
  explicitly referencing the user's constraints.

- bullets:
  EXACTLY 3 bullets.
  Each bullet must:
  - Mention a specific title.
  - Explain how it aligns with the user's constraints.
  - Use honest framing (strong match vs partial/tonal match).
  - Avoid plot claims or factual specifics.

IMPORTANT:
- If a movie is not clearly romantic or emotionally grounded,
  describe it as a tonal or adjacent match rather than a direct one.
- Do NOT try to justify poor matches.

RETURN ONLY THIS JSON SCHEMA:
{{
  "one_liner": "string",
  "bullets": ["string", "string", "string"]
}}
"""

        schema_hint = {
            "one_liner": "These films focus on grounded, emotionally honest storytelling rather than glossy or exaggerated romance.",
            "bullets": [
                "Get Real (1998) is often described as emotionally raw and understated, which aligns well with a preference for romance that feels genuine rather than performative.",
                "All the Real Girls (2003) leans into quiet, character-driven moments, favoring emotional realism over dramatic spectacle.",
                "Blue Valentine (2010) is known for its unvarnished tone, offering a more difficult but deeply authentic take on romantic relationships.",
            ],
        }

        resp = self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_json=True,
            schema_hint=schema_hint,
            max_retries=1,
        )

        data = resp["data"] if resp["ok"] else {}

        one_liner = data.get("one_liner", "")
        bullets = data.get("bullets", [])

        if not isinstance(one_liner, str):
            one_liner = ""

        if not isinstance(bullets, list):
            bullets = []

        bullets = [b for b in bullets if isinstance(b, str)]

        # Ensure exactly 3 bullets (safe padding)
        while len(bullets) < 3:
            bullets.append("")
        bullets = bullets[:3]

        return {
            "one_liner": one_liner,
            "bullets": bullets,
        }
