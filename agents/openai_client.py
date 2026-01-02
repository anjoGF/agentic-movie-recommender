import json
from typing import Any, Dict, Optional, Tuple
from openai import OpenAI


class OpenAIJSONClient:
    """
    OpenAI helper with:
    - JSON-only enforcement
    - bounded repair
    - structured verbosity (NO chain-of-thought)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        timeout: Optional[float] = None,
        verbose: bool = False,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[OpenAIJSONClient] {msg}")

    def _call(self, system_prompt: str, user_prompt: str, force_json: bool = True) -> str:
        self._log("Calling OpenAI API")

        kwargs: Dict[str, Any] = {}
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            **kwargs,
        )

        raw = resp.choices[0].message.content.strip()
        self._log(f"Raw response length: {len(raw)} chars")
        return raw

    @staticmethod
    def _try_parse(raw: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not raw or not raw.strip():
            return None, False

        try:
            return json.loads(raw), True
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start:end + 1]), True
                except Exception:
                    return None, False
        return None, False

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        force_json: bool = True,
        schema_hint: Optional[Dict[str, Any]] = None,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Returns:
        {
            "ok": bool,
            "data": dict,
            "raw": str,
            "trace": list[str]
        }
        """

        trace = []
        raw = ""

        # Attempt 1
        try:
            raw = self._call(system_prompt, user_prompt, force_json)
            obj, ok = self._try_parse(raw)
            if ok and isinstance(obj, dict):
                self._log("JSON parsed successfully")
                return {"ok": True, "data": obj, "raw": raw, "trace": trace}
            trace.append("json_parse_failed")
            self._log("JSON parse failed")
        except Exception as e:
            trace.append(f"openai_call_failed:{type(e).__name__}")
            self._log(f"OpenAI call failed: {e}")

        # Repair attempt
        for _ in range(max_retries):
            repair_system = (
                "You repair invalid JSON.\n"
                "Return ONLY valid JSON.\n"
                "Do NOT include commentary.\n"
            )

            repair_user = (
                "The previous output was not valid JSON.\n"
                "Fix it and return ONLY JSON.\n"
                "If content is missing, use safe defaults.\n"
            )

            if schema_hint:
                repair_user += f"\nJSON schema hint:\n{json.dumps(schema_hint, indent=2)}\n"

            repair_user += f"\nInvalid output:\n{raw}\n"

            try:
                self._log("Attempting JSON repair")
                raw2 = self._call(repair_system, repair_user, force_json)
                obj2, ok2 = self._try_parse(raw2)
                if ok2 and isinstance(obj2, dict):
                    trace.append("json_repair_success")
                    self._log("JSON repair succeeded")
                    return {"ok": True, "data": obj2, "raw": raw2, "trace": trace}
                trace.append("json_repair_failed")
                raw = raw2
            except Exception as e:
                trace.append(f"openai_repair_failed:{type(e).__name__}")
                self._log(f"Repair failed: {e}")

        self._log("Falling back to safe empty JSON")
        return {"ok": False, "data": {}, "raw": raw, "trace": trace}
