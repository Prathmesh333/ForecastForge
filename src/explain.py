"""Grounded deterministic summary plus an optional OpenAI-compatible LLM adapter."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def deterministic_explanation(decision: dict) -> dict:
    probability = float(decision["probability_target"])
    observed = (
        f"The median attributed-revenue forecast is ${decision['revenue_p50']:,.0f}, with a "
        f"median ROAS of {decision['roas_p50']:.2f}. The modeled probability of meeting the "
        f"{decision['target_roas']:.2f} ROAS target is {probability:.0%}."
    )
    return {
        "mode": "deterministic evidence summary",
        "observed_evidence": observed,
        "model_inference": (
            f"ForecastForge recommends {decision['recommendation']}. The portfolio evidence status is "
            f"{decision['support_status']}."
        ),
        "causal_hypothesis": decision["experiment"]["hypothesis"],
        "risk_and_limitations": (
            "The budget response is a conditional planning scenario based on attributed history; "
            "it is not proof of incremental causality."
        ),
        "recommended_test": decision["experiment"]["test_design"],
    }


def _prompt(decision: dict) -> str:
    return (
        "You are a cautious ecommerce marketing analyst. Use only the supplied JSON evidence. "
        "Never invent promotions, competitor events, attribution changes, or causal certainty. "
        "Return JSON with observed_evidence, model_inference, causal_hypothesis, "
        "risk_and_limitations, and recommended_test. Explicitly label uncertainty.\n\n"
        + json.dumps(decision, indent=2)
    )


def generate_llm_explanation(decision: dict) -> dict:
    endpoint = os.getenv("RANGE_LLM_ENDPOINT", "").strip()
    api_key = os.getenv("RANGE_LLM_API_KEY", "").strip()
    model = os.getenv("RANGE_LLM_MODEL", "").strip()
    if not endpoint or not api_key or not model:
        return deterministic_explanation(decision)

    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": "Ground every claim in supplied evidence and return valid JSON only.",
            },
            {"role": "user", "content": _prompt(decision)},
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    fallback = deterministic_explanation(decision)
    required_fields = (
        "observed_evidence",
        "model_inference",
        "causal_hypothesis",
        "risk_and_limitations",
        "recommended_test",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
        if not isinstance(result, dict):
            raise TypeError("LLM response must be a JSON object")
        grounded = {
            field: str(result.get(field) or fallback[field])
            for field in required_fields
        }
        grounded["mode"] = "LLM evidence summary"
        return grounded
    except (urllib.error.URLError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        fallback["mode"] = "deterministic fallback after LLM error"
        return fallback
