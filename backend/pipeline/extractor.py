"""
Entity-Relation Triple Extractor using the LLM adapter.
Extracts (subject, predicate, object) triples from raw text.
"""
import json
import re
from typing import List
from .llm import call_llm

# Alpaca instruction — matches fine-tuned model's training format exactly
EXTRACTION_INSTRUCTION = "Extract entity-relation triples from this text as JSON."


async def extract_triples(text: str) -> List[dict]:
    """Extract entity-relation triples using specialized reasoning modes."""
    # Determine the best instruction based on content
    text_lower = text.lower()
    
    if any(k in text_lower for k in ["security", "firewall", "port", "access", "vulnerability"]):
        instr = "Identify security risks and access control relations as JSON."
    elif any(k in text_lower for k in ["error", "oom", "crash", "log", "fail", "trace"]):
        instr = "Analyze log causality and system error relations as JSON."
    elif any(k in text_lower for k in ["table", "column", "schema", "database", "sql"]):
        instr = "Extract SQL-to-ERD mapping and database relations as JSON."
    elif any(k in text_lower for k in ["architecture", "module", "service", "integration", "migration"]):
        instr = "Extract architecture component and system integration relations as JSON."
    else:
        instr = "Extract entity-relation triples from this text as JSON."

    result = await call_llm(
        prompt      = text[:3000],
        instruction = instr,
        max_tokens  = 1024,
    )
    answer = result["answer"]

    # Try to parse JSON from response
    triples = _parse_json_triples(answer)
    if not triples:
        # Fallback: regex extraction
        triples = _regex_fallback(text)

    return triples


def _parse_json_triples(text: str) -> List[dict]:
    """Parse JSON triples from LLM output."""
    # Try direct JSON parse
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [_normalize_triple(t) for t in data if _is_valid_triple(t)]
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in text
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [_normalize_triple(t) for t in data if _is_valid_triple(t)]
        except json.JSONDecodeError:
            pass

    return []


def _is_valid_triple(t: dict) -> bool:
    required = ["subject", "predicate", "object"]
    return all(k in t and t[k] for k in required)


def _normalize_triple(t: dict) -> dict:
    return {
        "subject": str(t.get("subject", "")).strip(),
        "predicate": str(t.get("predicate", "")).strip().replace(" ", "_").lower(),
        "object": str(t.get("object", "")).strip(),
        "confidence": float(t.get("confidence", 0.85)),
    }


def _regex_fallback(text: str) -> List[dict]:
    """Simple regex-based extraction as last resort."""
    triples = []
    # Look for "X is/are/uses/enables Y" patterns
    patterns = [
        r"(\w[\w\s]{1,30})\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w[\w\s]{1,30})",
        r"(\w[\w\s]{1,30})\s+(uses|enables|requires|extends|based on|introduces)\s+(\w[\w\s]{1,30})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE)[:5]:
            groups = m.groups()
            if len(groups) == 3:
                triples.append({"subject": groups[0].strip(), "predicate": groups[1].strip(), "object": groups[2].strip(), "confidence": 0.6})
            elif len(groups) == 2:
                triples.append({"subject": groups[0].strip(), "predicate": "is_a", "object": groups[1].strip(), "confidence": 0.5})
    return triples[:10]
