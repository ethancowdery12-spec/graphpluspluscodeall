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

    if any(k in text_lower for k in ["safety", "alignment", "sycophancy", "corrigib", "decepti",
                                      "manipulation", "honest", "refusal", "harmful", "cheating",
                                      "feature activation", "interpretability", "model behavior",
                                      "constitutional", "rlhf", "reward", "red team"]):
        instr = (
            "Extract named entity and causal-event triples from this AI safety text as JSON. "
            "Focus on: model behaviors, safety properties, evaluation results, named techniques, "
            "researchers, and causal relationships (X caused Y, X led to Z). "
            "Subjects and objects must be proper nouns or short noun phrases, NOT sentence fragments."
        )
    elif any(k in text_lower for k in ["security", "firewall", "port", "access", "vulnerability"]):
        instr = "Identify security risks and access control relations as JSON."
    elif any(k in text_lower for k in ["error", "oom", "crash", "log", "fail", "trace"]):
        instr = "Analyze log causality and system error relations as JSON."
    elif any(k in text_lower for k in ["table", "column", "schema", "database", "sql"]):
        instr = "Extract SQL-to-ERD mapping and database relations as JSON."
    elif any(k in text_lower for k in ["architecture", "module", "service", "integration", "migration"]):
        instr = "Extract architecture component and system integration relations as JSON."
    else:
        instr = (
            "Extract entity-relation triples from this text as JSON. "
            "Subjects and objects must be proper nouns or short noun phrases, NOT partial sentences."
        )

    result = await call_llm(
        prompt      = text[:3000],
        instruction = instr,
        max_tokens  = 1024,
    )

    # Try answer first, then fall back to thinking block in case the model
    # put the JSON inside <think>...</think> with non-empty prose after it.
    triples = _parse_json_triples(result.get("answer", ""))
    if not triples and result.get("thinking"):
        triples = _parse_json_triples(result["thinking"])
    if not triples:
        triples = _regex_fallback(text)

    return triples


def _parse_json_triples(text: str) -> List[dict]:
    """Parse JSON triples from LLM output.

    Handles:
    - Clean JSON arrays / objects
    - Markdown code fences
    - Reasoning prose containing [subject, predicate, object] before the real array
    - JSON inside thinking blocks (caller passes thinking separately)
    """
    if not text:
        return []

    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE).strip()

    # 1. Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            triples = [_normalize_triple(t) for t in data if _is_valid_triple(t)]
            if triples:
                return triples
        if isinstance(data, dict):
            for key in ("triples", "result", "data", "output"):
                if isinstance(data.get(key), list):
                    triples = [_normalize_triple(t) for t in data[key] if _is_valid_triple(t)]
                    if triples:
                        return triples
    except json.JSONDecodeError:
        pass

    # 2. Iterate every [...] span left-to-right; take the first that yields valid triples.
    #    Non-greedy so we get each bracket pair independently, not one giant blob.
    for m in re.finditer(r'\[.*?\]', text, re.DOTALL):
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                triples = [_normalize_triple(t) for t in data if _is_valid_triple(t)]
                if triples:
                    return triples
        except json.JSONDecodeError:
            pass

    # 3. Greedy fallback — captures outermost array when inner arrays appear in values
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                triples = [_normalize_triple(t) for t in data if _is_valid_triple(t)]
                if triples:
                    return triples
        except json.JSONDecodeError:
            pass

    return []


def _is_valid_triple(t: dict) -> bool:
    required = ["subject", "predicate", "object"]
    if not all(k in t and t[k] for k in required):
        return False
    subj = str(t["subject"]).strip()
    obj  = str(t["object"]).strip()
    # Reject sentence fragments masquerading as entities
    if _is_fragment(subj) or _is_fragment(obj):
        return False
    # Reject trivially short fields
    if len(subj) < 3 or len(obj) < 3:
        return False
    return True


def _is_fragment(text: str) -> bool:
    """Return True if the text looks like a mid-sentence fragment, not a real entity."""
    import re
    t = text.strip()
    if len(t) < 3:
        return True
    # Starts with a lowercase letter that's clearly mid-word/sentence
    if t[0].islower() and t[0] not in ('a', 'i'):
        return True
    # Ends with a dangling function word (open prepositional phrase, article, conjunction)
    dangling = re.compile(
        r'\b(a|an|the|of|in|on|at|to|for|with|by|from|and|or|that|which|'
        r'its|this|these|those|not|without|after|before|when|as|is|are|was|were)\s*$',
        re.IGNORECASE
    )
    if dangling.search(t):
        return True
    return False


def _normalize_triple(t: dict) -> dict:
    conf = float(t.get("confidence", 0.85))
    # Tier: explicit override > infer from confidence
    tier = t.get("confidence_tier") or ("INFERRED" if conf >= 0.7 else "AMBIGUOUS")
    return {
        "subject":          str(t.get("subject", "")).strip(),
        "predicate":        str(t.get("predicate", "")).strip().replace(" ", "_").lower(),
        "object":           str(t.get("object", "")).strip(),
        "confidence":       conf,
        "confidence_tier":  tier,
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
        for m in list(re.finditer(pat, text, re.IGNORECASE))[:5]:
            groups = m.groups()
            if len(groups) == 3:
                triples.append({"subject": groups[0].strip(), "predicate": groups[1].strip(), "object": groups[2].strip(), "confidence": 0.6})
            elif len(groups) == 2:
                triples.append({"subject": groups[0].strip(), "predicate": "is_a",
                                "object": groups[1].strip(), "confidence": 0.5,
                                "confidence_tier": "AMBIGUOUS"})
    # Tag all regex results as AMBIGUOUS
    for t in triples:
        t.setdefault("confidence_tier", "AMBIGUOUS")
    return triples[:10]
