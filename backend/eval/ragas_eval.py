"""
RAGAS evaluation harness for GraphRAG++.

Usage:
    # Run evaluation against the live backend (must be running on port 8000)
    python -m eval.ragas_eval

    # Run against a specific golden set
    python -m eval.ragas_eval --golden eval/golden_set.json --output eval/results.json

    # Dry-run: query the pipeline only, skip RAGAS scoring
    python -m eval.ragas_eval --no-ragas

Metrics produced (RAGAS):
    - faithfulness       : claims in answer grounded in retrieved context
    - answer_relevancy   : answer is relevant to the question
    - context_precision  : retrieved chunks are useful (low noise)
    - context_recall     : retrieved chunks cover everything needed

RAGAS requires an LLM judge. Configure via environment:
    HF_TOKEN + HF_MODEL_ID  -> uses HuggingFace Inference API as judge
    OPENAI_API_KEY          -> uses OpenAI as judge (if available)
    USE_SIMULATION=true     -> skips RAGAS (no LLM judge available)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

BACKEND_URL = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000")
DEFAULT_GOLDEN = Path(__file__).parent / "golden_set.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results.json"


async def run_query(client: httpx.AsyncClient, question: str) -> Dict[str, Any]:
    """Run a single question through the GraphRAG++ query pipeline."""
    resp = await client.post(
        f"{BACKEND_URL}/query",
        json={"question": question},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()


def _build_ragas_dataset(pairs: List[Dict], pipeline_results: List[Dict]):
    """Convert pipeline results + golden answers into a RAGAS Dataset."""
    from datasets import Dataset

    rows = {
        "question":          [],
        "answer":            [],
        "contexts":          [],
        "ground_truth":      [],
    }
    for pair, result in zip(pairs, pipeline_results):
        rows["question"].append(pair["question"])
        rows["answer"].append(result.get("answer", ""))
        # Retrieved context = top passages + graph paths text
        passages = [p["text"] for p in result.get("passages", [])]
        paths    = [
            " -> ".join(
                f"{h.get('from', {}).get('label', '')} [{h.get('edge', {}).get('predicate', '')}] {h.get('to', {}).get('label', '')}"
                for h in path
            )
            for path in result.get("paths", [])
        ]
        rows["contexts"].append(passages + paths or ["(no context retrieved)"])
        rows["ground_truth"].append(pair["ground_truth"])

    return Dataset.from_dict(rows)


def _make_ragas_llm():
    """Return a RAGAS-compatible LLM judge, preferring HF API over OpenAI."""
    hf_token = os.getenv("HF_TOKEN")
    hf_model  = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")

    if hf_token:
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            llm = HuggingFaceEndpoint(
                repo_id=hf_model,
                huggingfacehub_api_token=hf_token,
                max_new_tokens=512,
                temperature=0.1,
            )
            print(f"[RAGAS] Using HF endpoint as LLM judge: {hf_model}")
            return llm
        except Exception as e:
            print(f"[RAGAS] HF judge setup failed: {e}")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            print("[RAGAS] Using OpenAI gpt-4o-mini as LLM judge")
            return llm
        except Exception as e:
            print(f"[RAGAS] OpenAI judge setup failed: {e}")

    return None


async def evaluate(
    golden_path: Path = DEFAULT_GOLDEN,
    output_path: Path = DEFAULT_OUTPUT,
    use_ragas: bool = True,
    max_pairs: int = 20,
) -> Dict[str, Any]:
    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    pairs = golden["pairs"][:max_pairs]
    print(f"[Eval] Running {len(pairs)} queries against {BACKEND_URL}…")

    pipeline_results = []
    async with httpx.AsyncClient() as client:
        # Check backend is up
        try:
            await client.get(f"{BACKEND_URL}/health", timeout=5.0)
        except Exception:
            print(f"[Eval] ERROR: backend not reachable at {BACKEND_URL}")
            sys.exit(1)

        for i, pair in enumerate(pairs):
            t0 = time.time()
            try:
                result = await run_query(client, pair["question"])
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(pairs)}] {pair['id']} — {elapsed:.1f}s — {len(result.get('answer',''))} chars")
                pipeline_results.append(result)
            except Exception as e:
                print(f"  [{i+1}/{len(pairs)}] {pair['id']} — FAILED: {e}")
                pipeline_results.append({"answer": "", "passages": [], "paths": []})

    output = {
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend_url":   BACKEND_URL,
        "num_questions": len(pairs),
        "pipeline_results": [
            {
                "id":              pair["id"],
                "question":        pair["question"],
                "ground_truth":    pair["ground_truth"],
                "answer":          res.get("answer", ""),
                "total_ms":        res.get("total_ms", 0),
                "llm_source":      res.get("source", "unknown"),
                "passages_used":   len(res.get("passages", [])),
                "paths_found":     len(res.get("paths", [])),
            }
            for pair, res in zip(pairs, pipeline_results)
        ],
    }

    if use_ragas and os.getenv("USE_SIMULATION") != "true":
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            dataset = _build_ragas_dataset(pairs, pipeline_results)
            llm = _make_ragas_llm()

            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            kwargs: Dict[str, Any] = {"dataset": dataset, "metrics": metrics}
            if llm:
                kwargs["llm"] = llm

            print("\n[RAGAS] Running evaluation (this may take a few minutes)…")
            ragas_result = ragas_evaluate(**kwargs)
            scores = ragas_result.to_pandas().mean(numeric_only=True).to_dict()

            output["ragas_scores"] = {
                "faithfulness":      round(scores.get("faithfulness", 0), 4),
                "answer_relevancy":  round(scores.get("answer_relevancy", 0), 4),
                "context_precision": round(scores.get("context_precision", 0), 4),
                "context_recall":    round(scores.get("context_recall", 0), 4),
            }
            print("\n[RAGAS] Results:")
            for k, v in output["ragas_scores"].items():
                print(f"  {k:<22} {v:.4f}")

        except ImportError:
            print("[RAGAS] ragas package not installed — skipping metric computation")
            print("        Run: pip install ragas datasets langchain-huggingface")
        except Exception as e:
            print(f"[RAGAS] Evaluation failed: {e}")
    else:
        print("[Eval] RAGAS scoring skipped (USE_SIMULATION=true or --no-ragas)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[Eval] Results written to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="GraphRAG++ RAGAS evaluation harness")
    parser.add_argument("--golden",   default=str(DEFAULT_GOLDEN), help="Path to golden set JSON")
    parser.add_argument("--output",   default=str(DEFAULT_OUTPUT),  help="Path for results JSON output")
    parser.add_argument("--no-ragas", action="store_true",           help="Skip RAGAS metric computation")
    parser.add_argument("--max",      type=int, default=20,          help="Max questions to evaluate (default 20)")
    args = parser.parse_args()

    asyncio.run(evaluate(
        golden_path=Path(args.golden),
        output_path=Path(args.output),
        use_ragas=not args.no_ragas,
        max_pairs=args.max,
    ))


if __name__ == "__main__":
    main()
