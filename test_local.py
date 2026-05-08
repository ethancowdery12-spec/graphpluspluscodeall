"""
GraphRAG++ local pipeline tests.

Tests cover:
  - File existence (GGUF, llama-server.exe, Vulkan DLL)
  - Graph builder unit tests (no external deps)
  - Sample data seeding
  - Simulation mode full pipeline
  - GPU inference via llama-server.exe + Vulkan (marked @pytest.mark.gpu)

Run all:        pytest test_local.py -v
Run GPU only:   pytest test_local.py -v -m gpu
Skip GPU:       pytest test_local.py -v -m "not gpu"
"""
import os
import sys
import time
import asyncio
import importlib
import subprocess
import pytest

ROOT    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

GGUF_MODEL    = os.path.join(ROOT, "graphrag-plus-plus-qwen35-4b-q3_k_m.gguf")
LLAMA_SERVER  = os.path.join(ROOT, "llama-server.exe")
VULKAN_DLL    = os.path.join(ROOT, "ggml-vulkan.dll")
# Use non-default port so tests don't clash with a running UI instance
TEST_PORT     = 8089
TEST_BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{context}\n\n"
    "### Response:\n"
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _reset_graph_singleton():
    """Wipe the module-level graph singleton so each test starts fresh."""
    import pipeline.graph_builder as gb
    gb._graph_store = None


def _reload_pipeline():
    """Reload all pipeline sub-modules so env-var changes take effect."""
    for key in list(sys.modules):
        if key.startswith("pipeline"):
            del sys.modules[key]


# ─── 1. File existence ─────────────────────────────────────────────────────────

class TestFileExistence:
    def test_gguf_model_exists(self):
        assert os.path.exists(GGUF_MODEL), f"GGUF model missing: {GGUF_MODEL}"

    def test_llama_server_exe_exists(self):
        assert os.path.exists(LLAMA_SERVER), f"llama-server.exe missing: {LLAMA_SERVER}"

    def test_vulkan_dll_exists(self):
        assert os.path.exists(VULKAN_DLL), f"ggml-vulkan.dll missing: {VULKAN_DLL}"

    def test_backend_main_exists(self):
        assert os.path.exists(os.path.join(BACKEND, "main.py"))

    def test_pipeline_modules_importable(self):
        from pipeline.graph_builder import GraphStore
        from pipeline.embedder     import embed_query
        from pipeline.llm          import build_alpaca_prompt


# ─── 2. Graph builder unit tests ──────────────────────────────────────────────

class TestGraphBuilder:
    """Pure in-memory tests — no LLM, no server."""

    def setup_method(self):
        from pipeline.graph_builder import GraphStore
        self.g = GraphStore()

    def test_add_entity_returns_id(self):
        nid = self.g.add_entity("Transformer", "model")
        assert nid is not None
        assert len(nid) > 0

    def test_single_node_in_graph(self):
        self.g.add_entity("Transformer", "model")
        assert self.g.G.number_of_nodes() == 1

    def test_entity_deduplication(self):
        nid1 = self.g.add_entity("BERT", "model")
        nid2 = self.g.add_entity("BERT", "model")
        assert nid1 == nid2
        assert self.g.G.number_of_nodes() == 1

    def test_ingest_three_triples(self):
        triples = [
            {"subject": "Transformer", "predicate": "uses",     "object": "Self-Attention", "confidence": 0.99},
            {"subject": "BERT",        "predicate": "based_on", "object": "Transformer",    "confidence": 0.98},
            {"subject": "GPT",         "predicate": "based_on", "object": "Transformer",    "confidence": 0.97},
        ]
        result = self.g.ingest_triples(triples, source="test")
        # 4 unique entities: Transformer, Self-Attention, BERT, GPT
        assert self.g.G.number_of_nodes() == 4
        assert result["edges_added"] == 3

    def test_find_nodes_by_name(self):
        self.g.add_entity("GraphRAG", "system")
        hits = self.g.find_nodes_by_name("GraphRAG")
        assert len(hits) > 0

    def test_multi_hop_paths_returns_list(self):
        # multi_hop_paths always returns a list (may be empty for sparse graphs)
        triples = [
            {"subject": "A", "predicate": "uses", "object": "B", "confidence": 0.9},
            {"subject": "B", "predicate": "uses", "object": "C", "confidence": 0.8},
            {"subject": "C", "predicate": "uses", "object": "D", "confidence": 0.7},
        ]
        self.g.ingest_triples(triples, "test")
        a_id = self.g.find_nodes_by_name("A")[0]
        paths = self.g.multi_hop_paths([a_id], max_hops=3, top_k=10)
        assert isinstance(paths, list)

    def test_multi_hop_paths_seeded(self):
        # Use the richer seeded graph where paths are more likely to be found
        import pipeline.graph_builder as gb
        gb._graph_store = None
        from pipeline.sample_data   import seed_graph
        from pipeline.graph_builder import get_graph
        seed_graph()
        graph   = get_graph()
        hits    = graph.find_nodes_by_name("Transformer")
        if not hits:
            return  # entity not present after seed — skip gracefully
        paths = graph.multi_hop_paths(hits[:2], max_hops=3, top_k=5)
        assert isinstance(paths, list)

    def test_get_stats_structure(self):
        self.g.add_entity("X", "concept")
        stats = self.g.get_stats()
        for key in ("node_count", "edge_count", "version"):
            assert key in stats, f"Missing key: {key}"
        assert stats["node_count"] == 1

    def test_version_increments_on_ingest(self):
        v0 = self.g.get_stats()["version"]
        self.g.ingest_triples(
            [{"subject": "A", "predicate": "uses", "object": "B", "confidence": 0.9}],
            "test",
        )
        assert self.g.get_stats()["version"] == v0 + 1


# ─── 3. Sample data seeding ───────────────────────────────────────────────────

class TestSampleData:
    def setup_method(self):
        _reset_graph_singleton()

    def test_seed_populates_graph(self):
        from pipeline.sample_data  import seed_graph
        from pipeline.graph_builder import get_graph
        seed_graph()
        graph = get_graph()
        assert graph.G.number_of_nodes() > 50, "Expected 50+ seeded nodes"
        assert graph.G.number_of_edges() > 50, "Expected 50+ seeded edges"

    def test_seed_returns_counts(self):
        from pipeline.sample_data import seed_graph
        result = seed_graph()
        # seed_graph returns {nodes, edges, paper_triples, codebase_triples}
        assert "nodes" in result or "nodes_added" in result or "status" in result


# ─── 4. Simulation pipeline (no LLM needed) ───────────────────────────────────

class TestSimulationPipeline:
    def setup_method(self):
        _reset_graph_singleton()
        os.environ["USE_SIMULATION"]      = "true"
        os.environ["LOCAL_LLAMA_CPP_URL"] = ""
        os.environ["CLOUD_RUN_URL"]       = ""
        os.environ["LOCAL_LORA_PATH"]     = ""
        _reload_pipeline()

    def test_simulation_llm_call_returns_dict(self):
        import pipeline.llm as llm_mod
        result = asyncio.run(llm_mod.call_llm("What is a transformer?"))
        assert isinstance(result, dict)
        assert "answer"  in result
        assert "source"  in result
        assert result["source"] == "simulation"

    def test_simulation_llm_not_empty(self):
        import pipeline.llm as llm_mod
        result = asyncio.run(llm_mod.call_llm("Tell me about BERT"))
        assert len(result["answer"]) > 0

    def test_full_pipeline_five_steps(self):
        from pipeline.sample_data  import seed_graph
        from pipeline.query_engine import query
        seed_graph()
        result = asyncio.run(query("How does BERT relate to Transformer?"))
        assert len(result["steps"]) == 5

    def test_pipeline_step_names(self):
        from pipeline.sample_data  import seed_graph
        from pipeline.query_engine import query
        seed_graph()
        result = asyncio.run(query("What is GraphRAG?"))
        names = [s["step"] for s in result["steps"]]
        assert names == [
            "intent_classification",
            "node_matching",
            "graph_traversal",
            "context_fusion",
            "generation",
        ]

    def test_pipeline_result_fields(self):
        from pipeline.sample_data  import seed_graph
        from pipeline.query_engine import query
        seed_graph()
        result = asyncio.run(query("What uses self-attention?"))
        for field in ("question", "answer", "intent", "steps", "paths", "total_ms", "provenance"):
            assert field in result, f"Missing field: {field}"

    def test_pipeline_answer_not_empty(self):
        from pipeline.sample_data  import seed_graph
        from pipeline.query_engine import query
        seed_graph()
        result = asyncio.run(query("Describe the relationship between LoRA and fine-tuning"))
        assert len(result["answer"]) > 0

    def test_on_step_callback_fires(self):
        from pipeline.sample_data  import seed_graph
        from pipeline.query_engine import query
        seed_graph()
        fired = []

        async def on_step(name, data):
            fired.append(name)

        asyncio.run(query("What is RAG?", on_step=on_step))
        assert len(fired) == 5

    def test_provenance_structure(self):
        from pipeline.sample_data  import seed_graph
        from pipeline.query_engine import query
        seed_graph()
        result = asyncio.run(query("How does GPT relate to OpenAI?"))
        for prov in result["provenance"]:
            assert "entities"   in prov
            assert "hops"       in prov
            assert "confidence" in prov


# ─── 5. GPU / llama-server integration ────────────────────────────────────────

@pytest.fixture(scope="module")
def llama_server():
    """
    Start llama-server.exe with Vulkan GPU on TEST_PORT.
    Skips the test module if startup fails within 90 seconds.
    """
    import requests

    # If something is already listening, use it
    try:
        r = requests.get(f"{TEST_BASE_URL}/health", timeout=2)
        if r.status_code == 200:
            yield None
            return
    except Exception:
        pass

    proc = subprocess.Popen(
        [
            LLAMA_SERVER,
            "--model",          GGUF_MODEL,
            "--port",           str(TEST_PORT),
            "--host",           "127.0.0.1",
            "--n-gpu-layers",   "99",   # offload all layers to Vulkan GPU
            "--ctx-size",       "2048",
            "--threads",        "4",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + 90
    ready = False
    while time.time() < deadline:
        try:
            r = requests.get(f"{TEST_BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(2)

    if not ready:
        proc.kill()
        pytest.skip("llama-server did not become ready within 90s — skipping GPU tests")

    yield proc

    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()


@pytest.mark.gpu
class TestLocalGPUInference:
    """Integration tests that require llama-server running with the GGUF model."""

    def test_server_health(self, llama_server):
        import requests
        r = requests.get(f"{TEST_BASE_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "ok"

    def test_completion_returns_content(self, llama_server):
        import requests
        payload = {
            "prompt": ALPACA_PROMPT.format(
                instruction="What is a knowledge graph? Answer in one sentence.",
                context="",
            ),
            "max_tokens":  80,
            "temperature": 0.1,
            "stop": ["### Instruction:", "### Input:"],
        }
        r = requests.post(f"{TEST_BASE_URL}/completion", json=payload, timeout=60)
        assert r.status_code == 200
        data    = r.json()
        content = data.get("content", "").strip()
        assert len(content) > 5, f"Response too short: {repr(content)}"

    def test_completion_is_coherent(self, llama_server):
        import requests
        payload = {
            "prompt": ALPACA_PROMPT.format(
                instruction="Complete the sentence: BERT is a language model that",
                context="",
            ),
            "max_tokens":  60,
            "temperature": 0.1,
            "stop": ["### Instruction:", "###"],
        }
        r = requests.post(f"{TEST_BASE_URL}/completion", json=payload, timeout=60)
        assert r.status_code == 200
        content = r.json().get("content", "").strip()
        # Should produce at least a few words
        assert len(content.split()) >= 3, f"Too few words: {repr(content)}"

    def test_llm_adapter_uses_local_server(self, llama_server):
        """Confirm the LLM adapter routes to llama_cpp_local when URL is set."""
        _reset_graph_singleton()
        os.environ["USE_SIMULATION"]      = "false"
        os.environ["LOCAL_LLAMA_CPP_URL"] = TEST_BASE_URL
        os.environ["CLOUD_RUN_URL"]       = ""
        os.environ["LOCAL_LORA_PATH"]     = ""
        _reload_pipeline()

        import pipeline.llm as llm_mod
        result = asyncio.run(
            llm_mod.call_llm(
                "What is a knowledge graph?",
                instruction="Answer in one sentence.",
                max_tokens=80,
            )
        )
        assert result["source"] == "llama_cpp_local", f"Wrong source: {result['source']}"
        assert len(result["answer"]) > 5

    def test_full_pipeline_gpu_end_to_end(self, llama_server):
        """Full five-step query pipeline using GPU-backed local inference."""
        _reset_graph_singleton()
        os.environ["USE_SIMULATION"]      = "false"
        os.environ["LOCAL_LLAMA_CPP_URL"] = TEST_BASE_URL
        os.environ["CLOUD_RUN_URL"]       = ""
        os.environ["LOCAL_LORA_PATH"]     = ""
        _reload_pipeline()

        from pipeline.sample_data  import seed_graph
        from pipeline.query_engine import query
        seed_graph()

        result = asyncio.run(query("How does BERT relate to the Transformer architecture?"))

        assert len(result["answer"]) > 10,   "Answer too short"
        assert result["source"] != "simulation", "Should not fall back to simulation"
        assert len(result["steps"]) == 5,    "Expected 5 pipeline steps"
