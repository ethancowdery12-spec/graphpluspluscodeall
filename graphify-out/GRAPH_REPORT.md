# Graph Report - GraphRAG  (2026-05-14)

## Corpus Check
- 60 files · ~558,703 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1055 nodes · 1280 edges · 42 communities (35 shown, 7 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 75 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b0551c76`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 39|Community 39]]

## God Nodes (most connected - your core abstractions)
1. `entity_index` - 494 edges
2. `GraphStore` - 24 edges
3. `get_graph()` - 22 edges
4. `query()` - 22 edges
5. `seed_graph()` - 17 edges
6. `compilerOptions` - 16 edges
7. `TestGraphBuilder` - 13 edges
8. `extract_code_triples()` - 12 edges
9. `call_llm()` - 12 edges
10. `GraphRAG++ — Claude Context File` - 12 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `get_chunk_store()`  [INFERRED]
  chunk_mythos.py → backend/pipeline/chunk_store.py
- `_init_pipeline()` --calls--> `seed_graph()`  [INFERRED]
  local_ui.py → backend/pipeline/sample_data.py
- `_init_pipeline()` --calls--> `get_graph()`  [INFERRED]
  local_ui.py → backend/pipeline/graph_builder.py
- `_graph_stats_md()` --calls--> `get_graph()`  [INFERRED]
  local_ui.py → backend/pipeline/graph_builder.py
- `TestFileExistence` --uses--> `GraphStore`  [INFERRED]
  test_local.py → backend/pipeline/graph_builder.py

## Communities (42 total, 7 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.0
Nodes (494): entity_index, 175b parameters, 2 destructiveness evaluation, 4-bit quantization, 95, a combination of opaque reasoni, a small number of\nexperimental, above (+486 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (36): llama_server(), GraphRAG++ local pipeline tests.  Tests cover:   - File existence (GGUF, llama-s, Start llama-server.exe with Vulkan GPU on TEST_PORT.     Skips the test module i, Integration tests that require llama-server running with the GGUF model., Confirm the LLM adapter routes to llama_cpp_local when URL is set., Full five-step query pipeline using GPU-backed local inference., Wipe the module-level graph singleton so each test starts fresh., Reload all pipeline sub-modules so env-var changes take effect. (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (49): collect(), collect2(), Full test suite for all new backend functions. Run with: python test_all.py, _route(), extract_code_triples(), _extract_go(), _extract_java(), _extract_js() (+41 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (40): _boot(), _check_server_up(), do_ingest(), do_query(), do_reset(), _graph_stats_md(), _init_pipeline(), _log() (+32 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (14): Pure in-memory tests — no LLM, no server., TestGraphBuilder, _cosine_sim(), _count_types(), GraphStore, _infer_entity_type(), Knowledge Graph Builder NetworkX multigraph with PageRank, Louvain community det, Add a directed relation edge with confidence tier. (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (38): Add a new pipeline step, Architecture Overview, Check what the LLM is actually sending/receiving, Chunk Store (`pipeline/chunk_store.py`), code:block1 (frontend/           Next.js 16 + React 19 + D3.js + Chart.js), code:bash (curl -X POST http://localhost:8000/graph/reset), code:powershell (python -m py_compile backend/pipeline/query_engine.py), code:powershell (.\start.ps1) (+30 more)

### Community 6 - "Community 6"
Cohesion: 0.11
Nodes (36): _broadcast(), chunk_stats(), _embed_all_nodes(), get_graph_data(), get_metrics(), get_stats(), health(), ingest_chunks_only() (+28 more)

### Community 7 - "Community 7"
Cohesion: 0.08
Nodes (31): GraphResponse, IngestDirRequest, IngestDirResponse, IngestRequest, IngestResponse, IngestURLRequest, MetricsResponse, PassageHit (+23 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (26): export(), _get_training_data(), GraphRAG++ Fine-Tuning on Modal  (A100 80GB) ===================================, Return the full GraphRAG++ training dataset., Fine-tune Qwen3.5-4B on GraphRAG++ data, then export GGUF., Manually export the model to GGUF, specifically bypassing multimodal/vision erro, Test the fine-tuned model on batch of 150+ fresh examples across all 6 categorie, test_batch() (+18 more)

### Community 9 - "Community 9"
Cohesion: 0.09
Nodes (21): dependencies, chart.js, d3, framer-motion, next, react, react-chartjs-2, react-dom (+13 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (21): build_alpaca_prompt(), _call_cloud_run(), _call_gradio(), _call_hf_api(), _call_llama_cpp(), call_llm(), _call_local(), _get_hf_client() (+13 more)

### Community 11 - "Community 11"
Cohesion: 0.14
Nodes (16): ACCEPT, Mode, Props, TIER_COLORS, Mode, GraphEdge, HopPath, ingestDirectory() (+8 more)

### Community 12 - "Community 12"
Cohesion: 0.1
Nodes (19): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+11 more)

### Community 13 - "Community 13"
Cohesion: 0.14
Nodes (9): InferenceLoaderProps, PHASES, createWebSocket(), QueryResult, runQuery(), WSHandle, EXAMPLE_QUERIES, GraphViewer (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.17
Nodes (9): Props, TIER_OPACITY, TIER_STROKE, GraphViewer, TIER_COLORS, TIER_DESC, fetchGraph(), GraphData (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (7): BENCHMARK_LABELS, GRAPHRAG_SCORES, LATENCY_DIST, LATENCY_LABELS, MiniChart, RAG_SCORES, fetchMetrics()

### Community 16 - "Community 16"
Cohesion: 0.29
Nodes (6): graph, directed, graph, links, multigraph, nodes

### Community 17 - "Community 17"
Cohesion: 0.29
Nodes (4): PERF_DATA, fetchStats(), GraphStats, reseedGraph()

### Community 18 - "Community 18"
Cohesion: 0.29
Nodes (3): metadata, LINKS, fetchHealth()

### Community 19 - "Community 19"
Cohesion: 0.57
Nodes (6): patch(), patch_gguf.py — Fix qwen35 GGUF metadata so llama.cpp can load it.  The model wa, read_str(), read_value(), write_str(), write_value()

### Community 20 - "Community 20"
Cohesion: 0.38
Nodes (6): main(), Upload fine-tuned GraphRAG++ model to HuggingFace Hub ==========================, Upload LoRA adapter to HuggingFace Hub., Upload GGUF quantized model to HuggingFace Hub., upload_gguf(), upload_lora()

### Community 21 - "Community 21"
Cohesion: 0.4
Nodes (4): code:bash (npm run dev), Deploy on Vercel, Getting Started, Learn More

### Community 24 - "Community 24"
Cohesion: 0.4
Nodes (4): handler(), Download and load models on container start., RunPod endpoint handler logic.     Accepts job["input"] with the payload., setup_models()

## Knowledge Gaps
- **710 isolated node(s):** `Retroactively chunk the Claude Mythos PDF into the ChunkStore. Runs pdfplumber p`, `GraphRAG++ Fine-Tuning on Modal  (A100 80GB) ===================================`, `Return the full GraphRAG++ training dataset.`, `Fine-tune Qwen3.5-4B on GraphRAG++ data, then export GGUF.`, `Manually export the model to GGUF, specifically bypassing multimodal/vision erro` (+705 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `entity_index` connect `Community 0` to `Community 16`?**
  _High betweenness centrality (0.237) - this node is a cross-community bridge._
- **Why does `get_graph()` connect `Community 6` to `Community 1`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `GraphStore` connect `Community 4` to `Community 1`, `Community 6`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `GraphStore` (e.g. with `TestFileExistence` and `TestGraphBuilder`) actually correct?**
  _`GraphStore` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `get_graph()` (e.g. with `_init_pipeline()` and `_graph_stats_md()`) actually correct?**
  _`get_graph()` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `query()` (e.g. with `.test_full_pipeline_five_steps()` and `.test_pipeline_step_names()`) actually correct?**
  _`query()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `seed_graph()` (e.g. with `_init_pipeline()` and `.test_multi_hop_paths_seeded()`) actually correct?**
  _`seed_graph()` has 14 INFERRED edges - model-reasoned connections that need verification._