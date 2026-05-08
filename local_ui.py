"""
GraphRAG++ Local Interface

Replaces the Next.js web UI with a pure-Python Gradio app.
Starts llama-server.exe with Vulkan GPU automatically, then wires the full
pipeline (graph builder → embedder → query engine) directly — no FastAPI
process needed.

Run:  python local_ui.py
"""
import os
import sys
import time
import threading
import subprocess
import asyncio
import importlib

ROOT    = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

# Try the patched GGUF first (qwen35 → qwen2 keys renamed),
# then the original, then any other Qwen2 GGUF in the directory.
GGUF_CANDIDATES = [
    os.path.join(ROOT, "graphrag-plus-plus-qwen35-4b-q3_k_m-patched.gguf"),
    os.path.join(ROOT, "graphrag-plus-plus-qwen35-4b-q3_k_m.gguf"),
    os.path.join(ROOT, "test-qwen-0.5b.gguf"),
]
GGUF_MODEL = next((p for p in GGUF_CANDIDATES if os.path.exists(p)), GGUF_CANDIDATES[0])

# Prefer llama-latest (has full CPU backend DLLs), fall back to llama-fresh, then root.
LLAMA_SERVER_CANDIDATES = [
    os.path.join(ROOT, "llama-latest", "llama-server.exe"),
    os.path.join(ROOT, "llama-fresh",  "llama-server.exe"),
    os.path.join(ROOT, "llama-server.exe"),
]
LLAMA_SERVER = next(
    (p for p in LLAMA_SERVER_CANDIDATES if os.path.exists(p)),
    LLAMA_SERVER_CANDIDATES[-1],
)
LLAMA_PORT = 8080
LLAMA_URL  = f"http://127.0.0.1:{LLAMA_PORT}"

# ─── startup state ─────────────────────────────────────────────────────────────

_llama_proc   = None
_boot_log     = []
_boot_done    = False
_boot_ok      = False


def _log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _boot_log.append(line)
    print(line, flush=True)


def _check_server_up() -> bool:
    import requests
    try:
        r = requests.get(f"{LLAMA_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _start_llama_server() -> bool:
    """Launch llama-server.exe with Vulkan GPU offload, wait until ready."""
    global _llama_proc

    if not os.path.exists(LLAMA_SERVER):
        _log(f"ERROR: llama-server.exe not found at {LLAMA_SERVER}")
        return False
    if not os.path.exists(GGUF_MODEL):
        _log(f"ERROR: GGUF model not found at {GGUF_MODEL}")
        return False

    if _check_server_up():
        _log("llama-server already running on port 8080 — reusing")
        return True

    _log("Starting llama-server.exe with Vulkan GPU (--n-gpu-layers 99)...")
    server_cwd = os.path.dirname(LLAMA_SERVER) or ROOT
    _log(f"  using server: {LLAMA_SERVER}")
    _log(f"  using model:  {os.path.basename(GGUF_MODEL)}")
    _llama_proc = subprocess.Popen(
        [
            LLAMA_SERVER,
            "--model",         GGUF_MODEL,
            "--port",          str(LLAMA_PORT),
            "--host",          "127.0.0.1",
            "--n-gpu-layers",  "99",    # offload everything to Vulkan GPU
            "--ctx-size",      "2048",
            "--threads",       "6",
        ],
        cwd=server_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream server logs to console so GPU/layer info is visible
    def _stream():
        for line in _llama_proc.stdout:
            stripped = line.rstrip()
            if stripped:
                _log(f"  llama-server: {stripped}")
    threading.Thread(target=_stream, daemon=True).start()

    _log("Waiting for model to load (up to 90 s)...")
    deadline = time.time() + 90
    while time.time() < deadline:
        if _check_server_up():
            _log("llama-server ready — model loaded on GPU")
            return True
        time.sleep(2)

    _log("ERROR: llama-server did not become ready in 90 s")
    return False


def _init_pipeline():
    """Seed the graph and embed nodes; must run after env vars are set."""
    _log("Seeding knowledge graph with sample AI/ML data...")
    from pipeline.sample_data   import seed_graph
    result = seed_graph()
    _log(f"Graph seeded: {result}")

    _log("Embedding graph nodes into FAISS index...")
    from pipeline.graph_builder import get_graph
    from pipeline.embedder      import embed_text, add_to_index
    graph  = get_graph()
    nodes  = [(nid, d.get("label", nid)) for nid, d in graph.G.nodes(data=True)]

    batch = 32
    for i in range(0, len(nodes), batch):
        chunk  = nodes[i : i + batch]
        labels = [c[1] for c in chunk]
        try:
            loop   = asyncio.new_event_loop()
            embs   = loop.run_until_complete(embed_text(labels))
            loop.close()
            for (nid, _), emb in zip(chunk, embs):
                graph.set_embedding(nid, emb)
                add_to_index(nid, emb)
        except Exception as e:
            _log(f"  Embedding batch {i//batch} failed: {e}")

    stats = graph.get_stats()
    _log(f"Ready: {stats['node_count']} nodes, {stats['edge_count']} edges")


def _boot():
    """Full startup sequence, runs once in a background thread."""
    global _boot_done, _boot_ok

    _log("=== GraphRAG++ Local UI booting ===")

    ok = _start_llama_server()
    if not ok:
        _log("WARNING: GPU inference unavailable — falling back to simulation")
        os.environ["USE_SIMULATION"]      = "true"
        os.environ["LOCAL_LLAMA_CPP_URL"] = ""
    else:
        os.environ["USE_SIMULATION"]      = "false"
        os.environ["LOCAL_LLAMA_CPP_URL"] = LLAMA_URL
        os.environ["CLOUD_RUN_URL"]       = ""
        os.environ["LOCAL_LORA_PATH"]     = ""

    # Reload pipeline with updated env
    for key in list(sys.modules):
        if key.startswith("pipeline"):
            del sys.modules[key]

    _init_pipeline()
    _boot_done = True
    _boot_ok   = ok
    _log("=== Boot complete — UI is ready ===")


# Start boot in background so Gradio can show a loading state
_boot_thread = threading.Thread(target=_boot, daemon=True)
_boot_thread.start()


# ─── query helpers ──────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from a sync Gradio handler."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _graph_stats_md() -> str:
    try:
        from pipeline.graph_builder import get_graph
        s = get_graph().get_stats()
        entity_types = s.get("entity_types", {})
        type_lines   = "\n".join(
            f"  - {t}: {c}" for t, c in sorted(entity_types.items(), key=lambda x: -x[1])
        )
        return (
            f"### Knowledge Graph\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Nodes | **{s['node_count']}** |\n"
            f"| Edges | **{s['edge_count']}** |\n"
            f"| Queries run | {s.get('total_queries', 0)} |\n"
            f"| Version | {s.get('version', 0)} |\n\n"
            f"**Entity types:**\n{type_lines}"
        )
    except Exception as e:
        return f"_Graph not ready: {e}_"


def _system_status_md() -> str:
    server_status = "Running" if _check_server_up() else "Offline"
    inference     = "GPU (Vulkan)" if _check_server_up() else (
                    "Simulation" if os.environ.get("USE_SIMULATION") == "true" else "Unknown"
                  )
    boot_status   = "Ready" if _boot_done else "Booting..."
    return (
        f"| | |\n|---|---|\n"
        f"| **Boot** | {boot_status} |\n"
        f"| **llama-server** | {server_status} |\n"
        f"| **Inference** | {inference} |\n"
        f"| **Model** | graphrag-plus-plus-qwen35-4b-q3_k_m |\n"
        f"| **GPU** | AMD Radeon 840M (Vulkan) |\n"
        f"| **Context** | 4096 tokens |\n"
    )


# ─── Gradio UI ─────────────────────────────────────────────────────────────────

import gradio as gr


def do_query(question: str, history: list):
    """Main query handler — runs the full 5-step pipeline."""
    if not question.strip():
        return history, "", "", "", ""

    if not _boot_done:
        history.append({"role": "assistant", "content": "Still booting — wait a moment and try again."})
        return history, "", "", "", ""

    history.append({"role": "user", "content": question})

    step_lines  = []
    path_lines  = []
    thinking    = ""

    async def _query():
        from pipeline.query_engine import query

        async def on_step(name, data):
            label = data.get("label", name)
            ms    = data.get("ms", 0)
            step_lines.append(f"✓  {label}  ({ms} ms)")

        return await query(question, on_step=on_step)

    try:
        result  = _run_async(_query())
        answer  = result.get("answer", "").strip() or "_(no answer generated)_"
        thinking = result.get("thinking", "").strip()
        source   = result.get("source", "unknown")
        total_ms = result.get("total_ms", 0)

        # Format provenance paths
        for p in result.get("provenance", []):
            entities = " → ".join(p.get("entities", []))
            conf     = p.get("confidence", 0)
            hops     = p.get("hops", 0)
            path_lines.append(f"Path {p['path_id']}: {entities}  ({hops} hops, conf {conf:.2f})")

        badge = f"[{source} • {total_ms} ms]"
        history.append({"role": "assistant", "content": f"**{badge}**\n\n{answer}"})

    except Exception as e:
        history.append({"role": "assistant", "content": f"**Error:** {e}"})

    trace_text = "\n".join(step_lines) if step_lines else "No steps recorded."
    path_text  = "\n".join(path_lines) if path_lines else "No paths found."

    return (
        history,
        trace_text,
        thinking,
        path_text,
        _graph_stats_md(),
    )


def do_ingest(raw_text: str, source_name: str):
    """Extract triples from text and add them to the graph."""
    if not raw_text.strip():
        return "Nothing to ingest.", _graph_stats_md()

    if not _boot_done:
        return "Still booting — try again in a moment.", _graph_stats_md()

    async def _ingest():
        from pipeline.extractor     import extract_triples
        from pipeline.graph_builder import get_graph
        from pipeline.embedder      import embed_text, add_to_index

        triples = await extract_triples(raw_text[:8000])
        graph   = get_graph()
        result  = graph.ingest_triples(triples, source=source_name or "manual")

        # Embed new nodes
        nodes  = [(nid, d.get("label", nid)) for nid, d in graph.G.nodes(data=True)]
        labels = [n[1] for n in nodes[-result["nodes_added"]:]] if result["nodes_added"] else []
        if labels:
            embs = await embed_text(labels)
            for (nid, _), emb in zip(nodes[-result["nodes_added"]:], embs):
                graph.set_embedding(nid, emb)
                add_to_index(nid, emb)

        return triples, result

    try:
        triples, result = _run_async(_ingest())
        lines = [
            f"Extracted {len(triples)} triples",
            f"Nodes added: {result['nodes_added']}",
            f"Edges added: {result['edges_added']}",
            f"Graph version: {result['version']}",
            "",
            "Triples:",
        ]
        for t in triples[:20]:
            lines.append(f"  {t['subject']}  --[{t['predicate']}]-->  {t['object']}  (conf {t.get('confidence', 0):.2f})")
        if len(triples) > 20:
            lines.append(f"  ... and {len(triples)-20} more")
        return "\n".join(lines), _graph_stats_md()
    except Exception as e:
        return f"Ingest error: {e}", _graph_stats_md()


def do_reset():
    """Reset and re-seed the graph."""
    try:
        import pipeline.graph_builder as gb
        gb._graph_store = None
        _init_pipeline()
        return "Graph reset and re-seeded.", _graph_stats_md()
    except Exception as e:
        return f"Reset error: {e}", _graph_stats_md()


def get_boot_log() -> str:
    return "\n".join(_boot_log[-60:]) if _boot_log else "Booting..."


# ─── build UI ──────────────────────────────────────────────────────────────────

CSS = """
#chatbot { font-size: 14px; }
.step-box textarea { font-family: monospace; font-size: 13px; }
.thinking-box textarea { font-family: monospace; font-size: 12px; color: #888; }
"""

with gr.Blocks(
    title="GraphRAG++ Local",
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    css=CSS,
) as demo:

    gr.Markdown(
        "## GraphRAG++ — Local Interface\n"
        "Knowledge graph reasoning with GPU inference via llama.cpp + Vulkan.\n"
        "No cloud. No Next.js. Just the pipeline."
    )

    with gr.Tabs():

        # ── Query tab ────────────────────────────────────────────────────────
        with gr.Tab("Query"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        type="messages",
                        height=420,
                        elem_id="chatbot",
                    )
                    with gr.Row():
                        query_box = gr.Textbox(
                            placeholder="Ask anything about the knowledge graph...",
                            label="",
                            scale=5,
                            lines=1,
                        )
                        submit_btn = gr.Button("Query ▶", variant="primary", scale=1)

                    gr.Examples(
                        examples=[
                            ["How does BERT relate to the Transformer architecture?"],
                            ["What does LoRA enable in fine-tuning?"],
                            ["Trace the path from Self-Attention to Modern NLP"],
                            ["Compare RAG and GraphRAG"],
                            ["What is the relationship between GPT and OpenAI?"],
                        ],
                        inputs=query_box,
                    )

                with gr.Column(scale=2):
                    stats_md = gr.Markdown(value=_graph_stats_md(), label="Graph")
                    status_md = gr.Markdown(value=_system_status_md())
                    refresh_btn = gr.Button("Refresh Stats", size="sm")

            with gr.Row():
                with gr.Column():
                    trace_box = gr.Textbox(
                        label="Pipeline trace",
                        lines=6,
                        interactive=False,
                        elem_classes=["step-box"],
                    )
                with gr.Column():
                    thinking_box = gr.Textbox(
                        label="Model thinking  (<think> block)",
                        lines=6,
                        interactive=False,
                        elem_classes=["thinking-box"],
                    )

            path_box = gr.Textbox(
                label="Knowledge graph paths used",
                lines=4,
                interactive=False,
            )

            # Wire up
            submit_btn.click(
                fn=do_query,
                inputs=[query_box, chatbot],
                outputs=[chatbot, trace_box, thinking_box, path_box, stats_md],
            )
            query_box.submit(
                fn=do_query,
                inputs=[query_box, chatbot],
                outputs=[chatbot, trace_box, thinking_box, path_box, stats_md],
            )
            refresh_btn.click(
                fn=lambda: (_graph_stats_md(), _system_status_md()),
                outputs=[stats_md, status_md],
            )

        # ── Ingest tab ───────────────────────────────────────────────────────
        with gr.Tab("Ingest text"):
            gr.Markdown(
                "Paste any text and GraphRAG++ will extract knowledge triples "
                "and add them to the graph."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    ingest_text_box = gr.Textbox(
                        label="Text",
                        placeholder="Paste a paragraph, paper abstract, doc excerpt...",
                        lines=8,
                    )
                    source_box = gr.Textbox(label="Source label", value="manual", lines=1)
                    ingest_btn = gr.Button("Extract & Ingest", variant="primary")
                with gr.Column(scale=1):
                    ingest_result_box = gr.Textbox(
                        label="Result",
                        lines=20,
                        interactive=False,
                    )
                    ingest_stats = gr.Markdown()

            ingest_btn.click(
                fn=do_ingest,
                inputs=[ingest_text_box, source_box],
                outputs=[ingest_result_box, ingest_stats],
            )

        # ── Graph management tab ─────────────────────────────────────────────
        with gr.Tab("Graph"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Graph management")
                    reset_btn    = gr.Button("Reset & Re-seed graph", variant="stop")
                    reset_status = gr.Textbox(label="Status", lines=2, interactive=False)
                    graph_stats2 = gr.Markdown(value=_graph_stats_md())
                    reset_btn.click(
                        fn=do_reset,
                        outputs=[reset_status, graph_stats2],
                    )

        # ── Boot log tab ──────────────────────────────────────────────────────
        with gr.Tab("Boot log"):
            gr.Markdown("Live startup log — GPU layer offload info appears here.")
            log_box = gr.Textbox(
                value=get_boot_log,
                label="",
                lines=25,
                interactive=False,
                every=3,  # auto-refresh every 3 s while booting
            )


# ─── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GraphRAG++ Local UI")
    print("  Starting llama-server in background...")
    print("  Interface will open at http://127.0.0.1:7860")
    print("=" * 60 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )
