"""
Sample data seeder — AI/ML research papers and real codebases.
Seeds the knowledge graph with ~150 nodes and ~300 edges.
"""
from .graph_builder import get_graph

# ─── AI/ML Research Paper Triples ─────────────────────────────────────────────
PAPER_TRIPLES = [
    # Attention Is All You Need
    {"subject": "Transformer", "predicate": "introduced_by", "object": "Vaswani et al. 2017", "confidence": 0.99},
    {"subject": "Transformer", "predicate": "uses", "object": "Self-Attention Mechanism", "confidence": 0.99},
    {"subject": "Self-Attention Mechanism", "predicate": "enables", "object": "Parallel Sequence Processing", "confidence": 0.97},
    {"subject": "Self-Attention Mechanism", "predicate": "replaces", "object": "Recurrent Neural Networks", "confidence": 0.95},
    {"subject": "Transformer", "predicate": "achieves", "object": "State-of-the-Art Translation", "confidence": 0.94},
    {"subject": "Transformer", "predicate": "uses", "object": "Positional Encoding", "confidence": 0.98},
    {"subject": "Transformer", "predicate": "consists_of", "object": "Encoder-Decoder Architecture", "confidence": 0.97},
    
    # BERT
    {"subject": "BERT", "predicate": "based_on", "object": "Transformer", "confidence": 0.99},
    {"subject": "BERT", "predicate": "introduced_by", "object": "Devlin et al. 2019", "confidence": 0.99},
    {"subject": "BERT", "predicate": "uses", "object": "Masked Language Modeling", "confidence": 0.98},
    {"subject": "BERT", "predicate": "uses", "object": "Next Sentence Prediction", "confidence": 0.96},
    {"subject": "BERT", "predicate": "trained_on", "object": "BookCorpus", "confidence": 0.95},
    {"subject": "BERT", "predicate": "trained_on", "object": "English Wikipedia", "confidence": 0.95},
    {"subject": "BERT", "predicate": "achieves", "object": "GLUE Benchmark SOTA", "confidence": 0.93},
    {"subject": "BERT", "predicate": "extended_by", "object": "RoBERTa", "confidence": 0.97},
    {"subject": "BERT", "predicate": "extended_by", "object": "DistilBERT", "confidence": 0.96},
    {"subject": "BERT", "predicate": "extended_by", "object": "ALBERT", "confidence": 0.95},
    
    # GPT Series
    {"subject": "GPT-4", "predicate": "developed_by", "object": "OpenAI", "confidence": 0.99},
    {"subject": "GPT-4", "predicate": "based_on", "object": "Transformer", "confidence": 0.99},
    {"subject": "GPT-4", "predicate": "uses", "object": "RLHF Training", "confidence": 0.97},
    {"subject": "GPT-4", "predicate": "outperforms", "object": "GPT-3.5", "confidence": 0.96},
    {"subject": "GPT-4", "predicate": "supports", "object": "Multi-modal Input", "confidence": 0.94},
    {"subject": "GPT-3", "predicate": "introduced", "object": "In-context Learning", "confidence": 0.97},
    {"subject": "GPT-3", "predicate": "has", "object": "175B Parameters", "confidence": 0.99},
    {"subject": "InstructGPT", "predicate": "uses", "object": "RLHF Training", "confidence": 0.98},
    {"subject": "InstructGPT", "predicate": "aligns", "object": "LLM with Human Preferences", "confidence": 0.96},
    
    # LLaMA
    {"subject": "LLaMA", "predicate": "released_by", "object": "Meta AI", "confidence": 0.99},
    {"subject": "LLaMA", "predicate": "based_on", "object": "Transformer", "confidence": 0.99},
    {"subject": "LLaMA-2", "predicate": "extends", "object": "LLaMA", "confidence": 0.97},
    {"subject": "LLaMA-2", "predicate": "supports", "object": "Chat Fine-tuning", "confidence": 0.95},
    {"subject": "Alpaca", "predicate": "fine-tuned_from", "object": "LLaMA", "confidence": 0.97},
    {"subject": "Alpaca", "predicate": "uses", "object": "Self-Instruct Data", "confidence": 0.94},
    {"subject": "Vicuna", "predicate": "fine-tuned_from", "object": "LLaMA", "confidence": 0.96},
    
    # RAG
    {"subject": "RAG", "predicate": "introduced_by", "object": "Lewis et al. 2020", "confidence": 0.98},
    {"subject": "RAG", "predicate": "combines", "object": "Dense Retrieval", "confidence": 0.97},
    {"subject": "RAG", "predicate": "combines", "object": "Generative LLM", "confidence": 0.97},
    {"subject": "RAG", "predicate": "uses", "object": "DPR Retriever", "confidence": 0.93},
    {"subject": "RAG", "predicate": "reduces", "object": "Hallucinations", "confidence": 0.89},
    {"subject": "GraphRAG", "predicate": "extends", "object": "RAG", "confidence": 0.96},
    {"subject": "GraphRAG", "predicate": "uses", "object": "Knowledge Graph", "confidence": 0.97},
    {"subject": "Knowledge Graph", "predicate": "enables", "object": "Multi-hop Reasoning", "confidence": 0.95},
    {"subject": "Knowledge Graph", "predicate": "stores", "object": "Entity-Relation Triples", "confidence": 0.98},
    
    # Fine-tuning Techniques
    {"subject": "LoRA", "predicate": "introduced_by", "object": "Hu et al. 2021", "confidence": 0.99},
    {"subject": "LoRA", "predicate": "enables", "object": "Parameter-Efficient Fine-tuning", "confidence": 0.97},
    {"subject": "LoRA", "predicate": "uses", "object": "Low-rank Decomposition", "confidence": 0.98},
    {"subject": "QLoRA", "predicate": "extends", "object": "LoRA", "confidence": 0.96},
    {"subject": "QLoRA", "predicate": "uses", "object": "4-bit Quantization", "confidence": 0.97},
    {"subject": "PEFT", "predicate": "includes", "object": "LoRA", "confidence": 0.95},
    {"subject": "PEFT", "predicate": "includes", "object": "Prefix Tuning", "confidence": 0.94},
    {"subject": "PEFT", "predicate": "developed_by", "object": "Hugging Face", "confidence": 0.95},
    {"subject": "Instruction Following", "predicate": "enabled_by", "object": "RLHF Training", "confidence": 0.94},
    
    # Chain-of-Thought
    {"subject": "Chain-of-Thought Prompting", "predicate": "introduced_by", "object": "Wei et al. 2022", "confidence": 0.98},
    {"subject": "Chain-of-Thought Prompting", "predicate": "improves", "object": "Mathematical Reasoning", "confidence": 0.95},
    {"subject": "Chain-of-Thought Prompting", "predicate": "improves", "object": "Multi-step Problem Solving", "confidence": 0.94},
    {"subject": "ReAct", "predicate": "combines", "object": "Chain-of-Thought Prompting", "confidence": 0.93},
    {"subject": "ReAct", "predicate": "combines", "object": "Tool Use", "confidence": 0.94},
    {"subject": "Tree of Thoughts", "predicate": "extends", "object": "Chain-of-Thought Prompting", "confidence": 0.92},
    
    # Embeddings
    {"subject": "Word2Vec", "predicate": "introduced_by", "object": "Mikolov et al. 2013", "confidence": 0.99},
    {"subject": "Word2Vec", "predicate": "produces", "object": "Word Embeddings", "confidence": 0.98},
    {"subject": "GloVe", "predicate": "alternative_to", "object": "Word2Vec", "confidence": 0.90},
    {"subject": "Sentence Transformers", "predicate": "produces", "object": "Sentence Embeddings", "confidence": 0.97},
    {"subject": "Sentence Transformers", "predicate": "based_on", "object": "BERT", "confidence": 0.95},
    {"subject": "Dense Retrieval", "predicate": "uses", "object": "Sentence Embeddings", "confidence": 0.94},
    {"subject": "FAISS", "predicate": "developed_by", "object": "Meta AI", "confidence": 0.96},
    {"subject": "FAISS", "predicate": "enables", "object": "Approximate Nearest Neighbor Search", "confidence": 0.97},
    
    # GNNs
    {"subject": "Graph Neural Networks", "predicate": "process", "object": "Graph-structured Data", "confidence": 0.97},
    {"subject": "GraphSAGE", "predicate": "type_of", "object": "Graph Neural Networks", "confidence": 0.97},
    {"subject": "GraphSAGE", "predicate": "uses", "object": "Neighborhood Sampling", "confidence": 0.95},
    {"subject": "GAT", "predicate": "type_of", "object": "Graph Neural Networks", "confidence": 0.96},
    {"subject": "GAT", "predicate": "uses", "object": "Attention Mechanism", "confidence": 0.95},
    {"subject": "GCN", "predicate": "type_of", "object": "Graph Neural Networks", "confidence": 0.96},
    
    # Scaling & MoE
    {"subject": "Mixture of Experts", "predicate": "enables", "object": "Sparse Activation", "confidence": 0.95},
    {"subject": "Mixture of Experts", "predicate": "improves", "object": "Scaling Efficiency", "confidence": 0.93},
    {"subject": "Mixtral-8x7B", "predicate": "uses", "object": "Mixture of Experts", "confidence": 0.97},
    {"subject": "Mixtral-8x7B", "predicate": "released_by", "object": "Mistral AI", "confidence": 0.98},
    {"subject": "Scaling Laws", "predicate": "described_by", "object": "Hoffmann et al. 2022", "confidence": 0.95},
    {"subject": "Scaling Laws", "predicate": "govern", "object": "LLM Training Efficiency", "confidence": 0.92},
    
    # Vector DBs & Tools
    {"subject": "Pinecone", "predicate": "type_of", "object": "Vector Database", "confidence": 0.95},
    {"subject": "Weaviate", "predicate": "type_of", "object": "Vector Database", "confidence": 0.94},
    {"subject": "Chroma", "predicate": "type_of", "object": "Vector Database", "confidence": 0.94},
    {"subject": "LangChain", "predicate": "framework_for", "object": "LLM Application Development", "confidence": 0.96},
    {"subject": "LlamaIndex", "predicate": "framework_for", "object": "RAG Pipelines", "confidence": 0.95},
    {"subject": "Haystack", "predicate": "framework_for", "object": "NLP Pipelines", "confidence": 0.91},
    
    # Benchmarks
    {"subject": "MMLU", "predicate": "evaluates", "object": "Multitask Language Understanding", "confidence": 0.97},
    {"subject": "HumanEval", "predicate": "evaluates", "object": "Code Generation", "confidence": 0.97},
    {"subject": "MATH Benchmark", "predicate": "evaluates", "object": "Mathematical Reasoning", "confidence": 0.96},
    {"subject": "BIG-Bench", "predicate": "evaluates", "object": "Diverse Reasoning Capabilities", "confidence": 0.94},
    {"subject": "GPT-4", "predicate": "achieves_high_score_on", "object": "MMLU", "confidence": 0.95},
    {"subject": "GPT-4", "predicate": "achieves_high_score_on", "object": "HumanEval", "confidence": 0.94},
]

# ─── Real Codebase Triples ─────────────────────────────────────────────────────
CODEBASE_TRIPLES = [
    # PyTorch
    {"subject": "PyTorch", "predicate": "developed_by", "object": "Meta AI", "confidence": 0.99},
    {"subject": "PyTorch", "predicate": "uses", "object": "Dynamic Computation Graphs", "confidence": 0.97},
    {"subject": "PyTorch", "predicate": "provides", "object": "Autograd Engine", "confidence": 0.97},
    {"subject": "PyTorch", "predicate": "competes_with", "object": "TensorFlow", "confidence": 0.90},
    {"subject": "TorchScript", "predicate": "part_of", "object": "PyTorch", "confidence": 0.95},
    {"subject": "CUDA Integration", "predicate": "enables", "object": "GPU Training in PyTorch", "confidence": 0.97},
    
    # Hugging Face Transformers
    {"subject": "Hugging Face Transformers", "predicate": "implements", "object": "BERT", "confidence": 0.97},
    {"subject": "Hugging Face Transformers", "predicate": "implements", "object": "GPT-2", "confidence": 0.97},
    {"subject": "Hugging Face Transformers", "predicate": "provides", "object": "Model Hub", "confidence": 0.96},
    {"subject": "Hugging Face Transformers", "predicate": "supports", "object": "PyTorch", "confidence": 0.96},
    {"subject": "Hugging Face Transformers", "predicate": "supports", "object": "TensorFlow", "confidence": 0.94},
    {"subject": "Hugging Face Transformers", "predicate": "enables", "object": "Fine-tuning", "confidence": 0.95},
    {"subject": "Accelerate", "predicate": "part_of", "object": "Hugging Face Transformers", "confidence": 0.93},
    {"subject": "Datasets Library", "predicate": "developed_by", "object": "Hugging Face", "confidence": 0.95},
    
    # vLLM
    {"subject": "vLLM", "predicate": "optimizes", "object": "LLM Inference", "confidence": 0.97},
    {"subject": "vLLM", "predicate": "uses", "object": "PagedAttention", "confidence": 0.97},
    {"subject": "PagedAttention", "predicate": "reduces", "object": "Memory Fragmentation", "confidence": 0.94},
    {"subject": "vLLM", "predicate": "achieves", "object": "High Throughput Serving", "confidence": 0.95},
    {"subject": "vLLM", "predicate": "supports", "object": "Continuous Batching", "confidence": 0.95},
    
    # llama.cpp
    {"subject": "llama.cpp", "predicate": "enables", "object": "CPU LLM Inference", "confidence": 0.97},
    {"subject": "llama.cpp", "predicate": "uses", "object": "GGUF Format", "confidence": 0.96},
    {"subject": "llama.cpp", "predicate": "implements", "object": "4-bit Quantization", "confidence": 0.95},
    {"subject": "Ollama", "predicate": "built_on", "object": "llama.cpp", "confidence": 0.92},
    {"subject": "Ollama", "predicate": "simplifies", "object": "Local LLM Deployment", "confidence": 0.93},
    
    # LangChain implementation
    {"subject": "LangChain", "predicate": "implements", "object": "ReAct Agent", "confidence": 0.93},
    {"subject": "LangChain", "predicate": "implements", "object": "RAG Pipeline", "confidence": 0.95},
    {"subject": "LangChain", "predicate": "integrates_with", "object": "Chroma", "confidence": 0.92},
    {"subject": "LangChain", "predicate": "integrates_with", "object": "Pinecone", "confidence": 0.92},
    {"subject": "LangChain Expression Language", "predicate": "part_of", "object": "LangChain", "confidence": 0.94},
    {"subject": "LangGraph", "predicate": "extends", "object": "LangChain", "confidence": 0.93},
    {"subject": "LangGraph", "predicate": "supports", "object": "Graph-based Agent Workflows", "confidence": 0.92},
    
    # NetworkX
    {"subject": "NetworkX", "predicate": "implements", "object": "PageRank Algorithm", "confidence": 0.97},
    {"subject": "NetworkX", "predicate": "implements", "object": "Louvain Community Detection", "confidence": 0.89},
    {"subject": "NetworkX", "predicate": "supports", "object": "Multi-hop Graph Traversal", "confidence": 0.95},
    {"subject": "NetworkX", "predicate": "used_for", "object": "Knowledge Graph Construction", "confidence": 0.91},
    
    # Neo4j
    {"subject": "Neo4j", "predicate": "is_a", "object": "Graph Database", "confidence": 0.99},
    {"subject": "Neo4j", "predicate": "uses", "object": "Cypher Query Language", "confidence": 0.98},
    {"subject": "Cypher Query Language", "predicate": "enables", "object": "Declarative Graph Queries", "confidence": 0.95},
    {"subject": "Neo4j", "predicate": "supports", "object": "ACID Transactions", "confidence": 0.95},
    {"subject": "Neo4j GDS", "predicate": "part_of", "object": "Neo4j", "confidence": 0.93},
    {"subject": "Neo4j GDS", "predicate": "implements", "object": "Graph Algorithms", "confidence": 0.94},
    
    # Ray
    {"subject": "Ray", "predicate": "enables", "object": "Distributed Python Computing", "confidence": 0.96},
    {"subject": "Ray", "predicate": "includes", "object": "Ray Train", "confidence": 0.94},
    {"subject": "Ray Train", "predicate": "supports", "object": "Distributed LLM Training", "confidence": 0.93},
    {"subject": "Ray Serve", "predicate": "enables", "object": "Scalable Model Serving", "confidence": 0.93},
    {"subject": "Ray", "predicate": "used_by", "object": "vLLM", "confidence": 0.9},
    
    # Kubernetes & MLOps
    {"subject": "Kubernetes", "predicate": "manages", "object": "Container Orchestration", "confidence": 0.98},
    {"subject": "KServe", "predicate": "built_on", "object": "Kubernetes", "confidence": 0.93},
    {"subject": "KServe", "predicate": "enables", "object": "ML Model Serving", "confidence": 0.94},
    {"subject": "MLflow", "predicate": "manages", "object": "ML Experiment Tracking", "confidence": 0.95},
    {"subject": "DVC", "predicate": "manages", "object": "Dataset Versioning", "confidence": 0.93},
    
    # Python AI/ML Stack
    {"subject": "NumPy", "predicate": "foundation_for", "object": "Scientific Python Stack", "confidence": 0.98},
    {"subject": "SciPy", "predicate": "built_on", "object": "NumPy", "confidence": 0.96},
    {"subject": "scikit-learn", "predicate": "built_on", "object": "NumPy", "confidence": 0.96},
    {"subject": "Pandas", "predicate": "used_for", "object": "Data Preprocessing", "confidence": 0.95},
    {"subject": "TensorFlow", "predicate": "developed_by", "object": "Google", "confidence": 0.98},
    {"subject": "JAX", "predicate": "developed_by", "object": "Google", "confidence": 0.97},
    {"subject": "Flax", "predicate": "built_on", "object": "JAX", "confidence": 0.95},
    
    # Unsloth (model's training tool)
    {"subject": "Unsloth", "predicate": "optimizes", "object": "LLM Fine-tuning Speed", "confidence": 0.96},
    {"subject": "Unsloth", "predicate": "reduces", "object": "VRAM Usage During Training", "confidence": 0.95},
    {"subject": "Unsloth", "predicate": "supports", "object": "LoRA Fine-tuning", "confidence": 0.97},
    {"subject": "Unsloth", "predicate": "used_to_train", "object": "Jackrong Qwen3.5 Model", "confidence": 0.99},
    {"subject": "Jackrong Qwen3.5 Model", "predicate": "based_on", "object": "Qwen3.5-27B", "confidence": 0.99},
    {"subject": "Jackrong Qwen3.5 Model", "predicate": "distilled_from", "object": "Claude Opus Reasoning", "confidence": 0.97},
    {"subject": "Jackrong Qwen3.5 Model", "predicate": "uses", "object": "Chain-of-Thought Prompting", "confidence": 0.98},
]


def seed_graph():
    """Seed the knowledge graph with AI/ML research and codebase data."""
    graph = get_graph()
    
    # Clear and reinitialize
    import networkx as nx
    graph.G = nx.MultiDiGraph()
    graph._entity_index = {}
    graph._embeddings = {}
    graph._version = 0
    graph._ingestion_log = []
    
    # Ingest paper triples
    r1 = graph.ingest_triples(PAPER_TRIPLES, source="ai_ml_papers")
    
    # Ingest codebase triples
    r2 = graph.ingest_triples(CODEBASE_TRIPLES, source="real_codebases")
    
    total_nodes = len(graph.G.nodes())
    total_edges = len(graph.G.edges())
    
    print(f"[Seeder] Graph seeded: {total_nodes} nodes, {total_edges} edges")
    return {
        "nodes": total_nodes,
        "edges": total_edges,
        "paper_triples": len(PAPER_TRIPLES),
        "codebase_triples": len(CODEBASE_TRIPLES),
    }
