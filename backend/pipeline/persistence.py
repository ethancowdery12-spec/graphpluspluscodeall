"""
Graph Persistence — atomic JSON save/load for GraphStore.
Writes to a .tmp file then os.replace() so a crash never corrupts the store.
"""
import os
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_builder import GraphStore

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.getenv("GRAPH_STORE_PATH", "data/graph_store.json"))


def save_graph(graph_store: "GraphStore") -> None:
    """Serialize GraphStore to JSON on disk. Called after every ingest."""
    try:
        import networkx as nx
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "graph": nx.node_link_data(graph_store.G),
            "entity_index": graph_store._entity_index,
            "ingestion_log": graph_store._ingestion_log[-500:],
            "metrics": graph_store._metrics,
            "version": graph_store._version,
            "file_hashes": graph_store._file_hashes,
        }

        tmp_path = str(STORE_PATH) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, STORE_PATH)   # atomic on NTFS + ext4
        logger.debug(f"[Persistence] Saved {graph_store.G.number_of_nodes()} nodes to {STORE_PATH}")
    except Exception as e:
        logger.warning(f"[Persistence] Save failed: {e}")


def load_graph(graph_store: "GraphStore") -> bool:
    """
    Load persisted state into an existing GraphStore in-place.
    Returns True if a store was found and loaded, False if starting fresh.
    """
    if not STORE_PATH.exists():
        return False
    try:
        import networkx as nx
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # node_link_graph needs directed=True, multigraph=True for MultiDiGraph
        graph_store.G = nx.node_link_graph(
            payload["graph"], directed=True, multigraph=True
        )
        graph_store._entity_index = payload.get("entity_index", {})
        graph_store._ingestion_log = payload.get("ingestion_log", [])
        graph_store._metrics      = {**graph_store._metrics, **payload.get("metrics", {})}
        graph_store._version      = payload.get("version", 0)
        graph_store._file_hashes  = payload.get("file_hashes", {})

        n = graph_store.G.number_of_nodes()
        e = graph_store.G.number_of_edges()
        logger.info(f"[Persistence] Loaded {n} nodes, {e} edges from {STORE_PATH}")
        return True
    except Exception as exc:
        logger.warning(f"[Persistence] Load failed ({exc}) — starting fresh.")
        return False
