"""
Pydantic models for GraphRAG++ API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw text to ingest")
    source: str = Field("manual", description="Source identifier")
    title: Optional[str] = None


class IngestURLRequest(BaseModel):
    url: str
    source: str = "url"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    max_hops: int = Field(3, ge=1, le=6)


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.85


class IngestResponse(BaseModel):
    status: str
    triples_extracted: int
    nodes_added: int
    edges_added: int
    version: int
    triples: List[Triple]
    source: str


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


class QueryStep(BaseModel):
    step: str
    label: str
    icon: str
    result: Dict[str, Any]
    thinking: Optional[str] = None
    ms: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    thinking: Optional[str] = None
    intent: Dict[str, Any]
    steps: List[Dict[str, Any]]
    paths: List[Any]
    total_ms: int
    source: str
    provenance: List[Dict[str, Any]]


class MetricsResponse(BaseModel):
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    version: int
    ingestion_count: int
    query_count: int
    entity_types: Dict[str, int]
    uptime_s: float
    llm_source: str
