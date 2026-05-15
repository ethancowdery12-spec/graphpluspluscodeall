"""
Tree-sitter AST Code Extractor
Converts source code into (subject, predicate, object) triples with
confidence_tier="EXTRACTED" — deterministic, no LLM involved.

Supports: Python, JavaScript, TypeScript, TSX, Go, Rust, Java, C, C++, Ruby
Falls back to regex heuristics if tree-sitter is unavailable.
"""
import re
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Extension → tree-sitter language name
LANG_MAP: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".go":   "go",
    ".rs":   "rust",
    ".java": "java",
    ".c":    "c",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".rb":   "ruby",
}

# Lazy parser cache {lang_name: Parser}
_parsers: dict = {}

# ─── Chunk extraction ─────────────────────────────────────────────────────────

# Node types whose source text forms one semantic chunk per instance
_CHUNK_NODE_TYPES: dict[str, set] = {
    "python":     {"function_definition", "async_function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "generator_function_declaration"},
    "tsx":        {"function_declaration", "class_declaration"},
    "typescript": {"function_declaration", "class_declaration", "interface_declaration"},
    "go":         {"function_declaration", "method_declaration", "type_declaration"},
    "rust":       {"function_item", "struct_item", "impl_item", "trait_item"},
    "java":       {"class_declaration", "method_declaration"},
}

_MAX_CHUNK_CHARS = 2000   # hard cap per chunk before add_text splits further


def extract_code_chunks(source: str, filename: str) -> list[dict]:
    """Return semantic code chunks (one per function/class) for the ChunkStore.

    Each chunk: {"text": str, "source": str, "page": 0}
    Falls back to line-based splitting when tree-sitter is unavailable.
    """
    ext  = Path(filename).suffix.lower()
    lang = LANG_MAP.get(ext)
    if lang is None:
        return _line_chunks(source, filename)

    parser = _get_parser(lang)
    if parser is None:
        return _line_chunks(source, filename)

    try:
        src_bytes = source.encode("utf-8", errors="replace")
        tree      = parser.parse(src_bytes)
        chunks    = _collect_chunks(tree.root_node, src_bytes, lang, filename)
        return chunks if chunks else _line_chunks(source, filename)
    except Exception as e:
        logger.debug(f"[CodeExtractor] chunk extraction failed for {filename}: {e}")
        return _line_chunks(source, filename)


def _collect_chunks(root, src: bytes, lang: str, filename: str) -> list[dict]:
    target_types = _CHUNK_NODE_TYPES.get(lang, set())
    if not target_types:
        return []

    chunks: list[dict] = []

    def visit(node, depth: int = 0):
        if depth > 4:
            return
        if node.type in target_types:
            name_node = node.child_by_field_name("name")
            name      = _text(name_node, src) if name_node else node.type
            body_text = src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            header    = f"# {name} [{filename}]\n"
            chunk_text = (header + body_text)[:_MAX_CHUNK_CHARS]
            chunks.append({"text": chunk_text, "source": filename, "page": 0})
            # Don't recurse into class/impl bodies — methods are already in the class chunk
            if node.type in ("class_definition", "class_declaration",
                             "impl_item", "interface_declaration"):
                return
        for child in node.children:
            visit(child, depth + 1)

    visit(root)
    return chunks


def _line_chunks(source: str, filename: str, lines_per_chunk: int = 60) -> list[dict]:
    lines  = source.splitlines()
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        text = "\n".join(lines[i : i + lines_per_chunk])
        if text.strip():
            chunks.append({"text": text, "source": filename, "page": 0})
    return chunks


def _get_parser(lang: str):
    """Return a cached tree-sitter parser for *lang*, or None on failure."""
    if lang in _parsers:
        return _parsers[lang]
    try:
        from tree_sitter_languages import get_parser
        p = get_parser(lang)
        _parsers[lang] = p
        return p
    except Exception as e:
        logger.debug(f"[CodeExtractor] tree-sitter parser unavailable for {lang}: {e}")
        _parsers[lang] = None
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_code_triples(source: str, filename: str) -> List[dict]:
    """
    Parse *source* as the language inferred from *filename*'s extension.
    Returns a list of triples, all with confidence_tier='EXTRACTED'.
    Falls back to regex extraction if tree-sitter fails.
    """
    ext = Path(filename).suffix.lower()
    lang = LANG_MAP.get(ext)
    if not lang:
        return []

    parser = _get_parser(lang)
    if parser is None:
        return _regex_extract(source, filename)

    try:
        source_bytes = source.encode("utf-8", errors="replace")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        if lang == "python":
            return _extract_python(root, source_bytes, filename)
        elif lang in ("javascript", "tsx"):
            return _extract_js(root, source_bytes, filename)
        elif lang == "typescript":
            return _extract_typescript(root, source_bytes, filename)
        elif lang == "go":
            return _extract_go(root, source_bytes, filename)
        elif lang == "rust":
            return _extract_rust(root, source_bytes, filename)
        elif lang == "java":
            return _extract_java(root, source_bytes, filename)
        else:
            return _generic_extract(root, source_bytes, filename)
    except Exception as e:
        logger.warning(f"[CodeExtractor] AST parse failed for {filename}: {e}")
        return _regex_extract(source, filename)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()

def _triple(subj: str, pred: str, obj: str, conf: float = 1.0) -> dict:
    return {
        "subject": subj,
        "predicate": pred,
        "object": obj,
        "confidence": conf,
        "confidence_tier": "EXTRACTED",
    }

def _modname(filename: str) -> str:
    """'src/pipeline/llm.py' → 'llm'"""
    return Path(filename).stem


# ─── Python ───────────────────────────────────────────────────────────────────

def _extract_python(root, src: bytes, filename: str) -> List[dict]:
    triples: List[dict] = []
    mod = _modname(filename)
    _visit_python(root, src, filename, mod, triples, parent_class=None)
    return triples


def _visit_python(node, src: bytes, filename: str, mod: str,
                  triples: list, parent_class: str | None,
                  _decorators: list | None = None):
    t = node.type

    if t == "decorated_definition":
        # Collect decorator names, then recurse into the wrapped definition
        decorators = [
            _text(c, src).lstrip("@").split("(")[0]
            for c in node.children if c.type == "decorator"
        ]
        for child in node.children:
            if child.type in ("function_definition", "async_function_definition",
                              "class_definition"):
                _visit_python(child, src, filename, mod, triples,
                              parent_class=parent_class, _decorators=decorators)
        return

    if t == "class_definition":
        name_node = node.child_by_field_name("name")
        if name_node:
            cls = _text(name_node, src)
            full = f"{mod}.{cls}"
            triples.append(_triple(full, "defined_in", filename))
            triples.append(_triple(full, "is_a", "class"))
            if parent_class:
                triples.append(_triple(full, "nested_in", parent_class))

            # Base classes
            args = node.child_by_field_name("superclasses") or node.child_by_field_name("arguments")
            if args:
                for child in args.named_children:
                    if child.type in ("identifier", "attribute"):
                        base = _text(child, src).split(".")[-1]
                        if base not in ("object", "Exception", "BaseException", "ABC"):
                            triples.append(_triple(full, "inherits_from", base, 0.99))

            # Recurse into body with this class as parent
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    _visit_python(child, src, filename, mod, triples, parent_class=full)
        return  # handled children above

    elif t in ("function_definition", "async_function_definition"):
        name_node = node.child_by_field_name("name")
        if name_node:
            fn = _text(name_node, src)
            if parent_class:
                full = f"{parent_class}.{fn}"
                triples.append(_triple(full, "belongs_to", parent_class))
            else:
                full = f"{mod}.{fn}"
            triples.append(_triple(full, "defined_in", filename))
            triples.append(_triple(full, "is_a", "function"))

            # Decorators passed from decorated_definition parent
            for dec_text in (_decorators or []):
                triples.append(_triple(full, "decorated_with", dec_text, 0.95))
        return

    elif t == "import_statement":
        for child in node.named_children:
            if child.type in ("dotted_name", "identifier"):
                triples.append(_triple(filename, "imports", _text(child, src), 0.98))
                break

    elif t == "import_from_statement":
        mod_node = node.child_by_field_name("module_name")
        if mod_node:
            triples.append(_triple(filename, "imports", _text(mod_node, src), 0.98))

    # Recurse
    for child in node.children:
        _visit_python(child, src, filename, mod, triples, parent_class)


# ─── JavaScript / JSX ────────────────────────────────────────────────────────

def _extract_js(root, src: bytes, filename: str) -> List[dict]:
    triples: List[dict] = []
    mod = _modname(filename)

    def visit(node, parent_class=None):
        t = node.type

        if t == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                cls = _text(name_node, src)
                full = f"{mod}.{cls}"
                triples.append(_triple(full, "defined_in", filename))
                triples.append(_triple(full, "is_a", "class"))
                # extends — field name varies by grammar version; scan by node type as fallback
                heritage = node.child_by_field_name("heritage")
                if heritage is None:
                    heritage = next((c for c in node.children if c.type == "class_heritage"), None)
                if heritage:
                    for h in heritage.children:
                        if h.type == "identifier":
                            triples.append(_triple(full, "inherits_from", _text(h, src), 0.99))
                body = node.child_by_field_name("body")
                if body:
                    for c in body.children:
                        visit(c, parent_class=full)
            return

        elif t in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                fn = _text(name_node, src)
                full = f"{mod}.{fn}"
                triples.append(_triple(full, "defined_in", filename))
                triples.append(_triple(full, "is_a", "function"))

        elif t == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node and parent_class:
                fn = _text(name_node, src)
                full = f"{parent_class}.{fn}"
                triples.append(_triple(full, "belongs_to", parent_class))
                triples.append(_triple(full, "is_a", "function"))

        elif t == "import_statement":
            src_node = node.child_by_field_name("source")
            if src_node:
                module = _text(src_node, src).strip("'\"")
                triples.append(_triple(filename, "imports", module, 0.98))

        elif t == "export_statement":
            decl = node.child_by_field_name("declaration")
            if decl:
                visit(decl, parent_class)

        elif t == "lexical_declaration":
            for dec in node.named_children:
                if dec.type == "variable_declarator":
                    val = dec.child_by_field_name("value")
                    name_n = dec.child_by_field_name("name")
                    if val and name_n and val.type in ("arrow_function", "function"):
                        fn = _text(name_n, src)
                        full = f"{mod}.{fn}"
                        triples.append(_triple(full, "defined_in", filename))
                        triples.append(_triple(full, "is_a", "function"))

        for child in node.children:
            visit(child, parent_class)

    visit(root)
    return triples


# ─── TypeScript ───────────────────────────────────────────────────────────────

def _extract_typescript(root, src: bytes, filename: str) -> List[dict]:
    """TypeScript = JS extraction + interface/type alias support."""
    triples = _extract_js(root, src, filename)
    mod = _modname(filename)

    def visit_ts(node):
        t = node.type
        if t == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                iface = _text(name_node, src)
                full = f"{mod}.{iface}"
                triples.append(_triple(full, "defined_in", filename))
                triples.append(_triple(full, "is_a", "interface"))
                # extends — field absent; scan children by type
                ext = (node.child_by_field_name("extends_clause")
                       or next((c for c in node.children
                                if c.type in ("extends_type_clause", "extends_clause")), None))
                if ext:
                    for c in ext.children:
                        if c.type == "type_identifier":
                            triples.append(_triple(full, "extends", _text(c, src), 0.99))
        elif t == "type_alias_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                alias = _text(name_node, src)
                triples.append(_triple(f"{mod}.{alias}", "defined_in", filename))
                triples.append(_triple(f"{mod}.{alias}", "is_a", "type"))
        for child in node.children:
            visit_ts(child)

    visit_ts(root)
    return triples


# ─── Go ───────────────────────────────────────────────────────────────────────

def _extract_go(root, src: bytes, filename: str) -> List[dict]:
    triples: List[dict] = []
    mod = _modname(filename)

    def visit(node):
        t = node.type
        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                fn = _text(name_node, src)
                full = f"{mod}.{fn}"
                triples.append(_triple(full, "defined_in", filename))
                triples.append(_triple(full, "is_a", "function"))

        elif t == "method_declaration":
            name_node = node.child_by_field_name("name")
            recv = node.child_by_field_name("receiver")
            if name_node:
                fn = _text(name_node, src)
                if recv:
                    # Receiver type — may be wrapped in pointer_type (*Server → type_identifier)
                    def _find_type_id(n):
                        if n.type == "type_identifier":
                            return n
                        for ch in n.children:
                            res = _find_type_id(ch)
                            if res:
                                return res
                        return None
                    type_id = _find_type_id(recv)
                    if type_id:
                        recv_type = _text(type_id, src)
                        full = f"{mod}.{recv_type}.{fn}"
                        triples.append(_triple(full, "belongs_to", f"{mod}.{recv_type}"))
                        triples.append(_triple(full, "is_a", "function"))

        elif t == "type_declaration":
            for spec in node.named_children:
                if spec.type == "type_spec":
                    name_n = spec.child_by_field_name("name")
                    type_n = spec.child_by_field_name("type")
                    if name_n:
                        tname = _text(name_n, src)
                        full = f"{mod}.{tname}"
                        kind = "struct" if (type_n and "struct" in _text(type_n, src)[:6]) else "type"
                        triples.append(_triple(full, "defined_in", filename))
                        triples.append(_triple(full, "is_a", kind))

        elif t == "import_declaration":
            for spec in node.named_children:
                if spec.type == "import_spec":
                    path_node = spec.child_by_field_name("path") or (spec.named_children[0] if spec.named_children else None)
                    if path_node:
                        path = _text(path_node, src).strip("\"")
                        triples.append(_triple(filename, "imports", path.split("/")[-1], 0.98))

        for child in node.children:
            visit(child)

    visit(root)
    return triples


# ─── Rust ─────────────────────────────────────────────────────────────────────

def _extract_rust(root, src: bytes, filename: str) -> List[dict]:
    triples: List[dict] = []
    mod = _modname(filename)

    def visit(node, parent_impl=None):
        t = node.type
        if t == "function_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                fn = _text(name_node, src)
                full = f"{parent_impl}.{fn}" if parent_impl else f"{mod}.{fn}"
                triples.append(_triple(full, "defined_in", filename))
                triples.append(_triple(full, "is_a", "function"))
                if parent_impl:
                    triples.append(_triple(full, "belongs_to", parent_impl))

        elif t == "struct_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                s = _text(name_node, src)
                full = f"{mod}.{s}"
                triples.append(_triple(full, "defined_in", filename))
                triples.append(_triple(full, "is_a", "struct"))

        elif t == "impl_item":
            type_node = node.child_by_field_name("type")
            if type_node:
                impl_type = _text(type_node, src)
                body = node.child_by_field_name("body")
                if body:
                    for c in body.children:
                        visit(c, parent_impl=f"{mod}.{impl_type}")
            return

        elif t == "use_declaration":
            tree_n = node.child_by_field_name("argument")
            if tree_n:
                triples.append(_triple(filename, "imports", _text(tree_n, src), 0.97))

        elif t == "trait_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                tr = _text(name_node, src)
                triples.append(_triple(f"{mod}.{tr}", "defined_in", filename))
                triples.append(_triple(f"{mod}.{tr}", "is_a", "trait"))

        for child in node.children:
            visit(child, parent_impl)

    visit(root)
    return triples


# ─── Java ─────────────────────────────────────────────────────────────────────

def _extract_java(root, src: bytes, filename: str) -> List[dict]:
    triples: List[dict] = []
    mod = _modname(filename)

    def visit(node, parent_class=None):
        t = node.type
        if t == "class_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                cls = _text(name_node, src)
                full = f"{mod}.{cls}"
                triples.append(_triple(full, "defined_in", filename))
                triples.append(_triple(full, "is_a", "class"))
                sup = node.child_by_field_name("superclass")
                if sup:
                    for c in sup.named_children:
                        triples.append(_triple(full, "inherits_from", _text(c, src), 0.99))
                body = node.child_by_field_name("body")
                if body:
                    for c in body.children:
                        visit(c, parent_class=full)
            return

        elif t == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node and parent_class:
                fn = _text(name_node, src)
                full = f"{parent_class}.{fn}"
                triples.append(_triple(full, "belongs_to", parent_class))
                triples.append(_triple(full, "is_a", "function"))

        elif t == "import_declaration":
            for child in node.named_children:
                if child.type in ("scoped_identifier", "identifier"):
                    triples.append(_triple(filename, "imports", _text(child, src), 0.98))
                    break

        for child in node.children:
            visit(child, parent_class)

    visit(root)
    return triples


# ─── Generic (C, C++, Ruby, etc.) ────────────────────────────────────────────

def _generic_extract(root, src: bytes, filename: str) -> List[dict]:
    """Best-effort extraction for languages without specialized visitors."""
    triples: List[dict] = []
    mod = _modname(filename)
    FUNC_TYPES = {"function_definition", "function_declaration", "method_definition",
                  "function_item", "method", "def"}
    CLASS_TYPES = {"class_definition", "class_declaration", "class", "struct_specifier"}

    def visit(node, depth=0):
        if depth > 20:
            return
        t = node.type
        name_node = node.child_by_field_name("name") or node.child_by_field_name("declarator")
        if t in FUNC_TYPES and name_node:
            fn = _text(name_node, src).split("(")[0]
            triples.append(_triple(f"{mod}.{fn}", "defined_in", filename))
            triples.append(_triple(f"{mod}.{fn}", "is_a", "function"))
        elif t in CLASS_TYPES and name_node:
            cls = _text(name_node, src)
            triples.append(_triple(f"{mod}.{cls}", "defined_in", filename))
            triples.append(_triple(f"{mod}.{cls}", "is_a", "class"))
        for child in node.children:
            visit(child, depth + 1)

    visit(root)
    return triples


# ─── Regex fallback ──────────────────────────────────────────────────────────

def _regex_extract(source: str, filename: str) -> List[dict]:
    """Lightweight regex extraction when tree-sitter is unavailable."""
    triples: List[dict] = []
    mod = _modname(filename)
    ext = Path(filename).suffix.lower()

    if ext == ".py":
        for m in re.finditer(r"^(?:async\s+)?def\s+(\w+)\s*\(", source, re.MULTILINE):
            triples.append(_triple(f"{mod}.{m.group(1)}", "defined_in", filename, 0.85))
            triples.append(_triple(f"{mod}.{m.group(1)}", "is_a", "function", 0.85))
        for m in re.finditer(r"^class\s+(\w+)", source, re.MULTILINE):
            triples.append(_triple(f"{mod}.{m.group(1)}", "defined_in", filename, 0.85))
            triples.append(_triple(f"{mod}.{m.group(1)}", "is_a", "class", 0.85))
        for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", source, re.MULTILINE):
            triples.append(_triple(filename, "imports", m.group(1), 0.8))

    elif ext in (".js", ".ts", ".tsx", ".jsx"):
        for m in re.finditer(r"(?:function|const|let|var)\s+(\w+)\s*[=\(]", source):
            triples.append(_triple(f"{mod}.{m.group(1)}", "defined_in", filename, 0.8))
            triples.append(_triple(f"{mod}.{m.group(1)}", "is_a", "function", 0.8))
        for m in re.finditer(r"^class\s+(\w+)", source, re.MULTILINE):
            triples.append(_triple(f"{mod}.{m.group(1)}", "defined_in", filename, 0.8))
            triples.append(_triple(f"{mod}.{m.group(1)}", "is_a", "class", 0.8))
        for m in re.finditer(r'(?:import|from)\s+["\']([^"\']+)["\']', source):
            triples.append(_triple(filename, "imports", m.group(1), 0.8))

    # Always add a file node
    triples.insert(0, _triple(filename, "is_a", "file", 0.99))
    # confidence_tier stays EXTRACTED for regex (still deterministic, just lower confidence)
    for t in triples:
        t["confidence_tier"] = "EXTRACTED"

    return triples[:50]  # cap to avoid noise from minified files
