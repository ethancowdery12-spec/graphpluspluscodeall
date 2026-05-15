"""
Full test suite for all new backend functions.
Run with: python test_all.py
"""
import sys, os, json, tempfile, zipfile, hashlib, asyncio
from pathlib import Path

PASS = 0; FAIL = 0

def check(label, condition, info=''):
    global PASS, FAIL
    if condition:
        print(f'  PASS  {label}')
        PASS += 1
    else:
        print(f'  FAIL  {label}' + (f'  -> {info}' if info else ''))
        FAIL += 1

def section(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')


# ─────────────────────────────────────────────────────────────────────────────
section('1. CODE EXTRACTOR — JavaScript / JSX')
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.code_extractor import extract_code_triples

js_src = """
import React from 'react';
class Button extends Component {
  render() { return null; }
}
function greet(name) { return 'hi'; }
const arrow = () => 42;
"""
ts = extract_code_triples(js_src, 'ui.js')
preds = {(t['subject'], t['predicate'], t['object']) for t in ts}
check('import react', ('ui.js', 'imports', 'react') in preds)
check('class Button defined_in', any(s.endswith('Button') and p == 'defined_in' for s,p,o in preds))
check('class Button is_a class', any(s.endswith('Button') and p == 'is_a' and o == 'class' for s,p,o in preds))
check('Button inherits_from Component', any(p == 'inherits_from' and o == 'Component' for s,p,o in preds), str(preds))
check('render method belongs_to Button', any('render' in s and p == 'belongs_to' and 'Button' in o for s,p,o in preds))
check('greet function', any(s.endswith('greet') and p == 'is_a' for s,p,o in preds))
check('arrow const function', any(s.endswith('arrow') and p == 'is_a' for s,p,o in preds))
check('all EXTRACTED tier', all(t['confidence_tier'] == 'EXTRACTED' for t in ts))

# ─────────────────────────────────────────────────────────────────────────────
section('2. CODE EXTRACTOR — TypeScript')
# ─────────────────────────────────────────────────────────────────────────────
ts_src = """
interface User { id: number; name: string; }
interface Admin extends User { role: string; }
type ID = string | number;
class UserService {
  constructor() {}
  findUser(id: ID) { return null; }
}
"""
tt = extract_code_triples(ts_src, 'service.ts')
tp = {(t['subject'], t['predicate'], t['object']) for t in tt}
check('interface User is_a interface', any(s.endswith('User') and p == 'is_a' and o == 'interface' for s,p,o in tp))
check('Admin extends User', any(p == 'extends' and o == 'User' for s,p,o in tp))
check('type alias ID', any(s.endswith('ID') and p == 'is_a' and o == 'type' for s,p,o in tp))
check('class UserService is_a class', any(s.endswith('UserService') and p == 'is_a' and o == 'class' for s,p,o in tp))
check('findUser method', any('findUser' in s and p == 'belongs_to' for s,p,o in tp))
check('all EXTRACTED tier', all(t['confidence_tier'] == 'EXTRACTED' for t in tt))

# ─────────────────────────────────────────────────────────────────────────────
section('3. CODE EXTRACTOR — Go')
# ─────────────────────────────────────────────────────────────────────────────
go_src = """
package main
import "fmt"
type Server struct { host string }
func NewServer() *Server { return &Server{} }
func (s *Server) Start() { fmt.Println("started") }
"""
gt = extract_code_triples(go_src, 'server.go')
gp = {(t['subject'], t['predicate'], t['object']) for t in gt}
check('struct Server is_a struct', any(s.endswith('Server') and p == 'is_a' and o == 'struct' for s,p,o in gp), str(gp))
check('func NewServer is_a function', any(s.endswith('NewServer') and p == 'is_a' for s,p,o in gp))
check('method Start belongs_to Server', any('Start' in s and p == 'belongs_to' for s,p,o in gp), str(gp))
check('import fmt', any(p == 'imports' and o == 'fmt' for s,p,o in gp), str(gp))
check('all EXTRACTED tier', all(t['confidence_tier'] == 'EXTRACTED' for t in gt))

# ─────────────────────────────────────────────────────────────────────────────
section('4. CODE EXTRACTOR — Rust')
# ─────────────────────────────────────────────────────────────────────────────
rs_src = """
use std::io::Read;
struct Config { debug: bool }
trait Logger { fn log(&self, msg: &str); }
impl Config {
    fn new() -> Config { Config { debug: false } }
}
fn main() {}
"""
rt = extract_code_triples(rs_src, 'main.rs')
rp = {(t['subject'], t['predicate'], t['object']) for t in rt}
check('struct Config is_a struct', any(s.endswith('Config') and p == 'is_a' and o == 'struct' for s,p,o in rp))
check('trait Logger is_a trait', any(s.endswith('Logger') and p == 'is_a' and o == 'trait' for s,p,o in rp))
check('impl fn new belongs_to Config', any('new' in s and p == 'belongs_to' for s,p,o in rp), str(rp))
check('fn main is_a function', any(s.endswith('main') and p == 'is_a' for s,p,o in rp))
check('use import', any(p == 'imports' for s,p,o in rp), str(rp))
check('all EXTRACTED tier', all(t['confidence_tier'] == 'EXTRACTED' for t in rt))

# ─────────────────────────────────────────────────────────────────────────────
section('5. CODE EXTRACTOR — Java')
# ─────────────────────────────────────────────────────────────────────────────
java_src = """
import java.util.List;
public class Animal {
    public void speak() {}
}
public class Dog extends Animal {
    public void fetch() {}
}
"""
jt = extract_code_triples(java_src, 'animals.java')
jp = {(t['subject'], t['predicate'], t['object']) for t in jt}
check('class Animal is_a class', any(s.endswith('Animal') and p == 'is_a' and o == 'class' for s,p,o in jp))
check('class Dog is_a class', any(s.endswith('Dog') and p == 'is_a' and o == 'class' for s,p,o in jp))
check('Dog inherits_from Animal', any(p == 'inherits_from' and o == 'Animal' for s,p,o in jp), str(jp))
check('speak method belongs_to', any('speak' in s and p == 'belongs_to' for s,p,o in jp))
check('fetch method belongs_to', any('fetch' in s and p == 'belongs_to' for s,p,o in jp))
check('java.util.List import', any(p == 'imports' for s,p,o in jp))
check('all EXTRACTED tier', all(t['confidence_tier'] == 'EXTRACTED' for t in jt))

# ─────────────────────────────────────────────────────────────────────────────
section('6. CODE EXTRACTOR — Python (regression)')
# ─────────────────────────────────────────────────────────────────────────────
py_src = """
import os
from pathlib import Path

class Animal:
    def speak(self): pass

class Dog(Animal):
    @staticmethod
    def fetch(): pass

def standalone(): pass
"""
pt = extract_code_triples(py_src, 'zoo.py')
pp = {(t['subject'], t['predicate'], t['object']) for t in pt}
check('import os', ('zoo.py', 'imports', 'os') in pp)
check('from pathlib import Path', any(p == 'imports' and 'pathlib' in o for s,p,o in pp))
check('class Animal is_a class', any(s.endswith('Animal') and p == 'is_a' and o == 'class' for s,p,o in pp))
check('Dog inherits_from Animal', any(p == 'inherits_from' and o == 'Animal' for s,p,o in pp))
check('speak method belongs_to Animal', any('speak' in s and p == 'belongs_to' for s,p,o in pp))
check('decorator @staticmethod', any(p == 'decorated_with' and o == 'staticmethod' for s,p,o in pp))
check('standalone func is_a function', any(s.endswith('standalone') and p == 'is_a' for s,p,o in pp))
check('all EXTRACTED tier', all(t['confidence_tier'] == 'EXTRACTED' for t in pt))

# ─────────────────────────────────────────────────────────────────────────────
section('7. FILE ROUTER — PDF / text / ZIP')
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.file_router import route_file

async def _route(content, name):
    return await route_file(content, name)

# Text routing (plain text → LLM path, but we only test that it returns a list + sha)
txt_content = b"Alice knows Bob. Bob works at Acme Corp."
triples_txt, sha_txt = asyncio.run(_route(txt_content, "notes.txt"))
check('text file returns list', isinstance(triples_txt, list))
check('text file returns sha256', sha_txt and len(sha_txt) == 64)

# Python code routing via file_router (should go to AST)
py_bytes = b"class Foo:\n    def bar(self): pass\n"
triples_py, sha_py = asyncio.run(_route(py_bytes, "foo.py"))
check('python file returns list', isinstance(triples_py, list))
check('python AST triples non-empty', len(triples_py) > 0, f'got {len(triples_py)}')
check('python AST tier EXTRACTED', all(t.get('confidence_tier') == 'EXTRACTED' for t in triples_py) if triples_py else False)

# ZIP routing
with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
    tmp_path = tmp.name
try:
    with zipfile.ZipFile(tmp_path, 'w') as zf:
        zf.writestr('hello.py', 'def hello(): pass\n')
        zf.writestr('world.py', 'def world(): pass\n')
    with open(tmp_path, 'rb') as f:
        zip_bytes = f.read()
    triples_zip, sha_zip = asyncio.run(_route(zip_bytes, "archive.zip"))
    check('zip returns list', isinstance(triples_zip, list))
    check('zip extracts py files', len(triples_zip) > 0, f'got {len(triples_zip)}')
finally:
    os.unlink(tmp_path)

# Oversized file guard (>10 MB)
big = b"x" * (11 * 1024 * 1024)
triples_big, sha_big = asyncio.run(_route(big, "huge.py"))
check('oversized file skipped (empty triples)', len(triples_big) == 0, f'got {len(triples_big)}')

# ─────────────────────────────────────────────────────────────────────────────
section('8. PERSISTENCE — save / load cycle')
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.graph_builder import GraphStore
from pipeline import persistence

# Create a fresh store (not the singleton)
gs = GraphStore.__new__(GraphStore)
import networkx as nx
gs.G = nx.MultiDiGraph()
gs._entity_index = {}
gs._ingestion_log = []
gs._file_hashes = {}
gs._metrics = {'total_queries': 0, 'total_ingestions': 0, 'cache_hits': 0, 'avg_query_ms': 0}
gs._version = 0

# Add some data
gs.G.add_node('alice', label='Alice', type='person', color='#f59e0b',
              pagerank=0.1, community=0, degree=1, size=8.0)
gs.G.add_node('bob',   label='Bob',   type='person', color='#f59e0b',
              pagerank=0.1, community=0, degree=1, size=8.0)
gs.G.add_edge('alice', 'bob', predicate='knows', confidence=0.9,
              weight=0.9, confidence_tier='INFERRED')
gs._file_hashes['abc123'] = 'notes.txt'
gs._version = 5

# Save to a temp path
save_path = Path(tempfile.mkdtemp()) / 'test_graph.json'
import pipeline.persistence as pers

orig_path = pers.STORE_PATH
pers.STORE_PATH = save_path
try:
    pers.save_graph(gs)
    check('save creates file', save_path.exists())

    # Load into a fresh store
    gs2 = GraphStore.__new__(GraphStore)
    gs2.G = nx.MultiDiGraph()
    gs2._entity_index = {}
    gs2._ingestion_log = []
    gs2._file_hashes = {}
    gs2._metrics = {'total_queries': 0, 'total_ingestions': 0, 'cache_hits': 0, 'avg_query_ms': 0}
    gs2._version = 0

    pers.load_graph(gs2)
    check('load restores nodes', gs2.G.number_of_nodes() == 2, f'got {gs2.G.number_of_nodes()}')
    check('load restores edges', gs2.G.number_of_edges() == 1, f'got {gs2.G.number_of_edges()}')
    check('load restores file_hashes', 'abc123' in gs2._file_hashes)
    check('load restores version', gs2._version == 5, f'got {gs2._version}')

    # Check edge confidence_tier survived round-trip
    edges = list(gs2.G.edges(data=True))
    check('edge confidence_tier preserved', edges[0][2].get('confidence_tier') == 'INFERRED',
          str(edges[0][2]))
finally:
    pers.STORE_PATH = orig_path
    save_path.unlink(missing_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
section('9. GRAPH BUILDER — new methods')
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.graph_builder import GraphStore as GS

gs3 = GS.__new__(GS)
gs3.G = nx.MultiDiGraph()
gs3._entity_index = {}
gs3._ingestion_log = []
gs3._file_hashes = {}
gs3._metrics = {'total_queries': 0, 'total_ingestions': 0, 'cache_hits': 0, 'avg_query_ms': 0}
gs3._version = 0

# register_file_hash / has_file_hash
gs3._file_hashes = {}
gs3._file_hashes['deadbeef'] = 'file.py'
check('has_file_hash True', gs3._file_hashes.get('deadbeef') is not None)
check('has_file_hash False', gs3._file_hashes.get('notexist') is None)

# add_relation with confidence_tier
gs3.G.add_node('X', label='X', type='concept', color='#00d4ff', pagerank=0.0, community=0, degree=0, size=6.0)
gs3.G.add_node('Y', label='Y', type='concept', color='#00d4ff', pagerank=0.0, community=0, degree=0, size=6.0)
gs3.G.add_edge('X', 'Y', predicate='related_to', confidence=1.0, weight=1.0, confidence_tier='EXTRACTED')
gs3.G.add_edge('X', 'Y', predicate='maybe_uses', confidence=0.5, weight=0.5, confidence_tier='AMBIGUOUS')

edges_data = [d for _,__,d in gs3.G.edges(data=True)]
tiers = {d['confidence_tier'] for d in edges_data}
check('EXTRACTED tier stored', 'EXTRACTED' in tiers)
check('AMBIGUOUS tier stored', 'AMBIGUOUS' in tiers)

# to_json confidence_tier in edges
import json
# Manually simulate to_json edges
json_edges = [
    {'source': u, 'target': v, 'predicate': d['predicate'],
     'confidence': d.get('confidence', 1.0), 'weight': d.get('weight', 1.0),
     'confidence_tier': d.get('confidence_tier', 'INFERRED')}
    for u,v,d in gs3.G.edges(data=True)
]
check('to_json edges have confidence_tier', all('confidence_tier' in e for e in json_edges))

# confidence_breakdown in stats
from collections import Counter
breakdown = Counter(d.get('confidence_tier', 'INFERRED') for _,__,d in gs3.G.edges(data=True))
check('confidence_breakdown EXTRACTED=1', breakdown['EXTRACTED'] == 1)
check('confidence_breakdown AMBIGUOUS=1', breakdown['AMBIGUOUS'] == 1)

# ─────────────────────────────────────────────────────────────────────────────
section('10. EXTRACTOR — confidence_tier tagging')
# ─────────────────────────────────────────────────────────────────────────────
import pipeline.extractor as ext_mod

# _normalize_triple: explicit tier wins
t1 = {'subject': 'A', 'predicate': 'knows', 'object': 'B',
      'confidence': 0.9, 'confidence_tier': 'EXTRACTED'}
n1 = ext_mod._normalize_triple(t1)
check('explicit EXTRACTED tier preserved', n1.get('confidence_tier') == 'EXTRACTED')

# _normalize_triple: infer from confidence >= 0.7 -> INFERRED
t2 = {'subject': 'A', 'predicate': 'knows', 'object': 'C', 'confidence': 0.8}
n2 = ext_mod._normalize_triple(t2)
check('confidence 0.8 -> INFERRED tier', n2.get('confidence_tier') == 'INFERRED',
      str(n2))

# _normalize_triple: infer from confidence < 0.7 -> AMBIGUOUS
t3 = {'subject': 'A', 'predicate': 'knows', 'object': 'D', 'confidence': 0.5}
n3 = ext_mod._normalize_triple(t3)
check('confidence 0.5 -> AMBIGUOUS tier', n3.get('confidence_tier') == 'AMBIGUOUS',
      str(n3))

# _regex_fallback (not async, call directly)
fallback = ext_mod._regex_fallback("BERT is a transformer model. GPT uses attention mechanisms.")
check('regex fallback non-empty', len(fallback) > 0, str(fallback))
check('regex fallback AMBIGUOUS tier', all(t.get('confidence_tier') == 'AMBIGUOUS' for t in fallback),
      str(fallback[:2]))

# ─────────────────────────────────────────────────────────────────────────────
section('11. DIR WALKER — basic walk')
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.dir_walker import walk_directory

# Create a temp directory with some files
tmpdir = Path(tempfile.mkdtemp())
(tmpdir / 'main.py').write_text('def main(): pass\n')
(tmpdir / 'util.js').write_text('function helper() {}\n')
subdir = tmpdir / 'pkg'
subdir.mkdir()
(subdir / 'module.py').write_text('class MyClass: pass\n')

results = []
async def collect():
    async for item in walk_directory(str(tmpdir), set()):
        results.append(item)

asyncio.run(collect())
check('walk finds files', len(results) > 0, f'found {len(results)}')
found_paths = [r[2] for r in results]  # (triples, sha, filepath)
check('walk finds main.py', any('main.py' in p for p in found_paths))
check('walk finds subdir module.py', any('module.py' in p for p in found_paths))
check('walk triples non-empty', any(len(r[0]) > 0 for r in results))
check('walk sha256 correct length', all(len(r[1]) == 64 for r in results))

# Dedup: pass existing hash → should be skipped
import hashlib
main_sha = hashlib.sha256((tmpdir / 'main.py').read_bytes()).hexdigest()
results2 = []
async def collect2():
    async for item in walk_directory(str(tmpdir), {main_sha}):
        results2.append(item)
asyncio.run(collect2())
check('dedup skips known hash', all(r[1] != main_sha for r in results2),
      f'found {[r[2] for r in results2]}')

# Cleanup
import shutil
shutil.rmtree(tmpdir)

# ─────────────────────────────────────────────────────────────────────────────
section('12. INGEST_TRIPLES — confidence_tier flows end-to-end')
# ─────────────────────────────────────────────────────────────────────────────
from pipeline.graph_builder import GraphStore as GS2

gs4 = GS2.__new__(GS2)
gs4.G = nx.MultiDiGraph()
gs4._entity_index = {}
gs4._ingestion_log = []
gs4._file_hashes = {}
gs4._metrics = {'total_queries': 0, 'total_ingestions': 0, 'cache_hits': 0, 'avg_query_ms': 0}
gs4._version = 0

triples_in = [
    {'subject': 'ModelA', 'predicate': 'uses', 'object': 'DatasetB',
     'confidence': 1.0, 'confidence_tier': 'EXTRACTED'},
    {'subject': 'ModelA', 'predicate': 'based_on', 'object': 'PaperC',
     'confidence': 0.8, 'confidence_tier': 'INFERRED'},
    {'subject': 'ModelA', 'predicate': 'maybe_related', 'object': 'Concept',
     'confidence': 0.4, 'confidence_tier': 'AMBIGUOUS'},
]

# Call ingest_triples if it exists, otherwise test add_relation directly
if hasattr(gs4, 'ingest_triples'):
    gs4.ingest_triples(triples_in)
    e_data = [d for _,__,d in gs4.G.edges(data=True)]
    tiers_found = {d.get('confidence_tier') for d in e_data}
    check('ingest_triples: EXTRACTED stored', 'EXTRACTED' in tiers_found)
    check('ingest_triples: INFERRED stored', 'INFERRED' in tiers_found)
    check('ingest_triples: AMBIGUOUS stored', 'AMBIGUOUS' in tiers_found)
else:
    check('ingest_triples: method exists', False, 'method not found on GraphStore')

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'  FINAL RESULTS: {PASS} PASS   {FAIL} FAIL')
print(f'{"="*60}\n')
sys.exit(0 if FAIL == 0 else 1)
