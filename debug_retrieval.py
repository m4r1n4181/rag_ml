"""
Show what retrieval returns for one question. No LLM call, no API cost.

    python debug_retrieval.py "Objasniti regularizaciju kod linearne regresije"
    python debug_retrieval.py "..." --hybrid
"""
import sys

from src.retrieval.vector_store import load_vector_store, search

args = [a for a in sys.argv[1:] if not a.startswith("--")]
query = args[0] if args else "regularizacija linearna regresija L1 L2"
use_hybrid = "--hybrid" in sys.argv

results = search(query, load_vector_store(), n_results=10, use_hybrid=use_hybrid)

print(f"\n'{query}'  (hybrid={use_hybrid})\n" + "=" * 60)
for i, r in enumerate(results, 1):
    meta = r["metadata"]
    where = f"slajd {meta.get('slide')}" if meta.get("type") == "presentation" else f"strana {meta.get('page')}"
    print(f"\n[{i}] {meta.get('source')} - {where}   cos={r.get('similarity')}")
    print(f"    {r['content'][:150]!r}")
