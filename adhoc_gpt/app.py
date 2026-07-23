"""Phase 4: the AdHoc-GPT drafting application (RAG + generation).

    # draft a resolution, conditioned on retrieved precedent clauses
    python -m adhoc_gpt.app draft --topic "climate resilience"

    # interactive drafting session
    python -m adhoc_gpt.app repl

    # local web UI on http://127.0.0.1:8000
    python -m adhoc_gpt.app serve

Everything the model writes is *synthetic*: it is trained on the templated
corpus in ``adhoc_gpt/domain/corpus.py``, so its output imitates the register of
UN drafting without being authentic practice. The disclaimer travels with the
output in both the CLI and the web UI.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from .generate import load_model
from .rag import (BM25Index, EmbeddingIndex, HybridRetriever, fit_prompt, load_library,
                  mmr)

DISCLAIMER = (
    "Synthetic draft produced by AdHoc-GPT, a small model trained on a templated corpus. "
    "It imitates the register of multilateral drafting and is not authentic UN text."
)

DEFAULT_CKPT = "runs/adhoc-lm-domain/ckpt.pt"
DEFAULT_LIBRARY = "data/clause_library.json"


class DraftingEngine:
    """Retrieval + generation, wired together."""

    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        library: str | Path = DEFAULT_LIBRARY,
        device: str = "auto",
        dense: bool = True,
    ):
        self.model, self.tokenizer, self.device = load_model(ckpt, device)
        self.docs = load_library(library)
        bm25 = BM25Index(self.docs)
        emb = EmbeddingIndex(self.docs, self.model, self.tokenizer) if dense else None
        self.retriever = HybridRetriever(bm25, emb) if dense else bm25
        self.bm25 = bm25

    def retrieve(self, topic: str, k: int = 4, kind: str | None = None, diverse: bool = True):
        """Retrieve k clauses, de-duplicated by MMR so the precedents differ."""
        pool = self.retriever.search(topic, k=k * 5 if diverse else k, kind=kind)
        return mmr(pool, k=k) if diverse else pool

    def draft(
        self,
        topic: str,
        k: int = 4,
        tokens: int = 400,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float | None = None,
        organ: str = "The General Assembly",
        seed: int | None = None,
    ) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
        hits = self.retrieve(topic, k=k)
        # leave room in the context for the draft the model is about to write
        budget = max(int(self.model.config.block_size * 0.6), 32)
        prompt, hits = fit_prompt(topic, hits, self.tokenizer, budget, organ=organ)
        ids = self.tokenizer.encode(prompt)[-self.model.config.block_size + 1 :]
        idx = torch.tensor(ids or [0], dtype=torch.long, device=self.device)[None, ...]
        out = self.model.generate(
            idx, max_new_tokens=tokens, temperature=temperature, top_k=top_k, top_p=top_p
        )
        text = self.tokenizer.decode(out[0].tolist())
        continuation = text[len(prompt):] if text.startswith(prompt) else text
        return {
            "topic": topic,
            "prompt": prompt,
            "draft": continuation.split("<|endoftext|>")[0].rstrip(),
            "retrieved": [
                {"score": round(h.score, 4), "kind": h.doc.kind, "text": h.doc.text}
                for h in hits
            ],
            "disclaimer": DISCLAIMER,
        }


# --------------------------------------------------------------------------
# CLI modes
# --------------------------------------------------------------------------
def cmd_draft(a) -> None:
    engine = DraftingEngine(a.ckpt, a.library, a.device, dense=not a.no_dense)
    result = engine.draft(a.topic, a.k, a.tokens, a.temperature, a.top_k, a.top_p,
                          organ=a.organ, seed=a.seed)
    if a.json:
        print(json.dumps(result, indent=2))
        return
    print(f"# Draft resolution on {result['topic']}\n")
    print("## Retrieved precedent")
    for hit in result["retrieved"]:
        print(f"  [{hit['score']:.3f}] ({hit['kind']}) {hit['text'][:150]}")
    print("\n## Generated draft\n")
    print(result["prompt"].splitlines()[-2] if result["prompt"] else "")
    print(result["draft"])
    print(f"\n---\n{DISCLAIMER}")


def cmd_retrieve(a) -> None:
    docs = load_library(a.library)
    for hit in BM25Index(docs).search(a.topic, a.k, a.kind):
        print(f"[{hit.score:5.2f}] ({hit.doc.kind}) {hit.doc.text[:170]}")


def cmd_repl(a) -> None:
    engine = DraftingEngine(a.ckpt, a.library, a.device, dense=not a.no_dense)
    print("AdHoc-GPT drafting session. Enter a topic (or 'quit').")
    print(f"({DISCLAIMER})\n")
    while True:
        try:
            topic = input("topic> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not topic or topic in {"quit", "exit"}:
            break
        result = engine.draft(topic, a.k, a.tokens, a.temperature, a.top_k, a.top_p)
        print()
        print(result["draft"])
        print()


PAGE = """<!doctype html><meta charset="utf-8"><title>AdHoc-GPT</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 16px/1.55 system-ui, sans-serif; max-width: 60rem; margin: 2rem auto;
        padding: 0 1rem; }}
 h1 {{ font-size: 1.4rem; }}
 form {{ display: flex; gap: .5rem; flex-wrap: wrap; align-items: center; margin: 1rem 0; }}
 input, select, button {{ font: inherit; padding: .45rem .6rem; }}
 input[name=topic] {{ flex: 1 1 22rem; }}
 pre {{ white-space: pre-wrap; background: rgba(127,127,127,.12); padding: 1rem;
        border-radius: .5rem; overflow-x: auto; }}
 .note {{ font-size: .85rem; opacity: .75; border-left: 3px solid currentColor;
          padding-left: .7rem; }}
 li {{ margin-bottom: .35rem; font-size: .9rem; }}
</style>
<h1>AdHoc-GPT — resolution drafting</h1>
<p class="note">{disclaimer}</p>
<form onsubmit="go(event)">
  <input name="topic" placeholder="e.g. climate resilience" required>
  <label>tokens <input name="tokens" type="number" value="400" min="50" max="2000"
    style="width:6rem"></label>
  <label>temp <input name="temperature" type="number" value="0.8" step="0.05" min="0.1"
    max="2" style="width:5rem"></label>
  <button>Draft</button>
</form>
<div id="out"></div>
<script>
async function go(e) {{
  e.preventDefault();
  const f = new FormData(e.target), out = document.getElementById('out');
  out.innerHTML = '<p>drafting…</p>';
  const r = await fetch('/api/draft', {{method: 'POST', headers: {{'Content-Type':'application/json'}},
    body: JSON.stringify(Object.fromEntries(f))}});
  const d = await r.json();
  if (d.error) {{ out.innerHTML = '<pre>' + d.error + '</pre>'; return; }}
  out.innerHTML = '<h2>Retrieved precedent</h2><ul>' +
    d.retrieved.map(h => '<li><b>' + h.score.toFixed(3) + '</b> (' + h.kind + ') ' +
      h.text.replace(/</g,'&lt;') + '</li>').join('') +
    '</ul><h2>Generated draft</h2><pre>' + d.draft.replace(/</g,'&lt;') + '</pre>' +
    '<p class="note">' + d.disclaimer + '</p>';
}}
</script>
"""


def make_server(engine: DraftingEngine, host: str = "127.0.0.1", port: int = 8000):
    """Build (but do not start) the drafting HTTP server."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    page = PAGE.format(disclaimer=html.escape(DISCLAIMER))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, page.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path != "/api/draft":
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                result = engine.draft(
                    str(req.get("topic", ""))[:200],
                    k=int(req.get("k", 4)),
                    tokens=min(int(req.get("tokens", 400)), 2000),
                    temperature=float(req.get("temperature", 0.8)),
                    top_k=int(req.get("top_k", 50)),
                )
                body = json.dumps(result).encode()
            except Exception as e:  # keep the server alive on bad input
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(200, body, "application/json")

        def log_message(self, *args):  # quieter console
            pass

    return ThreadingHTTPServer((host, port), Handler)


def cmd_serve(a) -> None:
    engine = DraftingEngine(a.ckpt, a.library, a.device, dense=not a.no_dense)
    server = make_server(engine, a.host, a.port)
    print(f"serving on http://{a.host}:{server.server_address[1]}  (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AdHoc-GPT drafting application")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--ckpt", default=DEFAULT_CKPT)
        sp.add_argument("--library", default=DEFAULT_LIBRARY)
        sp.add_argument("--device", default="auto")
        sp.add_argument("--no-dense", action="store_true",
                        help="lexical retrieval only (skip embedding retrieval)")
        sp.add_argument("-k", type=int, default=4, help="clauses to retrieve")
        sp.add_argument("--tokens", type=int, default=400)
        sp.add_argument("--temperature", type=float, default=0.8)
        sp.add_argument("--top-k", type=int, default=50)
        sp.add_argument("--top-p", type=float, default=None)
        return sp

    d = common(sub.add_parser("draft", help="draft a resolution on a topic"))
    d.add_argument("--topic", required=True)
    d.add_argument("--organ", default="The General Assembly")
    d.add_argument("--seed", type=int, default=None)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_draft)

    r = sub.add_parser("retrieve", help="query the clause library only")
    r.add_argument("--topic", required=True)
    r.add_argument("--library", default=DEFAULT_LIBRARY)
    r.add_argument("-k", type=int, default=8)
    r.add_argument("--kind", default=None)
    r.set_defaults(func=cmd_retrieve)

    common(sub.add_parser("repl", help="interactive drafting session")).set_defaults(func=cmd_repl)

    s = common(sub.add_parser("serve", help="local web UI"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
