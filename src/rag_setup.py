"""
rag_setup.py  — v3
Key fixes over v2:
  - Resilient embedding wrapper: retries each embed call with exponential
    backoff + tunnel health poll, so individual entity upserts survive
    brief ngrok drops without failing the whole merge stage.
  - Reduced embedding concurrency (embedding_func_max_async=2) to stop
    8 simultaneous embed requests from overwhelming the free ngrok tunnel.
  - Post-insert verification: checks the knowledge graph actually grew
    before marking a file as checkpointed. LightRAG silently returns from
    ainsert even when the merge stage fails, so we can't rely on no-exception.
  - Retry loop now also re-checks health between attempts and waits for
    operator to refresh the tunnel URL.
"""

import asyncio
import json
import os
import sys
import time

import httpx
import ollama as _ollama
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKING_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag/index"))
GUIDELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag/guidelines"))
CHECKPOINT_FILE = os.path.join(WORKING_DIR, ".ingested_files.json")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RAG_LLM     = "gemma4:e4b"

MAX_FILE_RETRIES  = 5     # retries per file on connection failure
EMBED_MAX_RETRIES = 12    # retries per individual embed call
EMBED_BASE_DELAY  = 3.0   # seconds — doubled on each retry (capped at 60s)
HEALTH_TIMEOUT_S  = 10    # seconds for pre-flight HTTP check
POLL_INTERVAL_S   = 15    # seconds between tunnel-recovery polls

# Limit concurrent embed requests — free ngrok tunnels choke on 8 parallel
# SSL connections.  2 is conservative but reliable.
EMBED_MAX_ASYNC   = 2


# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------
def _current_host():
    """Always read from env so operator can refresh without restarting."""
    return os.getenv("OLLAMA_HOST", OLLAMA_HOST)


def check_ollama_reachable(host=None):
    url = (host or _current_host()).rstrip("/") + "/api/tags"
    try:
        r = httpx.get(url, timeout=HEALTH_TIMEOUT_S)
        return r.status_code == 200
    except Exception as exc:
        print(f"[health-check] {url} -> {exc}")
        return False


def wait_for_tunnel(fname=""):
    """Block until Ollama is reachable, prompting operator if interactive."""
    prefix = f"[{fname}] " if fname else ""
    while not check_ollama_reachable():
        msg = (
            f"\n{prefix}Ollama unreachable at {_current_host()}\n"
            f"  The ngrok tunnel may have expired.\n"
            f"  1. Restart the ngrok cell in your Kaggle notebook.\n"
            f"  2. Re-export OLLAMA_HOST with the new URL in your shell:\n"
            f"       export OLLAMA_HOST='https://new-tunnel-url.ngrok-free.app'\n"
            f"  3. Press Enter here to retry, or Ctrl-C to abort.\n"
        )
        print(msg, end="")
        try:
            input("  Press Enter ... ")
        except EOFError:
            print(f"  Non-interactive -- retrying in {POLL_INTERVAL_S}s ...")
            time.sleep(POLL_INTERVAL_S)
    print(f"[health-check] OK  {_current_host()}")


def assert_ollama_or_die():
    if not check_ollama_reachable():
        wait_for_tunnel()


# ---------------------------------------------------------------------------
# Resilient embedding wrapper
# Retries each embed call individually so a transient drop doesn't cascade
# into failing 182 entities during the merge stage.
# ---------------------------------------------------------------------------
async def resilient_embed(texts):
    host = _current_host()
    delay = EMBED_BASE_DELAY
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            return await ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host=host,
            )
        except Exception as exc:
            err = str(exc)
            is_transient = any(k in err for k in (
                "ERR_NGROK", "503", "502", "ConnectionError",
                "Failed to connect to Ollama", "incomplete HTTP response",
                "EOF", "SSL",
            ))
            if not is_transient or attempt == EMBED_MAX_RETRIES:
                raise
            print(
                f"  [embed] transient error attempt {attempt}/{EMBED_MAX_RETRIES}: "
                f"{err[:80]} -- waiting {delay:.0f}s"
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
            # Re-read host in case operator updated it while we waited
            host = _current_host()
            # If tunnel still dead, keep polling
            while not check_ollama_reachable(host):
                print("  [embed] tunnel still down -- polling ...")
                await asyncio.sleep(POLL_INTERVAL_S)
                host = _current_host()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done):
    os.makedirs(WORKING_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(done), f, indent=2)


# ---------------------------------------------------------------------------
# Embedding dimension probe
# ---------------------------------------------------------------------------
def get_embedding_dim():
    client = _ollama.Client(host=_current_host())
    r = client.embeddings(model="nomic-embed-text", prompt="test")
    dim = len(r["embedding"])
    print(f"Embedding dim: {dim}  (host: {_current_host()})")
    return dim


# ---------------------------------------------------------------------------
# Build RAG
# ---------------------------------------------------------------------------
async def build_rag(dim):
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=RAG_LLM,
        llm_model_kwargs={
            "host": _current_host(),
            "options": {
                "num_ctx": 8192,
                "num_predict": 2048,
            },
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=dim,
            max_token_size=512,
            func=resilient_embed,           # <- resilient wrapper
        ),
        embedding_func_max_async=EMBED_MAX_ASYNC,  # <- throttle concurrency
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
    )
    await rag.initialize_storages()
    return rag


# ---------------------------------------------------------------------------
# Post-insert graph-growth verification
# LightRAG doesn't raise when merge fails, so we compare node counts.
# ---------------------------------------------------------------------------
def graph_node_count(rag):
    try:
        g = rag.chunk_entity_relation_graph._graph  # internal networkx graph
        return g.number_of_nodes()
    except Exception:
        return -1  # can't check — don't block on it


# ---------------------------------------------------------------------------
# Ingest one file with file-level retry
# ---------------------------------------------------------------------------
async def ingest_one(fpath, rag):
    fname = os.path.basename(fpath)
    with open(fpath, "r") as f:
        content = f.read()

    for attempt in range(1, MAX_FILE_RETRIES + 1):
        # Gate: tunnel must be up before we start
        if not check_ollama_reachable():
            wait_for_tunnel(fname)

        nodes_before = graph_node_count(rag)
        print(
            f"[{fname}] attempt {attempt}/{MAX_FILE_RETRIES} "
            f"({len(content):,} chars, graph nodes before: {nodes_before}) ..."
        )
        try:
            await rag.ainsert(content)
        except Exception as exc:
            print(f"[{fname}] exception from ainsert: {exc}")
            await asyncio.sleep(10)
            continue

        # Verify graph actually grew (catches silent merge failures)
        nodes_after = graph_node_count(rag)
        if nodes_after > nodes_before or nodes_before == -1:
            print(f"[{fname}] done  (graph nodes: {nodes_before} -> {nodes_after})")
            return True
        else:
            print(
                f"[{fname}] WARNING: graph unchanged after insert "
                f"({nodes_before} -> {nodes_after}). "
                f"Merge probably failed silently -- retrying ..."
            )
            await asyncio.sleep(10)

    print(f"[{fname}] FAILED after {MAX_FILE_RETRIES} attempt(s) -- will retry on next run.")
    return False


# ---------------------------------------------------------------------------
# Test query
# ---------------------------------------------------------------------------
async def test_query(rag):
    print("\nRunning test query ...")
    if not check_ollama_reachable():
        wait_for_tunnel("test_query")
    result = await rag.aquery(
        "What are the valid cholesterol and blood pressure ranges "
        "for a 55 year old male at heart disease risk?",
        param=QueryParam(mode="hybrid"),
    )
    print("\n--- Test Query Result ---")
    print(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    assert_ollama_or_die()
    dim = get_embedding_dim()

    files = sorted(
        os.path.join(GUIDELINES_DIR, f)
        for f in os.listdir(GUIDELINES_DIR)
        if f.endswith(".txt")
    )
    if not files:
        print(f"No .txt files found in {GUIDELINES_DIR}")
        return

    done = load_checkpoint()
    pending = [f for f in files if os.path.basename(f) not in done]
    if done:
        print(f"Checkpoint: {len(done)} file(s) already ingested, skipping.")
    if not pending:
        print("All files already ingested. Running test query only.")
    else:
        print(f"Files to ingest: {[os.path.basename(f) for f in pending]}")

    rag = await build_rag(dim)

    try:
        for fpath in pending:
            fname = os.path.basename(fpath)
            success = await ingest_one(fpath, rag)
            if success:
                done.add(fname)
                save_checkpoint(done)
            else:
                print(f"Skipping {fname} for this run.")

        if done:
            await test_query(rag)
        else:
            print("Index is empty -- skipping test query.")

    finally:
        await rag.finalize_storages()
        print("Storages finalized.")


if __name__ == "__main__":
    asyncio.run(main())