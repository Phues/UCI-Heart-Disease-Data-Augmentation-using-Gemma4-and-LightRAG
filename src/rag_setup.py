import asyncio
import os
import ollama as _ollama
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag/index"))
GUIDELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rag/guidelines"))

def get_embedding_dim() -> int:
    r = _ollama.embeddings(model="nomic-embed-text", prompt="test")
    dim = len(r["embedding"])
    print(f"Detected embedding dim: {dim}")
    return dim

async def build_rag() -> LightRAG:
    dim = get_embedding_dim()
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="gemma4:e4b",
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {
                "num_ctx": 4096,    # GPU can handle this now
                "num_gpu": 99,      # offload all layers to GPU
            }
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=dim,
            max_token_size=512,
            func=lambda texts: ollama_embed(
                texts,
                embed_model="nomic-embed-text",
                host="http://localhost:11434"
            )
        ),
        chunk_token_size=512,
        chunk_overlap_token_size=64,
        # entity_extract_max_gleaning restored to default (1)
    )
    await rag.initialize_storages()
    return rag

async def ingest_guidelines(rag: LightRAG):
    files = [f for f in os.listdir(GUIDELINES_DIR) if f.endswith(".txt")]
    if not files:
        print("No .txt files found in", GUIDELINES_DIR)
        return
    for fname in files:
        fpath = os.path.join(GUIDELINES_DIR, fname)
        with open(fpath, "r") as f:
            content = f.read()
        print(f"Ingesting: {fname} ({len(content)} chars)...")
        await rag.ainsert(content)
    print("Ingestion complete.")

async def test_query(rag: LightRAG):
    result = await rag.aquery(
        "What are the valid cholesterol and blood pressure ranges for a 55 year old male at heart disease risk?",
        param=QueryParam(mode="hybrid")  # full graph + vector search
    )
    print("\n--- Test Query Result ---")
    print(result)

async def main():
    rag = await build_rag()
    await ingest_guidelines(rag)
    await test_query(rag)
    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
