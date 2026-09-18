from datetime import datetime, timezone
import uuid

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import (
    DocStatus,
)
from lightrag.kg.hologres.doc_status import HologresDocStatusStorage
from lightrag.kg.hologres.graph import HologresGraphStorage
from lightrag.kg.hologres.kv import HologresKVStorage
from lightrag.kg.hologres.vector import (
    HologresVectorStorage,
)
from lightrag.utils import EmbeddingFunc, Tokenizer


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_lightrag_env_selected_hologres_stack_round_trip(
    hologres_live_client, tmp_path, monkeypatch
):
    """The API server selects backends by name from the environment; prove
    that exact path live: LightRAG builds every storage type from the
    ``Hologres*`` names, each storage resolves its connection from
    ``HOLOGRES_*`` variables, and basic round-trips succeed."""
    _client, schema = hologres_live_client
    monkeypatch.setenv("HOLOGRES_SCHEMA", schema)

    async def fixed_embedding(texts, **_kwargs):
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(texts))

    async def no_llm(*_args, **_kwargs):
        raise AssertionError("The storage smoke never calls the LLM")

    class _OrdinalTokenizer:
        def encode(self, content: str) -> list[int]:
            return [ord(ch) for ch in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(token) for token in tokens)

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"lightrag_test_stack_{uuid.uuid4().hex}",
        kv_storage="HologresKVStorage",
        vector_storage="HologresVectorStorage",
        graph_storage="HologresGraphStorage",
        doc_status_storage="HologresDocStatusStorage",
        embedding_func=EmbeddingFunc(embedding_dim=8, func=fixed_embedding),
        llm_model_func=no_llm,
        tokenizer=Tokenizer("live-smoke-tokenizer", _OrdinalTokenizer()),
    )
    await rag.initialize_storages()
    try:
        assert type(rag.full_docs) is HologresKVStorage
        assert type(rag.chunks_vdb) is HologresVectorStorage
        assert type(rag.chunk_entity_relation_graph) is HologresGraphStorage
        assert type(rag.doc_status) is HologresDocStatusStorage

        await rag.full_docs.upsert({"doc-1": {"content": "hello hologres"}})
        stored = await rag.full_docs.get_by_id("doc-1")
        assert stored["content"] == "hello hologres"

        await rag.chunk_entity_relation_graph.upsert_node(
            "Alice", {"entity_id": "Alice", "description": "smoke"}
        )
        assert await rag.chunk_entity_relation_graph.has_node("Alice")

        await rag.chunks_vdb.upsert(
            {
                "chunk-1": {
                    "content": "hello hologres",
                    "embedding": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "full_doc_id": "doc-1",
                    "file_path": "doc.txt",
                }
            }
        )
        chunk = await rag.chunks_vdb.get_by_id("chunk-1")
        assert chunk["full_doc_id"] == "doc-1"

        now = datetime.now(timezone.utc).isoformat()
        await rag.doc_status.upsert(
            {
                "doc-1": {
                    "content_summary": "summary",
                    "content_length": 14,
                    "file_path": "doc.txt",
                    "status": DocStatus.PENDING,
                    "created_at": now,
                    "updated_at": now,
                    "track_id": "track-1",
                    "chunks_count": 1,
                    "chunks_list": ["chunk-1"],
                    "error_msg": None,
                    "metadata": {},
                    "multimodal_processed": True,
                    "content_hash": None,
                    "producer_extension": {},
                }
            }
        )
        counts = await rag.doc_status.get_status_counts()
        assert counts[DocStatus.PENDING.value] == 1
    finally:
        try:
            for storage in (
                rag.full_docs,
                rag.chunks_vdb,
                rag.chunk_entity_relation_graph,
                rag.doc_status,
            ):
                await storage.drop()
        finally:
            await rag.finalize_storages()
