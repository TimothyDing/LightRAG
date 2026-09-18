import os
import uuid

import pytest

from lightrag.kg.hologres.client import (
    STREAM_COPY_MIN_ROWS,
    HologresClient,
)
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.kv import HologresKVStorage
from lightrag.kg.hologres.schema import (
    vector_schema_descriptors,
)
from lightrag.kg.hologres.vector import (
    HologresVectorError,
)
from lightrag.namespace import NameSpace
from lightrag.utils import compute_mdhash_id
from tests.kg.hologres_impl._live_support import _live_vector_storage


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_stream_copy_bulk_upserts_round_trip_on_live_hologres(
    hologres_live_client,
):
    _client, schema = hologres_live_client
    enabled_client = HologresClient(
        HologresConfig.from_env(
            {
                **os.environ,
                "HOLOGRES_SCHEMA": schema,
                "HOLOGRES_STREAM_COPY_ENABLED": "true",
            }
        )
    )
    await enabled_client.open()
    suffix = uuid.uuid4().hex
    kv = HologresKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        workspace=f"lightrag_test_copy_kv_{suffix}",
        global_config={},
        embedding_func=None,
        config=enabled_client.config,
        client=enabled_client,
    )
    vectors = _live_vector_storage(
        enabled_client,
        workspace=f"lightrag_test_copy_vec_{suffix}",
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
    )
    initialized = []
    try:
        for storage in (kv, vectors):
            await storage.initialize()
            initialized.append(storage)
        assert enabled_client.stream_copy_available is True

        data = {
            f"chunk-{index:04d}": {"value": index}
            for index in range(STREAM_COPY_MIN_ROWS)
        }
        await kv.upsert(data)
        stored = await kv.get_by_id_strict("chunk-0000")
        assert stored["value"] == 0
        assert isinstance(stored["create_time"], int)
        assert isinstance(stored["update_time"], int)
        await kv.upsert(
            {key: {**payload, "rev": 2} for key, payload in data.items()}
        )
        updated = await kv.get_by_id_strict("chunk-0000")
        assert (updated["value"], updated["rev"]) == (0, 2)
        updated_last = await kv.get_by_id_strict(
            f"chunk-{STREAM_COPY_MIN_ROWS - 1:04d}"
        )
        assert updated_last["value"] == STREAM_COPY_MIN_ROWS - 1
        assert updated_last["rev"] == 2
        assert await kv.filter_keys(set(data) | {"missing"}) == {"missing"}

        vector_data = {
            f"vec-{index:04d}": {
                "content": f"content-{index}",
                "embedding": [1.0, 0.0, 0.0],
            }
            for index in range(STREAM_COPY_MIN_ROWS)
        }
        await vectors.upsert(vector_data)
        stored = await vectors.get_by_id("vec-0000")
        assert stored is not None
        assert stored["content"] == "content-0"
        await vectors.upsert(
            {
                key: {
                    "content": payload["content"] + "-updated",
                    "embedding": [0.0, 1.0, 0.0],
                }
                for key, payload in vector_data.items()
            }
        )
        updated = await vectors.get_by_id("vec-0000")
        assert updated is not None
        assert updated["content"] == "content-0-updated"
    finally:
        for storage in initialized:
            try:
                await storage.drop()
            finally:
                await storage.finalize()
        await enabled_client.close()


async def test_hologres_vector_catalog_crud_and_live_similarity_query(
    hologres_live_client,
):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    workspace_a = f"lightrag_test_vector_a_{suffix}"
    workspace_b = f"lightrag_test_vector_b_{suffix}"
    primary = _live_vector_storage(
        client,
        workspace=workspace_a,
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
    )
    isolated = _live_vector_storage(
        client,
        workspace=workspace_b,
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
    )
    relations = _live_vector_storage(
        client,
        workspace=workspace_a,
        namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
    )
    entities = _live_vector_storage(
        client,
        workspace=workspace_a,
        namespace=NameSpace.VECTOR_STORE_ENTITIES,
    )
    initialized = []

    try:
        for storage in (primary, isolated, relations, entities):
            await storage.initialize()
            initialized.append(storage)

        for descriptor in vector_schema_descriptors(schema, 3):
            assert (
                await client.fetch_value(
                    descriptor.postcondition_sql,
                    *descriptor.postcondition_args,
                    descriptor="live.vector.catalog",
                )
                is True
            )

        wrong_dimension = _live_vector_storage(
            client,
            workspace=workspace_a,
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            dimension=2,
        )
        with pytest.raises(HologresVectorError, match="initialization failed"):
            await wrong_dimension.initialize()

        known_vectors = {
            "identical": {
                "content": "identical",
                "embedding": [1.0, 0.0, 0.0],
                "source": {"kind": "known", "rank": 1},
            },
            "orthogonal": {
                "content": "orthogonal",
                "embedding": [0.0, 1.0, 0.0],
                "source": {"kind": "known", "rank": 2},
            },
            "opposite": {
                "content": "opposite",
                "embedding": [-1.0, 0.0, 0.0],
                "source": {"kind": "known", "rank": 3},
            },
            "tie": {
                "content": "tie",
                "embedding": [1.0, 0.0, 0.0],
                "source": {"kind": "known", "rank": 4},
            },
        }
        await primary.upsert(known_vectors)
        await isolated.upsert(
            {
                "identical": {
                    "content": "isolated",
                    "embedding": [0.0, 0.0, 1.0],
                    "source": {"workspace": "other"},
                }
            }
        )

        assert await primary.get_by_id("identical") == {
            "id": "identical",
            "content": "identical",
            "source": {"kind": "known", "rank": 1},
        }
        assert await primary.get_by_ids(
            ["tie", "missing", "identical", "tie"]
        ) == [
            {
                "id": "tie",
                "content": "tie",
                "source": {"kind": "known", "rank": 4},
            },
            None,
            {
                "id": "identical",
                "content": "identical",
                "source": {"kind": "known", "rank": 1},
            },
            {
                "id": "tie",
                "content": "tie",
                "source": {"kind": "known", "rank": 4},
            },
        ]
        assert await primary.get_vectors_by_ids(
            ["opposite", "missing", "orthogonal", "identical"]
        ) == {
            "identical": [1.0, 0.0, 0.0],
            "opposite": [-1.0, 0.0, 0.0],
            "orthogonal": [0.0, 1.0, 0.0],
        }
        assert (await isolated.get_by_id("identical"))["content"] == "isolated"

        await primary.upsert(
            {
                "identical": {
                    "content": "updated",
                    "embedding": [0.0, 0.0, 1.0],
                    "source": {"replacement": True},
                }
            }
        )
        assert await primary.get_by_id("identical") == {
            "id": "identical",
            "content": "updated",
            "source": {"replacement": True},
        }
        assert await primary.get_vectors_by_ids(["identical"]) == {
            "identical": [0.0, 0.0, 1.0]
        }

        await relations.upsert(
            {
                "relation-a": {
                    "content": "relation-a",
                    "embedding": [1.0, 0.0, 0.0],
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                },
                "relation-b": {
                    "content": "relation-b",
                    "embedding": [0.0, 1.0, 0.0],
                    "src_id": "Carol",
                    "tgt_id": "Alice",
                },
            }
        )
        await relations.delete_entity_relation("Alice")
        assert await relations.get_by_ids(["relation-a", "relation-b"]) == [
            None,
            None,
        ]

        entity_id = compute_mdhash_id("Alice", prefix="ent-")
        await entities.upsert(
            {
                entity_id: {
                    "content": "Alice",
                    "embedding": [1.0, 0.0, 0.0],
                }
            }
        )
        await entities.delete_entity("Alice")
        assert await entities.get_by_id(entity_id) is None

        await primary.upsert(
            {
                "near": {
                    "content": "near",
                    "embedding": [0.8, 0.6, 0.0],
                    "source": {"kind": "known", "rank": 5},
                }
            }
        )
        # Frozen contract: approx_cosine_distance returns cosine similarity,
        # results ordered nearest-first, threshold (0.2) filters the rest.
        results = await primary.query(
            "known query", top_k=4, query_embedding=[1.0, 0.0, 0.0]
        )
        assert [row["id"] for row in results] == ["tie", "near"]
        assert results[0]["content"] == "tie"
        assert results[0]["source"] == {"kind": "known", "rank": 4}
        assert abs(results[0]["distance"] - 1.0) <= 1e-3
        assert results[1]["content"] == "near"
        assert abs(results[1]["distance"] - 0.8) <= 1e-3
        assert all(isinstance(row["created_at"], int) for row in results)
        top_one = await primary.query(
            "known query", top_k=1, query_embedding=[1.0, 0.0, 0.0]
        )
        assert [row["id"] for row in top_one] == ["tie"]
        assert (
            await isolated.query(
                "known query", top_k=4, query_embedding=[1.0, 0.0, 0.0]
            )
            == []
        )
        await primary.delete(["near"])

        await primary.delete(["opposite"])
        assert await primary.get_by_id("opposite") is None
        await primary.drop()
        assert await primary.get_by_ids(["identical", "orthogonal", "tie"]) == [
            None,
            None,
            None,
        ]
        assert (await isolated.get_by_id("identical"))["content"] == "isolated"
    finally:
        for storage in initialized:
            try:
                await storage.drop()
            finally:
                await storage.finalize()
