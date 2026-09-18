import uuid

import pytest

from lightrag.kg.hologres.kv import HologresKVStorage
from lightrag.kg.hologres.schema import (
    kv_schema_descriptors,
)
from lightrag.namespace import NameSpace


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


def _application_payload(row):
    return {
        key: value
        for key, value in row.items()
        if key not in {"create_time", "update_time"}
    }


async def test_hologres_kv_logical_partitions_and_crud(hologres_live_client):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    workspace_a = f"lightrag_test_kv_a_{suffix}"
    workspace_b = f"lightrag_test_kv_b_{suffix}"

    def storage(namespace, workspace):
        return HologresKVStorage(
            namespace=namespace,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=client.config,
            client=client,
        )

    primary = storage(NameSpace.KV_STORE_TEXT_CHUNKS, workspace_a)
    isolated = storage(NameSpace.KV_STORE_TEXT_CHUNKS, workspace_b)
    documents = storage(NameSpace.KV_STORE_FULL_DOCS, workspace_a)
    tracking = storage(NameSpace.KV_STORE_ENTITY_CHUNKS, workspace_a)
    entity_anchors = storage(NameSpace.KV_STORE_FULL_ENTITIES, workspace_a)
    relation_anchors = storage(NameSpace.KV_STORE_FULL_RELATIONS, workspace_a)
    storages = (
        primary,
        isolated,
        documents,
        tracking,
        entity_anchors,
        relation_anchors,
    )
    initialized = []

    try:
        for item in storages:
            await item.initialize()
            initialized.append(item)

        for descriptor in kv_schema_descriptors(schema):
            assert (
                await client.fetch_value(
                    descriptor.postcondition_sql,
                    *descriptor.postcondition_args,
                    descriptor="live.kv.catalog",
                )
                is True
            )

        await primary.upsert({"shared": {"value": "a", "old": True}})
        await isolated.upsert({"shared": {"value": "b"}})
        assert _application_payload(
            await primary.get_by_id_strict("shared")
        ) == {
            "value": "a",
            "old": True,
        }
        assert _application_payload(
            await isolated.get_by_id_strict("shared")
        ) == {"value": "b"}

        await primary.upsert({"shared": {"value": "replaced"}, "empty": {}})
        assert _application_payload(
            await primary.get_by_id_strict("shared")
        ) == {"value": "replaced"}
        assert [
            _application_payload(row) if row is not None else None
            for row in await primary.get_by_ids(["shared", "missing", "shared"])
        ] == [
            {"value": "replaced"},
            None,
            {"value": "replaced"},
        ]
        assert await primary.filter_keys({"shared", "missing"}) == {"missing"}

        protected_a = {
            "sidecar_location": "sidecar-a",
            "parse_format": "markdown",
            "content_hash": "hash-a",
            "process_options": {"mode": "a"},
            "parse_engine": "native",
            "chunk_options": {"size": 100},
        }
        await documents.upsert(
            {
                "doc": {
                    "content": "old",
                    **protected_a,
                    "ordinary": "old",
                }
            }
        )
        await documents.upsert(
            {
                "doc": {
                    "content": "",
                    "sidecar_location": None,
                    "parse_format": "",
                    "content_hash": "",
                    "process_options": None,
                    "parse_engine": "",
                    "chunk_options": {},
                    "ordinary": "new",
                }
            }
        )
        assert _application_payload(
            await documents.get_by_id_strict("doc")
        ) == {
            "content": "",
            **protected_a,
            "ordinary": "new",
        }

        await documents.upsert({"doc": {"ordinary": "newer"}})
        assert _application_payload(
            await documents.get_by_id_strict("doc")
        ) == {
            "content": "",
            **protected_a,
            "ordinary": "newer",
        }

        protected_b = {
            "sidecar_location": "sidecar-b",
            "parse_format": "html",
            "content_hash": "hash-b",
            "process_options": {"mode": "b"},
            "parse_engine": "docling",
            "chunk_options": {"size": 200},
        }
        await documents.upsert({"doc": protected_b})
        assert _application_payload(
            await documents.get_by_id_strict("doc")
        ) == {
            "content": "",
            **protected_b,
            "ordinary": "newer",
        }

        await documents.upsert(
            {
                "new-doc": {
                    "content": "new",
                    "sidecar_location": None,
                    "parse_format": "",
                    "content_hash": "",
                    "process_options": None,
                    "parse_engine": "",
                    "chunk_options": {},
                    "ordinary": "inserted",
                }
            }
        )
        assert _application_payload(
            await documents.get_by_id_strict("new-doc")
        ) == {
            "content": "new",
            "ordinary": "inserted",
        }

        await tracking.upsert({"tracking": {"chunk_ids": []}})
        await entity_anchors.upsert(
            {"entity-anchor": {"entity_names": [], "metadata": {}}}
        )
        await relation_anchors.upsert(
            {"relation-anchor": {"relation_pairs": [], "metadata": {}}}
        )
        assert _application_payload(
            await tracking.get_by_id_strict("tracking")
        ) == {"chunk_ids": []}
        assert _application_payload(
            await entity_anchors.get_by_id_strict("entity-anchor")
        ) == {
            "entity_names": [],
            "metadata": {},
        }
        assert _application_payload(
            await relation_anchors.get_by_id_strict("relation-anchor")
        ) == {
            "relation_pairs": [],
            "metadata": {},
        }

        await primary.delete(["empty"])
        assert await primary.get_by_id_strict("empty") is None
        assert await primary.is_empty() is False
        await primary.drop()
        assert await primary.is_empty() is True
        assert _application_payload(
            await isolated.get_by_id_strict("shared")
        ) == {"value": "b"}
        assert await documents.get_by_id_strict("doc") is not None
    finally:
        for item in initialized:
            try:
                await item.drop()
            finally:
                await item.finalize()
