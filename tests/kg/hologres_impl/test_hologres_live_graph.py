import os
import uuid

import pytest

from lightrag.kg.hologres.capabilities import (
    ProbeStatus,
    probe_age_graph_capability,
)
from lightrag.kg.hologres.client import (
    HologresClient,
)
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.graph_age import HologresAGEGraphStorage, _ID_CHUNK_SIZE
from lightrag.kg.hologres.schema import (
    graph_schema_descriptors,
)
from lightrag.namespace import NameSpace
from tests.kg.hologres_impl._live_support import _live_graph_storage


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_age_graph_capability_probe_on_live_hologres(hologres_live_client):
    client, schema = hologres_live_client
    del schema
    age_client = HologresClient(
        HologresConfig.from_env(
            {**os.environ, "HOLOGRES_AGE_SEARCH_PATH": "true"}
        )
    )
    await age_client.open()
    try:
        result = await probe_age_graph_capability(age_client)

        assert result.status is ProbeStatus.PASSED, result.detail_code
        assert result.detail_code == "age_graph_semantics_frozen"
        leaked = await client.fetch_value(
            "SELECT count(*) FROM pg_namespace "
            "WHERE nspname LIKE 'lightrag_test_age_%'",
            descriptor="live.age.leak_check",
        )
        assert leaked == 0
    finally:
        await age_client.close()


async def test_hologres_age_graph_contract(hologres_live_client):
    client, schema = hologres_live_client
    del schema
    suffix = uuid.uuid4().hex[:10]
    workspace_a = f"agews{suffix}a"
    workspace_b = f"agews{suffix}b"
    config = HologresConfig.from_env(dict(os.environ))

    def storage(workspace):
        return HologresAGEGraphStorage(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=workspace,
            global_config={"max_graph_nodes": 1000},
            embedding_func=None,
            config=config,
        )

    primary = storage(workspace_a)
    isolated = storage(workspace_b)
    initialized = []
    try:
        for item in (primary, isolated):
            await item.initialize()
            initialized.append(item)
        assert primary._delegate is None, "live AGE probe unexpectedly failed"

        tricky = "it's a \"tricky\" \\ value;\nwith 中文 🚀"
        await primary.upsert_node(
            "alpha", {"entity_id": "alpha", "description": tricky}
        )
        assert await primary.has_node("alpha") is True
        assert await primary.get_node("alpha") == {
            "entity_id": "alpha",
            "description": tricky,
        }
        # Second upsert merges: omitted keys survive, shared keys update.
        await primary.upsert_node(
            "alpha", {"entity_id": "alpha", "entity_type": "org"}
        )
        assert await primary.get_node("alpha") == {
            "entity_id": "alpha",
            "description": tricky,
            "entity_type": "org",
        }

        await primary.upsert_node("beta", {"entity_id": "beta"})
        await primary.upsert_edge(
            "beta", "alpha", {"weight": 2.5, "keywords": "k1"}
        )
        assert await primary.has_edge("alpha", "beta") is True
        assert await primary.get_edge("alpha", "beta") == {
            "weight": 2.5,
            "keywords": "k1",
        }
        assert await primary.get_edge("beta", "alpha") == {
            "weight": 2.5,
            "keywords": "k1",
        }
        # Edge properties REPLACE the stored map.
        await primary.upsert_edge("alpha", "beta", {"weight": 9.0})
        assert await primary.get_edge("beta", "alpha") == {"weight": 9.0}

        # Auto-created endpoint stubs and a self-loop counting twice.
        await primary.upsert_edge("alpha", "ghost", {"weight": 1.0})
        assert await primary.get_node("ghost") == {"entity_id": "ghost"}
        await primary.upsert_edge("alpha", "alpha", {"weight": 0.5})
        assert await primary.node_degree("alpha") == 4
        assert await primary.node_degrees_batch(["alpha", "beta", "missing"]) == {
            "alpha": 4,
            "beta": 1,
            "missing": 0,
        }
        assert await primary.edge_degree("alpha", "beta") == 5

        assert await primary.get_node_edges("missing") is None
        await primary.upsert_node("lonely", {"entity_id": "lonely"})
        assert await primary.get_node_edges("lonely") == []
        assert await primary.get_node_edges("alpha") == [
            ("alpha", "alpha"),
            ("alpha", "beta"),
            ("alpha", "ghost"),
        ]

        nodes = await primary.get_nodes_batch(["alpha", "beta", "missing"])
        assert set(nodes) == {"alpha", "beta"}
        assert await primary.has_nodes_batch(["alpha", "missing"]) == {"alpha"}
        edges = await primary.get_edges_batch(
            [{"src": "beta", "tgt": "alpha"}, {"src": "alpha", "tgt": "missing"}]
        )
        assert edges == {("beta", "alpha"): {"weight": 9.0}}

        assert await primary.get_all_labels() == [
            "alpha",
            "beta",
            "ghost",
            "lonely",
        ]
        assert (await primary.get_popular_labels(limit=2)) == ["alpha", "beta"]
        assert await primary.search_labels("alp") == ["alpha"]

        all_nodes = await primary.get_all_nodes()
        assert [node["id"] for node in all_nodes] == [
            "alpha",
            "beta",
            "ghost",
            "lonely",
        ]
        all_edges = await primary.get_all_edges()
        assert [(edge["source"], edge["target"]) for edge in all_edges] == [
            ("alpha", "alpha"),
            ("alpha", "beta"),
            ("alpha", "ghost"),
        ]

        kg = await primary.get_knowledge_graph("beta", max_depth=1)
        assert kg.nodes[0].id == "beta"
        assert {node.id for node in kg.nodes} == {"alpha", "beta"}
        assert kg.is_truncated is False
        wildcard = await primary.get_knowledge_graph("*", max_nodes=2)
        assert {node.id for node in wildcard.nodes} == {"alpha", "beta"}
        assert wildcard.is_truncated is True

        # Workspace isolation: the second graph sees none of it.
        assert await isolated.get_all_labels() == []
        await isolated.upsert_node("alpha", {"entity_id": "alpha"})
        assert await isolated.get_node("alpha") == {"entity_id": "alpha"}
        assert await isolated.node_degree("alpha") == 0

        await primary.remove_edges([("ghost", "alpha")])
        assert await primary.get_edge("alpha", "ghost") is None
        assert await primary.has_node("ghost") is True
        await primary.delete_node("ghost")
        assert await primary.has_node("ghost") is False
        await primary.remove_nodes(["lonely", "missing"])
        assert await primary.has_node("lonely") is False

        assert await primary.drop() == {
            "status": "success",
            "message": "data dropped",
        }
        assert await primary.get_all_labels() == []
        assert await isolated.get_node("alpha") == {"entity_id": "alpha"}
    finally:
        cleanup_client = HologresClient(
            HologresConfig.from_env(
                {**os.environ, "HOLOGRES_AGE_SEARCH_PATH": "true"}
            )
        )
        await cleanup_client.open()
        try:
            for workspace in (workspace_a, workspace_b):
                graph = f"lightrag_age_{workspace}"
                exists = await cleanup_client.fetch_value(
                    "SELECT EXISTS (SELECT 1 FROM pg_namespace "
                    "WHERE nspname = $1)",
                    graph,
                    descriptor="live.age.cleanup.check",
                )
                if exists:
                    await cleanup_client.call_age_procedure(
                        "drop_graph",
                        graph,
                        True,
                        descriptor="live.age.cleanup.drop",
                        replay_safe=False,
                    )
        finally:
            await cleanup_client.close()
            for item in initialized:
                await item.finalize()


async def test_hologres_age_graph_preserves_edges_across_hydration_chunks(
    hologres_live_client,
):
    _client, _schema = hologres_live_client
    workspace = f"agechunk{uuid.uuid4().hex[:10]}"
    graph_name = f"lightrag_age_{workspace}"
    storage = HologresAGEGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        config=HologresConfig.from_env(
            {
                **os.environ,
                "HOLOGRES_POOL_MIN_SIZE": "1",
                "HOLOGRES_POOL_MAX_SIZE": "2",
                "HOLOGRES_POOL_CLOSE_TIMEOUT": "60",
            }
        ),
    )
    neighbours = [f"n{index:03d}" for index in range(_ID_CHUNK_SIZE + 1)]
    expected_edges = {("hub", neighbour) for neighbour in neighbours}

    await storage.initialize()
    try:
        assert storage._delegate is None, "live AGE probe unexpectedly failed"
        await storage.upsert_nodes_batch(
            [("hub", {"entity_id": "hub"})]
            + [
                (neighbour, {"entity_id": neighbour})
                for neighbour in neighbours
            ]
        )
        await storage.upsert_edges_batch(
            [
                ("hub", neighbour, {"weight": 1})
                for neighbour in neighbours
            ]
        )

        graph = await storage.get_knowledge_graph(
            "hub", max_depth=1, max_nodes=_ID_CHUNK_SIZE + 2
        )

        assert {node.id for node in graph.nodes} == {"hub", *neighbours}
        assert graph.is_truncated is False
        assert {(edge.source, edge.target) for edge in graph.edges} == expected_edges
    finally:
        await storage.drop()
        await storage.finalize()

        cleanup_client = HologresClient(
            HologresConfig.from_env(
                {**os.environ, "HOLOGRES_AGE_SEARCH_PATH": "true"}
            )
        )
        await cleanup_client.open()
        try:
            graph_exists = await cleanup_client.fetch_value(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = $1)",
                graph_name,
                descriptor="live.age.chunk.cleanup.check",
            )
            if graph_exists:
                await cleanup_client.call_age_procedure(
                    "drop_graph",
                    graph_name,
                    True,
                    descriptor="live.age.chunk.cleanup.drop",
                    replay_safe=False,
                )
        finally:
            await cleanup_client.close()


async def test_hologres_graph_two_table_contract(hologres_live_client):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    primary = _live_graph_storage(
        client, workspace=f"lightrag_test_graph_a_{suffix}"
    )
    isolated = _live_graph_storage(
        client, workspace=f"lightrag_test_graph_b_{suffix}"
    )
    initialized = []

    try:
        for storage in (primary, isolated):
            await storage.initialize()
            initialized.append(storage)

        for descriptor in graph_schema_descriptors(schema):
            assert (
                await client.fetch_value(
                    descriptor.postcondition_sql,
                    *descriptor.postcondition_args,
                    descriptor="live.graph.catalog",
                )
                is True
            )

        # Node upsert MERGES properties and forces entity_id = node id.
        await primary.upsert_node(
            "Alice", {"entity_id": "stale", "description": "first", "keep": "x"}
        )
        await primary.upsert_node(
            "Alice", {"entity_id": "Alice", "description": "second"}
        )
        assert await primary.get_node("Alice") == {
            "entity_id": "Alice",
            "description": "second",
            "keep": "x",
        }

        await primary.upsert_nodes_batch(
            [
                ("Bob", {"entity_id": "Bob", "kind": "person"}),
                ("Carol", {"entity_id": "Carol"}),
                ("Bob", {"entity_id": "Bob", "kind": "engineer"}),
            ]
        )
        assert (await primary.get_node("Bob"))["kind"] == "engineer"
        assert await primary.has_nodes_batch(["Alice", "Bob", "Ghost"]) == {
            "Alice",
            "Bob",
        }
        assert await primary.get_nodes_batch(["Carol", "Ghost"]) == {
            "Carol": {"entity_id": "Carol"}
        }

        # Edge upsert REPLACES properties and stores the canonical order.
        await primary.upsert_edge("Bob", "Alice", {"weight": 1, "note": "ab"})
        assert await primary.get_edge("Bob", "Alice") == {"weight": 1, "note": "ab"}
        await primary.upsert_edge("Alice", "Bob", {"weight": 2})
        assert await primary.get_edge("Bob", "Alice") == {"weight": 2}
        assert await primary.has_edge("Alice", "Bob") is True

        await primary.upsert_edge("Alice", "Alice", {"loop": True})
        # Missing endpoints are auto-created as entity_id-only stubs.
        await primary.upsert_edge("Alice", "Zed", {"weight": 1})
        assert await primary.get_node("Zed") == {"entity_id": "Zed"}
        await primary.upsert_edges_batch(
            [("Carol", "Bob", {"w": 1}), ("Bob", "Carol", {"w": 9})]
        )
        assert await primary.get_edge("Carol", "Bob") == {"w": 9}

        # A self-loop counts twice in degree but appears once in adjacency.
        assert await primary.node_degree("Alice") == 4
        assert await primary.edge_degree("Alice", "Bob") == 6
        assert await primary.node_degrees_batch(["Alice", "Bob", "Ghost"]) == {
            "Alice": 4,
            "Bob": 2,
            "Ghost": 0,
        }
        assert await primary.get_node_edges("Alice") == [
            ("Alice", "Alice"),
            ("Alice", "Bob"),
            ("Alice", "Zed"),
        ]
        assert await primary.get_node_edges("Ghost") is None
        assert await primary.get_nodes_edges_batch(["Alice", "Bob"]) == {
            "Alice": [("Alice", "Alice"), ("Alice", "Bob"), ("Alice", "Zed")],
            "Bob": [("Bob", "Alice"), ("Bob", "Carol")],
        }
        assert await primary.get_edges_batch(
            [{"src": "Bob", "tgt": "Alice"}, {"src": "Alice", "tgt": "Ghost"}]
        ) == {("Bob", "Alice"): {"weight": 2}}

        # Workspace isolation plus bytewise ("C") ordering on the live server.
        await isolated.upsert_nodes_batch(
            [
                ("a", {"entity_id": "a"}),
                ("B", {"entity_id": "B"}),
                ("_x", {"entity_id": "_x"}),
                ("100%_sure", {"entity_id": "100%_sure"}),
                ("100abc", {"entity_id": "100abc"}),
            ]
        )
        assert await isolated.get_node("Alice") is None
        assert await isolated.get_all_edges() == []
        assert await isolated.get_popular_labels(limit=3) == ["100%_sure", "100abc", "B"]
        assert await isolated.get_all_labels() == [
            "100%_sure",
            "100abc",
            "B",
            "_x",
            "a",
        ]
        # LIKE wildcards in the query are escaped, so '%'/'_' match literally.
        assert await isolated.search_labels("100%_s") == ["100%_sure"]

        assert await primary.get_all_labels() == ["Alice", "Bob", "Carol", "Zed"]
        assert await primary.get_popular_labels(limit=2) == ["Alice", "Bob"]
        assert await primary.search_labels("alice") == ["Alice"]
        assert await primary.search_labels("nomatch") == []

        all_nodes = await primary.get_all_nodes()
        assert [node["id"] for node in all_nodes] == [
            "Alice",
            "Bob",
            "Carol",
            "Zed",
        ]
        assert all(node["entity_id"] == node["id"] for node in all_nodes)
        assert [
            (edge["source"], edge["target"]) for edge in await primary.get_all_edges()
        ] == [
            ("Alice", "Alice"),
            ("Alice", "Bob"),
            ("Alice", "Zed"),
            ("Bob", "Carol"),
        ]

        # Knowledge graph: seed pinned first, ranked by depth/degree/id.
        one_hop = await primary.get_knowledge_graph(
            "Alice", max_depth=1, max_nodes=10
        )
        assert [node.id for node in one_hop.nodes] == ["Alice", "Bob", "Zed"]
        assert one_hop.is_truncated is False
        assert [(edge.source, edge.target) for edge in one_hop.edges] == [
            ("Alice", "Alice"),
            ("Alice", "Bob"),
            ("Alice", "Zed"),
        ]
        assert all(edge.type == "DIRECTED" for edge in one_hop.edges)
        assert all(
            edge.id == f"{edge.source}-{edge.target}" for edge in one_hop.edges
        )

        two_hop = await primary.get_knowledge_graph(
            "Alice", max_depth=2, max_nodes=10
        )
        assert {node.id for node in two_hop.nodes} == {
            "Alice",
            "Bob",
            "Carol",
            "Zed",
        }
        assert two_hop.is_truncated is False

        truncated = await primary.get_knowledge_graph(
            "Alice", max_depth=2, max_nodes=2
        )
        assert [node.id for node in truncated.nodes] == ["Alice", "Bob"]
        assert truncated.is_truncated is True

        wildcard = await primary.get_knowledge_graph("*", max_nodes=10)
        assert [node.id for node in wildcard.nodes] == [
            "Alice",
            "Bob",
            "Carol",
            "Zed",
        ]
        assert wildcard.is_truncated is False

        missing = await primary.get_knowledge_graph("Ghost")
        assert missing.nodes == [] and missing.edges == []
        assert missing.is_truncated is False

        # Deletions: edges always go before their nodes.
        await primary.remove_edges([("Bob", "Alice")])
        assert await primary.has_edge("Alice", "Bob") is False
        assert await primary.has_node("Bob") is True
        await primary.delete_node("Zed")
        assert await primary.has_node("Zed") is False
        assert await primary.has_edge("Alice", "Zed") is False
        await primary.remove_nodes(["Carol"])
        assert await primary.get_node_edges("Bob") == []
        assert await primary.node_degree("Alice") == 2

        assert await primary.drop() == {
            "status": "success",
            "message": "data dropped",
        }
        assert await primary.get_all_labels() == []
        assert await primary.get_all_edges() == []
        assert await isolated.get_node("a") == {"entity_id": "a"}
    finally:
        for storage in initialized:
            try:
                await storage.drop()
            finally:
                await storage.finalize()
