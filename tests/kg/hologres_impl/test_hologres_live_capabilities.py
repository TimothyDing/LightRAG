import math
import os

import pytest

from lightrag.kg.hologres.capabilities import (
    ProbeKind,
    ProbeStatus,
    probe_production_capabilities,
    prove_stream_copy_capability,
    run_initial_isolated_probes,
)
from lightrag.kg.hologres.client import (
    HologresClient,
    quote_qualified_identifier,
)
from lightrag.kg.hologres.config import HologresConfig


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_initial_hologres_capabilities(hologres_live_client):
    client, schema = hologres_live_client

    production_report = await probe_production_capabilities(client)
    isolated_report = await run_initial_isolated_probes(client, schema)

    assert production_report.version.major >= 5
    assert isolated_report.supports(ProbeKind.SINGLE_AUTOCOMMIT_DDL)
    assert isolated_report.supports(ProbeKind.ASYNCPG_SETUP_RESET_BINDINGS)
    assert isolated_report.supports(ProbeKind.JSONB_ON_CONFLICT_ARRAYS_RECONNECT)
    assert isolated_report.supports(ProbeKind.LOGICAL_PARTITION)
    assert isolated_report.supports(ProbeKind.GRAPH_ADJACENCY_EXPLAIN)
    assert isolated_report.blocking_failures == ()
    by_kind = {result.kind: result for result in isolated_report.results}
    assert (
        by_kind[ProbeKind.LOGICAL_PARTITION].detail_code
        == "logical_partition_semantics_frozen"
    )
    assert (
        by_kind[ProbeKind.GRAPH_ADJACENCY_EXPLAIN].detail_code
        == "graph_adjacency_plan_partition_pruned"
    )
    assert isolated_report.supports(ProbeKind.STREAM_COPY)
    assert (
        by_kind[ProbeKind.STREAM_COPY].detail_code
        == "stream_copy_conflict_update_frozen"
    )
    hgraph_result = next(
        result
        for result in isolated_report.results
        if result.kind is ProbeKind.HGRAPH
    )
    assert hgraph_result.status is ProbeStatus.PASSED
    assert hgraph_result.detail_code == "hgraph_semantics_frozen"
    assert hgraph_result.evidence is not None
    assert [
        identifier
        for identifier, _raw_score in hgraph_result.evidence.ordered_raw_scores
    ] == [1, 4, 2, 3]
    observed_scores = dict(hgraph_result.evidence.ordered_raw_scores)
    for identifier, expected in {1: 1.0, 4: 1.0, 2: 0.0, 3: -1.0}.items():
        assert math.isfinite(observed_scores[identifier])
        assert abs(observed_scores[identifier] - expected) <= 1e-3
    assert hgraph_result.evidence.vector_filter_used is False


async def test_stream_copy_capability_proof_on_live_hologres(hologres_live_client):
    client, schema = hologres_live_client

    await client.execute_one(
        f"CREATE SCHEMA {quote_qualified_identifier(schema)}",
        descriptor="live.streamcopy.schema",
        replay_safe=False,
    )
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
    try:
        assert enabled_client.stream_copy_available is False
        version_report = await probe_production_capabilities(enabled_client)
        proven = await prove_stream_copy_capability(enabled_client, version_report)
        by_kind = {result.kind: result for result in proven.results}
        assert by_kind[ProbeKind.STREAM_COPY].status is ProbeStatus.PASSED
        assert (
            by_kind[ProbeKind.STREAM_COPY].detail_code
            == "stream_copy_conflict_update_frozen"
        )
        enabled_client.apply_capabilities(proven)
        assert enabled_client.stream_copy_available is True
        cached = await prove_stream_copy_capability(enabled_client, version_report)
        assert cached.supports(ProbeKind.STREAM_COPY) is True
    finally:
        await enabled_client.close()
