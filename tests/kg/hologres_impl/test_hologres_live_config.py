import os

import pytest

from lightrag.kg.hologres.client import HologresClient, quote_qualified_identifier
from lightrag.kg.hologres.config import HologresConfig


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_env_selected_configuration_reaches_the_live_server(
    hologres_live_client,
):
    client, schema = hologres_live_client
    await client.execute_one(
        f"CREATE SCHEMA {quote_qualified_identifier(schema)}",
        descriptor="live.config.schema",
        replay_safe=False,
    )
    expected = HologresConfig.from_env(
        {**os.environ, "HOLOGRES_SCHEMA": schema}
    )

    assert client.config == expected
    assert os.environ["HOLOGRES_PASSWORD"] not in repr(expected)

    configured = HologresConfig.from_env(
        {
            **os.environ,
            "HOLOGRES_SCHEMA": schema,
            "HOLOGRES_POOL_MIN_SIZE": "1",
            "HOLOGRES_POOL_MAX_SIZE": "2",
            "HOLOGRES_CONNECT_TIMEOUT": "15",
            "HOLOGRES_COMMAND_TIMEOUT": "15",
            "HOLOGRES_POOL_ACQUIRE_TIMEOUT": "5",
            "HOLOGRES_POOL_CLOSE_TIMEOUT": "2",
            "HOLOGRES_CONNECTION_RETRIES": "0",
            "HOLOGRES_RETRY_BACKOFF": "0",
            "HOLOGRES_STATEMENT_CACHE_SIZE": "16",
        }
    )
    probe = HologresClient(configured)
    await probe.open()
    try:
        assert await probe.fetch_value(
            "SELECT current_schema::text",
            descriptor="live.config.current_schema",
        ) == schema
        assert await probe.fetch_value(
            "SELECT current_database()::text",
            descriptor="live.config.current_database",
        ) == os.environ["HOLOGRES_DATABASE"]
    finally:
        await probe.close()
