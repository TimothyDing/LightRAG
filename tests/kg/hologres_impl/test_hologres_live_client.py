import pytest

from lightrag.kg.hologres.client import quote_qualified_identifier


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_hologres_client_round_trips_parameterized_rows(
    hologres_live_client,
):
    client, schema = hologres_live_client
    await client.execute_one(
        f"CREATE SCHEMA {quote_qualified_identifier(schema)}",
        descriptor="live.client.schema",
        replay_safe=False,
    )
    table = quote_qualified_identifier(schema, "client_round_trip")

    await client.execute_one(
        f"CREATE TABLE {table} ("
        "id integer PRIMARY KEY, value text NOT NULL)",
        descriptor="live.client.create",
        replay_safe=False,
    )
    await client.execute_one(
        f"INSERT INTO {table} (id, value) VALUES ($1, $2), ($3, $4)",
        1,
        "one",
        2,
        "two",
        descriptor="live.client.insert",
        replay_safe=True,
    )

    row = await client.fetch_one(
        "SELECT id, value FROM client_round_trip WHERE id = $1",
        1,
        descriptor="live.client.fetch_one",
    )
    assert dict(row) == {"id": 1, "value": "one"}
    rows = await client.fetch_all(
        "SELECT id, value FROM client_round_trip "
        "WHERE id = ANY($1::integer[]) ORDER BY id",
        [2, 1],
        descriptor="live.client.fetch_all",
    )
    assert [dict(row) for row in rows] == [
        {"id": 1, "value": "one"},
        {"id": 2, "value": "two"},
    ]
    assert await client.fetch_value(
        "SELECT count(*) FROM client_round_trip WHERE value <> $1",
        "missing",
        descriptor="live.client.fetch_value",
    ) == 2

    await client.execute_one(
        "UPDATE client_round_trip SET value = $2 WHERE id = $1",
        2,
        "updated",
        descriptor="live.client.update",
        replay_safe=True,
    )
    assert await client.fetch_value(
        "SELECT value FROM client_round_trip WHERE id = $1",
        2,
        descriptor="live.client.updated",
    ) == "updated"
