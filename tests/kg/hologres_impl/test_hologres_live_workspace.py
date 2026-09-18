import pytest

from lightrag.kg.hologres.kv import HologresKVStorage
from lightrag.namespace import NameSpace


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_workspace_override_selects_an_isolated_live_partition(
    hologres_live_client,
    monkeypatch,
):
    client, _schema = hologres_live_client

    def storage(workspace):
        return HologresKVStorage(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=client.config,
            client=client,
        )

    monkeypatch.delenv("HOLOGRES_WORKSPACE", raising=False)
    instance = storage("instance-workspace")
    assert instance.workspace == "instance-workspace"
    assert storage("").workspace == "default"

    monkeypatch.setenv("HOLOGRES_WORKSPACE", "override-workspace")
    overridden = storage("instance-workspace")
    assert overridden.workspace == "override-workspace"

    initialized = []
    try:
        for item in (instance, overridden):
            await item.initialize()
            initialized.append(item)

        await instance.upsert({"shared": {"value": "instance"}})
        await overridden.upsert({"shared": {"value": "override"}})

        instance_row = await instance.get_by_id_strict("shared")
        override_row = await overridden.get_by_id_strict("shared")
        assert instance_row is not None
        assert override_row is not None
        assert instance_row["value"] == "instance"
        assert override_row["value"] == "override"
    finally:
        for item in initialized:
            try:
                await item.drop()
            finally:
                await item.finalize()
