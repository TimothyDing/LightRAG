"""Shared helpers for isolated Hologres live integration modules."""


from lightrag.kg.hologres.graph import HologresGraphStorage
from lightrag.kg.hologres.vector import HologresVectorStorage
from lightrag.namespace import NameSpace


class _LiveVectorEmbedding:
    def __init__(self, dimension=3):
        self.embedding_dim = dimension

    async def __call__(self, _texts, **_kwargs):
        raise AssertionError("Live vector tests supply normalized embeddings")


def _live_vector_storage(client, *, workspace, namespace, dimension=3):
    return HologresVectorStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 2,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2
            },
        },
        embedding_func=_LiveVectorEmbedding(dimension),
        meta_fields={"content", "source", "src_id", "tgt_id"},
        config=client.config,
        client=client,
    )


def _live_graph_storage(client, *, workspace):
    return HologresGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        config=client.config,
        client=client,
    )
