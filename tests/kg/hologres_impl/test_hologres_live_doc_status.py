from datetime import datetime, timedelta, timezone
import uuid

import pytest

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.kg.hologres.doc_status import HologresDocStatusStorage
from lightrag.kg.hologres.schema import (
    doc_status_schema_descriptors,
)
from lightrag.namespace import NameSpace


pytestmark = [pytest.mark.integration, pytest.mark.hologres_live]


async def test_hologres_doc_status_contract(hologres_live_client):
    client, schema = hologres_live_client
    suffix = uuid.uuid4().hex
    workspace_a = f"lightrag_test_doc_status_a_{suffix}"
    workspace_b = f"lightrag_test_doc_status_b_{suffix}"

    def storage(workspace):
        return HologresDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            workspace=workspace,
            global_config={},
            embedding_func=None,
            config=client.config,
            client=client,
        )

    def document(
        *,
        status,
        created_at,
        file_path,
        content_hash=None,
        metadata=None,
        chunks_list=None,
    ):
        return {
            "content_summary": f"summary:{file_path}",
            "content_length": len(file_path),
            "file_path": file_path,
            "status": status,
            "created_at": created_at.isoformat(),
            "updated_at": created_at.isoformat(),
            "track_id": f"track:{file_path}",
            "chunks_count": 1,
            "chunks_list": chunks_list or [f"chunk:{file_path}"],
            "error_msg": None,
            "metadata": metadata or {},
            "multimodal_processed": True,
            "content_hash": content_hash,
            "producer_extension": {"source": "live-test"},
        }

    primary = storage(workspace_a)
    isolated = storage(workspace_b)
    initialized = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tied = start + timedelta(minutes=1)

    try:
        for item in (primary, isolated):
            await item.initialize()
            initialized.append(item)

        for descriptor in doc_status_schema_descriptors(schema):
            assert (
                await client.fetch_value(
                    descriptor.postcondition_sql,
                    *descriptor.postcondition_args,
                    descriptor="live.doc_status.catalog",
                )
                is True
            )

        await primary.upsert(
            {
                "unique": document(
                    status=DocStatus.PENDING,
                    created_at=start,
                    file_path="unique.md",
                ),
                "tie-a": document(
                    status=DocStatus.ANALYZING,
                    created_at=tied,
                    file_path="tie-a.md",
                    chunks_list=["tie-a-1", "tie-a-2"],
                ),
                "tie-b": document(
                    status=DocStatus.ANALYZING,
                    created_at=tied,
                    file_path="tie-b.md",
                ),
                "conflict-a": document(
                    status=DocStatus.PARSING,
                    created_at=start + timedelta(minutes=2),
                    file_path="conflict.md",
                ),
                "conflict-b": document(
                    status=DocStatus.PARSING,
                    created_at=start + timedelta(minutes=3),
                    file_path="conflict.md",
                ),
                "hash-primary": document(
                    status=DocStatus.PROCESSED,
                    created_at=start + timedelta(minutes=4),
                    file_path="hash-primary.md",
                    content_hash="same-hash",
                ),
                "hash-pointer": document(
                    status=DocStatus.PROCESSED,
                    created_at=start + timedelta(minutes=5),
                    file_path="hash-pointer.md",
                    content_hash="same-hash",
                    metadata={
                        "is_duplicate": True,
                        "original_doc_id": "hash-primary",
                    },
                ),
                "hash-third": document(
                    status=DocStatus.PROCESSED,
                    created_at=start + timedelta(minutes=6),
                    file_path="hash-third.md",
                    content_hash="same-hash",
                ),
                "delete-me": document(
                    status=DocStatus.FAILED,
                    created_at=start + timedelta(minutes=7),
                    file_path="delete.md",
                ),
            }
        )
        await isolated.upsert(
            {
                "unique": document(
                    status=DocStatus.FAILED,
                    created_at=start,
                    file_path="isolated.md",
                )
            }
        )

        counts = await primary.get_status_counts()
        assert counts[DocStatus.PENDING.value] == 1
        assert counts[DocStatus.ANALYZING.value] == 2
        assert counts[DocStatus.PARSING.value] == 2
        assert counts[DocStatus.PROCESSED.value] == 3
        assert counts[DocStatus.FAILED.value] == 1
        assert (await primary.get_all_status_counts())["all"] == 9
        assert (await isolated.get_status_counts())[DocStatus.FAILED.value] == 1

        first = await primary.get_docs_by_statuses_page(
            [DocStatus.ANALYZING], limit=1, position=CURSOR_START, strict=True
        )
        assert list(first.docs) == ["tie-a"]
        assert isinstance(first.next_position, CursorAfter)
        second = await primary.get_docs_by_statuses_page(
            [DocStatus.ANALYZING],
            limit=1,
            position=first.next_position,
            strict=True,
        )
        assert list(second.docs) == ["tie-b"]
        assert isinstance(second.next_position, CursorAfter)
        exhausted = await primary.get_docs_by_statuses_page(
            [DocStatus.ANALYZING],
            limit=1,
            position=second.next_position,
            strict=True,
        )
        assert exhausted.docs == {}
        assert exhausted.next_position is CURSOR_END

        scheduling = await primary.get_docs_by_ids(
            ["tie-b", "missing", "tie-a", "tie-b"], strict=True
        )
        assert list(scheduling) == ["tie-b", "tie-a"]
        assert scheduling["tie-a"].status is DocStatus.ANALYZING
        hydrated = await primary.get_full_docs_by_ids(
            ["tie-a", "missing"], strict=True
        )
        assert hydrated["tie-a"].chunks_list == ["tie-a-1", "tie-a-2"]
        assert "missing" not in hydrated

        holder = await primary.get_doc_by_content_hash("same-hash")
        assert holder is not None and holder[0] == "hash-primary"
        excluded = await primary.get_doc_by_content_hash(
            "same-hash", exclude_doc_id="hash-primary"
        )
        assert excluded is not None and excluded[0] == "hash-third"

        assert isinstance(
            await primary.resolve_doc_source_strict("absent.md"), SourceAbsent
        )
        unique = await primary.resolve_doc_source_strict("unique.md")
        assert isinstance(unique, SourceUnique)
        assert unique.doc_id == "unique"
        conflict = await primary.resolve_doc_source_strict("conflict.md")
        assert isinstance(conflict, SourceConflict)
        assert conflict.candidate_count == 2
        assert conflict.sample_doc_ids == ("conflict-a", "conflict-b")

        conflicts = await primary.list_source_conflicts_page(
            limit=1, position=CURSOR_START
        )
        assert [entry.canonical_source_key for entry in conflicts.conflicts] == [
            "conflict.md"
        ]
        assert isinstance(conflicts.next_position, CursorAfter)
        no_more_conflicts = await primary.list_source_conflicts_page(
            limit=1, position=conflicts.next_position
        )
        assert no_more_conflicts.conflicts == ()
        assert no_more_conflicts.next_position is CURSOR_END

        dry_run = await primary.repair_source_conflict(
            "conflict.md",
            primary_doc_id="conflict-a",
            expected_candidate_count=0,
            expected_candidate_fingerprint="",
            dry_run=True,
        )
        assert dry_run.candidate_count == 2
        assert dry_run.demoted_sample_doc_ids == ("conflict-b",)
        assert dry_run.committed is False
        repaired = await primary.repair_source_conflict(
            "conflict.md",
            primary_doc_id="conflict-a",
            expected_candidate_count=dry_run.candidate_count,
            expected_candidate_fingerprint=dry_run.fingerprint,
            dry_run=False,
        )
        assert repaired.committed is True
        resolved = await primary.resolve_doc_source_strict("conflict.md")
        assert isinstance(resolved, SourceUnique)
        assert resolved.doc_id == "conflict-a"
        demoted = await primary.get_by_id_strict("conflict-b")
        assert demoted is not None
        assert demoted["metadata"] == {
            "is_duplicate": True,
            "original_doc_id": "conflict-a",
        }
        assert demoted["chunks_list"] == ["chunk:conflict.md"]

        before_update = await primary.get_by_id_strict("unique")
        assert before_update is not None
        await primary.update_doc_status_fields(
            "unique",
            {
                "status": DocStatus.PROCESSING,
                "updated_at": start + timedelta(days=1),
                "metadata": {"targeted": True},
            },
        )
        after_update = await primary.get_by_id_strict("unique")
        assert after_update is not None
        assert after_update["status"] == DocStatus.PROCESSING
        assert after_update["created_at"] == before_update["created_at"]
        assert after_update["chunks_list"] == before_update["chunks_list"]
        assert after_update["metadata"] == {"targeted": True}

        await primary.delete(["delete-me"])
        assert await primary.get_by_id_strict("delete-me") is None
        await primary.drop()
        assert await primary.is_empty() is True
        isolated_row = await isolated.get_by_id_strict("unique")
        assert isolated_row is not None
        assert isolated_row["file_path"] == "isolated.md"
    finally:
        for item in initialized:
            try:
                await item.drop()
            finally:
                await item.finalize()
