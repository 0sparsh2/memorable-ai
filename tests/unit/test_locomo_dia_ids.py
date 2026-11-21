"""LOCOMO evidence id collection from retrieved memories."""

from benchmarks.locomo.judge import collect_dia_ids


def test_collect_dia_ids_from_metadata_lists():
    memories = [
        {"content": "fact about support group", "metadata": {"dia_ids": ["D1:3", "D1:4"]}},
        {"content": "see D2:1 for detail", "metadata": {"dia_id": "D2:1"}},
    ]
    ids = collect_dia_ids(memories)
    assert ids == {"D1:3", "D1:4", "D2:1"}
