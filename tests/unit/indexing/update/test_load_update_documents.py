# Copyright (C) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for update document loading around removed input documents.

Documents that disappear from the input must never be ignored silently:
after `graphrag update` the removed documents (and their derived text units,
entities, relationships and community reports) stay indexed and retrievable,
so the update workflow must warn that a full re-index is required to honor
removals (see microsoft/graphrag#2590).
"""

import logging
from pathlib import Path

import pandas as pd
import pytest
from graphrag.index.run.utils import create_run_context
from graphrag.index.workflows.load_update_documents import (
    load_update_documents,
    run_workflow,
)
from graphrag_input import InputConfig
from graphrag_input.input_reader_factory import create_input_reader
from graphrag_storage.file_storage import FileStorage
from graphrag_storage.memory_storage import MemoryStorage
from graphrag_storage.tables.parquet_table_provider import ParquetTableProvider

from tests.unit.config.utils import get_default_graphrag_config

_WARN_LOGGER = "graphrag.index.workflows.load_update_documents"


def _removal_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return warning messages that flag removed input documents."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "full re-index" in record.getMessage()
    ]


async def _previous_provider(*titles: str) -> ParquetTableProvider:
    """Create a previous-run documents table containing the given titles."""
    documents = pd.DataFrame({
        "id": [f"id-{title}" for title in titles],
        "human_readable_id": list(range(len(titles))),
        "title": list(titles),
        "text": [f"text of {title}" for title in titles],
    })
    provider = ParquetTableProvider(storage=MemoryStorage())
    await provider.write_dataframe("documents", documents)
    return provider


def _input_storage(tmp_path: Path, *titles: str) -> FileStorage:
    """Create input storage with one .txt file per title."""
    storage = FileStorage(base_dir=str(tmp_path))
    for title in titles:
        (tmp_path / title).write_text(f"text of {title}", encoding="utf-8")
    return storage


async def test_load_update_documents_warns_on_removed_documents(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """An update with both additions and removals must warn about the removals."""
    previous = await _previous_provider("a.txt", "b.txt")
    input_storage = _input_storage(tmp_path, "a.txt", "c.txt")
    input_reader = create_input_reader(InputConfig(), input_storage)

    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        output = await load_update_documents(input_reader, previous)

    assert output["title"].tolist() == ["c.txt"]
    warnings = _removal_warnings(caplog)
    assert len(warnings) == 1
    assert "b.txt" in warnings[0]
    assert "1 document" in warnings[0]
    assert "remain indexed and retrievable" in warnings[0]
    assert "full re-index" in warnings[0]


async def test_run_workflow_warns_on_pure_removal_update(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A removal-only update must warn even though it finds no new documents."""
    previous = await _previous_provider("a.txt", "b.txt")
    input_storage = _input_storage(tmp_path, "a.txt")
    config = get_default_graphrag_config()
    context = create_run_context(
        input_storage=input_storage,
        previous_table_provider=previous,
    )

    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        output = await run_workflow(config, context)

    # the pipeline still stops early: there is nothing new to index ...
    assert output.stop is True
    assert "No new update documents found." in caplog.text
    # ... but the removed document must be called out explicitly
    warnings = _removal_warnings(caplog)
    assert len(warnings) == 1
    assert "b.txt" in warnings[0]
    assert "remain indexed and retrievable" in warnings[0]
    assert "full re-index" in warnings[0]


async def test_run_workflow_warns_when_all_documents_removed(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """An update whose input lost every document must also warn."""
    previous = await _previous_provider("a.txt", "b.txt")
    input_storage = _input_storage(tmp_path)
    config = get_default_graphrag_config()
    context = create_run_context(
        input_storage=input_storage,
        previous_table_provider=previous,
    )

    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        output = await run_workflow(config, context)

    assert output.stop is True
    warnings = _removal_warnings(caplog)
    assert len(warnings) == 1
    assert "2 document" in warnings[0]
    assert "a.txt" in warnings[0]
    assert "b.txt" in warnings[0]
    assert "full re-index" in warnings[0]


async def test_load_update_documents_no_warning_on_additions(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """An update that only adds documents must not warn about removals."""
    previous = await _previous_provider("a.txt")
    input_storage = _input_storage(tmp_path, "a.txt", "c.txt")
    input_reader = create_input_reader(InputConfig(), input_storage)

    with caplog.at_level(logging.WARNING, logger=_WARN_LOGGER):
        output = await load_update_documents(input_reader, previous)

    assert output["title"].tolist() == ["c.txt"]
    assert _removal_warnings(caplog) == []
