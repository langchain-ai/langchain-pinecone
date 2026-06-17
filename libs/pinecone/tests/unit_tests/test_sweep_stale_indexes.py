"""Unit tests for _sweep_stale_langchain_test_indexes.

The function is defined in the integration test module; these tests verify that the
name/age filter is correct without making any network calls.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pinecone  # type: ignore[import-untyped]
from pytest_mock import MockerFixture

from tests.integration_tests.test_vectorstores import (
    _sweep_stale_langchain_test_indexes,
)

PREFIX = "langchain-test-vectorstores-"


def _make_idx(name: str) -> MagicMock:
    idx = MagicMock()
    idx.name = name
    return idx


def _ts(delta: timedelta) -> str:
    return (datetime.utcnow() + delta).strftime("%Y%m%d%H%M%S")


# ---------------------------------------------------------------------------
# filter correctness


def test_deletes_stale_matching_index(mocker: MockerFixture) -> None:
    client = mocker.Mock(spec=pinecone.Pinecone)
    old_name = f"{PREFIX}{_ts(timedelta(hours=-3))}"
    client.list_indexes.return_value = [_make_idx(old_name)]

    _sweep_stale_langchain_test_indexes(client, PREFIX, age_threshold_s=7200)

    client.delete_index.assert_called_once_with(old_name)


def test_skips_recent_matching_index(mocker: MockerFixture) -> None:
    client = mocker.Mock(spec=pinecone.Pinecone)
    recent_name = f"{PREFIX}{_ts(timedelta(minutes=-5))}"
    client.list_indexes.return_value = [_make_idx(recent_name)]

    _sweep_stale_langchain_test_indexes(client, PREFIX, age_threshold_s=7200)

    client.delete_index.assert_not_called()


def test_skips_non_matching_prefix(mocker: MockerFixture) -> None:
    client = mocker.Mock(spec=pinecone.Pinecone)
    foreign_name = f"idx-{_ts(timedelta(days=-7))}-abcdef"
    client.list_indexes.return_value = [_make_idx(foreign_name)]

    _sweep_stale_langchain_test_indexes(client, PREFIX, age_threshold_s=7200)

    client.delete_index.assert_not_called()


def test_skips_prefix_match_with_non_timestamp_suffix(mocker: MockerFixture) -> None:
    client = mocker.Mock(spec=pinecone.Pinecone)
    bad_suffix = f"{PREFIX}not-a-timestamp"
    client.list_indexes.return_value = [_make_idx(bad_suffix)]

    _sweep_stale_langchain_test_indexes(client, PREFIX, age_threshold_s=7200)

    client.delete_index.assert_not_called()


def test_deletes_multiple_stale_skips_recent(mocker: MockerFixture) -> None:
    client = mocker.Mock(spec=pinecone.Pinecone)
    old1 = f"{PREFIX}{_ts(timedelta(hours=-4))}"
    old2 = f"{PREFIX}{_ts(timedelta(hours=-3, minutes=-1))}"
    recent = f"{PREFIX}{_ts(timedelta(minutes=-10))}"
    foreign = f"idx-{_ts(timedelta(days=-5))}-deadbeef"
    client.list_indexes.return_value = [
        _make_idx(old1),
        _make_idx(old2),
        _make_idx(recent),
        _make_idx(foreign),
    ]

    _sweep_stale_langchain_test_indexes(client, PREFIX, age_threshold_s=7200)

    assert client.delete_index.call_count == 2
    client.delete_index.assert_any_call(old1)
    client.delete_index.assert_any_call(old2)


# ---------------------------------------------------------------------------
# error resilience


def test_list_indexes_exception_is_swallowed(mocker: MockerFixture) -> None:
    client = mocker.Mock(spec=pinecone.Pinecone)
    client.list_indexes.side_effect = Exception("network error")

    _sweep_stale_langchain_test_indexes(client, PREFIX, age_threshold_s=7200)

    client.delete_index.assert_not_called()


def test_delete_index_exception_continues(mocker: MockerFixture) -> None:
    client = mocker.Mock(spec=pinecone.Pinecone)
    old1 = f"{PREFIX}{_ts(timedelta(hours=-3))}"
    old2 = f"{PREFIX}{_ts(timedelta(hours=-4))}"
    client.list_indexes.return_value = [_make_idx(old1), _make_idx(old2)]
    client.delete_index.side_effect = Exception("delete failed")

    _sweep_stale_langchain_test_indexes(client, PREFIX, age_threshold_s=7200)

    assert client.delete_index.call_count == 2
