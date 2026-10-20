"""Regression tests for discovery-only news evidence metadata."""

from __future__ import annotations

import json
from unittest.mock import patch

from backtest.loaders import yahoo_client
import mcp_server
from src.tools.stock_news_tool import StockNewsTool


def test_yahoo_news_exposes_cutoff_and_related_ticker_evidence() -> None:
    provider_articles = [
        {
            "title": "Apple filing",
            "publisher": "Reuters",
            "link": "https://example.invalid/old",
            "providerPublishTime": 1704067200,
            "summary": "An older discovery result.",
            "relatedTickers": ["AAPL"],
        },
        {
            "title": "Other company",
            "publisher": "Example Media",
            "link": "https://example.invalid/new",
            "providerPublishTime": 1704153600,
            "relatedTickers": ["MSFT"],
        },
        {
            "title": "Undated result",
            "publisher": "Example Media",
            "link": "https://example.invalid/undated",
        },
    ]

    with patch.object(yahoo_client, "search_news", return_value=provider_articles) as search:
        out = json.loads(
            StockNewsTool().execute(
                code="AAPL.US",
                limit=10,
                cutoff="2024-01-01T12:00:00Z",
                require_related_ticker=True,
            )
        )

    search.assert_called_once_with("AAPL", 10)
    assert out["ok"] is True
    assert len(out["data"]["articles"]) == 1
    evidence = out["data"]["evidence"]
    assert evidence["status"] == "discovery_only"
    assert evidence["source_role"] == "news_discovery"
    assert evidence["requires_direct_read"] is True
    assert evidence["cutoff"] == "2024-01-01T12:00:00Z"
    assert evidence["cutoff_applied"] is True
    assert evidence["related_ticker_filter"] == "required"
    assert evidence["related_ticker_matches"] == 1
    assert evidence["related_ticker_mismatches"] == 0
    assert evidence["related_ticker_unverified"] == 0
    assert evidence["cutoff_excluded"] == 2
    assert evidence["relevance_rejected"] == 0
    assert out["data"]["retrieved_at"].endswith("Z")


def test_yahoo_news_reports_mismatch_without_silently_claiming_relevance() -> None:
    provider_articles = [
        {
            "title": "Mixed result",
            "publisher": "Example Media",
            "link": "https://example.invalid/mixed",
            "providerPublishTime": 1704067200,
            "relatedTickers": ["MSFT"],
        },
        {
            "title": "Unverified result",
            "publisher": "Example Media",
            "link": "https://example.invalid/unverified",
            "providerPublishTime": 1704067200,
        },
    ]

    with patch.object(yahoo_client, "search_news", return_value=provider_articles):
        out = json.loads(StockNewsTool().execute(code="AAPL.US", limit=10))

    evidence = out["data"]["evidence"]
    assert evidence["related_ticker_filter"] == "reported_only"
    assert evidence["related_ticker_matches"] == 0
    assert evidence["related_ticker_mismatches"] == 1
    assert evidence["related_ticker_unverified"] == 1
    assert len(out["data"]["articles"]) == 2


def test_news_cutoff_requires_timezone() -> None:
    out = json.loads(StockNewsTool().execute(code="AAPL.US", cutoff="2024-01-01T12:00:00"))
    assert out["ok"] is False
    assert "timezone" in out["error"]


def test_mcp_front_door_forwards_news_evidence_arguments() -> None:
    with patch.object(mcp_server, "_get_registry") as get_registry:
        registry = get_registry.return_value
        registry.execute.return_value = '{"ok": true}'
        result = mcp_server.get_stock_news(
            code="AAPL.US",
            scope="stock",
            limit=5,
            cutoff="2024-01-01T12:00:00Z",
            require_related_ticker=True,
        )

    assert result == '{"ok": true}'
    registry.execute.assert_called_once_with(
        "get_stock_news",
        {
            "scope": "stock",
            "limit": 5,
            "cutoff": "2024-01-01T12:00:00Z",
            "require_related_ticker": True,
            "code": "AAPL.US",
        },
    )
