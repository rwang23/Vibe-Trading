"""The v0.1.15 registry front door retains full market-data evidence."""

from __future__ import annotations

import json

import mcp_server
from src.tools import market_data_tool


_GET_MARKET_DATA = getattr(mcp_server.get_market_data, "fn", None) or mcp_server.get_market_data


def test_mcp_market_data_preserves_registry_loader_hook(monkeypatch):
    calls = {}

    class Registry:
        def execute(self, name, params):
            calls["name"] = name
            calls.update(params)
            return json.dumps({"ok": True})

    monkeypatch.setattr(mcp_server, "_get_registry", lambda: Registry())
    result = _GET_MARKET_DATA(
        codes=["AAPL.US"],
        start_date="2026-07-30",
        end_date="2026-08-03",
        source="auto",
        interval="1D",
        max_rows=0,
    )

    assert json.loads(result)["ok"] is True
    assert calls["name"] == "get_market_data"
    assert calls["loader_resolver"] is mcp_server._get_loader


def test_registry_market_data_tool_requests_full_provenance(monkeypatch):
    calls = {}

    def fake_fetch_market_data_json(**kwargs):
        calls.update(kwargs)
        return json.dumps({"_provenance": {"schema_version": "market-data-provenance-v1"}})

    monkeypatch.setattr(market_data_tool, "fetch_market_data_json", fake_fetch_market_data_json)
    result = market_data_tool.MarketDataTool().execute(
        codes=["AAPL.US"],
        start_date="2026-07-30",
        end_date="2026-08-03",
        source="auto",
        interval="1D",
        max_rows=0,
        loader_resolver=mcp_server._get_loader,
    )

    assert json.loads(result)["_provenance"]["schema_version"] == "market-data-provenance-v1"
    assert calls["include_provenance"] is True
    assert calls["provenance_detail"] == "full"
    assert calls["loader_resolver"] is mcp_server._get_loader
