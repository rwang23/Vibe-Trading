"""Regression tests for the market-data provenance envelope."""

from __future__ import annotations

import json

import pandas as pd

from backtest.loaders.base import NoAvailableSourceError
from src.market_data import PROVENANCE_KEY, fetch_market_data


class _FailingLoader:
    def fetch(self, *_args, **_kwargs):
        raise RuntimeError("401 Unauthorized")


class _SuccessfulLoader:
    volume_units = {"us_equity": "shares"}

    def fetch(self, codes, *_args, **_kwargs):
        frame = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "close": [10.2, 11.2, 12.2],
                "volume": [1000, 1100, 1200],
            },
            index=pd.to_datetime(["2026-07-30", "2026-07-31", "2026-08-03"]),
        )
        return {codes[0]: frame}


def test_provenance_records_fallback_attempts_and_display_truncation() -> None:
    def resolver(source: str):
        return {"yahoo": _FailingLoader, "stooq": _SuccessfulLoader}[source]

    out = fetch_market_data(
        codes=["AAPL.US"],
        start_date="2026-07-30",
        end_date="2026-08-03",
        source="auto",
        interval="1D",
        max_rows=2,
        loader_resolver=resolver,
        fallback_chain_provider=lambda _source: ["yahoo", "stooq"],
        max_fallback_attempts=2,
        include_provenance=True,
        provenance_detail="full",
    )

    provenance = out[PROVENANCE_KEY]
    request = provenance["requests"][0]
    assert request["detected_source"] == "yahoo"
    assert request["used_source"] == "stooq"
    assert request["provider_family"] == "stooq"
    assert request["fallback_used"] is True
    assert [attempt["status"] for attempt in request["attempts"]] == [
        "fetch_error",
        "ok",
    ]
    assert request["attempts"][0]["error_type"] == "RuntimeError"
    assert request["attempts"][0]["error_code"] == "http_401"
    assert "401 Unauthorized" not in json.dumps(provenance)
    assert "http_401" in json.dumps(provenance)
    assert request["latest_returned_bar_by_symbol"]["AAPL.US"].startswith("2026-08-03")
    assert request["row_counts"]["AAPL.US"] == 3
    assert request["truncated_symbols"] == ["AAPL.US"]
    assert provenance["calculation_warning"]
    assert provenance["retrieved_at"].endswith("Z")
    assert provenance["symbol_provenance"]["AAPL.US"]["volume_unit"] == "shares"
    assert provenance["AAPL.US"]["volume_unit"] == "shares"
    assert "adjustment" in provenance["symbol_provenance"]["AAPL.US"]
    assert out["AAPL.US"]["truncated"] is True
    json.dumps(out, allow_nan=False)


def test_provenance_marks_unresolved_provider_attempts_without_raw_errors() -> None:
    def resolver(_source: str):
        raise NoAvailableSourceError("provider unavailable")

    out = fetch_market_data(
        codes=["AAPL.US"],
        start_date="2026-07-30",
        end_date="2026-08-03",
        source="auto",
        loader_resolver=resolver,
        fallback_chain_provider=lambda _source: ["yahoo", "stooq"],
        max_fallback_attempts=2,
        include_provenance=True,
        provenance_detail="full",
    )

    assert out["_unresolved"] == ["AAPL.US"]
    request = out[PROVENANCE_KEY]["requests"][0]
    assert request["status"] == "unresolved"
    assert request["used_source"] is None
    assert [attempt["status"] for attempt in request["attempts"]] == [
        "unavailable",
        "unavailable",
    ]
    assert "provider unavailable" not in json.dumps(out[PROVENANCE_KEY])


def test_full_series_request_has_no_truncation_warning() -> None:
    out = fetch_market_data(
        codes=["AAPL.US"],
        start_date="2026-07-30",
        end_date="2026-08-03",
        source="auto",
        max_rows=0,
        loader_resolver=lambda _source: _SuccessfulLoader,
        fallback_chain_provider=lambda _source: ["yahoo"],
        include_provenance=True,
        provenance_detail="full",
    )

    assert isinstance(out["AAPL.US"], list)
    assert out[PROVENANCE_KEY]["truncated_symbols"] == []
    assert out[PROVENANCE_KEY]["calculation_warning"] is None


def test_per_symbol_fallback_preserves_each_serving_source() -> None:
    class FirstLoader(_SuccessfulLoader):
        def fetch(self, codes, *_args, **_kwargs):
            return {"AAPL.US": super().fetch(["AAPL.US"])["AAPL.US"]} if "AAPL.US" in codes else {}

    class SecondLoader(_SuccessfulLoader):
        def fetch(self, codes, *_args, **_kwargs):
            return {"MSFT.US": super().fetch(["MSFT.US"])["MSFT.US"]} if "MSFT.US" in codes else {}

    out = fetch_market_data(
        codes=["AAPL.US", "MSFT.US"],
        start_date="2026-07-30",
        end_date="2026-08-03",
        loader_resolver=lambda name: {"yahoo": FirstLoader, "stooq": SecondLoader}[name],
        fallback_chain_provider=lambda _source: ["yahoo", "stooq"],
        max_fallback_attempts=2,
        include_provenance=True,
        provenance_detail="full",
    )

    envelope = out[PROVENANCE_KEY]
    request = envelope["requests"][0]
    assert request["used_source"] is None  # Mixed sources must not claim one provider.
    assert request["used_source_by_symbol"] == {
        "AAPL.US": "yahoo",
        "MSFT.US": "stooq",
    }
    assert [attempt["requested_symbols"] for attempt in request["attempts"]] == [
        ["AAPL.US", "MSFT.US"],
        ["MSFT.US"],
    ]
    assert envelope["symbol_provenance"]["AAPL.US"]["source"] == "yahoo"
    assert envelope["symbol_provenance"]["MSFT.US"]["source"] == "stooq"
    assert envelope["quality"] == "degraded_fallback"


def test_basic_provenance_keeps_upstream_shape() -> None:
    out = fetch_market_data(
        codes=["AAPL.US"],
        start_date="2026-07-30",
        end_date="2026-08-03",
        loader_resolver=lambda _source: _SuccessfulLoader,
        fallback_chain_provider=lambda _source: ["yahoo"],
        include_provenance=True,
    )
    assert out[PROVENANCE_KEY]["AAPL.US"]["volume_unit"] == "shares"
    assert "schema_version" not in out[PROVENANCE_KEY]
