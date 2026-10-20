"""Read-only news tool: per-stock and global financial headlines.

Two public, no-auth news surfaces are wrapped behind one BaseTool contract:

* China A-share (and general China-market finance) headlines come from
  Eastmoney's free ``search-api`` news-list endpoint. Like every Eastmoney
  surface it rate-limits by source IP, so the request routes through the frozen,
  IP-throttled :mod:`backtest.loaders.eastmoney_client` rather than touching the
  host directly.
* US / HK headlines come from Yahoo Finance's public v1 search-news surface via
  the frozen, IP-throttled :mod:`backtest.loaders.yahoo_client`.

The tool never re-implements provider plumbing and never issues an un-throttled
request: every outbound call goes through a frozen client.

Scopes:

* ``stock`` (default) — headlines for a single security named by ``code``.
* ``global`` — broad market headlines, no ``code`` required.

A failure for one upstream is reported as an error envelope; the tool never
raises out of :meth:`StockNewsTool.execute`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from backtest.loaders import eastmoney_client, yahoo_client

from src.agent.tools import BaseTool

logger = logging.getLogger(__name__)

# Eastmoney free news search endpoint (JSON list of CMS articles). It is the
# same surface the site's search box calls; no auth, IP-throttled.
_EM_NEWS_URL = "https://search-api-web.eastmoney.com/search/jsonp"

# A-share / China-market suffixes that route to the Eastmoney news surface.
_EM_SUFFIXES = ("SH", "SZ", "BJ")
# Suffixes that route to Yahoo's search-news surface.
_YAHOO_SUFFIXES = ("US", "HK", "SA")

# Default broad-market query used when ``scope='global'`` carries no code.
_GLOBAL_QUERY = "财经"

# Bounds so a noisy upstream can never return an unbounded payload.
_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50
# Per-article body trim so the envelope stays compact for the LLM.
_SNIPPET_CHARS = 280


def _clamp_limit(raw: Any) -> int:
    """Coerce a caller-supplied ``limit`` into the supported ``1.._MAX_LIMIT`` range.

    Args:
        raw: The raw ``limit`` value from the tool arguments (any type).

    Returns:
        An integer in ``[1, _MAX_LIMIT]``, falling back to ``_DEFAULT_LIMIT``
        when *raw* is missing or non-numeric.
    """
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_LIMIT
    if value < 1:
        return 1
    return min(value, _MAX_LIMIT)


def _parse_cutoff(raw: Any) -> datetime | None:
    """Parse an optional timezone-aware ISO-8601 cutoff into UTC."""
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ValueError("cutoff must be a timezone-aware ISO-8601 string")
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("cutoff must be a timezone-aware ISO-8601 string") from exc
    if parsed.tzinfo is None:
        raise ValueError("cutoff must include a timezone offset")
    return parsed.astimezone(timezone.utc)


def _iso_or_none(value: datetime | None) -> str | None:
    """Serialize an optional UTC datetime as an explicit ISO value."""
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _discovery_evidence(
    *,
    source: str,
    query: str | None = None,
    cutoff: datetime | None = None,
    cutoff_applied: bool = False,
    cutoff_note: str | None = None,
    related_ticker_filter: str | None = None,
    related_ticker_matches: int | None = None,
    related_ticker_mismatches: int | None = None,
    related_ticker_unverified: int | None = None,
    cutoff_excluded: int | None = None,
) -> dict[str, Any]:
    """Build a common discovery-only evidence status for news payloads."""
    evidence: dict[str, Any] = {
        "status": "discovery_only",
        "source_role": "news_discovery",
        "source": source,
        "requires_direct_read": True,
        "cutoff": _iso_or_none(cutoff),
        "cutoff_applied": cutoff_applied,
    }
    if query is not None:
        evidence["query"] = query
    if cutoff_note:
        evidence["cutoff_note"] = cutoff_note
    if related_ticker_filter is not None:
        evidence["related_ticker_filter"] = related_ticker_filter
    if related_ticker_matches is not None:
        evidence["related_ticker_matches"] = related_ticker_matches
    if related_ticker_mismatches is not None:
        evidence["related_ticker_mismatches"] = related_ticker_mismatches
    if related_ticker_unverified is not None:
        evidence["related_ticker_unverified"] = related_ticker_unverified
    if cutoff_excluded is not None:
        evidence["cutoff_excluded"] = cutoff_excluded
    return evidence


def _suffix_of(code: str) -> str:
    """Return the upper-cased exchange suffix of a symbol, or ``""`` when none."""
    if "." not in code:
        return ""
    return code.rpartition(".")[2].strip().upper()


def _bare_query(code: str) -> str:
    """Strip any exchange suffix to the bare code used as a news search term."""
    return code.strip().split(".", 1)[0].strip()


def _snippet(text: Any) -> str:
    """Trim an article body to a bounded plain-text snippet.

    Args:
        text: Raw body/summary value (any type).

    Returns:
        A whitespace-collapsed snippet capped at ``_SNIPPET_CHARS`` characters,
        or ``""`` when ``text`` is not a usable string.
    """
    if not isinstance(text, str):
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    return collapsed[:_SNIPPET_CHARS].rstrip() + "…"


def _decode_jsonp(payload: Any) -> Any:
    """Decode an Eastmoney response that may arrive JSON or JSONP-wrapped.

    The search endpoint usually returns a JSON object, but can echo a
    ``callback(...)`` JSONP envelope. A single outer call wrapper is stripped
    before parsing.

    Args:
        payload: The decoded body from the throttled client (``dict`` already, or
            a raw ``str`` when JSONP-wrapped).

    Returns:
        The decoded object, or ``None`` when nothing parseable is found.
    """
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return None
    start = payload.find("(")
    end = payload.rfind(")")
    inner = payload[start + 1 : end] if start != -1 and end > start else payload
    try:
        return json.loads(inner)
    except (ValueError, TypeError):
        return None


def _em_article(raw: dict[str, Any]) -> dict[str, Any]:
    """Project one Eastmoney CMS article into a compact, named record.

    Args:
        raw: A single article dict from ``result.cmsArticleWebOld``.

    Returns:
        A flat ``{title, url, source, published, snippet}`` record.
    """
    return {
        "title": _snippet(raw.get("title")),
        "url": raw.get("url"),
        "source": raw.get("mediaName"),
        "published": raw.get("date"),
        "snippet": _snippet(raw.get("content")),
    }


def _fetch_eastmoney_news(query: str, limit: int) -> list[dict[str, Any]]:
    """Fetch China-market news headlines for a query from Eastmoney.

    Args:
        query: Free-text search term (bare code or keyword).
        limit: Maximum number of articles to return.

    Returns:
        A capped list of compact article records; empty when none.

    Raises:
        requests.RequestException: Network failure, propagated to the caller.
        requests.HTTPError: Non-2xx response status.
        ValueError: Body is not valid JSON.
    """
    param = json.dumps(
        {
            "uid": "",
            "keyword": query,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": limit,
                }
            },
        },
        ensure_ascii=False,
    )
    payload = eastmoney_client.get_json(
        _EM_NEWS_URL,
        params={"cb": "", "param": param, "_": "0"},
    )
    decoded = _decode_jsonp(payload)
    if not isinstance(decoded, dict):
        return []
    result = decoded.get("result")
    if not isinstance(result, dict):
        return []
    articles = result.get("cmsArticleWebOld")
    if not isinstance(articles, list):
        return []
    return [_em_article(a) for a in articles if isinstance(a, dict)][:limit]


def _yahoo_article(raw: dict[str, Any]) -> dict[str, Any]:
    """Project one Yahoo search-news item into the shared article shape.

    Args:
        raw: A single news dict from :func:`yahoo_client.search_news`.

    Returns:
        A flat ``{title, url, source, published, snippet}`` record.
    """
    published = None
    try:
        timestamp = float(raw.get("providerPublishTime"))
        published = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (TypeError, ValueError, OSError, OverflowError):
        pass
    return {
        "title": _snippet(raw.get("title")),
        "url": raw.get("link"),
        "source": raw.get("publisher"),
        "published": published,
        "snippet": _snippet(raw.get("summary")),
    }


def _related_ticker_match(raw: dict[str, Any], query: str) -> bool | None:
    """Return Yahoo's explicit related-ticker match, or ``None`` if absent."""
    related = raw.get("relatedTickers")
    if not isinstance(related, list):
        return None
    normalized = {
        str(item).strip().upper().split(".", 1)[0]
        for item in related
        if isinstance(item, (str, int, float))
    }
    return query.strip().upper() in normalized


def _fetch_yahoo_news(
    query: str,
    limit: int,
    *,
    cutoff: datetime | None = None,
    require_related_ticker: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fetch US/HK news and expose cutoff/ticker-relevance accounting.

    Yahoo Search remains a discovery source. The returned article records retain
    the historical shape for compatibility; the second return value records
    how many provider related-ticker flags matched, mismatched, were missing,
    or were excluded by a caller cutoff.
    """
    articles = yahoo_client.search_news(query, limit)
    stats = {
        "related_ticker_matches": 0,
        "related_ticker_mismatches": 0,
        "related_ticker_unverified": 0,
        "cutoff_excluded": 0,
        "relevance_rejected": 0,
    }
    cutoff_epoch = cutoff.timestamp() if cutoff is not None else None
    selected: list[dict[str, Any]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        try:
            published_raw = article.get("providerPublishTime")
            published_epoch = (
                float(published_raw) if published_raw is not None else None
            )
        except (TypeError, ValueError):
            published_epoch = None
        if cutoff_epoch is not None and (
            published_epoch is None or published_epoch > cutoff_epoch
        ):
            stats["cutoff_excluded"] += 1
            continue

        relation = _related_ticker_match(article, query)
        if relation is True:
            stats["related_ticker_matches"] += 1
        elif relation is False:
            stats["related_ticker_mismatches"] += 1
        else:
            stats["related_ticker_unverified"] += 1
        if require_related_ticker and relation is not True:
            stats["relevance_rejected"] += 1
            continue
        selected.append(_yahoo_article(article))
        if len(selected) >= limit:
            break
    return selected, stats


class StockNewsTool(BaseTool):
    """Read-only per-stock and global financial news headlines."""

    name = "get_stock_news"
    description = (
        "Fetch recent financial news headlines, read-only and no auth. Yahoo "
        "Finance Search and Eastmoney are discovery sources only: returned items "
        "must be directly read and independently verified before being stated as "
        "company, regulatory, or issuer facts. Markets: China A-share "
        "(SH/SZ/BJ) returns Eastmoney news ARTICLES; US (.US), Hong Kong (.HK) "
        "and Brazil (B3 .SA) return Yahoo Finance news articles. Use scope "
        "'stock' with a 'code', or scope 'global' (no code) for broad "
        "China-market finance articles. Optional cutoff is timezone-aware "
        "ISO-8601; Yahoo can also require an explicit related-ticker match. "
        'Example: {"code": "AAPL.US", "scope": "stock", "limit": 10, '
        '"cutoff": "2026-08-03T21:00:00Z"}.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": (
                    "Symbol whose news to fetch, e.g. '600519.SH', 'AAPL.US', "
                    "'00700.HK'. Required when scope='stock'; ignored when "
                    "scope='global'. The exchange suffix selects the upstream: "
                    "SH/SZ/BJ -> Eastmoney, US/HK -> Yahoo Finance."
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["stock", "global"],
                "description": (
                    "'stock' (default) for one security named by 'code'; "
                    "'global' for broad China-market finance headlines."
                ),
                "default": "stock",
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of headlines to return (1-50). Default 20."
                ),
                "default": _DEFAULT_LIMIT,
            },
            "cutoff": {
                "type": "string",
                "description": (
                    "Optional timezone-aware ISO-8601 publication cutoff. Yahoo "
                    "filters providerPublishTime at or before this cutoff; a "
                    "timezone is required."
                ),
            },
            "require_related_ticker": {
                "type": "boolean",
                "description": (
                    "Yahoo only: when true, retain articles whose provider "
                    "relatedTickers explicitly include the queried ticker. "
                    "Default false because this is a discovery filter, not issuer verification."
                ),
                "default": False,
            },
        },
        "required": [],
    }

    def execute(self, **kwargs: Any) -> str:
        """Fetch news headlines for one stock or the broad market.

        Args:
            **kwargs: ``scope`` ('stock' | 'global', default 'stock'), ``code``
                (required when scope='stock'), optional ``limit`` (1-50), an
                optional timezone-aware ISO-8601 ``cutoff``, and the Yahoo-only
                ``require_related_ticker`` filter.

        Returns:
            A JSON string envelope. On success:
            ``{"ok": true, "market": <market>, "source": <source>,
            "data": {...}}``. On failure: ``{"ok": false, "error": "..."}``.
        """
        scope = kwargs.get("scope", "stock")
        if scope not in ("stock", "global"):
            return self._error(
                f"invalid scope: {scope!r}; expected 'stock' or 'global'"
            )

        try:
            cutoff = _parse_cutoff(kwargs.get("cutoff"))
        except ValueError as exc:
            return self._error(str(exc))
        limit = _clamp_limit(kwargs.get("limit"))
        require_related_ticker = kwargs.get("require_related_ticker") is True

        if scope == "global":
            return self._run_global(limit, cutoff)
        return self._run_stock(
            kwargs.get("code"), limit, cutoff, require_related_ticker
        )

    def _run_global(self, limit: int, cutoff: datetime | None = None) -> str:
        """Fetch broad China-market headlines from Eastmoney."""
        try:
            articles = _fetch_eastmoney_news(_GLOBAL_QUERY, limit)
        except Exception as exc:  # noqa: BLE001 - surface any fetch failure as envelope
            logger.warning("global news fetch failed: %s", exc)
            return self._error(f"eastmoney news fetch failed: {exc}")
        evidence = _discovery_evidence(
            source="eastmoney",
            query=_GLOBAL_QUERY,
            cutoff=cutoff,
            cutoff_note=(
                "Eastmoney publication timestamps are not normalized by this tool; "
                "caller cutoff was not applied"
                if cutoff is not None
                else None
            ),
        )
        return self._ok(
            "global",
            "eastmoney",
            {
                "scope": "global",
                "articles": articles,
                "evidence": evidence,
                "retrieved_at": _iso_or_none(datetime.now(timezone.utc)),
            },
        )

    def _run_stock(
        self,
        code_arg: Any,
        limit: int,
        cutoff: datetime | None = None,
        require_related_ticker: bool = False,
    ) -> str:
        """Fetch single-security headlines, routing by exchange suffix."""
        if not isinstance(code_arg, str) or not code_arg.strip():
            return self._error(
                "missing required parameter: code (required when scope='stock')"
            )

        code = code_arg.strip()
        suffix = _suffix_of(code)
        query = _bare_query(code)
        if not query:
            return self._error(f"invalid code: {code!r}")

        if suffix in _EM_SUFFIXES:
            return self._stock_via_eastmoney(code, query, limit, cutoff)
        if suffix in _YAHOO_SUFFIXES:
            # A bare B3 code (PETR4) is ambiguous on Yahoo search; its own
            # spelling (PETR4.SA) is the listing.
            if suffix == "SA":
                query = code.upper()
            return self._stock_via_yahoo(
                code, query, limit, cutoff, require_related_ticker
            )
        return self._error(
            f"unsupported market for code {code!r}; expected suffix in "
            f"{_EM_SUFFIXES + _YAHOO_SUFFIXES}"
        )

    def _stock_via_eastmoney(
        self,
        code: str,
        query: str,
        limit: int,
        cutoff: datetime | None = None,
    ) -> str:
        """Fetch A-share headlines from Eastmoney for one code."""
        try:
            articles = _fetch_eastmoney_news(query, limit)
        except Exception as exc:  # noqa: BLE001 - surface any fetch failure as envelope
            logger.warning("eastmoney news fetch failed for %s: %s", code, exc)
            return self._error(f"eastmoney news fetch failed: {exc}")
        evidence = _discovery_evidence(
            source="eastmoney",
            query=query,
            cutoff=cutoff,
            cutoff_note=(
                "Eastmoney publication timestamps are not normalized by this tool; "
                "caller cutoff was not applied"
                if cutoff is not None
                else None
            ),
        )
        return self._ok(
            "a_share",
            "eastmoney",
            {
                "scope": "stock",
                "code": code,
                "articles": articles,
                "evidence": evidence,
                "retrieved_at": _iso_or_none(datetime.now(timezone.utc)),
            },
        )

    def _stock_via_yahoo(
        self,
        code: str,
        query: str,
        limit: int,
        cutoff: datetime | None = None,
        require_related_ticker: bool = False,
    ) -> str:
        """Fetch US/HK news articles from Yahoo for one code."""
        market = {"HK": "hk", "SA": "br"}.get(_suffix_of(code), "us")
        try:
            articles, stats = _fetch_yahoo_news(
                query,
                limit,
                cutoff=cutoff,
                require_related_ticker=require_related_ticker,
            )
        except Exception as exc:  # noqa: BLE001 - surface any fetch failure as envelope
            logger.warning("yahoo news fetch failed for %s: %s", code, exc)
            return self._error(f"yahoo news fetch failed: {exc}")
        evidence = _discovery_evidence(
            source="yahoo_finance_search",
            query=query,
            cutoff=cutoff,
            cutoff_applied=cutoff is not None,
            related_ticker_filter=(
                "required" if require_related_ticker else "reported_only"
            ),
            related_ticker_matches=stats["related_ticker_matches"],
            related_ticker_mismatches=stats["related_ticker_mismatches"],
            related_ticker_unverified=stats["related_ticker_unverified"],
            cutoff_excluded=stats["cutoff_excluded"],
        )
        evidence["relevance_rejected"] = stats["relevance_rejected"]
        return self._ok(
            market,
            "yahoo",
            {
                "scope": "stock",
                "code": code,
                "articles": articles,
                "evidence": evidence,
                "retrieved_at": _iso_or_none(datetime.now(timezone.utc)),
            },
        )

    @staticmethod
    def _ok(market: str, source: str, data: dict[str, Any]) -> str:
        """Render a success envelope as a JSON string.

        Args:
            market: Market label (e.g. ``"a_share"``, ``"us"``, ``"global"``).
            source: Upstream provider name (``"eastmoney"`` or ``"yahoo"``).
            data: The payload mapping.

        Returns:
            ``{"ok": true, "market": ..., "source": ..., "data": ...}`` as JSON.
        """
        return json.dumps(
            {"ok": True, "market": market, "source": source, "data": data},
            ensure_ascii=False,
        )

    @staticmethod
    def _error(message: str) -> str:
        """Render a failure envelope as a JSON string.

        Args:
            message: Human-readable error text.

        Returns:
            ``{"ok": false, "error": message}`` as a JSON string.
        """
        return json.dumps({"ok": False, "error": message}, ensure_ascii=False)
