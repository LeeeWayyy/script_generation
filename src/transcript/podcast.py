"""Podcast / RSS resolution for ``audio_extraction`` (plan §C).

Feed parse is PRIMARY (``feedparser``, lazy); yt-dlp's ``%(id)s`` does not reliably
surface the RSS ``<guid>``. Feed parse yields ``feed_url``/``episode_guid``/
``enclosure_url``/``published``; yt-dlp is then handed the **selected
enclosure_url** (given a page URL it may pick a different asset).

This module holds the **pure, deterministic selection + canonicalization logic**
(unit-tested without network): episode selection precedence, enclosure choice,
and byte-defined URL canonicalization. The network side-effects (fetch + parse +
download) live in :func:`resolve_podcast`, which lazy-imports feedparser.

Matching is NOT raw URL equality — enclosure URLs are tracking-wrapped and
redirect through CDNs. Prefer a stabler key first (``episode_guid`` + enclosure
``length``+``type``); URL canonicalization is only a tiebreak, and following
redirects to a final host+path is **best-effort** (matching correctness must not
depend on it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlsplit, urlunsplit

# Known feedburner / tracking redirector hosts we unwrap to the wrapped target.
_TRACKING_HOST_MARKERS = ("feedproxy.google.com", "feeds.feedburner.com", "pdst.fm",
                          "chtbl.com", "podtrac.com", "dts.podtrac.com")


@dataclass
class Enclosure:
    url: str
    length: Optional[int] = None  # bytes, from <enclosure length>
    type: Optional[str] = None  # MIME, from <enclosure type>


@dataclass
class Episode:
    guid: Optional[str]
    title: Optional[str]
    published: Optional[str]
    link: Optional[str] = None  # the episode *page* URL (<item><link>)
    enclosures: list[Enclosure] = field(default_factory=list)


class PodcastResolutionError(Exception):
    """Structured resolution failure. ``reason`` distinguishes the cases the
    consumer must tell apart (esp. ``ambiguous`` vs ``stale_selector``)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def canonicalize_url(url: str, *, feed_is_https: bool = False) -> str:
    """Canonicalize an enclosure/episode URL to the byte (tiebreak only).

    Rules (pinned): lower-case scheme+host; strip a leading ``www.``; upgrade
    http→https ONLY when the feed itself declares https; drop the fragment;
    collapse a single trailing slash on the path; decode safe percent-escapes;
    unwrap one layer of known feedburner/tracking redirectors.
    """
    url = _unwrap_tracking(url)
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = parts.hostname.lower() if parts.hostname else ""
    if host.startswith("www."):
        host = host[4:]
    if scheme == "http" and feed_is_https:
        scheme = "https"
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    path = _safe_unquote(parts.path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _safe_unquote(path: str) -> str:
    """Decode only *safe* percent-escapes, preserving structure-changing ones.

    Plain ``unquote`` would decode ``%2F``→``/`` (and ``%5C``→``\\``), turning
    ``a%2Fb`` into a two-segment path and false-matching/ambiguating selectors.
    We shield those reserved escapes behind sentinels across the decode."""
    shielded = (path.replace("%2F", "\x00SLASH\x00").replace("%2f", "\x00SLASH\x00")
                .replace("%5C", "\x00BSL\x00").replace("%5c", "\x00BSL\x00"))
    decoded = unquote(shielded)
    return decoded.replace("\x00SLASH\x00", "%2F").replace("\x00BSL\x00", "%5C")


def _unwrap_tracking(url: str) -> str:
    """Unwrap one layer of a known tracking redirector that embeds the target
    URL as a trailing path segment (best-effort, structural)."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not any(marker in host for marker in _TRACKING_HOST_MARKERS):
        return url
    # These wrappers embed the real URL as a trailing ``.../https://real/...``.
    tail = parts.path
    for scheme in ("https://", "http://"):
        idx = tail.find(scheme)
        if idx != -1:
            return tail[idx:]
    return url


def select_enclosure(enclosures: list[Enclosure]) -> Enclosure:
    """Pick the audio enclosure. Prefer audio MIME; fail if multiple audio remain."""
    if not enclosures:
        raise PodcastResolutionError("feed_identity_unavailable", "episode has no enclosure")
    audio = [e for e in enclosures if (e.type or "").lower().startswith("audio/")]
    pool = audio if audio else enclosures
    if len(pool) > 1 and audio:
        raise PodcastResolutionError(
            "multiple_audio_enclosures",
            f"{len(pool)} audio enclosures; cannot pick deterministically",
        )
    if len(pool) > 1:
        raise PodcastResolutionError(
            "multiple_audio_enclosures",
            "multiple enclosures and none declare an audio MIME type",
        )
    return pool[0]


def select_episode(
    entries: list[Episode],
    *,
    episode_guid: Optional[str] = None,
    episode_url: Optional[str] = None,
    episode_title: Optional[str] = None,
    episode_published: Optional[str] = None,
    feed_is_https: bool = False,
) -> Episode:
    """Select one episode per the precedence contract (plan §C).

    * Both ``episode_guid`` and ``episode_url`` supplied → ``episode_guid`` wins;
      a mismatch (the matched entry's canonical enclosure/episode URL disagrees)
      raises ``guid_url_mismatch``.
    * GUID supplied → match by GUID. Duplicated GUIDs → ``ambiguous``; zero
      matches → ``stale_selector`` (the GUID-instability case).
    * No GUID: fall to ``episode_url`` (canonical match), then to
      ``(episode_title, episode_published)``. Multiple → ``ambiguous``; zero →
      ``stale_selector``.
    * No selector at all → ``missing_selector``.
    """
    guid = (episode_guid or "").strip() or None  # tolerate stray feed whitespace
    # The (title, published) fallback requires BOTH fields — one alone is too weak
    # a selector to mint provenance from (plan §C: the pair, not either).
    has_title_pair = bool(episode_title) and bool(episode_published)

    if not guid and not episode_url and not has_title_pair:
        raise PodcastResolutionError(
            "missing_selector",
            "a feed URL alone does not identify an episode; supply episode_guid, "
            "episode_url, or the full (episode_title, episode_published) pair",
        )

    # A given episode_url may name the episode *page* (<item><link>) OR the audio
    # enclosure — match against both, canonicalized.
    def _url_candidates(e: Episode) -> set[str]:
        urls = {canonicalize_url(en.url, feed_is_https=feed_is_https) for en in e.enclosures}
        if e.link:
            urls.add(canonicalize_url(e.link, feed_is_https=feed_is_https))
        return urls

    if guid:
        matches = [e for e in entries if (e.guid or "").strip() == guid]
        if len(matches) > 1:
            raise PodcastResolutionError(
                "ambiguous", f"GUID {guid!r} is duplicated in the feed"
            )
        if not matches:
            raise PodcastResolutionError(
                "stale_selector",
                f"GUID {guid!r} matches no entry (feed may have rewritten GUIDs)",
            )
        chosen = matches[0]
        if episode_url:
            want = canonicalize_url(episode_url, feed_is_https=feed_is_https)
            if want not in _url_candidates(chosen):
                raise PodcastResolutionError(
                    "guid_url_mismatch",
                    "episode_guid and episode_url select different episodes",
                )
        return chosen

    # GUID-less: match by canonical episode page/enclosure URL, then fall back to
    # (title, published) — the last-resort selector for feeds that carry neither.
    if episode_url:
        want = canonicalize_url(episode_url, feed_is_https=feed_is_https)
        matches = [e for e in entries if want in _url_candidates(e)]
        selector_desc = "episode_url"
    else:
        # Both fields required (guarded above): match the exact pair.
        matches = [e for e in entries
                   if e.title == episode_title and e.published == episode_published]
        selector_desc = "(episode_title, episode_published)"
    if len(matches) > 1:
        raise PodcastResolutionError(
            "ambiguous", f"{selector_desc} matches {len(matches)} entries"
        )
    if not matches:
        raise PodcastResolutionError(
            "stale_selector", f"{selector_desc} matches no entry"
        )
    return matches[0]


@dataclass
class EnclosureDownload:
    """Result of actually downloading the enclosure (plan §C). Carries the final
    size so an authoritative length mismatch can be detected; on any failure
    ``ok`` is False and the caller falls back to the opaque yt-dlp download."""

    path: Optional[Path] = None
    final_url: Optional[str] = None  # post-redirect host+path
    redirect_chain: list[str] = field(default_factory=list)
    content_length: Optional[int] = None  # response Content-Length header
    downloaded_size: Optional[int] = None  # bytes actually written to disk
    ranged: bool = False
    ok: bool = False


# Cap a direct enclosure download so a hostile/misconfigured feed can't fill disk.
MAX_ENCLOSURE_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB
# Feed fetch bounds — a slow/hanging feed must not block the single worker.
FEED_FETCH_TIMEOUT = 30.0
MAX_FEED_BYTES = 16 * 1024 * 1024  # 16 MiB is already huge for an RSS feed


def _assert_public_url(url: str) -> None:
    """SSRF guard (shared with the yt-dlp paths via ``ingest.assert_public_url``):
    reject a feed/enclosure URL whose host is private/loopback/link-local, on the
    initial URL and every redirect target. Raises ``PodcastResolutionError``."""
    from .ingest import SsrfError, assert_public_url
    try:
        assert_public_url(url)
    except SsrfError as exc:
        raise PodcastResolutionError("feed_identity_unavailable", str(exc)) from exc


def _ssrf_redirect_handler(chain: "Optional[list[str]]" = None):
    """HTTPRedirectHandler that re-validates every redirect target against the SSRF
    guard — letting a blocked host RAISE (so it surfaces with the host detail and
    fails the job, rather than being swallowed) — and optionally records the chain."""
    import urllib.request

    class _Guard(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            _assert_public_url(newurl)
            if chain is not None:
                chain.append(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return _Guard()


class _EnclosureTooLarge(Exception):
    """Internal sentinel: the download exceeded the byte cap (fatal, no fallback)."""


def download_enclosure(url: str, dest_dir: "Path", *, timeout: float = 60.0,
                       max_bytes: int = MAX_ENCLOSURE_BYTES) -> EnclosureDownload:
    """Stream-download a direct RSS ``<enclosure>`` to ``dest_dir``, capturing the
    redirect chain + Content-Length + actual downloaded size for an auditable,
    authoritative length check (plan §C).

    For a direct media-file enclosure this is exactly how a downloader fetches it,
    and owning the transfer is the only way to learn the *complete* downloaded
    size. RAISES ``PodcastResolutionError`` for a policy violation that must fail
    the job — an SSRF-blocked host (``feed_identity_unavailable``) or a size-cap
    breach (``enclosure_too_large``). A best-effort NETWORK failure (refused/
    timeout/parse) instead returns ``ok=False`` so the caller can decide.
    """
    import urllib.request

    # SSRF guard: http(s) only + no private/loopback/link-local host (file://,
    # ftp://, localhost, cloud-metadata, LAN are all refused).
    _assert_public_url(url)

    chain: list[str] = []

    dest = dest_dir / "enclosure.bin"
    try:
        opener = urllib.request.build_opener(_ssrf_redirect_handler(chain))
        req = urllib.request.Request(url, headers={"User-Agent": "transcript/0.1"})
        with opener.open(req, timeout=timeout) as resp:
            cl = resp.headers.get("Content-Length")
            content_length = int(cl) if cl and cl.isdigit() else None
            final_url = resp.geturl()
            # A CDN can return 206/Content-Range even without a Range request; a
            # partial response is NOT an authoritative size (plan §C), so flag it.
            ranged = (getattr(resp, "status", None) == 206
                      or bool(resp.headers.get("Content-Range")))
            size = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise _EnclosureTooLarge()
                    out.write(chunk)
    except _EnclosureTooLarge as exc:
        # A too-large enclosure must FAIL the job, not switch downloaders and
        # bypass this enclosure-specific cap and redirect-validation policy.
        try:
            dest.unlink()
        except OSError:
            pass
        raise PodcastResolutionError(
            "enclosure_too_large",
            f"enclosure exceeds the {max_bytes}-byte download cap",
        ) from exc
    except PodcastResolutionError:
        # A redirect SSRF block (raised from _Track) is a policy failure — keep its
        # detailed host message instead of swallowing it into ok=False.
        try:
            dest.unlink()
        except OSError:
            pass
        raise
    except Exception:  # noqa: BLE001 — genuine network failure is best-effort
        try:
            dest.unlink()
        except OSError:
            pass
        return EnclosureDownload(redirect_chain=chain, ok=False)
    return EnclosureDownload(
        path=dest, final_url=final_url, redirect_chain=chain,
        content_length=content_length, downloaded_size=size, ranged=ranged, ok=True,
    )


def length_is_authoritative(*, redirected: bool, ranged: bool, fully_downloaded: bool) -> bool:
    """The feed ``<enclosure length>`` mismatch is fatal ONLY when the length is
    authoritative: a complete (non-ranged, fully downloaded, non-redirected-to-
    transformed) response. Stale/missing lengths and CDN range/transform
    responses are recorded as observations, not failures."""
    return fully_downloaded and not ranged and not redirected


# ---------------------------------------------------------------------------
# Network side — lazy feedparser; returns the selected enclosure + provenance.
# ---------------------------------------------------------------------------


@dataclass
class PodcastResolution:
    enclosure_url: str  # the feed's selected <enclosure> (observation)
    resolution_source: str  # feed_parse | yt-dlp_info_json | user_supplied
    feed_url: Optional[str] = None
    episode_guid: Optional[str] = None
    published: Optional[str] = None
    enclosure_length: Optional[int] = None
    enclosure_type: Optional[str] = None


def _parse_feed(feed_url: str) -> tuple[list[Episode], bool]:
    # SSRF guard (BEFORE importing feedparser, which would otherwise parse a local
    # path / file:// URL): http(s) only + no private/loopback/link-local host.
    _assert_public_url(feed_url)

    import feedparser

    # Fetch the feed OURSELVES with a timeout + size cap (feedparser.parse(url)
    # has no timeout and would block the single worker on a slow/hanging host),
    # then hand the bytes to feedparser. Redirects are re-validated for SSRF.
    import urllib.request

    try:
        opener = urllib.request.build_opener(_ssrf_redirect_handler())
        req = urllib.request.Request(feed_url, headers={"User-Agent": "transcript/0.1"})
        with opener.open(req, timeout=FEED_FETCH_TIMEOUT) as resp:
            raw = resp.read(MAX_FEED_BYTES + 1)
    except PodcastResolutionError:
        raise
    except Exception as exc:  # noqa: BLE001 — network/timeout → structured failure
        raise PodcastResolutionError(
            "feed_identity_unavailable", f"could not fetch feed {feed_url!r}: {exc}"
        ) from exc
    if len(raw) > MAX_FEED_BYTES:
        raise PodcastResolutionError(
            "feed_identity_unavailable", f"feed exceeds {MAX_FEED_BYTES}-byte cap"
        )

    parsed = feedparser.parse(raw)
    feed_is_https = feed_url.lower().startswith("https")
    episodes: list[Episode] = []
    for entry in parsed.entries:
        encs = []
        for link in entry.get("links", []):
            if link.get("rel") == "enclosure" or link.get("type", "").startswith("audio/"):
                length = link.get("length")
                try:
                    length = int(length) if length is not None else None
                except (TypeError, ValueError):
                    length = None
                encs.append(Enclosure(url=link.get("href", ""), length=length,
                                      type=link.get("type")))
        episodes.append(Episode(
            guid=entry.get("id") or entry.get("guid"),
            title=entry.get("title"),
            published=entry.get("published"),
            link=entry.get("link"),  # the episode page URL (for episode_url match)
            enclosures=encs,
        ))
    return episodes, feed_is_https


def resolve_podcast(
    feed_url: str,
    *,
    episode_guid: Optional[str] = None,
    episode_url: Optional[str] = None,
    episode_title: Optional[str] = None,
    episode_published: Optional[str] = None,
) -> PodcastResolution:
    """Resolve a feed + selector to a concrete enclosure + provenance (no download).

    Raises :class:`PodcastResolutionError` (with ``reason``) on any unresolvable
    selector — never silently mints weak provenance.
    """
    episodes, feed_is_https = _parse_feed(feed_url)
    if not episodes:
        raise PodcastResolutionError(
            "feed_identity_unavailable", f"no entries parsed from feed {feed_url!r}"
        )
    episode = select_episode(
        episodes, episode_guid=episode_guid, episode_url=episode_url,
        episode_title=episode_title, episode_published=episode_published,
        feed_is_https=feed_is_https,
    )
    enclosure = select_enclosure(episode.enclosures)
    return PodcastResolution(
        enclosure_url=enclosure.url,
        resolution_source="feed_parse",
        feed_url=feed_url,
        episode_guid=episode.guid,
        published=episode.published,
        enclosure_length=enclosure.length,
        enclosure_type=enclosure.type,
    )
