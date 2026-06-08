"""Podcast/RSS pure selection + canonicalization logic (plan §C). No network."""

import pytest

from transcript.podcast import (Enclosure, Episode, PodcastResolutionError,
                                canonicalize_url, length_is_authoritative,
                                select_enclosure, select_episode)


# --- URL canonicalization (defined to the byte) -----------------------------


def test_canonicalize_lowercases_and_strips_www_and_fragment():
    assert canonicalize_url("HTTP://WWW.Example.COM/Path/#frag") == "http://example.com/Path"


def test_canonicalize_collapses_trailing_slash_and_decodes_escapes():
    assert canonicalize_url("https://h.com/a%20b/") == "https://h.com/a b"


def test_canonicalize_http_to_https_only_when_feed_declares():
    assert canonicalize_url("http://h.com/x", feed_is_https=False) == "http://h.com/x"
    assert canonicalize_url("http://h.com/x", feed_is_https=True) == "https://h.com/x"


def test_canonicalize_unwraps_tracking_redirector():
    wrapped = "https://pdst.fm/e/https://cdn.example.com/ep1.mp3"
    assert canonicalize_url(wrapped) == "https://cdn.example.com/ep1.mp3"


def test_canonicalize_preserves_encoded_slash():
    # %2F must NOT decode to '/', which would change the path structure and
    # false-match selectors. A safe escape (%20→space) still decodes.
    assert canonicalize_url("https://h.com/a%2Fb%20c") == "https://h.com/a%2Fb c"


def test_resolve_podcast_rejects_non_http_feed_scheme():
    # SSRF / LFD guard: feedparser will read local files / file:// — refuse them.
    from transcript.podcast import resolve_podcast
    for bad in ("file:///etc/passwd", "/etc/passwd"):
        with pytest.raises(PodcastResolutionError) as ei:
            resolve_podcast(bad, episode_guid="g")
        assert ei.value.reason == "feed_identity_unavailable"


def test_download_enclosure_rejects_non_http_scheme(tmp_path):
    # SSRF / local-file-disclosure guard: file:// (and any non-http(s)) is refused.
    from transcript.podcast import download_enclosure
    with pytest.raises(PodcastResolutionError) as ei:
        download_enclosure("file:///etc/passwd", tmp_path)
    assert ei.value.reason == "feed_identity_unavailable"


def test_download_enclosure_blocks_loopback_ssrf(tmp_path):
    # SSRF guard: a loopback/private host is refused (raises), not fetched.
    from transcript.podcast import download_enclosure
    with pytest.raises(PodcastResolutionError) as ei:
        download_enclosure("http://127.0.0.1:9/x.mp3", tmp_path, timeout=0.5)
    assert ei.value.reason == "feed_identity_unavailable"


def test_download_enclosure_network_failure_is_soft(tmp_path, monkeypatch):
    # With the SSRF opt-out, a genuine http failure (refused) is best-effort → ok=False.
    monkeypatch.setenv("TRANSCRIPT_ALLOW_PRIVATE_FETCH", "1")
    from transcript.podcast import download_enclosure
    d = download_enclosure("http://127.0.0.1:9/x.mp3", tmp_path, timeout=0.5)
    assert d.ok is False and d.path is None


def test_resolve_podcast_blocks_loopback_feed_ssrf():
    from transcript.podcast import resolve_podcast
    with pytest.raises(PodcastResolutionError) as ei:
        resolve_podcast("http://127.0.0.1/feed.xml", episode_guid="g")
    assert ei.value.reason == "feed_identity_unavailable"


def test_enclosure_too_large_is_fatal_not_fallback(monkeypatch):
    # A too-large enclosure FAILS the job (no silent fall-back to unlimited yt-dlp).
    import transcript.podcast as podcast
    from transcript.extract import extract_audio_extraction
    from transcript.podcast import PodcastResolution

    monkeypatch.setattr(podcast, "resolve_podcast", lambda *a, **k: PodcastResolution(
        enclosure_url="https://cdn/ep.mp3", resolution_source="feed_parse", feed_url="f"))

    def boom_download(url, dest, **k):
        raise PodcastResolutionError("enclosure_too_large", "too big")
    monkeypatch.setattr(podcast, "download_enclosure", boom_download)

    with pytest.raises(PodcastResolutionError) as ei:
        extract_audio_extraction(feed_url="f", engine=None,
                                 transcribe_fn=lambda *a, **k: (_ for _ in ()).throw(
                                     AssertionError("must not transcribe after size-cap")))
    assert ei.value.reason == "enclosure_too_large"


def test_authoritative_length_mismatch_is_fatal(monkeypatch):
    # A COMPLETE, non-ranged, non-redirected download whose size disagrees with
    # the feed <enclosure length> is fatal (plan §C); otherwise it's observation.
    from pathlib import Path
    import transcript.podcast as podcast
    from transcript.extract import extract_audio_extraction
    from transcript.podcast import EnclosureDownload, PodcastResolution

    monkeypatch.setattr(podcast, "resolve_podcast", lambda *a, **k: PodcastResolution(
        enclosure_url="https://cdn/ep.mp3", resolution_source="feed_parse",
        feed_url="f", enclosure_length=100))
    monkeypatch.setattr(podcast, "download_enclosure", lambda url, dest, **k: EnclosureDownload(
        path=Path("/tmp/x"), downloaded_size=200, content_length=200, ok=True))  # != 100

    from transcript.types import Transcript
    with pytest.raises(PodcastResolutionError) as ei:
        extract_audio_extraction(feed_url="f", engine=None,
                                 transcribe_fn=lambda *a, **k: Transcript())
    assert ei.value.reason == "length_mismatch"


def test_ranged_download_size_mismatch_is_not_fatal(monkeypatch):
    # A 206/ranged response is NOT authoritative — a size mismatch is recorded as
    # an observation, never fatal (plan §C). Also confirms `source` (temp path)
    # is dropped from the envelope.
    from pathlib import Path
    import transcript.podcast as podcast
    from transcript.extract import extract_audio_extraction
    from transcript.podcast import EnclosureDownload, PodcastResolution
    from transcript.types import Transcript

    monkeypatch.setattr(podcast, "resolve_podcast", lambda *a, **k: PodcastResolution(
        enclosure_url="https://cdn/ep.mp3", resolution_source="feed_parse",
        feed_url="f", enclosure_length=100))
    monkeypatch.setattr(podcast, "download_enclosure", lambda url, dest, **k: EnclosureDownload(
        path=Path("/tmp/x"), downloaded_size=200, ranged=True, ok=True))  # ranged → not authoritative

    def fake_transcribe(source, **k):
        t = Transcript()
        t.meta["source"] = source  # the temp file path
        return t

    result, _ = extract_audio_extraction(feed_url="f", engine=None, transcribe_fn=fake_transcribe)
    assert result.meta["length_authoritative"] is False
    assert result.meta["length_matches"] is False  # recorded, not raised
    assert "source" not in result.meta  # transient temp path must not leak


# --- enclosure selection -----------------------------------------------------


def test_select_enclosure_prefers_audio_mime():
    enc = select_enclosure([
        Enclosure("a.mp3", type="audio/mpeg"),
        Enclosure("a.jpg", type="image/jpeg"),
    ])
    assert enc.url == "a.mp3"


def test_select_enclosure_multiple_audio_fails():
    with pytest.raises(PodcastResolutionError) as ei:
        select_enclosure([Enclosure("a.mp3", type="audio/mpeg"),
                          Enclosure("b.m4a", type="audio/mp4")])
    assert ei.value.reason == "multiple_audio_enclosures"


# --- episode selection precedence contract ----------------------------------


def _eps():
    return [
        Episode(guid="g1", title="One", published="2024", enclosures=[Enclosure("u1.mp3", type="audio/mpeg")]),
        Episode(guid="g2", title="Two", published="2024", enclosures=[Enclosure("u2.mp3", type="audio/mpeg")]),
    ]


def test_missing_selector_errors():
    with pytest.raises(PodcastResolutionError) as ei:
        select_episode(_eps())
    assert ei.value.reason == "missing_selector"


def test_guid_wins_and_matches():
    ep = select_episode(_eps(), episode_guid="g2")
    assert ep.title == "Two"


def test_duplicate_guid_is_ambiguous():
    eps = _eps() + [Episode(guid="g2", title="Dup", published="x", enclosures=[])]
    with pytest.raises(PodcastResolutionError) as ei:
        select_episode(eps, episode_guid="g2")
    assert ei.value.reason == "ambiguous"


def test_unknown_guid_is_stale_selector_not_ambiguous():
    with pytest.raises(PodcastResolutionError) as ei:
        select_episode(_eps(), episode_guid="nope")
    assert ei.value.reason == "stale_selector"


def test_guid_url_mismatch_is_structured_error():
    with pytest.raises(PodcastResolutionError) as ei:
        select_episode(_eps(), episode_guid="g1", episode_url="https://h/u2.mp3")
    assert ei.value.reason == "guid_url_mismatch"


def test_guidless_match_by_canonical_url():
    eps = [Episode(guid=None, title="One", published="x",
                   enclosures=[Enclosure("https://WWW.h.com/u1.mp3", type="audio/mpeg")])]
    # www-stripping makes these canonically equal even though one carries www.
    ep = select_episode(eps, episode_url="https://h.com/u1.mp3")
    assert ep.title == "One"


def test_episode_url_matches_page_link_not_just_enclosure():
    # episode_url naming the episode PAGE (<item><link>) must match, even though
    # the enclosure URL is a different CDN URL.
    eps = [Episode(guid="g1", title="One", published="x",
                   link="https://show.example.com/ep/1",
                   enclosures=[Enclosure("https://cdn.example.com/1.mp3", type="audio/mpeg")])]
    ep = select_episode(eps, episode_url="https://show.example.com/ep/1")
    assert ep.title == "One"
    # And guid+page_url agree → no mismatch.
    assert select_episode(eps, episode_guid="g1",
                          episode_url="https://show.example.com/ep/1").title == "One"


def test_guid_whitespace_tolerated():
    eps = [Episode(guid=" g1 ", title="One", published="x", enclosures=[])]
    assert select_episode(eps, episode_guid="g1").title == "One"


def test_guidless_title_published_fallback():
    eps = [
        Episode(guid=None, title="Ep One", published="2024-01-01", enclosures=[]),
        Episode(guid=None, title="Ep Two", published="2024-02-01", enclosures=[]),
    ]
    ep = select_episode(eps, episode_title="Ep Two", episode_published="2024-02-01")
    assert ep.title == "Ep Two"


def test_title_alone_is_missing_selector():
    # One of the (title, published) pair is too weak — require both.
    eps = [Episode(guid=None, title="Solo", published="2024", enclosures=[])]
    with pytest.raises(PodcastResolutionError) as ei:
        select_episode(eps, episode_title="Solo")
    assert ei.value.reason == "missing_selector"


def test_title_published_ambiguous_when_duplicated():
    eps = [
        Episode(guid=None, title="Dup", published="2024", enclosures=[]),
        Episode(guid=None, title="Dup", published="2024", enclosures=[]),
    ]
    with pytest.raises(PodcastResolutionError) as ei:
        select_episode(eps, episode_title="Dup", episode_published="2024")
    assert ei.value.reason == "ambiguous"


def test_guidless_url_matches_multiple_is_ambiguous():
    eps = [
        Episode(guid=None, title="A", published="x", enclosures=[Enclosure("https://h/u.mp3", type="audio/mpeg")]),
        Episode(guid=None, title="B", published="x", enclosures=[Enclosure("https://h/u.mp3", type="audio/mpeg")]),
    ]
    with pytest.raises(PodcastResolutionError) as ei:
        select_episode(eps, episode_url="https://h/u.mp3")
    assert ei.value.reason == "ambiguous"


# --- length authority --------------------------------------------------------


def test_length_authoritative_only_for_complete_response():
    assert length_is_authoritative(redirected=False, ranged=False, fully_downloaded=True)
    assert not length_is_authoritative(redirected=True, ranged=False, fully_downloaded=True)
    assert not length_is_authoritative(redirected=False, ranged=True, fully_downloaded=True)
    assert not length_is_authoritative(redirected=False, ranged=False, fully_downloaded=False)
