from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import quote

import aiohttp


@dataclass(frozen=True)
class LyricsResult:
    title: str
    artist: str
    pages: list[str]
    source_url: str | None = None


class LyricsService:
    LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
    LRCLIB_GET_URL = "https://lrclib.net/api/get"
    LYRICS_OVH_URL = "https://api.lyrics.ovh/v1/{artist}/{title}"
    ARTIST_ALIASES = {
        "하츠투하츠": "Hearts2Hearts",
        "hearts2hearts": "Hearts2Hearts",
    }

    def __init__(self):
        self._cache: dict[str, LyricsResult] = {}

    async def search(
        self,
        query: str,
        *,
        artist: str | None = None,
        duration: int | None = None,
    ) -> LyricsResult | None:
        cleaned = self._clean_query(query)
        cleaned_artist = self._clean_artist(artist or "")
        cache_key = f"{cleaned_artist.lower()}::{cleaned.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        headers = {"User-Agent": "ai-music-discord-bot/1.0"}
        candidates = self._build_candidates(cleaned, cleaned_artist)
        try:
            timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=16)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                result = await self._search_providers(session, candidates, duration)
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            print(f"Lyrics lookup failed for {cleaned!r}: {type(exc).__name__}: {exc}")
            return None

        if result:
            self._cache[cache_key] = result
            return result

        return None

    async def _search_providers(
        self,
        session: aiohttp.ClientSession,
        candidates: list[tuple[str, str | None]],
        duration: int | None,
    ) -> LyricsResult | None:
        tasks = [
            asyncio.create_task(self._search_lrclib(session, candidates, duration)),
            asyncio.create_task(self._search_lyrics_ovh(session, candidates)),
        ]
        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                if result:
                    return result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return None

    async def _search_lrclib(
        self,
        session: aiohttp.ClientSession,
        candidates: list[tuple[str, str | None]],
        duration: int | None,
    ) -> LyricsResult | None:
        tasks = []
        for title, artist in candidates[:4]:
            if artist:
                tasks.append(asyncio.create_task(self._search_lrclib_exact(session, title, artist, duration)))
            tasks.append(asyncio.create_task(self._search_lrclib_query(session, title, artist, duration)))

        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                if result:
                    return result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return None

    async def _search_lrclib_exact(
        self,
        session: aiohttp.ClientSession,
        title: str,
        artist: str,
        duration: int | None,
    ) -> LyricsResult | None:
        params: dict[str, str | int] = {
            "track_name": title,
            "artist_name": artist,
        }
        if duration:
            params["duration"] = duration
        item = await self._get_json(session, self.LRCLIB_GET_URL, params)
        return self._result_from_lrclib_item(item)

    async def _search_lrclib_query(
        self,
        session: aiohttp.ClientSession,
        title: str,
        artist: str | None,
        duration: int | None,
    ) -> LyricsResult | None:
        search_params = {"q": f"{artist} {title}".strip() if artist else title}
        data = await self._get_json(session, self.LRCLIB_SEARCH_URL, search_params)
        return self._best_lrclib_result(data, title, artist, duration)

    async def _search_lyrics_ovh(
        self,
        session: aiohttp.ClientSession,
        candidates: list[tuple[str, str | None]],
    ) -> LyricsResult | None:
        tasks = []
        for title, artist in candidates[:4]:
            if not artist:
                continue
            tasks.append(asyncio.create_task(self._search_lyrics_ovh_candidate(session, title, artist)))

        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                if result:
                    return result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return None

    async def _search_lyrics_ovh_candidate(
        self,
        session: aiohttp.ClientSession,
        title: str,
        artist: str,
    ) -> LyricsResult | None:
        url = self.LYRICS_OVH_URL.format(
            artist=quote(artist, safe=""),
            title=quote(title, safe=""),
        )
        data = await self._get_json(session, url)
        if not isinstance(data, dict):
            return None
        lyrics = data.get("lyrics")
        if not lyrics:
            return None
        return LyricsResult(
            title=title,
            artist=artist,
            pages=self._paginate(lyrics),
            source_url=url,
        )

    @staticmethod
    async def _get_json(
        session: aiohttp.ClientSession,
        url: str,
        params: dict[str, str | int] | None = None,
    ):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status not in (200, 404):
                    print(f"Lyrics provider returned HTTP {resp.status}: {resp.url}")
                if resp.status != 200:
                    return None
                return await resp.json(content_type=None)
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            print(f"Lyrics provider request failed: {type(exc).__name__}: {url}")
            return None
        except ValueError as exc:
            print(f"Lyrics provider JSON parse failed: {type(exc).__name__}: {url}")
            return None

    def _best_lrclib_result(
        self,
        data,
        title: str,
        artist: str | None,
        duration: int | None,
    ) -> LyricsResult | None:
        if not isinstance(data, list):
            return None

        best_item = None
        best_score = 0.0
        for item in data:
            if not isinstance(item, dict) or not item.get("plainLyrics"):
                continue
            score = self._score_lrclib_item(item, title, artist, duration)
            if score > best_score:
                best_score = score
                best_item = item

        if best_item and best_score >= 0.42:
            return self._result_from_lrclib_item(best_item)
        return None

    def _score_lrclib_item(
        self,
        item: dict,
        title: str,
        artist: str | None,
        duration: int | None,
    ) -> float:
        item_title = self._normalize_for_match(item.get("trackName") or item.get("name") or "")
        item_artist = self._normalize_for_match(item.get("artistName") or "")
        wanted_title = self._normalize_for_match(title)
        wanted_artist = self._normalize_for_match(artist or "")

        if not wanted_artist:
            artist_hint = self._artist_hint_from_query(title)
            if artist_hint and not self._artist_matches_hint(item_artist, artist_hint):
                return -1.0

        score = SequenceMatcher(None, wanted_title, item_title).ratio()
        if not wanted_artist and item_artist:
            combined_item = f"{item_artist} {item_title}".strip()
            combined_score = SequenceMatcher(None, wanted_title, combined_item).ratio()
            if wanted_title and wanted_title in combined_item:
                combined_score += 0.2
            score = max(score, combined_score)
        if wanted_artist and item_artist:
            artist_score = SequenceMatcher(None, wanted_artist, item_artist).ratio()
            if artist_score < 0.45:
                return -1.0
            score = (score * 0.65) + (artist_score * 0.35)
        elif wanted_artist:
            score -= 0.2
        if duration and item.get("duration"):
            diff = abs(float(item["duration"]) - float(duration))
            if diff <= 3:
                score += 0.1
            elif diff > 20:
                score -= 0.15
        if item.get("instrumental"):
            score -= 0.2
        return score

    @classmethod
    def _artist_hint_from_query(cls, query: str) -> str | None:
        normalized_query = cls._normalize_for_match(query)
        for source, replacement in cls.ARTIST_ALIASES.items():
            source_match = cls._normalize_for_match(source)
            replacement_match = cls._normalize_for_match(replacement)
            if source_match in normalized_query or replacement_match in normalized_query:
                return replacement
        return None

    @classmethod
    def _artist_matches_hint(cls, item_artist: str, hint: str) -> bool:
        normalized_artist = cls._normalize_for_match(item_artist)
        hint_values = {cls._normalize_for_match(hint)}
        for source, replacement in cls.ARTIST_ALIASES.items():
            if replacement.lower() == hint.lower():
                hint_values.add(cls._normalize_for_match(source))
                hint_values.add(cls._normalize_for_match(replacement))
        return any(value and value in normalized_artist for value in hint_values)

    def _result_from_lrclib_item(self, item) -> LyricsResult | None:
        if not isinstance(item, dict):
            return None
        lyrics = item.get("plainLyrics")
        if not lyrics:
            return None
        return LyricsResult(
            title=item.get("trackName") or item.get("name") or "Unknown",
            artist=item.get("artistName") or "Unknown",
            pages=self._paginate(lyrics),
            source_url=f"https://lrclib.net/api/get/{item['id']}" if item.get("id") else None,
        )

    @classmethod
    def _build_candidates(cls, title: str, artist: str) -> list[tuple[str, str | None]]:
        candidates: list[tuple[str, str | None]] = []

        def add(candidate_title: str, candidate_artist: str | None = None) -> None:
            candidate_title = cls._clean_query(candidate_title)
            candidate_artist = cls._clean_artist(candidate_artist or "")
            if not candidate_title:
                return
            value = (candidate_title.lower(), candidate_artist.lower() or None)
            existing = [(t.lower(), a.lower() if a else None) for t, a in candidates]
            if value not in existing:
                candidates.append((candidate_title, candidate_artist or None))

        parsed_artist, parsed_title = cls._split_artist_title(title)
        if parsed_artist and parsed_title:
            add(parsed_title, parsed_artist)
            add(parsed_title, artist or parsed_artist)

        add(title, artist or None)
        for alias_title in cls._expand_aliases(title):
            add(alias_title, artist or None)
        if artist:
            add(title, artist)
            for alias_artist in cls._expand_aliases(artist):
                add(title, alias_artist)
        add(title, None)
        return candidates

    @classmethod
    def _expand_aliases(cls, value: str) -> list[str]:
        expanded: list[str] = []
        for source, replacement in cls.ARTIST_ALIASES.items():
            pattern = re.compile(re.escape(source), flags=re.I)
            if pattern.search(value):
                expanded.append(pattern.sub(replacement, value))
        return expanded

    @staticmethod
    def _split_artist_title(query: str) -> tuple[str | None, str | None]:
        for separator in (" - ", " – ", " — ", " | ", " / "):
            if separator in query:
                left, right = query.split(separator, 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    return left, right
        return None, None

    @staticmethod
    def _clean_query(query: str) -> str:
        query = re.sub(r"https?://\S+", "", query, flags=re.I)
        noisy_block = r"official|music video|lyric|lyrics|audio|mv|m/v|video|visualizer|performance|stage|color coded|han|rom|eng|가사|번역|해석|자막"
        query = re.sub(rf"\[[^\]]*({noisy_block})[^\]]*\]", "", query, flags=re.I)
        query = re.sub(rf"\([^\)]*({noisy_block})[^\)]*\)", "", query, flags=re.I)
        query = re.sub(rf"\s*[-|]\s*(official\s*)?({noisy_block}).*$", "", query, flags=re.I)
        query = re.sub(r"\s+(ft\.?|feat\.?|featuring)\s+.+$", "", query, flags=re.I)
        query = re.sub(r"\s*-\s*YouTube\s*$", "", query, flags=re.I)
        query = query.replace("_", " ")
        query = re.sub(r"\s+", " ", query)
        return query.strip(" -_")

    @staticmethod
    def _clean_artist(artist: str) -> str:
        artist = re.sub(r"\s*-\s*Topic\s*$", "", artist, flags=re.I)
        artist = re.sub(r"\s*VEVO\s*$", "", artist, flags=re.I)
        artist = re.sub(r"\s+", " ", artist)
        return artist.strip(" -_")

    @classmethod
    def _normalize_for_match(cls, value: str) -> str:
        value = cls._clean_query(value).lower()
        value = re.sub(r"[^0-9a-z가-힣ぁ-んァ-ン一-龥]+", " ", value, flags=re.I)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _paginate(text: str, limit: int = 1700) -> list[str]:
        lines = text.strip().splitlines()
        pages: list[str] = []
        current = ""
        for line in lines:
            candidate = f"{current}\n{line}".strip()
            if len(candidate) > limit and current:
                pages.append(current)
                current = line
            else:
                current = candidate
        if current:
            pages.append(current)
        return pages[:5]
