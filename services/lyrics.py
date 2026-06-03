from __future__ import annotations

import re
from dataclasses import dataclass

import aiohttp


@dataclass(frozen=True)
class LyricsResult:
    title: str
    artist: str
    pages: list[str]
    source_url: str | None = None


class LyricsService:
    API_URL = "https://lrclib.net/api/search"

    def __init__(self):
        self._cache: dict[str, LyricsResult] = {}

    async def search(self, query: str) -> LyricsResult | None:
        cleaned = self._clean_query(query)
        cache_key = cleaned.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        headers = {"User-Agent": "ai-music-discord-bot/1.0"}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(self.API_URL, params={"q": cleaned}, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

        for item in data:
            lyrics = item.get("plainLyrics")
            if not lyrics:
                continue
            result = LyricsResult(
                title=item.get("trackName") or cleaned,
                artist=item.get("artistName") or "Unknown",
                pages=self._paginate(lyrics),
                source_url=None,
            )
            self._cache[cache_key] = result
            return result

        return None

    @staticmethod
    def _clean_query(query: str) -> str:
        query = re.sub(r"\[[^\]]+\]|\([^\)]*(official|mv|lyrics|audio|가사)[^\)]*\)", "", query, flags=re.I)
        query = re.sub(r"\s+", " ", query)
        return query.strip(" -_")

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
