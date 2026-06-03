from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import yt_dlp


YDL_COMMON = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
}


def build_youtube_query(query: str) -> str:
    value = query.strip()
    if not value:
        raise ValueError("유튜브 URL 또는 검색어를 입력해 주세요.")

    lower_value = value.lower()
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value) or lower_value.startswith(
        ("www.", "youtube.com", "youtu.be", "m.youtube.com", "music.youtube.com")
    ):
        return value

    return f"ytsearch1:{value}"


@dataclass
class TrackInfo:
    title: str
    webpage_url: str
    duration: int
    thumbnail: str
    stream_url: str | None = None

    def duration_text(self) -> str:
        if not self.duration:
            return "라이브/알 수 없음"
        minutes, seconds = divmod(self.duration, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


class YouTubeService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    async def get_track(self, query: str) -> TrackInfo:
        def extract() -> TrackInfo:
            opts = {
                **YDL_COMMON,
                "format": "bestaudio/best",
                "noplaylist": True,
                "default_search": "ytsearch1",
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(build_youtube_query(query), download=False)
                if "entries" in info:
                    info = next((entry for entry in info["entries"] if entry), None)
                    if not info:
                        raise ValueError("검색 결과를 찾지 못했습니다.")
                return self._track_from_info(info)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, extract)

    async def get_playlist(self, url: str, limit: int) -> list[TrackInfo]:
        def extract() -> list[TrackInfo]:
            opts = {
                **YDL_COMMON,
                "extract_flat": "in_playlist",
                "ignoreerrors": True,
                "playlistend": limit,
                "noplaylist": False,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)

            entries = info.get("entries") or []
            tracks: list[TrackInfo] = []
            for entry in entries[:limit]:
                if not entry:
                    continue
                webpage_url = entry.get("webpage_url") or entry.get("url") or ""
                if entry.get("id") and "youtube" not in webpage_url:
                    webpage_url = f"https://www.youtube.com/watch?v={entry['id']}"
                tracks.append(
                    TrackInfo(
                        title=entry.get("title") or "알 수 없는 제목",
                        webpage_url=webpage_url,
                        duration=entry.get("duration") or 0,
                        thumbnail=entry.get("thumbnail") or "",
                    )
                )
            return tracks

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, extract)

    async def resolve_stream(self, track: TrackInfo) -> TrackInfo:
        if track.stream_url:
            return track

        resolved = await self.get_track(track.webpage_url)
        track.title = resolved.title or track.title
        track.duration = resolved.duration or track.duration
        track.thumbnail = resolved.thumbnail or track.thumbnail
        track.stream_url = resolved.stream_url
        return track

    @staticmethod
    def _track_from_info(info: dict) -> TrackInfo:
        webpage_url = info.get("webpage_url") or info.get("original_url") or ""
        if info.get("id") and "youtube" not in webpage_url and "youtu.be" not in webpage_url:
            webpage_url = f"https://www.youtube.com/watch?v={info['id']}"

        return TrackInfo(
            title=info.get("title") or "알 수 없는 제목",
            webpage_url=webpage_url,
            duration=info.get("duration") or 0,
            thumbnail=info.get("thumbnail") or "",
            stream_url=info.get("url"),
        )
