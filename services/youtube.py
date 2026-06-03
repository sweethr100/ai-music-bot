from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import yt_dlp


YDL_COMMON = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
}


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
                "default_search": "ytsearch",
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
                if "entries" in info:
                    info = next(entry for entry in info["entries"] if entry)
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
        return TrackInfo(
            title=info.get("title") or "알 수 없는 제목",
            webpage_url=info.get("webpage_url") or info.get("original_url") or "",
            duration=info.get("duration") or 0,
            thumbnail=info.get("thumbnail") or "",
            stream_url=info.get("url"),
        )
