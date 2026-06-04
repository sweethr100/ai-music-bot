from __future__ import annotations

import os
import shutil

import imageio_ffmpeg


class FFmpegResolver:
    def __init__(self, configured_path: str | None, normalizer_filter: str):
        self.configured_path = configured_path
        self.normalizer_filter = normalizer_filter
        self._cached_path: str | None = None

    def executable(self) -> str:
        if self._cached_path:
            return self._cached_path

        candidates = [
            self.configured_path,
            shutil.which("ffmpeg"),
        ]

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                self._cached_path = candidate
                return candidate

        self._cached_path = imageio_ffmpeg.get_ffmpeg_exe()
        return self._cached_path

    def playback_options(self, normalizer_enabled: bool) -> str:
        return "-vn -loglevel error"

    @staticmethod
    def reconnect_options() -> str:
        return "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -reconnect_at_eof 1"
