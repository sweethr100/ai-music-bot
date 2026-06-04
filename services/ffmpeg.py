from __future__ import annotations

import ctypes
import os
import shutil
from ctypes import wintypes

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
    def prioritize_playback_process(process) -> None:
        pid = getattr(process, "pid", None)
        if not pid:
            return

        if os.name == "nt":
            try:
                kernel32 = ctypes.windll.kernel32
                kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel32.OpenProcess.restype = wintypes.HANDLE
                kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel32.SetPriorityClass.restype = wintypes.BOOL
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle.restype = wintypes.BOOL

                process_set_information = 0x0200
                high_priority_class = 0x00000080
                handle = kernel32.OpenProcess(process_set_information, False, int(pid))
                if not handle:
                    return
                try:
                    kernel32.SetPriorityClass(handle, high_priority_class)
                finally:
                    kernel32.CloseHandle(handle)
            except OSError:
                pass
            return

        if hasattr(os, "setpriority") and hasattr(os, "PRIO_PROCESS"):
            try:
                os.setpriority(os.PRIO_PROCESS, int(pid), -5)
            except OSError:
                pass

    @staticmethod
    def reconnect_options() -> str:
        return "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -reconnect_at_eof 1"
