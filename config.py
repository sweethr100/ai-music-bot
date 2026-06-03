from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    discord_token: str
    guild_id: int | None
    ffmpeg_path: str | None
    normalizer_filter: str


def _optional_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_settings() -> Settings:
    ffmpeg_path = os.getenv("FFMPEG_PATH") or None

    return Settings(
        discord_token=os.getenv("DISCORD_TOKEN", ""),
        guild_id=_optional_int(os.getenv("GUILD_ID")),
        ffmpeg_path=ffmpeg_path,
        normalizer_filter=os.getenv("NORMALIZER_FILTER", "dynaudnorm=f=150:g=15"),
    )
