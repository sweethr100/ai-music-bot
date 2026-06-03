from __future__ import annotations

import discord
from discord.ext import commands

from config import load_settings
from services.ai_audio import AIProcessor
from services.ffmpeg import FFmpegResolver
from services.lyrics import LyricsService
from services.youtube import YouTubeService


class AIMusicBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.voice_states = True

        super().__init__(command_prefix="!", intents=intents)
        self.settings = load_settings()
        self.ffmpeg = FFmpegResolver(self.settings.ffmpeg_path, self.settings.normalizer_filter)
        self.youtube = YouTubeService()
        self.lyrics = LyricsService()
        self.ai_processor = AIProcessor(self.ffmpeg)

    async def setup_hook(self) -> None:
        await self.load_extension("cogs.music")

        if self.settings.guild_id:
            guild = discord.Object(id=self.settings.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            print(f"Synced slash commands to guild {self.settings.guild_id}")
        else:
            await self.tree.sync()
            print("Synced global slash commands")

    async def on_ready(self) -> None:
        print(f"Ready as {self.user} ({self.user.id})")

    async def close(self) -> None:
        await self.youtube.close()
        await super().close()


if __name__ == "__main__":
    bot = AIMusicBot()
    if not bot.settings.discord_token:
        raise RuntimeError("DISCORD_TOKEN is missing. Copy .env.example to .env and fill it in.")
    bot.run(bot.settings.discord_token)
