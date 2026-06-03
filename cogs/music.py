from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from services.ai_audio import (
    ORIGINAL_SINGER_NAME,
    ORIGINAL_SINGER_VALUE,
    OptionalFeatureMissing,
    is_original_singer_voice,
)
from services.voice_patch import PatchedVoiceRecvClient
from services.youtube import TrackInfo


REPEAT_CHOICES = [
    app_commands.Choice(name="꺼짐", value="off"),
    app_commands.Choice(name="현재곡 반복", value="one"),
    app_commands.Choice(name="전체 반복", value="all"),
]

YOUTUBE_MODE_CHOICES = [
    app_commands.Choice(name="원본 재생", value="original"),
    app_commands.Choice(name="MR만 재생 (보컬 제거)", value="instrumental"),
    app_commands.Choice(name="보컬만 재생 (반주 제거)", value="vocal"),
    app_commands.Choice(name="보컬 강조 믹스", value="vocal_boost"),
    app_commands.Choice(name="AI 듀엣", value="ai_duet"),
]

PITCH_CHOICES = [
    app_commands.Choice(name="그대로 (0)", value=0),
    app_commands.Choice(name="높게: 남자목소리->여자목소리 (+6)", value=6),
    app_commands.Choice(name="낮게: 여자목소리->남자목소리 (-6)", value=-6),
    app_commands.Choice(name="많이 높게 (+12)", value=12),
    app_commands.Choice(name="많이 낮게 (-12)", value=-12),
]


@dataclass
class Song:
    track: TrackInfo
    requester_id: int
    requester_name: str
    local_path: str | None = None

    @property
    def title(self) -> str:
        return self.track.title

    @property
    def url(self) -> str:
        return self.track.webpage_url

    @property
    def is_local(self) -> bool:
        return bool(self.local_path)

    def duration_text(self) -> str:
        return self.track.duration_text()


class GuildMusicState:
    def __init__(self):
        self.queue: deque[Song] = deque()
        self.current: Song | None = None
        self.is_playing = False
        self.repeat_mode = "off"
        self.normalizer_enabled = False
        self.skip_requested = False
        self.generation = 0


class LyricsView(discord.ui.View):
    def __init__(self, title: str, artist: str, pages: list[str]):
        super().__init__(timeout=180)
        self.title = title
        self.artist = artist
        self.pages = pages
        self.index = 0

    def embed(self) -> discord.Embed:
        embed = discord.Embed(
            title=f"가사: {self.title}",
            description=self.pages[self.index],
            color=discord.Color.blurple(),
        )
        embed.set_footer(text=f"{self.artist} | {self.index + 1}/{len(self.pages)}")
        return embed

    @discord.ui.button(label="이전", style=discord.ButtonStyle.secondary)
    async def previous(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.index = max(0, self.index - 1)
        await interaction.response.edit_message(embed=self.embed(), view=self)

    @discord.ui.button(label="다음", style=discord.ButtonStyle.secondary)
    async def next(self, interaction: discord.Interaction, _: discord.ui.Button):
        self.index = min(len(self.pages) - 1, self.index + 1)
        await interaction.response.edit_message(embed=self.embed(), view=self)


class MusicCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.states: dict[int, GuildMusicState] = {}

    def state_for(self, guild_id: int) -> GuildMusicState:
        if guild_id not in self.states:
            self.states[guild_id] = GuildMusicState()
        return self.states[guild_id]

    async def _ensure_voice(self, interaction: discord.Interaction) -> discord.VoiceClient | None:
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.followup.send("먼저 음성 채널에 들어가 주세요.", ephemeral=True)
            return None

        channel = interaction.user.voice.channel
        voice_client = interaction.guild.voice_client
        if voice_client and voice_client.is_connected():
            if voice_client.channel != channel:
                await voice_client.move_to(channel)
            return voice_client

        return await channel.connect(cls=PatchedVoiceRecvClient, self_deaf=True)

    async def _start_if_idle(self, guild_id: int) -> None:
        state = self.state_for(guild_id)
        if not state.is_playing:
            await self._play_next(guild_id)

    async def _play_next(self, guild_id: int) -> None:
        state = self.state_for(guild_id)
        guild = self.bot.get_guild(guild_id)
        if not guild or not guild.voice_client:
            state.is_playing = False
            return

        if not state.queue:
            state.current = None
            state.is_playing = False
            return

        song = state.queue.popleft()
        state.current = song
        state.is_playing = True

        try:
            source = await self._make_source(song, state.normalizer_enabled)
        except Exception as exc:
            print(f"Failed to prepare source for {song.title}: {exc}")
            self._cleanup_song(song)
            state.current = None
            state.is_playing = False
            await self._play_next(guild_id)
            return

        generation = state.generation
        guild.voice_client.play(
            source,
            after=lambda error: self._after_track(guild_id, generation, error),
        )

    async def _make_source(self, song: Song, normalizer_enabled: bool):
        ffmpeg = self.bot.ffmpeg
        options = ffmpeg.playback_options(normalizer_enabled)

        if song.is_local:
            return discord.FFmpegPCMAudio(
                song.local_path,
                executable=ffmpeg.executable(),
                options=options,
            )

        song.track.stream_url = None
        await self.bot.youtube.resolve_stream(song.track)
        return discord.FFmpegOpusAudio(
            song.track.stream_url,
            executable=ffmpeg.executable(),
            before_options=ffmpeg.reconnect_options(),
            options=options,
        )

    def _after_track(self, guild_id: int, generation: int, error: Exception | None) -> None:
        if error:
            print(f"Playback error in guild {guild_id}: {error}")
        asyncio.run_coroutine_threadsafe(
            self._advance_after_finish(guild_id, generation),
            self.bot.loop,
        )

    async def _advance_after_finish(self, guild_id: int, generation: int) -> None:
        await asyncio.sleep(0.4)
        state = self.state_for(guild_id)
        if generation != state.generation:
            return

        finished = state.current
        skip_requested = state.skip_requested
        state.skip_requested = False

        if finished:
            if state.repeat_mode == "one" and not skip_requested:
                state.queue.appendleft(finished)
            elif state.repeat_mode == "all":
                state.queue.append(finished)
            else:
                self._cleanup_song(finished)

        state.current = None
        state.is_playing = False
        await self._play_next(guild_id)

    def _cleanup_song(self, song: Song | None) -> None:
        if not song or not song.local_path:
            return

        path = Path(song.local_path)
        try:
            processed_root = (Path.cwd() / "data" / "processed").resolve()
            resolved = path.resolve()
            resolved.relative_to(processed_root)
        except Exception:
            return

        try:
            if path.exists():
                path.unlink()
        except OSError:
            self.bot.loop.create_task(self._retry_delete(path))

    @staticmethod
    async def _retry_delete(path: Path) -> None:
        for _ in range(5):
            await asyncio.sleep(2)
            try:
                if path.exists():
                    path.unlink()
                return
            except OSError:
                continue

    async def _enqueue(self, interaction: discord.Interaction, song: Song) -> None:
        state = self.state_for(interaction.guild_id)
        state.queue.append(song)
        await self._start_if_idle(interaction.guild_id)

    @app_commands.command(name="play", description="음악 재생, MR/보컬 분리, AI 커버/듀엣을 한 번에 처리합니다.")
    @app_commands.describe(
        query="유튜브 URL 또는 검색어",
        mode="원본, MR, 보컬, 보컬 강조, AI 커버, AI 듀엣 중 선택",
        target_voice="AI 커버/듀엣 1번 목소리 이름",
        duet_voice="AI 듀엣 2번 목소리 이름",
        duet_parts="AI 듀엣 파트. 예: 0:00-0:35:1,0:35-1:10:2,1:10-1:25:both",
        pitch_shift="피치 조절. 예: -6 여자목소리->남자목소리, +6 남자목소리->여자목소리",
    )
    @app_commands.choices(mode=YOUTUBE_MODE_CHOICES, pitch_shift=PITCH_CHOICES)
    async def play(
        self,
        interaction: discord.Interaction,
        query: str,
        mode: str = "original",
        target_voice: str | None = None,
        duet_voice: str | None = None,
        duet_parts: str | None = None,
        pitch_shift: int = 0,
    ):
        await interaction.response.defer()
        voice_client = await self._ensure_voice(interaction)
        if not voice_client:
            return

        effective_mode = self._effective_play_mode(mode, target_voice)
        try:
            if effective_mode == "original" and pitch_shift == 0:
                track = await self.bot.youtube.get_track(query)
                song = Song(track, interaction.user.id, interaction.user.display_name)
                await self._enqueue(interaction, song)
                await self._send_enqueue_message(interaction, song)
                return

            status = await interaction.followup.send(f"{self._mode_label(effective_mode)} 모드 준비 중...")

            async def progress(text: str) -> None:
                try:
                    await status.edit(content=text)
                except discord.HTTPException:
                    pass

            processed = await self.bot.ai_processor.process_youtube(
                url=query,
                mode=effective_mode,
                target_voice=target_voice,
                pitch_shift=pitch_shift,
                progress=progress,
                duet_voice=duet_voice,
                duet_parts=duet_parts,
            )
            track = TrackInfo(
                title=f"[{self._mode_label(effective_mode)}] {processed.title}",
                webpage_url=processed.webpage_url,
                duration=processed.duration,
                thumbnail=processed.thumbnail,
            )
            song = Song(track, interaction.user.id, interaction.user.display_name, local_path=processed.path)
            await self._enqueue(interaction, song)
            await status.edit(content=f"대기열에 추가했습니다: **{song.title}**")

            size = Path(processed.path).stat().st_size
            if effective_mode == "ai_cover" and size < 24_000_000:
                await interaction.channel.send(file=discord.File(processed.path))
        except OptionalFeatureMissing as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception as exc:
            await interaction.followup.send(f"처리 중 오류가 발생했습니다.\n```{str(exc)[:1500]}```", ephemeral=True)

    @play.autocomplete("target_voice")
    async def target_voice_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._voice_autocomplete_choices(current, include_original=True)

    @play.autocomplete("duet_voice")
    async def duet_voice_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._voice_autocomplete_choices(current)

    @app_commands.command(name="playlist", description="유튜브 재생목록을 대기열에 추가합니다.")
    @app_commands.describe(url="유튜브 플레이리스트 URL", max_count="추가할 최대 곡 수")
    async def playlist(
        self,
        interaction: discord.Interaction,
        url: str,
        max_count: app_commands.Range[int, 1, 100] = 30,
    ):
        await interaction.response.defer()
        voice_client = await self._ensure_voice(interaction)
        if not voice_client:
            return

        tracks = await self.bot.youtube.get_playlist(url, max_count)
        if not tracks:
            await interaction.followup.send("가져올 수 있는 곡을 찾지 못했습니다.", ephemeral=True)
            return

        state = self.state_for(interaction.guild_id)
        for track in tracks:
            state.queue.append(Song(track, interaction.user.id, interaction.user.display_name))

        await self._start_if_idle(interaction.guild_id)
        first = tracks[0].title
        await interaction.followup.send(f"플레이리스트에서 **{len(tracks)}곡**을 추가했습니다.\n첫 곡: **{first}**")

    @app_commands.command(name="queue", description="현재 재생 중인 곡과 대기열을 봅니다.")
    async def queue(self, interaction: discord.Interaction):
        state = self.state_for(interaction.guild_id)
        embed = discord.Embed(title="재생 대기열", color=discord.Color.blurple())

        if state.current:
            embed.add_field(
                name="현재 재생 중",
                value=f"[{state.current.title}]({state.current.url}) - {state.current.duration_text()}",
                inline=False,
            )

        if state.queue:
            lines = []
            for index, song in enumerate(list(state.queue)[:10], start=1):
                lines.append(f"`{index}.` [{song.title}]({song.url}) - {song.duration_text()}")
            if len(state.queue) > 10:
                lines.append(f"... 그리고 {len(state.queue) - 10}곡 더")
            embed.add_field(name="대기 중", value="\n".join(lines), inline=False)
        elif not state.current:
            embed.description = "`/play` 또는 `/playlist`로 곡을 추가해 주세요."

        embed.set_footer(
            text=f"대기 {len(state.queue)}곡 | 반복 {self._repeat_label(state.repeat_mode)} | 노멀라이저 {'켜짐' if state.normalizer_enabled else '꺼짐'}"
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="remove", description="대기열에서 특정 번호의 곡을 삭제합니다.")
    @app_commands.describe(position="/queue에 표시되는 대기열 번호")
    async def remove(self, interaction: discord.Interaction, position: app_commands.Range[int, 1, 1000]):
        state = self.state_for(interaction.guild_id)
        if position > len(state.queue):
            await interaction.response.send_message("그 번호의 대기열 곡이 없습니다.", ephemeral=True)
            return

        songs = list(state.queue)
        removed = songs.pop(position - 1)
        state.queue = deque(songs)
        self._cleanup_song(removed)
        await interaction.response.send_message(f"삭제했습니다: `#{position}` **{removed.title}**")

    @app_commands.command(name="loop", description="반복재생 모드를 설정합니다.")
    @app_commands.choices(mode=REPEAT_CHOICES)
    async def loop(self, interaction: discord.Interaction, mode: str):
        state = self.state_for(interaction.guild_id)
        state.repeat_mode = mode
        await interaction.response.send_message(f"반복재생: **{self._repeat_label(mode)}**")

    @app_commands.command(name="normalizer", description="곡마다 들쭉날쭉한 음량을 자동 보정합니다.")
    async def normalizer(self, interaction: discord.Interaction, enabled: bool):
        state = self.state_for(interaction.guild_id)
        state.normalizer_enabled = enabled
        await interaction.response.send_message(
            f"노멀라이저를 **{'켰습니다' if enabled else '껐습니다'}**. 현재 곡 다음부터 적용됩니다."
        )

    @app_commands.command(name="skip", description="현재 곡을 건너뜁니다.")
    async def skip(self, interaction: discord.Interaction):
        state = self.state_for(interaction.guild_id)
        voice_client = interaction.guild.voice_client
        if not voice_client or not state.current:
            await interaction.response.send_message("현재 재생 중인 곡이 없습니다.", ephemeral=True)
            return

        title = state.current.title
        state.skip_requested = True
        voice_client.stop()
        await interaction.response.send_message(f"건너뛰었습니다: **{title}**")

    @app_commands.command(name="pause", description="현재 곡을 일시정지합니다.")
    async def pause(self, interaction: discord.Interaction):
        voice_client = interaction.guild.voice_client
        if not voice_client or not voice_client.is_playing():
            await interaction.response.send_message("일시정지할 곡이 없습니다.", ephemeral=True)
            return
        voice_client.pause()
        await interaction.response.send_message("일시정지했습니다.")

    @app_commands.command(name="resume", description="일시정지된 곡을 다시 재생합니다.")
    async def resume(self, interaction: discord.Interaction):
        voice_client = interaction.guild.voice_client
        if not voice_client or not voice_client.is_paused():
            await interaction.response.send_message("재개할 곡이 없습니다.", ephemeral=True)
            return
        voice_client.resume()
        await interaction.response.send_message("다시 재생합니다.")

    @app_commands.command(name="nowplaying", description="현재 재생 중인 곡 정보를 봅니다.")
    async def nowplaying(self, interaction: discord.Interaction):
        state = self.state_for(interaction.guild_id)
        if not state.current:
            await interaction.response.send_message("현재 재생 중인 곡이 없습니다.", ephemeral=True)
            return
        song = state.current
        embed = discord.Embed(title="지금 재생 중", description=f"[{song.title}]({song.url})", color=discord.Color.green())
        embed.add_field(name="재생 시간", value=song.duration_text(), inline=True)
        embed.add_field(name="요청자", value=song.requester_name, inline=True)
        embed.add_field(name="대기열", value=f"{len(state.queue)}곡", inline=True)
        if song.track.thumbnail:
            embed.set_thumbnail(url=song.track.thumbnail)
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="lyrics", description="현재 곡 또는 검색어로 가사를 찾습니다.")
    @app_commands.describe(query="비워두면 현재 재생 중인 곡 제목으로 검색합니다.")
    async def lyrics(self, interaction: discord.Interaction, query: str | None = None):
        await interaction.response.defer()
        state = self.state_for(interaction.guild_id)
        search_query = query or (state.current.title if state.current else None)
        if not search_query:
            await interaction.followup.send("검색어가 없고 현재 재생 중인 곡도 없습니다.", ephemeral=True)
            return

        result = await self.bot.lyrics.search(search_query)
        if not result:
            await interaction.followup.send("가사를 찾지 못했습니다.", ephemeral=True)
            return

        view = LyricsView(result.title, result.artist, result.pages)
        await interaction.followup.send(embed=view.embed(), view=view if len(result.pages) > 1 else None)

    async def _send_enqueue_message(self, interaction: discord.Interaction, song: Song) -> None:
        state = self.state_for(interaction.guild_id)
        is_current = state.current == song
        title = "지금 재생 중" if is_current else "대기열에 추가됨"
        embed = discord.Embed(title=title, description=f"[{song.title}]({song.url})", color=discord.Color.green())
        embed.add_field(name="재생 시간", value=song.duration_text(), inline=True)
        embed.add_field(name="요청자", value=song.requester_name, inline=True)
        if not is_current:
            embed.add_field(name="대기 순서", value=str(len(state.queue)), inline=True)
        if song.track.thumbnail:
            embed.set_thumbnail(url=song.track.thumbnail)
        await interaction.followup.send(embed=embed)

    @staticmethod
    def _repeat_label(mode: str) -> str:
        return {"off": "꺼짐", "one": "현재곡 반복", "all": "전체 반복"}.get(mode, mode)

    @staticmethod
    def _effective_play_mode(mode: str, target_voice: str | None) -> str:
        if mode == "ai_duet":
            return mode
        if target_voice and not is_original_singer_voice(target_voice):
            return "ai_cover"
        if mode == "ai_cover" and is_original_singer_voice(target_voice):
            return "vocal_boost"
        return mode

    def _voice_autocomplete_choices(self, current: str, include_original: bool = False):
        current_lower = current.lower()
        choices: list[app_commands.Choice[str]] = []
        if include_original:
            choices.append(app_commands.Choice(name=ORIGINAL_SINGER_NAME, value=ORIGINAL_SINGER_VALUE))

        choices.extend(
            app_commands.Choice(name=name, value=name)
            for name in self.bot.ai_processor.available_voices()
            if current_lower in name.lower()
        )
        return choices[:25]

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {
            "original": "피치변경",
            "instrumental": "MR",
            "vocal": "보컬",
            "vocal_boost": "보컬강조",
            "ai_cover": "AI커버",
            "ai_duet": "AI듀엣",
        }.get(mode, mode)


async def setup(bot: commands.Bot):
    await bot.add_cog(MusicCog(bot))
