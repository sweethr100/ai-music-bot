from __future__ import annotations

import asyncio
import shutil
import tempfile
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
]

DELIVERY_CHOICES = [
    app_commands.Choice(name="파일 올리고 통화방에서도 틀기", value="upload_and_play"),
    app_commands.Choice(name="파일만 채팅에 올리기", value="upload_only"),
    app_commands.Choice(name="통화방에서만 틀기", value="voice_only"),
]


@dataclass
class Song:
    track: TrackInfo
    requester_id: int
    requester_name: str
    local_path: str | None = None
    playback_path: str | None = None
    normalizer_task: asyncio.Task | None = None
    mode: str = "original"
    target_voice: str | None = None
    duet_voice: str | None = None
    vocal_pitch_shift: int = 0

    @property
    def title(self) -> str:
        return self.track.title

    @property
    def url(self) -> str:
        return self.track.webpage_url

    @property
    def is_local(self) -> bool:
        return bool(self.playback_path or self.local_path)

    def duration_text(self) -> str:
        return self.track.duration_text()

    def detail_text(self) -> str:
        details: list[str] = []
        if self.mode == "ai_cover" and self.target_voice:
            details.append(f"AI 목소리: {self.target_voice}")
        if self.mode == "ai_duet":
            duet_names = [name for name in (self.target_voice, self.duet_voice) if name]
            if duet_names:
                details.append(f"AI 듀엣: {' + '.join(duet_names)}")
        if self.vocal_pitch_shift:
            details.append(f"목소리 피치: {self.vocal_pitch_shift:+d}")
        return " | ".join(details)


class GuildMusicState:
    def __init__(self):
        self.queue: deque[Song] = deque()
        self.current: Song | None = None
        self.is_playing = False
        self.repeat_mode = "off"
        self.normalizer_enabled = True
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

    async def _ensure_voice(
        self,
        interaction: discord.Interaction,
        target_channel: discord.VoiceChannel | None = None,
    ) -> discord.VoiceClient | None:
        user_voice = getattr(interaction.user, "voice", None)
        channel = target_channel or getattr(user_voice, "channel", None)
        if not channel:
            await interaction.followup.send(
                "먼저 음성 채널에 들어가거나 `voice_channel`을 선택해 주세요.",
                ephemeral=True,
            )
            return None

        voice_client = interaction.guild.voice_client
        if voice_client and voice_client.is_connected():
            await interaction.guild.change_voice_state(channel=voice_client.channel, self_deaf=False)
            if voice_client.channel != channel:
                if self._voice_client_busy(interaction.guild_id, voice_client):
                    await interaction.followup.send(
                        f"이미 **{voice_client.channel.name}**에서 재생 중이라 "
                        "새 요청은 현재 재생 채널의 대기열에 넣습니다.",
                        ephemeral=True,
                    )
                    return voice_client
                await voice_client.move_to(channel)
                await interaction.guild.change_voice_state(channel=channel, self_deaf=False)
            return voice_client

        return await channel.connect(cls=PatchedVoiceRecvClient, self_deaf=False)

    def _voice_client_busy(self, guild_id: int, voice_client: discord.VoiceClient | None) -> bool:
        state = self.state_for(guild_id)
        return bool(
            state.current
            or state.is_playing
            or (voice_client and (voice_client.is_playing() or voice_client.is_paused()))
        )

    async def _resolve_voice_channel_option(
        self,
        interaction: discord.Interaction,
        connected_voice_channel: str | None,
    ) -> discord.VoiceChannel | None:
        if not connected_voice_channel:
            return None

        if not interaction.guild:
            await interaction.followup.send("서버 안에서만 음성 채널을 선택할 수 있습니다.", ephemeral=True)
            return None

        try:
            channel_id = int(connected_voice_channel)
        except ValueError:
            await interaction.followup.send("선택한 음성 채널 값을 읽지 못했습니다.", ephemeral=True)
            return None

        channel = interaction.guild.get_channel(channel_id)
        if not isinstance(channel, discord.VoiceChannel):
            await interaction.followup.send("선택한 음성 채널을 찾지 못했습니다.", ephemeral=True)
            return None

        return channel

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
            source_path = await self._prepare_playback_path(song, normalizer_enabled)
            source = discord.FFmpegPCMAudio(
                source_path,
                executable=ffmpeg.executable(),
                options=options,
            )
            ffmpeg.prioritize_playback_process(getattr(source, "_process", None))
            return source

        if normalizer_enabled:
            exported = await self.bot.ai_processor.export_youtube_audio(song.url, self._noop_progress)
            song.local_path = exported.path
            song.playback_path = str(
                await self.bot.ai_processor.normalize_for_playback(Path(exported.path), song.title)
            )
            source = discord.FFmpegPCMAudio(
                song.playback_path,
                executable=ffmpeg.executable(),
                options=options,
            )
            ffmpeg.prioritize_playback_process(getattr(source, "_process", None))
            return source

        song.track.stream_url = None
        await self.bot.youtube.resolve_stream(song.track)
        source = discord.FFmpegOpusAudio(
            song.track.stream_url,
            executable=ffmpeg.executable(),
            before_options=ffmpeg.reconnect_options(),
            options=options,
        )
        ffmpeg.prioritize_playback_process(getattr(source, "_process", None))
        return source

    async def _prepare_playback_path(self, song: Song, normalizer_enabled: bool) -> str:
        if not normalizer_enabled:
            source_path = song.playback_path or song.local_path
            if not source_path:
                raise RuntimeError("재생할 로컬 파일을 찾지 못했습니다.")
            return source_path
        if song.playback_path:
            return song.playback_path
        if song.normalizer_task:
            try:
                await song.normalizer_task
            except Exception as exc:
                print(f"Normalizer failed for {song.title}: {exc}")
            if song.playback_path:
                return song.playback_path
        if not song.local_path:
            raise RuntimeError("재생할 로컬 파일을 찾지 못했습니다.")

        await self._normalize_song_for_playback(song)
        return song.playback_path

    @staticmethod
    async def _noop_progress(_: str) -> None:
        return None

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
        if not song or not (song.local_path or song.playback_path):
            return

        if song.normalizer_task and not song.normalizer_task.done():
            song.normalizer_task.cancel()
        song.normalizer_task = None

        paths = [Path(value) for value in {song.local_path, song.playback_path} if value]
        try:
            processed_root = (Path.cwd() / "data" / "processed").resolve()
            paths = [
                path
                for path in paths
                if path.resolve().is_relative_to(processed_root)
            ]
        except Exception:
            paths = []

        for path in paths:
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
        self._prewarm_normalizer(interaction.guild_id, song)
        await self._start_if_idle(interaction.guild_id)

    def _prewarm_normalizer(self, guild_id: int, song: Song) -> None:
        state = self.state_for(guild_id)
        if state.normalizer_enabled and song.local_path and not song.playback_path and not song.normalizer_task:
            song.normalizer_task = self.bot.loop.create_task(self._normalize_song_for_playback(song))

    async def _normalize_song_for_playback(self, song: Song) -> None:
        if not song.local_path:
            return
        try:
            song.playback_path = str(
                await self.bot.ai_processor.normalize_for_playback(Path(song.local_path), song.title)
            )
        finally:
            song.normalizer_task = None

    @app_commands.command(name="play", description="음악 재생, MR/보컬 분리, AI 커버를 처리합니다.")
    @app_commands.describe(
        query="유튜브 URL 또는 검색어",
        mode="원본, MR, 보컬, 보컬 강조 중 선택",
        target_voice="AI 커버 목소리 이름",
        vocal_pitch_shift="목소리 피치만 조절. -12부터 +12까지 정수로 입력",
        delivery="결과 파일을 채팅에 올릴지, 통화방에서 틀지도 함께 선택",
        voice_channel="명령어 사용자가 음성 채널에 없어도 재생할 음성 채널",
        connected_voice_channel="현재 접속자가 있는 음성 채널 중에서 선택",
    )
    @app_commands.choices(mode=YOUTUBE_MODE_CHOICES, delivery=DELIVERY_CHOICES)
    async def play(
        self,
        interaction: discord.Interaction,
        query: str,
        mode: str = "original",
        target_voice: str | None = None,
        vocal_pitch_shift: app_commands.Range[int, -12, 12] = 0,
        delivery: str = "upload_and_play",
        voice_channel: discord.VoiceChannel | None = None,
        connected_voice_channel: str | None = None,
    ):
        await interaction.response.defer()
        upload_file = delivery in {"upload_and_play", "upload_only"}
        play_in_voice = delivery in {"upload_and_play", "voice_only"}

        if play_in_voice:
            selected_voice_channel = await self._resolve_voice_channel_option(interaction, connected_voice_channel)
            if connected_voice_channel and not selected_voice_channel:
                return
            target_channel = selected_voice_channel or voice_channel
            voice_client = await self._ensure_voice(interaction, target_channel)
            if not voice_client:
                return

        effective_mode = self._effective_play_mode(mode, target_voice)
        state = self.state_for(interaction.guild_id)
        try:
            if (
                effective_mode == "original"
                and vocal_pitch_shift == 0
                and delivery == "voice_only"
                and not state.normalizer_enabled
            ):
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
                vocal_pitch_shift=vocal_pitch_shift,
                progress=progress,
            )
            track = TrackInfo(
                title=f"[{self._mode_label(effective_mode)}] {processed.title}",
                webpage_url=processed.webpage_url,
                duration=processed.duration,
                thumbnail=processed.thumbnail,
            )
            song = Song(
                track,
                interaction.user.id,
                interaction.user.display_name,
                local_path=processed.path,
                mode=effective_mode,
                target_voice=target_voice if effective_mode == "ai_cover" else None,
                vocal_pitch_shift=vocal_pitch_shift if effective_mode != "instrumental" else 0,
            )

            if play_in_voice:
                self._prewarm_normalizer(interaction.guild_id, song)

            uploaded = True
            if upload_file:
                uploaded = await self._send_song_file(interaction, song)

            if play_in_voice:
                await self._enqueue(interaction, song)

            if play_in_voice and upload_file:
                if uploaded:
                    await status.edit(content=f"대기열에 추가하고 파일도 올렸습니다: **{song.title}**")
                else:
                    await status.edit(content=f"대기열에는 추가했지만 파일 업로드는 실패했습니다: **{song.title}**")
            elif play_in_voice:
                await status.edit(content=f"대기열에 추가했습니다: **{song.title}**")
            else:
                await status.edit(
                    content=(
                        f"파일을 올렸습니다: **{song.title}**"
                        if uploaded
                        else f"파일 업로드에 실패했습니다: **{song.title}**"
                    )
                )
                self._cleanup_song(song)
        except OptionalFeatureMissing as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception as exc:
            await interaction.followup.send(f"처리 중 오류가 발생했습니다.\n```{str(exc)[:1500]}```", ephemeral=True)

    @play.autocomplete("target_voice")
    async def target_voice_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._voice_autocomplete_choices(current, include_original=True)

    @play.autocomplete("connected_voice_channel")
    async def connected_voice_channel_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._connected_voice_channel_choices(interaction, current)

    @app_commands.command(name="play_file", description="업로드한 파일로 음악 재생, MR/보컬 분리, AI 커버를 처리합니다.")
    @app_commands.describe(
        file="재생하거나 처리할 오디오/동영상 파일",
        mode="원본, MR, 보컬, 보컬 강조 중 선택",
        target_voice="AI 커버 목소리 이름",
        vocal_pitch_shift="목소리 피치만 조절. -12부터 +12까지 정수로 입력",
        delivery="결과 파일을 채팅에 올릴지, 통화방에서 틀지도 함께 선택",
        voice_channel="명령어 사용자가 음성 채널에 없어도 재생할 음성 채널",
        connected_voice_channel="현재 접속자가 있는 음성 채널 중에서 선택",
    )
    @app_commands.choices(mode=YOUTUBE_MODE_CHOICES, delivery=DELIVERY_CHOICES)
    async def play_file(
        self,
        interaction: discord.Interaction,
        file: discord.Attachment,
        mode: str = "original",
        target_voice: str | None = None,
        vocal_pitch_shift: app_commands.Range[int, -12, 12] = 0,
        delivery: str = "upload_and_play",
        voice_channel: discord.VoiceChannel | None = None,
        connected_voice_channel: str | None = None,
    ):
        await interaction.response.defer()
        upload_file = delivery in {"upload_and_play", "upload_only"}
        play_in_voice = delivery in {"upload_and_play", "voice_only"}

        if play_in_voice:
            selected_voice_channel = await self._resolve_voice_channel_option(interaction, connected_voice_channel)
            if connected_voice_channel and not selected_voice_channel:
                return
            target_channel = selected_voice_channel or voice_channel
            voice_client = await self._ensure_voice(interaction, target_channel)
            if not voice_client:
                return

        effective_mode = self._effective_play_mode(mode, target_voice)
        source_dir: Path | None = None
        try:
            status = await interaction.followup.send(f"{self._mode_label(effective_mode)} 모드 준비 중...")

            async def progress(text: str) -> None:
                try:
                    await status.edit(content=text)
                except discord.HTTPException:
                    pass

            source_path, source_dir = await self._save_attachment(file)
            title = Path(file.filename).stem or "uploaded_audio"
            processed = await self.bot.ai_processor.process_file(
                source=source_path,
                title=title,
                webpage_url=file.url,
                mode=effective_mode,
                target_voice=target_voice,
                vocal_pitch_shift=vocal_pitch_shift,
                progress=progress,
            )
            track = TrackInfo(
                title=f"[{self._mode_label(effective_mode)}] {processed.title}",
                webpage_url=processed.webpage_url,
                duration=processed.duration,
                thumbnail=processed.thumbnail,
            )
            song = Song(
                track,
                interaction.user.id,
                interaction.user.display_name,
                local_path=processed.path,
                mode=effective_mode,
                target_voice=target_voice if effective_mode == "ai_cover" else None,
                vocal_pitch_shift=vocal_pitch_shift if effective_mode != "instrumental" else 0,
            )

            if play_in_voice:
                self._prewarm_normalizer(interaction.guild_id, song)

            uploaded = True
            if upload_file:
                uploaded = await self._send_song_file(interaction, song)

            if play_in_voice:
                await self._enqueue(interaction, song)

            if play_in_voice and upload_file:
                if uploaded:
                    await status.edit(content=f"대기열에 추가하고 파일도 올렸습니다: **{song.title}**")
                else:
                    await status.edit(content=f"대기열에는 추가했지만 파일 업로드는 실패했습니다: **{song.title}**")
            elif play_in_voice:
                await status.edit(content=f"대기열에 추가했습니다: **{song.title}**")
            else:
                await status.edit(
                    content=(
                        f"파일을 올렸습니다: **{song.title}**"
                        if uploaded
                        else f"파일 업로드에 실패했습니다: **{song.title}**"
                    )
                )
                self._cleanup_song(song)
        except OptionalFeatureMissing as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception as exc:
            await interaction.followup.send(f"처리 중 오류가 발생했습니다.\n```{str(exc)[:1500]}```", ephemeral=True)
        finally:
            if source_dir:
                shutil.rmtree(source_dir, ignore_errors=True)

    @play_file.autocomplete("target_voice")
    async def play_file_target_voice_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._voice_autocomplete_choices(current, include_original=True)

    @play_file.autocomplete("connected_voice_channel")
    async def play_file_connected_voice_channel_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._connected_voice_channel_choices(interaction, current)

    @app_commands.command(name="play_duet", description="AI 듀엣 커버를 가수별 자동 분리로 만듭니다.")
    @app_commands.describe(
        query="유튜브 URL 또는 검색어",
        voice1="AI 듀엣 1번 목소리 이름",
        voice2="AI 듀엣 2번 목소리 이름",
        vocal_pitch_shift="목소리 피치만 조절. -12부터 +12까지 정수로 입력",
        delivery="결과 파일을 채팅에 올릴지, 통화방에서 틀지도 함께 선택",
        voice_channel="명령어 사용자가 음성 채널에 없어도 재생할 음성 채널",
        connected_voice_channel="현재 접속자가 있는 음성 채널 중에서 선택",
    )
    @app_commands.choices(delivery=DELIVERY_CHOICES)
    async def play_duet(
        self,
        interaction: discord.Interaction,
        query: str,
        voice1: str,
        voice2: str,
        vocal_pitch_shift: app_commands.Range[int, -12, 12] = 0,
        delivery: str = "upload_and_play",
        voice_channel: discord.VoiceChannel | None = None,
        connected_voice_channel: str | None = None,
    ):
        await interaction.response.defer()
        upload_file = delivery in {"upload_and_play", "upload_only"}
        play_in_voice = delivery in {"upload_and_play", "voice_only"}

        if play_in_voice:
            selected_voice_channel = await self._resolve_voice_channel_option(interaction, connected_voice_channel)
            if connected_voice_channel and not selected_voice_channel:
                return
            target_channel = selected_voice_channel or voice_channel
            voice_client = await self._ensure_voice(interaction, target_channel)
            if not voice_client:
                return

        effective_mode = "ai_duet"
        voice1_label = ORIGINAL_SINGER_NAME if is_original_singer_voice(voice1) else voice1
        voice2_label = ORIGINAL_SINGER_NAME if is_original_singer_voice(voice2) else voice2
        try:
            status = await interaction.followup.send(f"{self._mode_label(effective_mode)} 모드 준비 중...")

            async def progress(text: str) -> None:
                try:
                    await status.edit(content=text)
                except discord.HTTPException:
                    pass

            processed = await self.bot.ai_processor.process_youtube(
                url=query,
                mode=effective_mode,
                target_voice=voice1,
                vocal_pitch_shift=vocal_pitch_shift,
                progress=progress,
                duet_voice=voice2,
            )
            track = TrackInfo(
                title=f"[{self._mode_label(effective_mode)}] {processed.title}",
                webpage_url=processed.webpage_url,
                duration=processed.duration,
                thumbnail=processed.thumbnail,
            )
            song = Song(
                track,
                interaction.user.id,
                interaction.user.display_name,
                local_path=processed.path,
                mode=effective_mode,
                target_voice=voice1_label,
                duet_voice=voice2_label,
                vocal_pitch_shift=vocal_pitch_shift,
            )

            if play_in_voice:
                self._prewarm_normalizer(interaction.guild_id, song)

            uploaded = True
            if upload_file:
                uploaded = await self._send_song_file(interaction, song)

            if play_in_voice:
                await self._enqueue(interaction, song)

            if play_in_voice and upload_file:
                if uploaded:
                    await status.edit(content=f"대기열에 추가하고 파일도 올렸습니다: **{song.title}**")
                else:
                    await status.edit(content=f"대기열에는 추가했지만 파일 업로드는 실패했습니다: **{song.title}**")
            elif play_in_voice:
                await status.edit(content=f"대기열에 추가했습니다: **{song.title}**")
            else:
                await status.edit(
                    content=(
                        f"파일을 올렸습니다: **{song.title}**"
                        if uploaded
                        else f"파일 업로드에 실패했습니다: **{song.title}**"
                    )
                )
                self._cleanup_song(song)
        except OptionalFeatureMissing as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception as exc:
            await interaction.followup.send(f"처리 중 오류가 발생했습니다.\n```{str(exc)[:1500]}```", ephemeral=True)

    @play_duet.autocomplete("voice1")
    async def play_duet_voice1_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._voice_autocomplete_choices(current, include_original=True)

    @play_duet.autocomplete("voice2")
    async def play_duet_voice2_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._voice_autocomplete_choices(current, include_original=True)

    @play_duet.autocomplete("connected_voice_channel")
    async def play_duet_connected_voice_channel_autocomplete(self, interaction: discord.Interaction, current: str):
        return self._connected_voice_channel_choices(interaction, current)

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
            current_detail = state.current.detail_text()
            current_value = f"[{state.current.title}]({state.current.url}) - {state.current.duration_text()}"
            if current_detail:
                current_value += f"\n{current_detail}"
            embed.add_field(
                name="현재 재생 중",
                value=current_value,
                inline=False,
            )

        if state.queue:
            lines = []
            for index, song in enumerate(list(state.queue)[:10], start=1):
                detail = song.detail_text()
                line = f"`{index}.` [{song.title}]({song.url}) - {song.duration_text()}"
                if detail:
                    line += f" ({detail})"
                lines.append(line)
            if len(state.queue) > 10:
                lines.append(f"... 그리고 {len(state.queue) - 10}곡 더")
            embed.add_field(name="대기 중", value="\n".join(lines), inline=False)
        elif not state.current:
            embed.description = "`/play`, `/play_file` 또는 `/playlist`로 곡을 추가해 주세요."

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

    @app_commands.command(name="normalizer", description="곡 전체를 분석해서 곡끼리 음량을 맞춥니다.")
    async def normalizer(self, interaction: discord.Interaction, enabled: bool):
        state = self.state_for(interaction.guild_id)
        state.normalizer_enabled = enabled
        await interaction.response.send_message(
            f"노멀라이저를 **{'켰습니다' if enabled else '껐습니다'}**. 현재 곡 다음부터 재생용 파일에 적용됩니다."
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

    async def _save_attachment(self, attachment: discord.Attachment) -> tuple[Path, Path]:
        temp_root = Path("tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        source_dir = Path(tempfile.mkdtemp(prefix="discord_upload_source_", dir=temp_root))
        original = Path(attachment.filename or "uploaded_audio")
        safe_stem = self._safe_filename_part(original.stem or "uploaded_audio")
        safe_suffix = "".join(
            ch if ch.isalnum() or ch in "._-" else "_"
            for ch in (original.suffix[:16] or ".bin")
        )
        if not safe_suffix.startswith("."):
            safe_suffix = f".{safe_suffix.lstrip('.')}"
        source_path = source_dir / f"{safe_stem}{safe_suffix}"
        try:
            await attachment.save(source_path)
            return source_path, source_dir
        except Exception:
            shutil.rmtree(source_dir, ignore_errors=True)
            raise

    async def _send_song_file(self, interaction: discord.Interaction, song: Song) -> bool:
        if not song.local_path:
            return False

        path = Path(song.local_path)
        upload_limit = self._upload_size_limit(interaction)

        try:
            if path.stat().st_size <= upload_limit:
                await self._send_file_message(interaction, f"노래 파일: **{song.title}**", path)
                return True

            chunks = await self._split_song_file_for_upload(path, song, upload_limit)
            if not chunks:
                return False

            try:
                total = len(chunks)
                for index, chunk in enumerate(chunks, start=1):
                    await self._send_file_message(
                        interaction,
                        f"노래 파일 ({index}/{total}): **{song.title}**",
                        chunk,
                    )
            finally:
                chunk_dirs = {chunk.parent for chunk in chunks}
                for chunk in chunks:
                    try:
                        chunk.unlink(missing_ok=True)
                    except OSError:
                        pass
                for chunk_dir in chunk_dirs:
                    try:
                        chunk_dir.rmdir()
                    except OSError:
                        pass
            return True
        except (OSError, RuntimeError, discord.HTTPException) as exc:
            await interaction.followup.send(
                f"파일 업로드에 실패했습니다.\n```{str(exc)[:800]}```",
                ephemeral=True,
            )
            return False

    async def _send_file_message(self, interaction: discord.Interaction, content: str, path: Path) -> None:
        if interaction.channel:
            await interaction.channel.send(content=content, file=discord.File(path))
        else:
            await interaction.followup.send(content=content, file=discord.File(path))

    async def _split_song_file_for_upload(self, path: Path, song: Song, upload_limit: int) -> list[Path]:
        temp_root = Path("tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="discord_upload_", dir=temp_root) as temp_dir:
            temp_path = Path(temp_dir)
            bitrate = 320_000
            bytes_per_second = bitrate / 8
            segment_seconds = max(30, int((upload_limit * 0.82) / bytes_per_second))
            if song.track.duration:
                segment_seconds = min(segment_seconds, max(30, song.track.duration))

            output_pattern = temp_path / "part_%03d.mp3"
            cmd = [
                self.bot.ffmpeg.executable(),
                "-y",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-vn",
                "-b:a",
                "320k",
                "-f",
                "segment",
                "-segment_time",
                str(segment_seconds),
                "-reset_timestamps",
                "1",
                str(output_pattern),
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(stderr.decode("utf-8", errors="ignore")[-1000:])

            chunks = sorted(temp_path.glob("part_*.mp3"))
            if not chunks:
                return []
            if any(chunk.stat().st_size > upload_limit for chunk in chunks):
                raise RuntimeError("분할된 파일도 Discord 업로드 제한보다 큽니다.")

            copied_chunks: list[Path] = []
            keep_dir = Path(tempfile.mkdtemp(prefix="discord_upload_chunks_", dir=temp_root))
            for chunk in chunks:
                target = keep_dir / f"{path.stem}_{chunk.name}"
                target.write_bytes(chunk.read_bytes())
                copied_chunks.append(target)
            return copied_chunks

    @staticmethod
    def _upload_size_limit(interaction: discord.Interaction) -> int:
        guild_limit = getattr(interaction.guild, "filesize_limit", None)
        if isinstance(guild_limit, int) and guild_limit > 0:
            return max(1_000_000, int(guild_limit * 0.92))
        return 23_000_000

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
        if target_voice and not is_original_singer_voice(target_voice):
            return "ai_cover"
        if mode == "ai_cover" and is_original_singer_voice(target_voice):
            return "vocal_boost"
        return mode

    def _voice_autocomplete_choices(
        self,
        current: str,
        include_original: bool = False,
    ):
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
    def _connected_voice_channel_choices(
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        if not interaction.guild:
            return []

        current_lower = current.lower()
        choices: list[app_commands.Choice[str]] = []
        for channel in interaction.guild.voice_channels:
            if not channel.members:
                continue

            category_name = channel.category.name if channel.category else ""
            searchable = f"{category_name} {channel.name}".lower()
            if current_lower and current_lower not in searchable:
                continue

            member_count = len(channel.members)
            label = f"{channel.name} ({member_count}명)"
            if category_name:
                label = f"{category_name} / {label}"
            choices.append(app_commands.Choice(name=label[:100], value=str(channel.id)))
            if len(choices) >= 25:
                break

        return choices

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {
            "original": "원본",
            "instrumental": "MR",
            "vocal": "보컬",
            "vocal_boost": "보컬강조",
            "ai_cover": "AI커버",
            "ai_duet": "AI듀엣",
        }.get(mode, mode)

    @staticmethod
    def _safe_filename_part(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in value).strip(" ._")
        return (safe or "audio")[:80]


async def setup(bot: commands.Bot):
    await bot.add_cog(MusicCog(bot))
