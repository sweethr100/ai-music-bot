from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands
import discord.ext.voice_recv as voice_recv

from services.voice_patch import PatchedVoiceRecvClient


RECORDING_ROOT = Path("data") / "recordings"
CHANNELS = 2
SAMPLE_WIDTH = 2
SAMPLE_RATE = 48_000
BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH
WAV_HEADER_SIZE = 44
SPLIT_SECONDS = 5 * 60
HEADER_FLUSH_SECONDS = 30


def _write_wav_header(file_obj, data_size: int = 0xFFFFFFFF - 36) -> None:
    file_obj.seek(0)
    file_obj.write(b"RIFF")
    file_obj.write(struct.pack("<I", data_size + 36))
    file_obj.write(b"WAVE")
    file_obj.write(b"fmt ")
    file_obj.write(struct.pack("<I", 16))
    file_obj.write(struct.pack("<H", 1))
    file_obj.write(struct.pack("<H", CHANNELS))
    file_obj.write(struct.pack("<I", SAMPLE_RATE))
    file_obj.write(struct.pack("<I", BYTES_PER_SECOND))
    file_obj.write(struct.pack("<H", CHANNELS * SAMPLE_WIDTH))
    file_obj.write(struct.pack("<H", SAMPLE_WIDTH * 8))
    file_obj.write(b"data")
    file_obj.write(struct.pack("<I", data_size))


def _update_wav_header(file_obj, data_size: int) -> None:
    file_obj.seek(4)
    file_obj.write(struct.pack("<I", data_size + 36))
    file_obj.seek(40)
    file_obj.write(struct.pack("<I", data_size))


def _safe_name(value: str, fallback: str = "user") -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", value).strip(" ._")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned or fallback)[:80]


def _dataset_dir_for(user: discord.abc.User) -> Path:
    display = getattr(user, "display_name", None) or getattr(user, "name", str(user.id))
    return RECORDING_ROOT / f"{_safe_name(display)}_{user.id}"


def _write_metadata(user_dir: Path, user: discord.abc.User) -> None:
    display = getattr(user, "display_name", None) or getattr(user, "name", str(user.id))
    metadata = {
        "user_id": user.id,
        "display_name": display,
        "username": getattr(user, "name", display),
        "updated_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    try:
        (user_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


class UserWavSink(voice_recv.sinks.AudioSink):
    def __init__(self, user: discord.abc.User):
        super().__init__()
        self.user = user
        self.user_dir = _dataset_dir_for(user)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        _write_metadata(self.user_dir, user)

        self.display_name = getattr(user, "display_name", None) or getattr(user, "name", str(user.id))
        self.split_bytes = SPLIT_SECONDS * BYTES_PER_SECOND
        self.wav_path: Path | None = None
        self.data_bytes = 0
        self.total_bytes = 0
        self._file = None
        self._closed = False
        self._last_header_flush = time.monotonic()
        self._open_resumable_file()

    def wants_opus(self) -> bool:
        return False

    @property
    def duration_seconds(self) -> float:
        return self.total_bytes / BYTES_PER_SECOND

    def write(self, user, data) -> None:
        if self._closed or data.pcm is None:
            return

        self._file.write(data.pcm)
        chunk_size = len(data.pcm)
        self.data_bytes += chunk_size
        self.total_bytes += chunk_size

        if self.data_bytes >= self.split_bytes:
            self._flush_header()
            self._file.close()
            self._open_new_file()
            return

        now = time.monotonic()
        if now - self._last_header_flush >= HEADER_FLUSH_SECONDS:
            self._flush_header()
            self._last_header_flush = now

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._file and not self._file.closed:
                self._flush_header()
                self._file.close()
        except OSError:
            pass

    def _open_resumable_file(self) -> None:
        wav_files = sorted(self.user_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime)
        if wav_files:
            last_file = wav_files[-1]
            data_size = max(0, last_file.stat().st_size - WAV_HEADER_SIZE)
            if data_size < self.split_bytes:
                self.wav_path = last_file
                self._file = last_file.open("r+b")
                self._file.seek(0, os.SEEK_END)
                self.data_bytes = data_size
                return

        self._open_new_file()

    def _open_new_file(self) -> None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_display = _safe_name(self.display_name)
        self.wav_path = self.user_dir / f"{safe_display}_{timestamp}.wav"
        self._file = self.wav_path.open("wb")
        _write_wav_header(self._file)
        self.data_bytes = 0
        self._last_header_flush = time.monotonic()

    def _flush_header(self) -> None:
        position = self._file.tell()
        _update_wav_header(self._file, self.data_bytes)
        self._file.seek(position)
        self._file.flush()


class MultiUserWavSink(voice_recv.sinks.AudioSink):
    def __init__(self):
        super().__init__()
        self.user_sinks: dict[int, UserWavSink] = {}

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data) -> None:
        if user is None:
            return
        if user.id not in self.user_sinks:
            self.user_sinks[user.id] = UserWavSink(user)
        self.user_sinks[user.id].write(user, data)

    def cleanup(self) -> None:
        for sink in list(self.user_sinks.values()):
            sink.cleanup()

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for sink in self.user_sinks.values():
            seconds = int(sink.duration_seconds)
            minutes, seconds = divmod(seconds, 60)
            lines.append(f"- {sink.display_name}: {minutes}분 {seconds}초")
        return lines


@dataclass
class ActiveRecording:
    channel_id: int
    channel_name: str
    started_at: float
    sink: MultiUserWavSink


class VoiceRecordingCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.active: dict[int, ActiveRecording] = {}
        RECORDING_ROOT.mkdir(parents=True, exist_ok=True)

    def is_recording(self, guild_id: int) -> bool:
        return guild_id in self.active

    @app_commands.command(name="record_start", description="음성 채널의 대화를 유저별 학습용 WAV로 녹음합니다.")
    async def record_start(self, interaction: discord.Interaction):
        if not interaction.guild:
            await interaction.response.send_message("서버에서만 사용할 수 있습니다.", ephemeral=True)
            return
        if not interaction.user.voice or not interaction.user.voice.channel:
            await interaction.response.send_message("먼저 음성 채널에 들어가 주세요.", ephemeral=True)
            return
        if interaction.guild_id in self.active:
            await interaction.response.send_message("이미 이 서버에서 녹음 중입니다.", ephemeral=True)
            return

        await interaction.response.defer()
        channel = interaction.user.voice.channel

        try:
            recv_client = await self._ensure_recordable_voice(interaction.guild, channel)
            sink = MultiUserWavSink()
            self.active[interaction.guild_id] = ActiveRecording(
                channel_id=channel.id,
                channel_name=channel.name,
                started_at=time.monotonic(),
                sink=sink,
            )
            recv_client.listen(sink, after=self._listen_callback(interaction.guild_id, sink))
            await interaction.followup.send(
                f"`{channel.name}` 녹음을 시작했습니다. 유저별 데이터는 `data/recordings`에 저장됩니다."
            )
        except Exception as exc:
            self.active.pop(interaction.guild_id, None)
            await interaction.followup.send(f"녹음을 시작하지 못했습니다.\n```{str(exc)[:1500]}```", ephemeral=True)

    @app_commands.command(name="record_stop", description="현재 녹음을 종료하고 저장합니다.")
    async def record_stop(self, interaction: discord.Interaction):
        if not interaction.guild or interaction.guild_id not in self.active:
            await interaction.response.send_message("현재 녹음 중이 아닙니다.", ephemeral=True)
            return

        await interaction.response.defer()
        active = self.active.pop(interaction.guild_id)
        voice_client = interaction.guild.voice_client

        if voice_client and hasattr(voice_client, "stop_listening"):
            try:
                voice_client.stop_listening()
            except Exception:
                pass

        active.sink.cleanup()
        await self._restore_voice_state_after_recording(interaction.guild)

        lines = active.sink.summary_lines()
        if not lines:
            await interaction.followup.send("녹음을 종료했습니다. 저장된 발화가 아직 없습니다.")
            return

        await interaction.followup.send("녹음을 저장했습니다.\n" + "\n".join(lines[:15]))

    @app_commands.command(name="record_status", description="현재 녹음 상태를 확인합니다.")
    async def record_status(self, interaction: discord.Interaction):
        if not interaction.guild or interaction.guild_id not in self.active:
            await interaction.response.send_message("현재 녹음 중이 아닙니다.", ephemeral=True)
            return

        active = self.active[interaction.guild_id]
        elapsed = int(time.monotonic() - active.started_at)
        minutes, seconds = divmod(elapsed, 60)
        users = len(active.sink.user_sinks)
        await interaction.response.send_message(
            f"`{active.channel_name}`에서 {minutes}분 {seconds}초째 녹음 중입니다. 감지된 유저: {users}명"
        )

    async def _ensure_recordable_voice(self, guild: discord.Guild, channel: discord.VoiceChannel):
        voice_client = guild.voice_client
        if voice_client and voice_client.is_connected():
            if voice_client.channel != channel:
                await voice_client.move_to(channel)
            if not hasattr(voice_client, "listen"):
                if voice_client.is_playing() or voice_client.is_paused():
                    raise RuntimeError("현재 음성 연결이 녹음을 지원하지 않습니다. 재생이 끝난 뒤 다시 시도해 주세요.")
                await voice_client.disconnect(force=True)
                voice_client = await channel.connect(cls=PatchedVoiceRecvClient, self_deaf=False)
        else:
            voice_client = await channel.connect(cls=PatchedVoiceRecvClient, self_deaf=False)

        await guild.change_voice_state(channel=channel, self_deaf=False)
        return voice_client

    async def _restore_voice_state_after_recording(self, guild: discord.Guild) -> None:
        voice_client = guild.voice_client
        if not voice_client:
            return

        music_cog = self.bot.get_cog("MusicCog")
        music_state = getattr(music_cog, "states", {}).get(guild.id) if music_cog else None
        music_active = bool(music_state and (music_state.is_playing or music_state.current or music_state.queue))

        if music_active:
            await guild.change_voice_state(channel=voice_client.channel, self_deaf=False)
            return

        await voice_client.disconnect()

    def _listen_callback(self, guild_id: int, sink: MultiUserWavSink):
        def after_listen(error: Exception | None) -> None:
            if error is None or guild_id not in self.active:
                return

            async def recover() -> None:
                await asyncio.sleep(0.5)
                guild = self.bot.get_guild(guild_id)
                active = self.active.get(guild_id)
                if not guild or not active or not guild.voice_client or not guild.voice_client.is_connected():
                    return

                try:
                    new_sink = MultiUserWavSink()
                    new_sink.user_sinks = sink.user_sinks
                    active.sink = new_sink
                    guild.voice_client.listen(new_sink, after=self._listen_callback(guild_id, new_sink))
                except Exception as exc:
                    print(f"Recording receive recovery failed in guild {guild_id}: {exc}")

            self.bot.loop.create_task(recover())

        return after_listen

    @commands.Cog.listener()
    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        if member.id != self.bot.user.id or before.channel is None or after.channel is not None:
            return

        guild_id = before.channel.guild.id
        if guild_id not in self.active:
            return

        async def delayed_cleanup() -> None:
            await asyncio.sleep(15)
            guild = self.bot.get_guild(guild_id)
            if guild and guild.voice_client is not None:
                return
            active = self.active.pop(guild_id, None)
            if active:
                active.sink.cleanup()

        self.bot.loop.create_task(delayed_cleanup())


async def setup(bot: commands.Bot):
    await bot.add_cog(VoiceRecordingCog(bot))
