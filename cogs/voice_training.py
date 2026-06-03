from __future__ import annotations

import json
import re
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from services.ai_audio import OptionalFeatureMissing


RECORDING_ROOT = Path("data") / "recordings"


def _safe_model_name(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", value).strip(" ._")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned or "voice")[:80]


def _default_model_name(dataset_name: str) -> str:
    match = re.match(r"(.+)_\d{8,}$", dataset_name)
    if match:
        return _safe_model_name(match.group(1))
    return _safe_model_name(dataset_name)


def _dataset_label(path: Path) -> str:
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            display_name = metadata.get("display_name")
            user_id = metadata.get("user_id")
            if display_name and user_id:
                return f"{display_name} ({user_id})"
        except (OSError, json.JSONDecodeError):
            pass
    return path.name


def _dataset_minutes(path: Path) -> float:
    total_bytes = 0
    for wav_path in path.glob("*.wav"):
        try:
            total_bytes += max(0, wav_path.stat().st_size - 44)
        except OSError:
            continue
    return total_bytes / (48_000 * 2 * 2) / 60


class VoiceTrainingCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.training_locks: set[str] = set()

    @app_commands.command(name="train_voice", description="녹음된 목소리 데이터로 AI 커버용 RVC 모델을 학습합니다.")
    @app_commands.describe(
        dataset="record_start로 모은 유저별 녹음 데이터",
        model_name="완성 후 /play target_voice에 표시될 이름. 비우면 유저 이름을 사용합니다.",
    )
    async def train_voice(
        self,
        interaction: discord.Interaction,
        dataset: str,
        model_name: str | None = None,
    ):
        await interaction.response.defer()

        dataset_dir = (RECORDING_ROOT / dataset).resolve()
        recording_root = RECORDING_ROOT.resolve()
        try:
            dataset_dir.relative_to(recording_root)
        except ValueError:
            await interaction.followup.send("잘못된 데이터셋 경로입니다.", ephemeral=True)
            return

        if not dataset_dir.exists() or not any(dataset_dir.glob("*.wav")):
            await interaction.followup.send("학습할 WAV 녹음 파일이 없습니다. 먼저 `/record_start`로 녹음해 주세요.", ephemeral=True)
            return

        final_model_name = _safe_model_name(model_name or _default_model_name(dataset))
        if final_model_name in self.training_locks:
            await interaction.followup.send(f"`{final_model_name}` 모델은 이미 학습 중입니다.", ephemeral=True)
            return

        minutes = _dataset_minutes(dataset_dir)
        embed = discord.Embed(
            title=f"AI 목소리 학습 시작: {final_model_name}",
            description="학습을 준비하고 있습니다.",
            color=discord.Color.gold(),
        )
        embed.add_field(name="데이터셋", value=f"`{dataset}`", inline=False)
        embed.add_field(name="녹음 분량", value=f"약 {minutes:.1f}분", inline=True)
        embed.add_field(name="결과 위치", value=f"`voice_models/{final_model_name}`", inline=True)
        status = await interaction.followup.send(embed=embed)

        async def progress(text: str) -> None:
            embed.description = text
            try:
                await status.edit(embed=embed)
            except discord.HTTPException as exc:
                if exc.code != 50027:
                    print(f"Failed to update training status: {exc}")

        self.training_locks.add(final_model_name)
        try:
            result = await self.bot.ai_processor.train_voice(dataset_dir, final_model_name, progress)
        except OptionalFeatureMissing as exc:
            embed.title = "AI 목소리 학습 준비 실패"
            embed.description = str(exc)
            embed.color = discord.Color.red()
            await status.edit(embed=embed)
        except Exception as exc:
            embed.title = "AI 목소리 학습 실패"
            embed.description = f"```{str(exc)[:1500]}```"
            embed.color = discord.Color.red()
            await status.edit(embed=embed)
        else:
            embed.title = f"AI 목소리 학습 완료: {result.model_name}"
            embed.description = "`/play`의 `target_voice` 또는 `duet_voice` 자동완성에서 바로 선택할 수 있습니다."
            embed.color = discord.Color.green()
            embed.clear_fields()
            embed.add_field(name="모델", value=f"`{result.model_path}`", inline=False)
            if result.index_path:
                embed.add_field(name="인덱스", value=f"`{result.index_path}`", inline=False)
            await status.edit(embed=embed)
        finally:
            self.training_locks.discard(final_model_name)

    @train_voice.autocomplete("dataset")
    async def dataset_autocomplete(self, interaction: discord.Interaction, current: str):
        if not RECORDING_ROOT.exists():
            return []

        current_lower = current.lower()
        choices: list[app_commands.Choice[str]] = []
        for path in sorted(RECORDING_ROOT.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir() or not any(path.glob("*.wav")):
                continue
            label = _dataset_label(path)
            search_text = f"{path.name} {label}".lower()
            if current_lower not in search_text:
                continue
            minutes = _dataset_minutes(path)
            choice_name = f"{label} - {minutes:.1f}분"[:100]
            choices.append(app_commands.Choice(name=choice_name, value=path.name))
            if len(choices) >= 25:
                break
        return choices


async def setup(bot: commands.Bot):
    await bot.add_cog(VoiceTrainingCog(bot))
