from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yt_dlp

from services.ffmpeg import FFmpegResolver
from services.youtube import build_youtube_query


class OptionalFeatureMissing(RuntimeError):
    pass


ORIGINAL_SINGER_NAME = "원본 가수"
ORIGINAL_SINGER_VALUE = "__original_singer__"


def is_original_singer_voice(voice: str | None) -> bool:
    if not voice:
        return False
    normalized = voice.strip().lower()
    return normalized in {
        ORIGINAL_SINGER_VALUE,
        ORIGINAL_SINGER_NAME.lower(),
        "original",
        "original singer",
        "source",
        "source singer",
        "원곡 가수",
    }


@dataclass(frozen=True)
class ProcessedAudio:
    path: str
    title: str
    duration: int
    webpage_url: str
    thumbnail: str


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    model_path: str
    index_path: str | None


@dataclass(frozen=True)
class DuetSegment:
    start: float
    end: float
    singer: str


class RVCEngine:
    def __init__(self, voice_root: Path | str = "voice_models"):
        self.voice_root = Path(voice_root)
        self.voice_root.mkdir(parents=True, exist_ok=True)

    def available_voices(self) -> list[str]:
        voices: list[str] = []
        for child in sorted(self.voice_root.iterdir()):
            if not child.is_dir():
                continue
            if any(child.glob("*.pth")):
                voices.append(child.name)
        return voices

    async def train(
        self,
        dataset_dir: Path,
        model_name: str,
        progress,
    ) -> TrainingResult:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"`{dataset_dir}` 폴더를 찾지 못했습니다.")
        if not any(dataset_dir.glob("*.wav")):
            raise ValueError("학습할 WAV 파일이 없습니다. 먼저 `/record_start`로 목소리를 녹음해 주세요.")

        safe_model = self._safe_name(model_name)
        applio_dir, python_exe, core_py = self._find_applio()
        cpu_cores = str(min(os.cpu_count() or 4, int(os.getenv("RVC_TRAIN_CPU_CORES", "12"))))
        total_epoch = os.getenv("RVC_TRAIN_EPOCHS", "300")
        batch_size = os.getenv("RVC_TRAIN_BATCH_SIZE", "16")

        await progress(f"[1/5] 데이터 전처리 중... 대상: {dataset_dir.name}")
        await self._run_applio(
            [
                python_exe,
                core_py,
                "preprocess",
                "--model_name",
                safe_model,
                "--dataset_path",
                str(dataset_dir.resolve()),
                "--sample_rate",
                "40000",
                "--cpu_cores",
                cpu_cores,
                "--cut_preprocess",
                "Automatic",
                "--process_effects",
                "False",
                "--noise_reduction",
                "False",
                "--noise_reduction_strength",
                "0.7",
                "--chunk_len",
                "3.0",
                "--overlap_len",
                "0.3",
                "--normalization_mode",
                "none",
            ],
            applio_dir,
        )

        await progress("[2/5] 음성 특징과 피치를 추출하는 중... (rmvpe/contentvec)")
        await self._run_applio(
            [
                python_exe,
                core_py,
                "extract",
                "--model_name",
                safe_model,
                "--f0_method",
                "rmvpe",
                "--cpu_cores",
                cpu_cores,
                "--gpu",
                "0",
                "--sample_rate",
                "40000",
                "--embedder_model",
                "contentvec",
                "--include_mutes",
                "2",
            ],
            applio_dir,
        )

        await progress(f"[3/5] RVC 모델 학습 시작... ({total_epoch} epoch)")
        await self._run_training(
            [
                python_exe,
                core_py,
                "train",
                "--model_name",
                safe_model,
                "--save_every_epoch",
                "50",
                "--total_epoch",
                total_epoch,
                "--sample_rate",
                "40000",
                "--batch_size",
                batch_size,
                "--gpu",
                "0",
                "--save_only_latest",
                "True",
                "--save_every_weights",
                "False",
                "--overtraining_detector",
                "False",
                "--overtraining_threshold",
                "50",
                "--pretrained",
                "True",
                "--custom_pretrained",
                "False",
                "--cleanup",
                "False",
                "--cache_data_in_gpu",
                "False",
                "--vocoder",
                "HiFi-GAN",
                "--checkpointing",
                "False",
                "--index_algorithm",
                "Auto",
            ],
            applio_dir,
            total_epoch,
            progress,
        )

        await progress("[4/5] 검색 인덱스를 생성하는 중...")
        await self._run_applio(
            [python_exe, core_py, "index", "--model_name", safe_model, "--index_algorithm", "Auto"],
            applio_dir,
        )

        await progress("[5/5] 모델 파일을 voice_models 폴더로 정리하는 중...")
        return await asyncio.to_thread(self._collect_trained_model, safe_model, Path(applio_dir))

    async def convert(self, vocals: Path, work_dir: Path, voice: str, pitch: int) -> Path:
        model_path, index_path = self._find_model_files(voice)
        applio_dir, python_exe, core_py = self._find_applio()
        output = work_dir / f"converted_{self._safe_name(voice)}.wav"

        cmd = [
            python_exe,
            core_py,
            "infer",
            "--pitch",
            str(pitch),
            "--f0_method",
            "rmvpe",
            "--index_rate",
            "0.25",
            "--volume_envelope",
            "0.75",
            "--protect",
            "0.5",
            "--input_path",
            str(vocals.resolve()),
            "--output_path",
            str(output.resolve()),
            "--pth_path",
            str(model_path.resolve()),
            "--index_path",
            str(index_path.resolve()) if index_path else "",
            "--export_format",
            "WAV",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=applio_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0 or not output.exists():
            message = (stderr or stdout).decode("utf-8", errors="ignore")
            raise RuntimeError(f"AI 커버 변환이 실패했습니다.\n{message[-1200:]}")
        return output

    def _find_model_files(self, voice: str) -> tuple[Path, Path | None]:
        voice_dir = self.voice_root / voice
        if not voice_dir.exists():
            raise OptionalFeatureMissing(
                f"`voice_models/{voice}` 폴더를 찾지 못했습니다. "
                "목소리 모델 폴더 안에 .pth 파일을 넣어 주세요."
            )

        pth_files = sorted(voice_dir.glob("*.pth"))
        if not pth_files:
            raise OptionalFeatureMissing(f"`voice_models/{voice}` 폴더에 .pth 모델 파일이 없습니다.")

        index_files = sorted(voice_dir.glob("*.index"))
        return pth_files[0], index_files[0] if index_files else None

    def _find_applio(self) -> tuple[str, str, str]:
        candidates: list[Path] = [
            Path("vendor") / "Applio",
            Path("vendor") / "ApplioV3.6.2",
            Path("..") / "Applio",
            Path("..") / "ApplioV3.6.2",
        ]

        env_dir = os.getenv("APPLIO_DIR")
        if env_dir:
            candidates.insert(0, Path(env_dir))

        for candidate in candidates:
            applio_dir = candidate.resolve()
            core_py = applio_dir / "core.py"
            python_candidates = [
                applio_dir / "env" / "Scripts" / "python.exe",
                applio_dir / "env" / "bin" / "python",
                applio_dir / ".venv" / "Scripts" / "python.exe",
                applio_dir / ".venv" / "bin" / "python",
            ]
            for python_exe in python_candidates:
                if core_py.exists() and python_exe.exists():
                    return str(applio_dir), str(python_exe), str(core_py)

        raise OptionalFeatureMissing(
            "AI 커버 엔진을 찾지 못했습니다. `scripts/install_ai.ps1` 또는 "
            "`scripts/install_ai.sh`를 실행해서 Applio 엔진을 설치해 주세요."
        )

    async def _run_applio(self, cmd: list[str], cwd: str) -> None:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = (stderr or stdout).decode("utf-8", errors="ignore")
            raise RuntimeError(f"Applio 작업이 실패했습니다.\n{message[-1500:]}")

    async def _run_training(self, cmd: list[str], cwd: str, total_epoch: str, progress) -> None:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        last_epoch = None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="ignore").strip()
            if "epoch=" not in text:
                continue

            match = re.search(r"epoch=(\d+)", text)
            if match and match.group(1) != last_epoch:
                last_epoch = match.group(1)
                await progress(f"[3/5] RVC 모델 학습 중... ({last_epoch}/{total_epoch} epoch)")

        await process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Applio 학습이 실패했습니다. return code: {process.returncode}")

    def _collect_trained_model(self, model_name: str, applio_dir: Path) -> TrainingResult:
        target_dir = self.voice_root / model_name
        target_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = applio_dir / "logs" / model_name
        pth_patterns = [
            logs_dir / f"{model_name}*.pth",
            applio_dir / "logs" / f"{model_name}*.pth",
            applio_dir / "assets" / "weights" / f"{model_name}*.pth",
        ]

        found_pth: Path | None = None
        for pattern in pth_patterns:
            matches = sorted(pattern.parent.glob(pattern.name), key=lambda item: item.stat().st_mtime, reverse=True)
            if matches:
                found_pth = matches[0]
                break

        if not found_pth:
            raise RuntimeError(f"학습은 끝났지만 `{model_name}` .pth 파일을 찾지 못했습니다.")

        found_index: Path | None = None
        if logs_dir.exists():
            index_files = sorted(logs_dir.glob("*.index"), key=lambda item: item.stat().st_mtime, reverse=True)
            if index_files:
                found_index = index_files[0]

        model_output = target_dir / f"{model_name}.pth"
        shutil.copy2(found_pth, model_output)

        index_output = None
        if found_index:
            index_output = target_dir / f"{model_name}.index"
            shutil.copy2(found_index, index_output)

        return TrainingResult(
            model_name=model_name,
            model_path=str(model_output),
            index_path=str(index_output) if index_output else None,
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        keep = [ch if ch.isalnum() or ch in "._- " else "_" for ch in value]
        safe = "".join(keep).strip(" ._")
        return (safe or "voice")[:80]


class AIProcessor:
    def __init__(self, ffmpeg: FFmpegResolver):
        self.ffmpeg = ffmpeg
        self.rvc = RVCEngine()
        self.output_dir = Path("data") / "processed"
        self.temp_dir = Path("tmp")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def available_voices(self) -> list[str]:
        return self.rvc.available_voices()

    async def train_voice(self, dataset_dir: Path, model_name: str, progress) -> TrainingResult:
        return await self.rvc.train(dataset_dir, model_name, progress)

    async def process_youtube(
        self,
        url: str,
        mode: str,
        target_voice: str | None,
        pitch_shift: int,
        progress,
        duet_voice: str | None = None,
        duet_parts: str | None = None,
    ) -> ProcessedAudio:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="ai_music_", dir=str(self.temp_dir)))
        try:
            await progress("유튜브 오디오를 다운로드하는 중...")
            input_wav, title, duration, webpage_url, thumbnail = await self._download_audio(url, work_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = self._safe_name(title)

            if mode == "original":
                output = self.output_dir / f"PITCH_{pitch_shift:+d}_{safe_title}_{timestamp}.mp3"
                await progress("피치를 조절하는 중...")
                await self._apply_pitch_shift(input_wav, output, pitch_shift)
                return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

            await progress("보컬과 반주를 분리하는 중... 처음 실행은 모델 다운로드 때문에 오래 걸릴 수 있습니다.")
            vocals, instrumental = await self._separate_with_demucs(input_wav, work_dir)

            if mode == "instrumental":
                output = self.output_dir / f"MR_{safe_title}_{timestamp}.wav"
                await self._copy_or_pitch_shift(instrumental, output, pitch_shift, progress)
                return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

            if mode == "vocal":
                output = self.output_dir / f"VOCAL_{safe_title}_{timestamp}.wav"
                await self._copy_or_pitch_shift(vocals, output, pitch_shift, progress)
                return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

            if mode == "ai_duet":
                if not target_voice:
                    raise ValueError("AI 듀엣 모드는 target_voice 값이 필요합니다.")
                if not duet_voice:
                    raise ValueError("AI 듀엣 모드는 duet_voice 값이 필요합니다.")

                first_is_original = is_original_singer_voice(target_voice)
                second_is_original = is_original_singer_voice(duet_voice)
                if first_is_original and second_is_original:
                    raise ValueError("AI 듀엣 모드는 원본 가수만 두 번 선택할 수 없습니다.")
                if not first_is_original and not second_is_original and target_voice == duet_voice:
                    raise ValueError("AI 듀엣 모드는 서로 다른 두 목소리를 선택해 주세요.")

                segments = self._parse_duet_parts(duet_parts, duration)
                if first_is_original:
                    first_vocal = vocals
                    first_name = ORIGINAL_SINGER_NAME
                else:
                    await progress(f"1번 파트를 {target_voice} 목소리로 변환하는 중...")
                    first_vocal = await self.rvc.convert(vocals, work_dir, target_voice, pitch_shift)
                    first_name = target_voice

                if second_is_original:
                    second_vocal = vocals
                    second_name = ORIGINAL_SINGER_NAME
                else:
                    await progress(f"2번 파트를 {duet_voice} 목소리로 변환하는 중...")
                    second_vocal = await self.rvc.convert(vocals, work_dir, duet_voice, pitch_shift)
                    second_name = duet_voice

                await progress("듀엣 파트를 합성하는 중...")

                output = self.output_dir / (
                    f"AI_DUET_{self._safe_name(first_name)}_{self._safe_name(second_name)}_"
                    f"{safe_title}_{timestamp}.mp3"
                )
                await self._merge_duet(first_vocal, second_vocal, instrumental, output, segments)
                return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

            converted_vocal = vocals
            prefix = "VOCAL_BOOST"
            if mode == "ai_cover":
                if not target_voice or is_original_singer_voice(target_voice):
                    raise ValueError("AI 커버 모드는 target_voice 값이 필요합니다.")
                await progress(f"{target_voice} 목소리로 변환하는 중...")
                converted_vocal = await self.rvc.convert(vocals, work_dir, target_voice, pitch_shift)
                prefix = f"AI_COVER_{self._safe_name(target_voice)}"

            output = self.output_dir / f"{prefix}_{safe_title}_{timestamp}.mp3"
            merge_output = work_dir / "merged.mp3" if mode != "ai_cover" and pitch_shift else output
            await progress("최종 오디오를 합성하는 중...")
            await self._merge(converted_vocal, instrumental, merge_output, vocal_boost=True)
            if merge_output != output:
                await progress("피치를 조절하는 중...")
                await self._apply_pitch_shift(merge_output, output, pitch_shift)
            return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _download_audio(self, url: str, work_dir: Path) -> tuple[Path, str, int, str, str]:
        output_template = str(work_dir / "input.%(ext)s")

        def download() -> tuple[Path, str, int, str, str]:
            opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "default_search": "ytsearch1",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }],
                "ffmpeg_location": os.path.dirname(self.ffmpeg.executable()),
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(build_youtube_query(url), download=True)
            if "entries" in info:
                info = next((entry for entry in info["entries"] if entry), None)
                if not info:
                    raise ValueError("검색 결과를 찾지 못했습니다.")

            webpage_url = info.get("webpage_url") or info.get("original_url") or ""
            if info.get("id") and "youtube" not in webpage_url and "youtu.be" not in webpage_url:
                webpage_url = f"https://www.youtube.com/watch?v={info['id']}"

            return (
                work_dir / "input.wav",
                info.get("title", "youtube_audio"),
                info.get("duration") or 0,
                webpage_url,
                info.get("thumbnail") or "",
            )

        return await asyncio.to_thread(download)

    def _parse_duet_parts(self, duet_parts: str | None, duration: int) -> list[DuetSegment]:
        if not duet_parts or not duet_parts.strip():
            raise ValueError(
                "AI 듀엣 모드는 duet_parts 값이 필요합니다. "
                "예: `0:00-0:35:1,0:35-1:10:2,1:10-1:25:both`"
            )

        segments: list[DuetSegment] = []
        for raw_part in re.split(r"[,;\n]+", duet_parts):
            part = raw_part.strip()
            if not part:
                continue

            pieces = part.rsplit(":", 1)
            if len(pieces) != 2:
                raise ValueError(f"듀엣 파트 형식이 올바르지 않습니다: `{part}`")

            time_range, singer_raw = pieces
            if "-" not in time_range:
                raise ValueError(f"듀엣 파트 시간 범위가 올바르지 않습니다: `{part}`")

            start_raw, end_raw = [value.strip() for value in time_range.split("-", 1)]
            start = self._parse_timestamp(start_raw)
            end = self._parse_timestamp(end_raw)
            singer = self._normalize_duet_singer(singer_raw.strip())

            if end <= start:
                raise ValueError(f"듀엣 파트 종료 시간이 시작 시간보다 커야 합니다: `{part}`")
            if duration and start >= duration:
                raise ValueError(f"듀엣 파트 시작 시간이 곡 길이를 넘었습니다: `{part}`")

            segments.append(DuetSegment(start=start, end=min(end, duration) if duration else end, singer=singer))

        if not segments:
            raise ValueError("사용할 수 있는 듀엣 파트를 찾지 못했습니다.")
        return segments

    @staticmethod
    def _parse_timestamp(value: str) -> float:
        if re.fullmatch(r"\d+(?:\.\d+)?", value):
            return float(value)

        parts = value.split(":")
        if not 2 <= len(parts) <= 3:
            raise ValueError(f"시간 형식이 올바르지 않습니다: `{value}`")

        try:
            seconds = float(parts[-1])
            minutes = int(parts[-2])
            hours = int(parts[-3]) if len(parts) == 3 else 0
        except ValueError as exc:
            raise ValueError(f"시간 형식이 올바르지 않습니다: `{value}`") from exc

        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def _normalize_duet_singer(value: str) -> str:
        normalized = value.lower().strip()
        if normalized in {"1", "1번", "a", "first", "target", "target_voice", "첫번째", "첫 번째"}:
            return "1"
        if normalized in {"2", "2번", "b", "second", "duet", "duet_voice", "두번째", "두 번째"}:
            return "2"
        if normalized in {"both", "all", "together", "1+2", "2+1", "둘다", "둘 다", "같이"}:
            return "both"
        raise ValueError(f"듀엣 파트 가수 값은 1, 2, both 중 하나여야 합니다: `{value}`")

    async def _separate_with_demucs(self, input_wav: Path, work_dir: Path) -> tuple[Path, Path]:
        output_root = work_dir / "demucs"
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems=vocals",
            "-d",
            "cuda",
            "--out",
            str(output_root),
            str(input_wav),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = (stderr or stdout).decode("utf-8", errors="ignore")
            if "TorchCodec" in message or "torchcodec" in message:
                raise OptionalFeatureMissing(
                    "보컬 분리 결과를 저장하는 데 필요한 TorchCodec이 없거나 PyTorch 버전과 맞지 않습니다. "
                    "`scripts/install_ai.ps1` 또는 `scripts/install_ai.sh`를 다시 실행해 주세요.\n"
                    f"{message[-700:]}"
                )
            if "Couldn't find appropriate backend" in message or "audio backend" in message:
                raise OptionalFeatureMissing(
                    "보컬 분리 결과를 저장할 오디오 백엔드를 찾지 못했습니다. "
                    "`scripts/install_ai.ps1` 또는 `scripts/install_ai.sh`를 다시 실행해 주세요.\n"
                    f"{message[-700:]}"
                )
            raise OptionalFeatureMissing(
                "CUDA 보컬 분리 엔진을 실행하지 못했습니다. NVIDIA 드라이버와 CUDA PyTorch 설치를 확인한 뒤 "
                "`scripts/install_ai.ps1` 또는 `scripts/install_ai.sh`를 다시 실행해 주세요.\n"
                f"{message[-700:]}"
            )

        matches = list(output_root.glob("*/input/vocals.wav"))
        if not matches:
            matches = list(output_root.glob("**/vocals.wav"))
        if not matches:
            raise RuntimeError("Demucs 결과에서 vocals.wav를 찾지 못했습니다.")

        vocals = matches[0]
        instrumental = vocals.with_name("no_vocals.wav")
        if not instrumental.exists():
            raise RuntimeError("Demucs 결과에서 no_vocals.wav를 찾지 못했습니다.")
        return vocals, instrumental

    async def _merge(self, vocals: Path, instrumental: Path, output: Path, vocal_boost: bool) -> None:
        vocal_volume = "1.35" if vocal_boost else "1.0"
        inst_volume = "0.55" if vocal_boost else "0.85"
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(vocals),
            "-i",
            str(instrumental),
            "-filter_complex",
            f"[0:a]volume={vocal_volume}[v];[1:a]volume={inst_volume}[i];[v][i]amix=inputs=2:duration=longest",
            "-b:a",
            "192k",
            str(output),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore")[-1000:])

    async def _copy_or_pitch_shift(self, source: Path, output: Path, pitch_shift: int, progress) -> None:
        if pitch_shift == 0:
            shutil.copy2(source, output)
            return

        await progress("피치를 조절하는 중...")
        await self._apply_pitch_shift(source, output, pitch_shift)

    async def _apply_pitch_shift(self, source: Path, output: Path, semitones: int) -> None:
        factor = 2 ** (semitones / 12)
        tempo = 1 / factor
        filter_complex = (
            "aresample=48000,"
            f"asetrate=48000*{factor:.8f},"
            "aresample=48000,"
            f"atempo={tempo:.8f}"
        )
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(source),
            "-af",
            filter_complex,
            "-b:a",
            "192k",
            str(output),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore")[-1000:])

    async def _merge_duet(
        self,
        first_vocal: Path,
        second_vocal: Path,
        instrumental: Path,
        output: Path,
        segments: list[DuetSegment],
    ) -> None:
        first_mask = self._duet_mask_expression(segments, {"1", "both"})
        second_mask = self._duet_mask_expression(segments, {"2", "both"})
        filter_complex = (
            f"[0:a]volume='{first_mask}':eval=frame[v1];"
            f"[1:a]volume='{second_mask}':eval=frame[v2];"
            "[2:a]volume=0.55[i];"
            "[v1][v2]amix=inputs=2:duration=longest:normalize=0[dv];"
            "[dv]volume=1.15[dv2];"
            "[dv2][i]amix=inputs=2:duration=longest:normalize=0"
        )
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(first_vocal),
            "-i",
            str(second_vocal),
            "-i",
            str(instrumental),
            "-filter_complex",
            filter_complex,
            "-b:a",
            "192k",
            str(output),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore")[-1000:])

    @staticmethod
    def _duet_mask_expression(segments: list[DuetSegment], singers: set[str]) -> str:
        ranges = [
            f"between(t,{segment.start:.3f},{segment.end:.3f})"
            for segment in segments
            if segment.singer in singers
        ]
        if not ranges:
            return "0"
        return f"if({'+'.join(ranges)},0.95,0)"

    @staticmethod
    def _safe_name(value: str) -> str:
        keep = [ch if ch.isalnum() or ch in "._- " else "_" for ch in value]
        safe = "".join(keep).strip(" ._")
        return (safe or "audio")[:80]
