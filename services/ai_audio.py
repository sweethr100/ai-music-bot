from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yt_dlp

from services.ffmpeg import FFmpegResolver
from services.youtube import build_youtube_query


class OptionalFeatureMissing(RuntimeError):
    pass


DEFAULT_SEPARATOR_MODEL = "mel_band_roformer_kim_ft_unwa.ckpt"
DEFAULT_MULTI_SINGER_SEPARATOR_MODEL = "Cyru5/MedleyVox"
DEFAULT_MULTI_SINGER_SEPARATOR_FILE = "vocals 100.pth"
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
class DuetSingerStems:
    first: Path
    second: Path


@dataclass(frozen=True)
class DiarizedSegment:
    start: float
    end: float
    speaker: str


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
        self._ensure_predictor_models(Path(applio_dir))
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
        self._ensure_predictor_models(Path(applio_dir))
        output = work_dir / f"converted_{self._safe_name(voice)}.wav"

        cmd = [
            python_exe,
            core_py,
            "infer",
            "--pitch",
            str(pitch),
            "--f0_method",
            os.getenv("RVC_INFER_F0_METHOD", "rmvpe"),
            "--index_rate",
            os.getenv("RVC_INFER_INDEX_RATE", "0.25"),
            "--volume_envelope",
            os.getenv("RVC_INFER_VOLUME_ENVELOPE", "0.75"),
            "--protect",
            os.getenv("RVC_INFER_PROTECT", "0.50"),
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
            "AI 커버 엔진을 찾지 못했습니다. `scripts/install.ps1` 또는 "
            "`scripts/install.sh`를 실행해서 Applio 엔진을 설치해 주세요."
        )

    @staticmethod
    def _ensure_predictor_models(applio_dir: Path) -> None:
        missing = [
            name
            for name in ("rmvpe.pt", "fcpe.pt")
            if not (applio_dir / "rvc" / "models" / "predictors" / name).exists()
        ]
        if missing:
            raise OptionalFeatureMissing(
                "AI 커버에 필요한 Applio 피치 추출 모델이 없습니다: "
                f"`{', '.join(missing)}`. `scripts/install.ps1` 또는 `scripts/install.sh`를 다시 실행해 주세요."
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

    async def export_youtube_audio(self, url: str, progress) -> ProcessedAudio:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="ai_music_", dir=str(self.temp_dir)))
        try:
            await progress("유튜브 오디오를 다운로드하는 중...")
            input_wav, title, duration, webpage_url, thumbnail = await self._download_audio(url, work_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = self._safe_name(title)
            output = self.output_dir / f"ORIGINAL_{safe_title}_{timestamp}.mp3"
            await progress("노래 파일을 준비하는 중...")
            await self._export_audio(input_wav, output)
            return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def normalize_for_playback(self, source: Path, title: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self._safe_name(title)
        output = self.output_dir / f"NORMALIZED_{safe_title}_{timestamp}.mp3"
        measured = await self._measure_loudnorm(source)
        await self._apply_loudnorm(source, output, measured)
        return output

    async def process_youtube(
        self,
        url: str,
        mode: str,
        target_voice: str | None,
        vocal_pitch_shift: int,
        progress,
        duet_voice: str | None = None,
    ) -> ProcessedAudio:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="ai_music_", dir=str(self.temp_dir)))
        try:
            await progress("유튜브 오디오를 다운로드하는 중...")
            input_wav, title, duration, webpage_url, thumbnail = await self._download_audio(url, work_dir)
            return await self._process_prepared_audio(
                input_wav=input_wav,
                title=title,
                duration=duration,
                webpage_url=webpage_url,
                thumbnail=thumbnail,
                mode=mode,
                target_voice=target_voice,
                vocal_pitch_shift=vocal_pitch_shift,
                progress=progress,
                work_dir=work_dir,
                duet_voice=duet_voice,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def process_file(
        self,
        source: Path,
        title: str,
        webpage_url: str,
        mode: str,
        target_voice: str | None,
        vocal_pitch_shift: int,
        progress,
        duet_voice: str | None = None,
    ) -> ProcessedAudio:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="ai_music_", dir=str(self.temp_dir)))
        try:
            await progress("업로드한 오디오 파일을 읽는 중...")
            input_wav = await self._prepare_uploaded_audio(source, work_dir)
            duration = await self._probe_duration(source)
            return await self._process_prepared_audio(
                input_wav=input_wav,
                title=title,
                duration=duration,
                webpage_url=webpage_url,
                thumbnail="",
                mode=mode,
                target_voice=target_voice,
                vocal_pitch_shift=vocal_pitch_shift,
                progress=progress,
                work_dir=work_dir,
                duet_voice=duet_voice,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def separate_youtube_singers(self, url: str, progress) -> list[ProcessedAudio]:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="ai_music_singers_", dir=str(self.temp_dir)))
        try:
            await progress("유튜브 오디오를 다운로드하는 중...")
            input_wav, title, duration, webpage_url, thumbnail = await self._download_audio(url, work_dir)

            await progress("보컬과 반주를 분리하는 중...")
            vocals, _ = await self._separate_vocals(input_wav, work_dir, progress)
            vocals = await self._prepare_vocals_for_conversion(vocals, work_dir, progress)

            await progress("가수별 보컬 stem을 분리하는 중...")
            stems = await self._separate_duet_singers(vocals, work_dir, progress)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = self._safe_name(title)
            outputs: list[ProcessedAudio] = []
            for index, stem in enumerate((stems.first, stems.second), start=1):
                output = self.output_dir / f"SINGER_{index}_{safe_title}_{timestamp}.mp3"
                await progress(f"{index}번 가수 stem을 MP3로 내보내는 중...")
                await self._export_audio(stem, output)
                outputs.append(
                    ProcessedAudio(
                        path=str(output),
                        title=f"{title} - singer{index}",
                        duration=duration,
                        webpage_url=webpage_url,
                        thumbnail=thumbnail,
                    )
                )
            return outputs
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _process_prepared_audio(
        self,
        input_wav: Path,
        title: str,
        duration: int,
        webpage_url: str,
        thumbnail: str,
        mode: str,
        target_voice: str | None,
        vocal_pitch_shift: int,
        progress,
        work_dir: Path,
        duet_voice: str | None = None,
    ) -> ProcessedAudio:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self._safe_name(title)

        if mode == "original" and vocal_pitch_shift == 0:
            output = self.output_dir / f"ORIGINAL_{safe_title}_{timestamp}.mp3"
            await progress("노래 파일을 준비하는 중...")
            await self._export_audio(input_wav, output)
            return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

        vocals, instrumental = await self._separate_vocals(input_wav, work_dir, progress)

        if mode == "original":
            vocals = await self._prepare_vocals_for_conversion(vocals, work_dir, progress)
            shifted_vocals = await self._vocal_with_pitch_shift(vocals, work_dir, vocal_pitch_shift, progress)
            output = self.output_dir / f"VOCAL_PITCH_{vocal_pitch_shift:+d}_{safe_title}_{timestamp}.mp3"
            await progress("목소리 피치를 바꾼 보컬과 반주를 합성하는 중...")
            await self._merge(shifted_vocals, instrumental, output, vocal_boost=False)
            return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

        if mode == "instrumental":
            output = self.output_dir / f"MR_{safe_title}_{timestamp}.wav"
            if vocal_pitch_shift:
                await progress("MR만 재생은 보컬이 없어 목소리 피치 옵션을 건너뜁니다.")
            shutil.copy2(instrumental, output)
            return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

        vocals = await self._prepare_vocals_for_conversion(vocals, work_dir, progress)

        if mode == "vocal":
            output = self.output_dir / f"VOCAL_{safe_title}_{timestamp}.wav"
            await self._copy_or_vocal_pitch_shift(vocals, output, vocal_pitch_shift, progress)
            return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

        if mode == "ai_duet":
            if not duet_voice:
                raise ValueError("AI 듀엣 모드는 duet_voice 값이 필요합니다.")
            target_voice = target_voice or ORIGINAL_SINGER_VALUE

            first_is_original = is_original_singer_voice(target_voice)
            second_is_original = is_original_singer_voice(duet_voice)
            if first_is_original and second_is_original:
                raise ValueError("AI 듀엣 모드는 원본 가수만 두 번 선택할 수 없습니다.")
            if not first_is_original and not second_is_original and target_voice == duet_voice:
                raise ValueError("AI 듀엣 모드는 서로 다른 두 목소리를 선택해 주세요.")

            duet_stems = await self._separate_duet_singers(vocals, work_dir, progress)
            if first_is_original:
                first_vocal = await self._vocal_with_pitch_shift(
                    duet_stems.first,
                    work_dir,
                    vocal_pitch_shift,
                    progress,
                )
                first_name = ORIGINAL_SINGER_NAME
            else:
                await progress(f"1번 파트를 {target_voice} 목소리로 변환하는 중...")
                first_vocal = await self.rvc.convert(duet_stems.first, work_dir, target_voice, vocal_pitch_shift)
                first_vocal = await self._polish_converted_vocal(first_vocal, work_dir)
                first_name = target_voice

            if second_is_original:
                second_vocal = await self._vocal_with_pitch_shift(
                    duet_stems.second,
                    work_dir,
                    vocal_pitch_shift,
                    progress,
                )
                second_name = ORIGINAL_SINGER_NAME
            else:
                await progress(f"2번 파트를 {duet_voice} 목소리로 변환하는 중...")
                second_vocal = await self.rvc.convert(duet_stems.second, work_dir, duet_voice, vocal_pitch_shift)
                second_vocal = await self._polish_converted_vocal(second_vocal, work_dir)
                second_name = duet_voice

            await progress("듀엣 파트를 합성하는 중...")

            output = self.output_dir / (
                f"AI_DUET_{self._safe_name(first_name)}_{self._safe_name(second_name)}_"
                f"{safe_title}_{timestamp}.mp3"
            )
            await self._merge_duet_sources(first_vocal, second_vocal, instrumental, output)
            return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

        converted_vocal = vocals
        prefix = "VOCAL_BOOST"
        if mode == "ai_cover":
            if not target_voice or is_original_singer_voice(target_voice):
                raise ValueError("AI 커버 모드는 target_voice 값이 필요합니다.")
            await progress(f"{target_voice} 목소리로 변환하는 중...")
            converted_vocal = await self.rvc.convert(vocals, work_dir, target_voice, vocal_pitch_shift)
            converted_vocal = await self._polish_converted_vocal(converted_vocal, work_dir)
            prefix = f"AI_COVER_{self._safe_name(target_voice)}"
        else:
            converted_vocal = await self._vocal_with_pitch_shift(vocals, work_dir, vocal_pitch_shift, progress)

        output = self.output_dir / f"{prefix}_{safe_title}_{timestamp}.mp3"
        await progress("최종 오디오를 합성하는 중...")
        await self._merge(
            converted_vocal,
            instrumental,
            output,
            vocal_boost=True,
            profile="cover" if mode == "ai_cover" else "boost",
        )
        return ProcessedAudio(str(output), title, duration, webpage_url, thumbnail)

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

    async def _prepare_uploaded_audio(self, source: Path, work_dir: Path) -> Path:
        output = work_dir / "input.wav"
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                "업로드한 파일에서 오디오를 읽지 못했습니다.\n"
                f"{stderr.decode('utf-8', errors='ignore')[-1000:]}"
            )
        return output

    async def _probe_duration(self, source: Path) -> int:
        cmd = [
            self.ffmpeg.executable(),
            "-hide_banner",
            "-i",
            str(source),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        message = (stderr or stdout).decode("utf-8", errors="ignore")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", message)
        if not match:
            return 0
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return int(hours * 3600 + minutes * 60 + seconds)

    async def _separate_duet_singers(self, vocals: Path, work_dir: Path, progress) -> DuetSingerStems:
        backend = os.getenv("MULTI_SINGER_SEPARATOR_BACKEND", "pyannote").strip().lower() or "pyannote"
        configured_model = os.getenv("MULTI_SINGER_SEPARATOR_MODEL", "").strip()
        model_name = configured_model or DEFAULT_MULTI_SINGER_SEPARATOR_MODEL
        command_template = os.getenv("MULTI_SINGER_SEPARATOR_COMMAND", "").strip()
        attempted_dedicated_separator = False
        last_separator_error: Exception | None = None

        if backend in {"local", "local_diarization", "local-diarization", "diarization", "no-login", "nologin"}:
            await progress("로그인 없는 로컬 보컬 시간표 분리기를 실행하는 중...")
            try:
                return await self._separate_duet_with_local_diarization(vocals, work_dir, progress)
            except Exception as exc:
                if self._env_enabled("MULTI_SINGER_SEPARATOR_REQUIRE_MODEL"):
                    raise RuntimeError(f"로컬 듀엣 파트 분석이 실패했습니다.\n{str(exc)[-1200:]}") from exc
                raise RuntimeError(f"로컬 듀엣 파트 분석이 실패했습니다.\n{str(exc)[-1200:]}") from exc

        if backend in {"pyannote", "speaker-diarization", "speaker_diarization"}:
            attempted_dedicated_separator = True
            await progress("PyAnnote Audio로 보컬 파트 시간표를 분석하는 중...")
            try:
                return await self._separate_duet_with_pyannote_diarization(vocals, work_dir, progress)
            except Exception as exc:
                raise RuntimeError(f"PyAnnote Audio 듀엣 파트 분석이 실패했습니다.\n{str(exc)[-1200:]}") from exc

        if backend in {"asteroid", "medleyvox", "huggingface", "hf"} or (
            backend == "auto" and configured_model and self._looks_like_huggingface_model(model_name)
        ):
            attempted_dedicated_separator = True
            await progress(f"가수 개별 보컬 분리 모델을 자동 실행하는 중... ({model_name})")
            try:
                stems = await self._separate_duet_with_asteroid(vocals, work_dir, model_name)
                if self._duet_stems_are_low_quality(stems):
                    if backend != "auto":
                        raise RuntimeError("가수 분리 모델 결과가 한쪽으로 몰렸거나 두 stem이 너무 비슷합니다.")
                    await progress("가수 분리 모델 결과가 충분히 갈라지지 않아 로컬 시간표 분리기로 다시 시도하는 중...")
                    return await self._separate_duet_with_local_diarization(vocals, work_dir, progress)
                return stems
            except Exception as exc:
                last_separator_error = exc
                if backend != "auto":
                    raise RuntimeError(f"가수 개별 보컬 분리가 실패했습니다.\n{str(exc)[-1200:]}") from exc
                print(f"Asteroid multi-singer separator failed, trying other backends: {exc}")

        if backend in {"auto", "audio-separator", "audio_separator"} and configured_model:
            attempted_dedicated_separator = True
            await progress(f"가수 개별 보컬 분리 모델을 실행하는 중... ({model_name})")
            try:
                return await self._separate_duet_with_audio_separator(vocals, work_dir, model_name)
            except Exception as exc:
                last_separator_error = exc
                if backend != "auto":
                    raise RuntimeError(f"가수 개별 보컬 분리가 실패했습니다.\n{str(exc)[-1200:]}") from exc
                print(f"Multi-singer audio-separator failed, trying command backend: {exc}")
        elif backend in {"audio-separator", "audio_separator"}:
            raise OptionalFeatureMissing(
                "audio-separator 방식의 AI 듀엣 가수 분리 모델명이 없습니다. "
                "`MULTI_SINGER_SEPARATOR_MODEL`에 audio-separator 호환 모델명을 넣어 주세요."
            )

        if backend in {"auto", "command", "external"} and command_template:
            attempted_dedicated_separator = True
            await progress("외부 가수 개별 보컬 분리 모델을 실행하는 중...")
            try:
                return await self._separate_duet_with_command(vocals, work_dir, command_template)
            except Exception as exc:
                last_separator_error = exc
                if backend != "auto":
                    raise RuntimeError(f"외부 가수 개별 보컬 분리가 실패했습니다.\n{str(exc)[-1200:]}") from exc
                print(f"Multi-singer command backend failed: {exc}")

        if self._env_enabled("MULTI_SINGER_SEPARATOR_REQUIRE_MODEL"):
            raise OptionalFeatureMissing(
                "AI 듀엣 자동 가수 분리 모델이 설정되어 있지 않습니다. "
                "`MULTI_SINGER_SEPARATOR_MODEL`에 audio-separator 호환 multi-singer 모델명을 넣거나, "
                "`MULTI_SINGER_SEPARATOR_COMMAND`로 외부 분리 명령을 지정해 주세요."
            )

        if attempted_dedicated_separator and last_separator_error:
            await progress(
                "전용 가수 분리 모델 실행이 실패해 로컬 시간표 분리기로 전환하는 중...\n"
                f"{str(last_separator_error)[-500:]}"
            )
        else:
            await progress("전용 가수 분리 모델 설정이 없어 로컬 시간표 분리기를 실행하는 중...")
        return await self._separate_duet_with_local_diarization(vocals, work_dir, progress)

    async def _separate_duet_with_local_diarization(
        self,
        vocals: Path,
        work_dir: Path,
        progress,
    ) -> DuetSingerStems:
        output_dir = work_dir / "multi_singer_local_diarization"
        output_dir.mkdir(parents=True, exist_ok=True)

        def diarize_and_cut() -> DuetSingerStems:
            try:
                import librosa
                import numpy as np
                import soundfile as sf
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
            except ImportError as exc:
                raise OptionalFeatureMissing(
                    "로컬 듀엣 시간표 분리에 필요한 `librosa`와 `scikit-learn`이 설치되어 있지 않습니다. "
                    "`scripts/install.ps1` 또는 `scripts/install.sh`를 다시 실행해 주세요."
                ) from exc

            y, sample_rate = librosa.load(vocals, sr=24000, mono=True)
            if y.size == 0:
                raise RuntimeError("듀엣 분리에 사용할 보컬 오디오가 비어 있습니다.")

            n_fft = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
            if rms.size < 8 or float(np.max(rms)) <= 1e-6:
                raise RuntimeError("로컬 시간표 분리에 사용할 충분한 보컬 에너지를 찾지 못했습니다.")

            active_floor = float(os.getenv("LOCAL_DIARIZATION_ACTIVE_FLOOR", "0.018"))
            percentile_gate = float(os.getenv("LOCAL_DIARIZATION_PERCENTILE_GATE", "35"))
            energy_threshold = max(
                float(np.max(rms)) * active_floor,
                float(np.percentile(rms, percentile_gate)) * 0.55,
                1e-6,
            )
            active = rms >= energy_threshold
            active_indices = np.flatnonzero(active)
            if active_indices.size < 12:
                raise RuntimeError("로컬 시간표 분리에 사용할 활성 보컬 프레임이 너무 적습니다.")

            mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
            delta = librosa.feature.delta(mfcc)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
            features = np.vstack(
                [
                    mfcc[1:13],
                    delta[1:13],
                    centroid,
                    bandwidth,
                    contrast,
                ]
            ).T

            frame_count = min(features.shape[0], rms.size)
            features = features[:frame_count]
            rms = rms[:frame_count]
            active = active[:frame_count]
            active_indices = np.flatnonzero(active)
            if active_indices.size < 12:
                raise RuntimeError("로컬 시간표 분리에 사용할 활성 보컬 프레임이 너무 적습니다.")

            active_features = features[active_indices]
            scaler = StandardScaler()
            active_features = scaler.fit_transform(active_features)

            model = KMeans(n_clusters=2, n_init=20, random_state=0)
            weights = np.maximum(rms[active_indices], 1e-6)
            labels = model.fit_predict(active_features, sample_weight=weights)

            frame_labels = np.full(frame_count, -1, dtype=np.int16)
            frame_labels[active_indices] = labels
            frame_labels = self._smooth_frame_labels(
                frame_labels,
                window_frames=max(3, int(round(float(os.getenv("LOCAL_DIARIZATION_SMOOTH_SECONDS", "0.45")) * sample_rate / hop_length))),
            )

            min_segment_seconds = float(os.getenv("LOCAL_DIARIZATION_MIN_SEGMENT_SECONDS", "0.35"))
            merge_gap_seconds = float(os.getenv("LOCAL_DIARIZATION_MERGE_GAP_SECONDS", "0.45"))
            segments = self._segments_from_frame_labels(
                frame_labels,
                sample_rate=sample_rate,
                hop_length=hop_length,
                min_segment_seconds=min_segment_seconds,
                merge_gap_seconds=merge_gap_seconds,
            )
            if not segments:
                raise RuntimeError("로컬 시간표 분리기가 두 보컬 구간을 만들지 못했습니다.")

            speaker_stats: dict[str, dict[str, float]] = {}
            for segment in segments:
                duration = segment.end - segment.start
                stats = speaker_stats.setdefault(
                    segment.speaker,
                    {"duration": 0.0, "first_start": segment.start},
                )
                stats["duration"] += duration
                stats["first_start"] = min(stats["first_start"], segment.start)
            if len(speaker_stats) < 2:
                raise RuntimeError("로컬 시간표 분리기가 두 명의 보컬 파트를 충분히 구분하지 못했습니다.")

            chosen_speakers = sorted(
                speaker_stats,
                key=lambda speaker: speaker_stats[speaker]["duration"],
                reverse=True,
            )[:2]
            chosen_speakers.sort(key=lambda speaker: speaker_stats[speaker]["first_start"])
            speaker_to_index = {speaker: index for index, speaker in enumerate(chosen_speakers)}

            masks = [np.zeros(len(y), dtype=np.float32), np.zeros(len(y), dtype=np.float32)]
            padding_seconds = float(os.getenv("LOCAL_DIARIZATION_SEGMENT_PADDING_SECONDS", "0.08"))
            fade_seconds = float(os.getenv("LOCAL_DIARIZATION_SEGMENT_FADE_SECONDS", "0.025"))
            fade_samples = max(1, int(fade_seconds * sample_rate))

            for segment in segments:
                index = speaker_to_index.get(segment.speaker)
                if index is None:
                    continue
                start = max(0, int(round((segment.start - padding_seconds) * sample_rate)))
                end = min(len(y), int(round((segment.end + padding_seconds) * sample_rate)))
                self._apply_segment_mask(masks[index], start, end, fade_samples)

            active_mask = masks[0] + masks[1]
            overlap = active_mask > 1.0
            if np.any(overlap):
                masks[0][overlap] /= active_mask[overlap]
                masks[1][overlap] /= active_mask[overlap]

            outputs: list[tuple[float, Path]] = []
            for index, mask in enumerate(masks, start=1):
                stem = y * mask
                output = output_dir / f"singer{index}.wav"
                sf.write(output, stem, sample_rate)
                rms_value = float(np.sqrt(np.mean(np.square(stem)))) if stem.size else 0.0
                outputs.append((rms_value, output))

            if not all(rms_value > 1e-5 for rms_value, _ in outputs):
                raise RuntimeError("로컬 시간표로 만든 stem 중 하나가 거의 비어 있습니다.")

            report = output_dir / "local_diarization_segments.txt"
            with report.open("w", encoding="utf-8") as file:
                for segment in sorted(segments, key=lambda item: item.start):
                    mapped = speaker_to_index.get(segment.speaker)
                    if mapped is None:
                        continue
                    file.write(
                        f"{self._format_seconds(segment.start)} - "
                        f"{self._format_seconds(segment.end)}: "
                        f"singer{mapped + 1} ({segment.speaker})\n"
                    )

            return DuetSingerStems(first=outputs[0][1], second=outputs[1][1])

        stems = await asyncio.to_thread(diarize_and_cut)
        await progress("로컬 시간표대로 1번/2번 보컬 파트를 잘라냈습니다.")
        return stems

    async def _separate_duet_with_pyannote_diarization(
        self,
        vocals: Path,
        work_dir: Path,
        progress,
    ) -> DuetSingerStems:
        output_dir = work_dir / "multi_singer_pyannote"
        output_dir.mkdir(parents=True, exist_ok=True)

        def diarize_and_cut() -> DuetSingerStems:
            try:
                import librosa
                import numpy as np
                import soundfile as sf
                import torch
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise OptionalFeatureMissing(
                    "PyAnnote Audio 듀엣 파트 분석에 필요한 `pyannote.audio`가 설치되어 있지 않습니다. "
                    "`scripts/install.ps1`을 다시 실행하거나 `python -m pip install -r requirements.txt`를 "
                    "실행해 주세요."
                ) from exc

            model_name = os.getenv(
                "PYANNOTE_DIARIZATION_MODEL",
                "pyannote/speaker-diarization-community-1",
            ).strip()
            explicit_token = (
                os.getenv("PYANNOTE_AUTH_TOKEN", "").strip()
                or os.getenv("HF_TOKEN", "").strip()
                or os.getenv("HUGGINGFACE_TOKEN", "").strip()
                or None
            )
            token = explicit_token or True

            kwargs = {"token": token}
            try:
                try:
                    pipeline = Pipeline.from_pretrained(model_name, **kwargs)
                except Exception as exc:
                    raise AIProcessor._pyannote_load_error(model_name, exc) from exc
            except TypeError:
                kwargs = {"use_auth_token": token}
                try:
                    pipeline = Pipeline.from_pretrained(model_name, **kwargs)
                except Exception as exc:
                    raise AIProcessor._pyannote_load_error(model_name, exc) from exc
            if pipeline is None:
                raise OptionalFeatureMissing(
                    "PyAnnote Audio 모델을 불러오지 못했습니다. Hugging Face 모델 사용 조건 동의와 "
                    "HF_TOKEN/PYANNOTE_AUTH_TOKEN 또는 `huggingface-cli login` 상태를 확인해 주세요."
                )

            device = os.getenv("PYANNOTE_DEVICE", "").strip()
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if hasattr(pipeline, "to"):
                pipeline.to(torch.device(device))

            y, sample_rate = librosa.load(vocals, sr=None, mono=True)
            if y.size == 0:
                raise RuntimeError("듀엣 분리에 사용할 보컬 오디오가 비어 있습니다.")

            waveform = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
            result = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                num_speakers=2,
            )
            annotation = getattr(result, "speaker_diarization", result)
            if not hasattr(annotation, "itertracks"):
                raise RuntimeError("PyAnnote Audio 결과에서 speaker diarization timeline을 찾지 못했습니다.")

            segments: list[DiarizedSegment] = []
            for segment, _, speaker in annotation.itertracks(yield_label=True):
                start = max(0.0, float(segment.start))
                end = max(start, float(segment.end))
                if end > start:
                    segments.append(DiarizedSegment(start=start, end=end, speaker=str(speaker)))

            if not segments:
                raise RuntimeError("PyAnnote Audio가 보컬 구간을 찾지 못했습니다.")

            speaker_stats: dict[str, dict[str, float]] = {}
            min_segment_seconds = float(os.getenv("PYANNOTE_MIN_SEGMENT_SECONDS", "0.25"))
            for segment in segments:
                duration = segment.end - segment.start
                if duration < min_segment_seconds:
                    continue
                stats = speaker_stats.setdefault(
                    segment.speaker,
                    {"duration": 0.0, "first_start": segment.start},
                )
                stats["duration"] += duration
                stats["first_start"] = min(stats["first_start"], segment.start)

            if len(speaker_stats) < 2:
                raise RuntimeError("PyAnnote Audio가 두 명의 보컬 파트를 충분히 구분하지 못했습니다.")

            chosen_speakers = sorted(
                speaker_stats,
                key=lambda speaker: speaker_stats[speaker]["duration"],
                reverse=True,
            )[:2]
            chosen_speakers.sort(key=lambda speaker: speaker_stats[speaker]["first_start"])
            speaker_to_index = {speaker: index for index, speaker in enumerate(chosen_speakers)}

            masks = [np.zeros(len(y), dtype=np.float32), np.zeros(len(y), dtype=np.float32)]
            padding_seconds = float(os.getenv("PYANNOTE_SEGMENT_PADDING_SECONDS", "0.08"))
            fade_seconds = float(os.getenv("PYANNOTE_SEGMENT_FADE_SECONDS", "0.025"))
            fade_samples = max(1, int(fade_seconds * sample_rate))

            for segment in segments:
                index = speaker_to_index.get(segment.speaker)
                if index is None or segment.end - segment.start < min_segment_seconds:
                    continue
                start = max(0, int(round((segment.start - padding_seconds) * sample_rate)))
                end = min(len(y), int(round((segment.end + padding_seconds) * sample_rate)))
                if end <= start:
                    continue
                self._apply_segment_mask(masks[index], start, end, fade_samples)

            active = masks[0] + masks[1]
            overlap = active > 1.0
            if np.any(overlap):
                masks[0][overlap] /= active[overlap]
                masks[1][overlap] /= active[overlap]

            outputs: list[tuple[float, Path]] = []
            for index, mask in enumerate(masks, start=1):
                stem = y * mask
                output = output_dir / f"singer{index}.wav"
                sf.write(output, stem, sample_rate)
                rms = float(np.sqrt(np.mean(np.square(stem)))) if stem.size else 0.0
                outputs.append((rms, output))

            if not all(rms > 1e-5 for rms, _ in outputs):
                raise RuntimeError("PyAnnote Audio 시간표로 만든 stem 중 하나가 거의 비어 있습니다.")

            report = output_dir / "diarization_segments.txt"
            with report.open("w", encoding="utf-8") as file:
                for segment in sorted(segments, key=lambda item: item.start):
                    mapped = speaker_to_index.get(segment.speaker)
                    if mapped is None:
                        continue
                    file.write(
                        f"{self._format_seconds(segment.start)} - "
                        f"{self._format_seconds(segment.end)}: "
                        f"singer{mapped + 1} ({segment.speaker})\n"
                    )

            return DuetSingerStems(first=outputs[0][1], second=outputs[1][1])

        stems = await asyncio.to_thread(diarize_and_cut)
        await progress("PyAnnote Audio 시간표대로 1번/2번 보컬 파트를 잘라냈습니다.")
        return stems

    async def _separate_duet_with_audio_separator(
        self,
        vocals: Path,
        work_dir: Path,
        model_name: str,
    ) -> DuetSingerStems:
        output_dir = work_dir / "multi_singer_audio_separator"
        output_dir.mkdir(parents=True, exist_ok=True)

        def separate() -> DuetSingerStems:
            try:
                from audio_separator.separator import Separator
            except ImportError as exc:
                raise OptionalFeatureMissing(
                    "`audio-separator`가 설치되어 있지 않습니다. `scripts/install.ps1` 또는 "
                    "`scripts/install.sh`를 다시 실행해 주세요."
                ) from exc

            separator = Separator(
                log_level=logging.ERROR,
                output_dir=str(output_dir),
                output_format="WAV",
            )
            separator.load_model(model_filename=model_name)
            stems = separator.separate(str(vocals))
            output_files = [output_dir / stem for stem in stems if stem]
            if not output_files:
                output_files = list(output_dir.glob("*.wav"))
            return self._pick_duet_singer_outputs(output_files)

        return await asyncio.to_thread(separate)

    async def _separate_duet_with_asteroid(
        self,
        vocals: Path,
        work_dir: Path,
        model_name: str,
    ) -> DuetSingerStems:
        output_dir = work_dir / "multi_singer_asteroid"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_input = work_dir / "multi_singer_input_mono.wav"
        await self._prepare_multi_singer_input(vocals, model_input)

        def separate() -> DuetSingerStems:
            try:
                import torch
            except ImportError as exc:
                raise OptionalFeatureMissing(
                    "AI 듀엣 가수별 분리 모델 실행에 필요한 `asteroid`가 설치되어 있지 않습니다. "
                    "`scripts/install.ps1` 또는 `scripts/install.sh`를 다시 실행해 주세요."
                ) from exc

            device = os.getenv("MULTI_SINGER_SEPARATOR_DEVICE", "").strip()
            if not device:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model = self._load_asteroid_separator_model(model_name)
            model.to(device)
            model.eval()
            with torch.inference_mode():
                model.separate(str(model_input), output_dir=str(output_dir), force_overwrite=True)

            return self._pick_duet_singer_outputs(list(output_dir.glob("**/*.wav")))

        return await asyncio.to_thread(separate)

    async def _prepare_multi_singer_input(self, source: Path, output: Path) -> None:
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
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
    def _load_asteroid_separator_model(model_name: str):
        if model_name == DEFAULT_MULTI_SINGER_SEPARATOR_MODEL:
            return AIProcessor._load_medleyvox_model()

        try:
            from asteroid.models import BaseModel
        except ImportError as exc:
            raise OptionalFeatureMissing(
                "AI 듀엣 가수별 분리 모델 실행에 필요한 `asteroid`가 설치되어 있지 않습니다. "
                "`scripts/install.ps1` 또는 `scripts/install.sh`를 다시 실행해 주세요."
            ) from exc

        return BaseModel.from_pretrained(model_name)

    @staticmethod
    def _load_medleyvox_model():
        try:
            import torch
            from asteroid.models import ConvTasNet
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise OptionalFeatureMissing(
                "AI 듀엣 MedleyVox 모델 실행에 필요한 패키지가 설치되어 있지 않습니다. "
                "`scripts/install.ps1` 또는 `scripts/install.sh`를 다시 실행해 주세요."
            ) from exc

        model_file = os.getenv("MULTI_SINGER_SEPARATOR_FILE", DEFAULT_MULTI_SINGER_SEPARATOR_FILE).strip()
        checkpoint_path = hf_hub_download(
            repo_id=DEFAULT_MULTI_SINGER_SEPARATOR_MODEL,
            filename=model_file,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        prefix = "ema_model.module."
        if not any(key.startswith(prefix) for key in checkpoint):
            prefix = "online_model.module."

        state_dict = {
            key.removeprefix(prefix): value
            for key, value in checkpoint.items()
            if key.startswith(prefix)
        }
        if not state_dict:
            raise RuntimeError(f"MedleyVox 체크포인트에서 모델 가중치를 찾지 못했습니다: {model_file}")

        model = ConvTasNet(
            n_src=2,
            in_chan=2050,
            out_chan=2050,
            n_blocks=8,
            n_repeats=3,
            bn_chan=256,
            hid_chan=1024,
            skip_chan=256,
            conv_kernel_size=3,
            fb_name="stft",
            kernel_size=2048,
            stride=512,
            n_filters=2048,
            sample_rate=44100,
        )
        model.load_state_dict(state_dict)
        return model

    async def _separate_duet_with_command(
        self,
        vocals: Path,
        work_dir: Path,
        command_template: str,
    ) -> DuetSingerStems:
        output_dir = work_dir / "multi_singer_external"
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered = command_template.format(
            input=self._shell_quote(str(vocals.resolve())),
            output_dir=self._shell_quote(str(output_dir.resolve())),
            raw_input=str(vocals.resolve()),
            raw_output_dir=str(output_dir.resolve()),
        )

        process = await asyncio.create_subprocess_shell(
            rendered,
            cwd=Path.cwd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = (stderr or stdout).decode("utf-8", errors="ignore")
            raise RuntimeError(message[-1500:])

        return self._pick_duet_singer_outputs(list(output_dir.glob("**/*.wav")))

    @staticmethod
    def _smooth_frame_labels(frame_labels, window_frames: int):
        import numpy as np

        labels = np.asarray(frame_labels).copy()
        if labels.size == 0:
            return labels

        window = max(3, int(window_frames))
        if window % 2 == 0:
            window += 1
        half = window // 2
        smoothed = labels.copy()
        for index in range(labels.size):
            if labels[index] < 0:
                continue
            start = max(0, index - half)
            end = min(labels.size, index + half + 1)
            local = labels[start:end]
            local = local[local >= 0]
            if local.size == 0:
                continue
            counts = np.bincount(local.astype(np.int16), minlength=2)
            smoothed[index] = int(np.argmax(counts))
        return smoothed

    @staticmethod
    def _segments_from_frame_labels(
        frame_labels,
        sample_rate: int,
        hop_length: int,
        min_segment_seconds: float,
        merge_gap_seconds: float,
    ) -> list[DiarizedSegment]:
        labels = list(frame_labels)
        raw_segments: list[DiarizedSegment] = []
        start_frame: int | None = None
        current_label: int | None = None

        for frame_index, label in enumerate(labels + [-1]):
            label = int(label)
            if label < 0:
                if current_label is not None and start_frame is not None:
                    start = start_frame * hop_length / sample_rate
                    end = frame_index * hop_length / sample_rate
                    if end - start >= min_segment_seconds:
                        raw_segments.append(DiarizedSegment(start, end, f"cluster_{current_label}"))
                current_label = None
                start_frame = None
                continue

            if current_label is None:
                current_label = label
                start_frame = frame_index
                continue

            if label != current_label:
                if start_frame is not None:
                    start = start_frame * hop_length / sample_rate
                    end = frame_index * hop_length / sample_rate
                    if end - start >= min_segment_seconds:
                        raw_segments.append(DiarizedSegment(start, end, f"cluster_{current_label}"))
                current_label = label
                start_frame = frame_index

        if not raw_segments:
            return []

        merged: list[DiarizedSegment] = []
        for segment in raw_segments:
            if (
                merged
                and merged[-1].speaker == segment.speaker
                and segment.start - merged[-1].end <= merge_gap_seconds
            ):
                previous = merged[-1]
                merged[-1] = DiarizedSegment(previous.start, segment.end, previous.speaker)
            else:
                merged.append(segment)
        return merged

    @staticmethod
    def _apply_segment_mask(mask, start: int, end: int, fade_samples: int) -> None:
        import numpy as np

        length = end - start
        if length <= 0:
            return

        window = np.ones(length, dtype=np.float32)
        fade = min(fade_samples, max(1, length // 2))
        if fade > 1:
            window[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            window[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        mask[start:end] = np.maximum(mask[start:end], window)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining = seconds - minutes * 60
        return f"{minutes:02d}:{remaining:05.2f}"

    @staticmethod
    def _shell_quote(value: str) -> str:
        if os.name == "nt":
            return subprocess.list2cmdline([value])
        return shlex.quote(value)

    @staticmethod
    def _env_enabled(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _pyannote_load_error(model_name: str, exc: Exception) -> OptionalFeatureMissing:
        message = str(exc)
        lowered = message.lower()
        hints: list[str] = []

        if os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
            hints.append("`HF_HUB_OFFLINE`가 켜져 있습니다. PowerShell에서 `Remove-Item Env:HF_HUB_OFFLINE` 후 다시 실행해 주세요.")
        if "local cache" in lowered or "internet connection" in lowered or "locate the file" in lowered:
            hints.append(
                "모델이 아직 로컬 캐시에 없거나 Hugging Face 연결에 실패했습니다. "
                "`huggingface-cli whoami`로 로그인 확인 후 "
                f"`huggingface-cli download {model_name} config.yaml`로 다운로드가 되는지 확인해 주세요. "
                "해당 명령이 없으면 `.\\.venv\\Scripts\\hf.exe auth whoami`와 "
                f"`.\\.venv\\Scripts\\hf.exe download {model_name} config.yaml`를 사용하세요."
            )
        if "invalid user token" in lowered or "token stored is invalid" in lowered:
            hints.append(
                "저장된 Hugging Face 토큰이 유효하지 않습니다. 권한을 켠 새 토큰으로 "
                "`huggingface-cli login` 또는 `.\\.venv\\Scripts\\hf.exe auth login --force`를 다시 실행해 주세요."
            )
        if "enable access to public gated repositories" in lowered or "fine-grained token" in lowered:
            hints.append(
                "현재 Hugging Face 토큰에서 public gated repositories 접근 권한이 꺼져 있습니다. "
                "토큰 설정에서 gated repo read 권한을 켜거나, 해당 권한이 있는 새 Read 토큰으로 "
                "`huggingface-cli login` 또는 `.\\.venv\\Scripts\\hf.exe auth login --force`를 다시 실행해 주세요."
            )
        if "403" in lowered or "gated" in lowered or "authorized" in lowered:
            hints.append(
                f"`https://huggingface.co/{model_name}`에서 같은 계정으로 사용 조건 동의/접근 승인을 완료해 주세요."
            )
        if not hints:
            hints.append("Hugging Face 로그인, 모델 사용 조건 동의, 인터넷 연결 상태를 확인해 주세요.")

        return OptionalFeatureMissing(
            "PyAnnote Audio 모델을 불러오지 못했습니다.\n"
            + "\n".join(f"- {hint}" for hint in hints)
            + f"\n\n원본 오류: {message[-800:]}"
        )

    @staticmethod
    def _looks_like_huggingface_model(model_name: str) -> bool:
        return "/" in model_name and not model_name.lower().endswith((".ckpt", ".pth", ".onnx"))

    @staticmethod
    def _pick_duet_singer_outputs(paths: list[Path]) -> DuetSingerStems:
        existing = sorted(
            [path for path in paths if path.exists() and path.stat().st_size > 44],
            key=lambda path: AIProcessor._audio_rms(path),
            reverse=True,
        )
        singer_like: list[Path] = []
        negative_tokens = {
            "instrumental",
            "accompaniment",
            "karaoke",
            "no_vocals",
            "no-vocals",
            "mixture",
            "mix",
            "input",
        }
        for path in existing:
            name = path.name.lower()
            if any(token in name for token in negative_tokens):
                continue
            singer_like.append(path)

        candidates = singer_like or existing
        if len(candidates) < 2:
            raise RuntimeError(
                "가수 개별 분리 결과에서 두 개의 보컬 stem을 찾지 못했습니다. "
                "결과 폴더에 두 가수 stem WAV가 생성되도록 모델/명령을 확인해 주세요."
            )
        return DuetSingerStems(first=candidates[0], second=candidates[1])

    @staticmethod
    def _duet_stems_are_low_quality(stems: DuetSingerStems) -> bool:
        return (
            AIProcessor._duet_stems_are_imbalanced(stems)
            or AIProcessor._duet_stems_are_too_similar(stems)
        )

    @staticmethod
    def _duet_stems_are_imbalanced(stems: DuetSingerStems) -> bool:
        first_rms = AIProcessor._audio_rms(stems.first)
        second_rms = AIProcessor._audio_rms(stems.second)
        stronger = max(first_rms, second_rms)
        weaker = min(first_rms, second_rms)
        if stronger <= 1e-6:
            return True

        min_ratio = float(os.getenv("MULTI_SINGER_SEPARATOR_MIN_RMS_RATIO", "0.08"))
        return weaker / stronger < min_ratio

    @staticmethod
    def _duet_stems_are_too_similar(stems: DuetSingerStems) -> bool:
        correlation = AIProcessor._audio_correlation(stems.first, stems.second)
        max_correlation = float(os.getenv("MULTI_SINGER_SEPARATOR_MAX_CORRELATION", "0.92"))
        return correlation >= max_correlation

    @staticmethod
    def _audio_correlation(first: Path, second: Path) -> float:
        try:
            import numpy as np
            import soundfile as sf

            first_audio, _ = sf.read(first, always_2d=True)
            second_audio, _ = sf.read(second, always_2d=True)
            length = min(len(first_audio), len(second_audio))
            if length < 1024:
                return 1.0

            first_mono = np.asarray(first_audio[:length], dtype=np.float32).mean(axis=1)
            second_mono = np.asarray(second_audio[:length], dtype=np.float32).mean(axis=1)
            if np.std(first_mono) <= 1e-8 or np.std(second_mono) <= 1e-8:
                return 1.0
            return float(np.corrcoef(first_mono, second_mono)[0, 1])
        except Exception:
            return 0.0

    @staticmethod
    def _audio_rms(path: Path) -> float:
        try:
            import numpy as np
            import soundfile as sf

            audio, _ = sf.read(path, always_2d=False)
            if audio.size == 0:
                return 0.0
            data = np.asarray(audio, dtype=np.float32)
            return float(np.sqrt(np.mean(np.square(data))))
        except Exception:
            try:
                return float(path.stat().st_size)
            except OSError:
                return 0.0

    async def _separate_vocals(self, input_wav: Path, work_dir: Path, progress) -> tuple[Path, Path]:
        backend = os.getenv("VOCAL_SEPARATOR_BACKEND", "auto").strip().lower()
        if backend in {"auto", "audio-separator", "audio_separator", "roformer"}:
            model_name = os.getenv("VOCAL_SEPARATOR_MODEL", DEFAULT_SEPARATOR_MODEL)
            await progress("보컬 분리 모델 실행하는 중...")
            try:
                return await self._separate_with_audio_separator(input_wav, work_dir, model_name)
            except OptionalFeatureMissing as exc:
                raise OptionalFeatureMissing("고품질 보컬 분리 엔진을 실행하지 못했습니다.") from exc
            except Exception as exc:
                raise RuntimeError(f"고품질 보컬 분리가 실패했습니다.\n{str(exc)[-1200:]}") from exc

        raise ValueError(f"지원하지 않는 VOCAL_SEPARATOR_BACKEND 값입니다: {backend}")

    async def _separate_with_audio_separator(
        self,
        input_wav: Path,
        work_dir: Path,
        model_name: str,
    ) -> tuple[Path, Path]:
        output_dir = work_dir / "audio_separator"
        output_dir.mkdir(parents=True, exist_ok=True)

        def separate() -> tuple[Path, Path]:
            try:
                from audio_separator.separator import Separator
            except ImportError as exc:
                raise OptionalFeatureMissing(
                    "고품질 보컬 분리용 `audio-separator`가 설치되어 있지 않습니다. "
                    "`scripts/install.ps1` 또는 `scripts/install.sh`를 다시 실행해 주세요."
                ) from exc

            separator = Separator(
                log_level=logging.ERROR,
                output_dir=str(output_dir),
                output_format="WAV",
            )
            separator.load_model(model_filename=model_name)
            stems = separator.separate(str(input_wav))
            output_files = [output_dir / stem for stem in stems if stem]
            if not output_files:
                output_files = list(output_dir.glob("*.wav"))

            vocals = self._pick_separator_output(output_files, "vocals")
            instrumental = self._pick_separator_output(output_files, "instrumental")
            if not vocals or not instrumental:
                raise RuntimeError(f"보컬 분리 결과가 불완전합니다. files={stems}")
            return vocals, instrumental

        return await asyncio.to_thread(separate)

    @staticmethod
    def _pick_separator_output(paths: list[Path], stem_type: str) -> Path | None:
        existing = [path for path in paths if path.exists()]
        if stem_type == "vocals":
            positives = ("vocal", "vocals", "voice")
            negatives = ("instrumental", "no_vocals", "no-vocals", "karaoke")
        else:
            positives = ("instrumental", "no_vocals", "no-vocals", "karaoke", "accompaniment")
            negatives = ("vocal", "vocals", "voice")

        if stem_type == "instrumental":
            for path in existing:
                name = path.name.lower()
                if any(token in name for token in ("no_vocals", "no-vocals", "instrumental", "karaoke")):
                    return path

        for path in existing:
            name = path.name.lower()
            if any(token in name for token in positives) and not any(token in name for token in negatives):
                return path

        if len(existing) == 2:
            vocal_like = [
                path
                for path in existing
                if any(token in path.name.lower() for token in ("vocal", "vocals", "voice"))
            ]
            instrumental_like = [path for path in existing if path not in vocal_like]
            if stem_type == "vocals" and vocal_like:
                return vocal_like[0]
            if stem_type == "instrumental" and instrumental_like:
                return instrumental_like[0]

        return None

    async def _prepare_vocals_for_conversion(self, vocals: Path, work_dir: Path, progress) -> Path:
        filter_chain = os.getenv(
            "AI_COVER_INPUT_VOCAL_FILTER",
            "none",
        ).strip()
        if not filter_chain or filter_chain.lower() in {"off", "false", "none"}:
            return vocals

        output = work_dir / f"prepared_{vocals.stem}.wav"
        await progress("변환용 보컬 소스를 정리하는 중...")
        await self._filter_audio(vocals, output, filter_chain, codec="pcm_s16le")
        return output

    async def _polish_converted_vocal(self, vocal: Path, work_dir: Path) -> Path:
        filter_chain = os.getenv(
            "AI_COVER_OUTPUT_VOCAL_FILTER",
            "none",
        ).strip()
        if not filter_chain or filter_chain.lower() in {"off", "false", "none"}:
            return vocal

        output = work_dir / f"polished_{vocal.stem}.wav"
        await self._filter_audio(vocal, output, filter_chain, codec="pcm_s16le")
        return output

    async def _filter_audio(self, source: Path, output: Path, filter_chain: str, codec: str | None = None) -> None:
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(source),
            "-af",
            filter_chain,
        ]
        if codec:
            cmd.extend(["-c:a", codec])
        cmd.append(str(output))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore")[-1000:])

    async def _merge(
        self,
        vocals: Path,
        instrumental: Path,
        output: Path,
        vocal_boost: bool,
        profile: str = "default",
    ) -> None:
        if profile == "cover":
            vocal_volume = "1.0"
            inst_volume = "0.80"
            master_filter = "alimiter=limit=0.98"
        elif profile == "boost" or vocal_boost:
            vocal_volume = "1.30"
            inst_volume = "0.58"
            master_filter = "alimiter=limit=0.98"
        else:
            vocal_volume = "1.0"
            inst_volume = "0.85"
            master_filter = "alimiter=limit=0.98"

        filter_complex = (
            f"[0:a]volume={vocal_volume}[v];"
            f"[1:a]volume={inst_volume}[i];"
            "[v][i]amix=inputs=2:duration=longest:normalize=0[m];"
            f"[m]{master_filter}"
        )
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(vocals),
            "-i",
            str(instrumental),
            "-filter_complex",
            filter_complex,
            "-b:a",
            "320k",
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

    async def _copy_or_vocal_pitch_shift(
        self,
        source: Path,
        output: Path,
        vocal_pitch_shift: int,
        progress,
    ) -> None:
        if vocal_pitch_shift == 0:
            shutil.copy2(source, output)
            return

        await progress("목소리 피치를 조절하는 중...")
        await self._apply_pitch_shift(source, output, vocal_pitch_shift)

    async def _vocal_with_pitch_shift(
        self,
        vocals: Path,
        work_dir: Path,
        vocal_pitch_shift: int,
        progress,
    ) -> Path:
        if vocal_pitch_shift == 0:
            return vocals

        output = work_dir / f"vocal_pitch_{vocal_pitch_shift:+d}.wav"
        await progress("목소리 피치를 조절하는 중...")
        await self._apply_pitch_shift(vocals, output, vocal_pitch_shift)
        return output

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
            "-c:a",
            "pcm_s16le",
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

    async def _export_audio(self, source: Path, output: Path) -> None:
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(source),
            "-vn",
            "-b:a",
            "320k",
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

    async def _measure_loudnorm(self, source: Path) -> dict[str, str]:
        cmd = [
            self.ffmpeg.executable(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-af",
            "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
            "-f",
            "null",
            os.devnull,
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        message = (stderr or stdout).decode("utf-8", errors="ignore")
        if process.returncode != 0:
            raise RuntimeError(message[-1000:])

        start = message.rfind("{")
        end = message.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("loudnorm 측정 결과를 읽지 못했습니다.")
        return json.loads(message[start : end + 1])

    async def _apply_loudnorm(self, source: Path, output: Path, measured: dict[str, str]) -> None:
        filter_complex = (
            "loudnorm=I=-14:TP=-1.5:LRA=11:"
            f"measured_I={measured['input_i']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:"
            "linear=true:print_format=summary"
        )
        cmd = [
            self.ffmpeg.executable(),
            "-y",
            "-i",
            str(source),
            "-af",
            filter_complex,
            "-b:a",
            "320k",
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

    async def _merge_duet_sources(
        self,
        first_vocal: Path,
        second_vocal: Path,
        instrumental: Path,
        output: Path,
    ) -> None:
        filter_complex = (
            "[0:a]volume=1.02[v1];"
            "[1:a]volume=1.02[v2];"
            "[2:a]volume=0.74[i];"
            "[v1][v2]amix=inputs=2:duration=longest:normalize=0[dv];"
            "[dv]volume=1.02[dv2];"
            "[dv2][i]amix=inputs=2:duration=longest:normalize=0[m];"
            "[m]loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.98"
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
            "320k",
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
    def _safe_name(value: str) -> str:
        keep = [ch if ch.isalnum() or ch in "._- " else "_" for ch in value]
        safe = "".join(keep).strip(" ._")
        return (safe or "audio")[:80]
