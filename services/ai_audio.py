from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yt_dlp

from services.ffmpeg import FFmpegResolver


class OptionalFeatureMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcessedAudio:
    path: str
    title: str
    duration: int


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
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def available_voices(self) -> list[str]:
        return self.rvc.available_voices()

    async def process_youtube(
        self,
        url: str,
        mode: str,
        target_voice: str | None,
        pitch_shift: int,
        progress,
    ) -> ProcessedAudio:
        work_dir = Path(tempfile.mkdtemp(prefix="ai_music_", dir="tmp"))
        try:
            await progress("유튜브 오디오를 다운로드하는 중...")
            input_wav, title, duration = await self._download_audio(url, work_dir)

            await progress("보컬과 반주를 분리하는 중... 처음 실행은 모델 다운로드 때문에 오래 걸릴 수 있습니다.")
            vocals, instrumental = await self._separate_with_demucs(input_wav, work_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = self._safe_name(title)

            if mode == "instrumental":
                output = self.output_dir / f"MR_{safe_title}_{timestamp}.wav"
                shutil.copy2(instrumental, output)
                return ProcessedAudio(str(output), title, duration)

            if mode == "vocal":
                output = self.output_dir / f"VOCAL_{safe_title}_{timestamp}.wav"
                shutil.copy2(vocals, output)
                return ProcessedAudio(str(output), title, duration)

            converted_vocal = vocals
            prefix = "VOCAL_BOOST"
            if mode == "ai_cover":
                if not target_voice:
                    raise ValueError("AI 커버 모드는 target_voice 값이 필요합니다.")
                await progress(f"{target_voice} 목소리로 변환하는 중...")
                converted_vocal = await self.rvc.convert(vocals, work_dir, target_voice, pitch_shift)
                prefix = f"AI_COVER_{self._safe_name(target_voice)}"

            await progress("최종 오디오를 합성하는 중...")
            output = self.output_dir / f"{prefix}_{safe_title}_{timestamp}.mp3"
            await self._merge(converted_vocal, instrumental, output, vocal_boost=True)
            return ProcessedAudio(str(output), title, duration)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _download_audio(self, url: str, work_dir: Path) -> tuple[Path, str, int]:
        output_template = str(work_dir / "input.%(ext)s")

        def download() -> tuple[Path, str, int]:
            opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }],
                "ffmpeg_location": os.path.dirname(self.ffmpeg.executable()),
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            return work_dir / "input.wav", info.get("title", "youtube_audio"), info.get("duration") or 0

        return await asyncio.to_thread(download)

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

    @staticmethod
    def _safe_name(value: str) -> str:
        keep = [ch if ch.isalnum() or ch in "._- " else "_" for ch in value]
        safe = "".join(keep).strip(" ._")
        return (safe or "audio")[:80]
