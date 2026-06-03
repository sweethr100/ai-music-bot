# AI Music Discord Bot

음악 재생과 AI 음악 처리를 `/play` 하나로 쓰는 Discord 음악 봇입니다. 기존 all-in-one 봇 코드를 복사하지 않고, 음악 기능만 새 구조로 다시 만들었습니다.

## 지원 기능

- `/play` 유튜브 URL/검색어 원본 재생
- `/play` MR만 재생, 보컬만 재생, 보컬 강조 믹스
- `/play` AI 커버 생성 및 재생
- `/playlist` 유튜브 재생목록 전체 추가
- `/queue` 현재 곡과 대기열 확인
- `/remove` 특정 대기열 곡 삭제
- `/loop` 반복재생: 꺼짐, 현재곡 반복, 전체 반복
- `/normalizer` 음량 노멀라이저 켜기/끄기
- `/lyrics` 현재 곡 또는 검색어 가사 조회
- `/skip`, `/pause`, `/resume`, `/nowplaying`

봇은 대기열이 비어도, 음성방에 사람이 없어도 자동으로 나가지 않습니다. 음성방에서 내보내고 싶으면 Discord에서 직접 연결을 끊거나 봇을 이동/강퇴하면 됩니다.

## 설치

이 봇은 NVIDIA GPU가 있는 컴퓨터를 기준으로 설치합니다. AI 기능은 CUDA로 실행되도록 고정되어 있고, 설치 스크립트는 CUDA PyTorch가 실제로 GPU를 잡는지 검사합니다.

권장 환경:

- Windows 10/11 또는 Linux
- Python 3.11 이상
- NVIDIA GPU
- 최신 NVIDIA 드라이버
- `git`

Windows PowerShell:

```powershell
cd C:\Users\sweet\Desktop\discord-ai-musicbot\ai-music-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\scripts\install_ai.ps1
```

Linux:

```bash
cd /path/to/ai-music-bot
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install_ai.sh
```

설치 스크립트가 음악 재생 패키지, AI 오디오 패키지, CUDA PyTorch, Applio RVC 엔진을 함께 준비합니다. FFmpeg는 시스템에 설치되어 있으면 그걸 쓰고, 없으면 `imageio-ffmpeg`의 번들 FFmpeg를 사용합니다.

## Discord 설정

`.env.example`을 `.env`로 복사하고 값을 채웁니다.

```env
DISCORD_TOKEN=put-your-bot-token-here
GUILD_ID=123456789012345678
```

`GUILD_ID`를 넣으면 해당 서버에 슬래시 명령어가 빠르게 동기화됩니다.

스크립트가 하는 일:

- 봇 실행에 필요한 전체 `requirements.txt` 설치
- CUDA 12.8 PyTorch 설치
- `torch.cuda.is_available()` 검증
- 봇 환경에 Demucs 설치
- `vendor/Applio`에 Applio RVC 엔진 다운로드
- Applio 전용 가상환경 생성
- Applio 환경에도 CUDA 12.8 PyTorch 설치
- Applio 의존성 설치
- `voice_models` 폴더 준비

CUDA 12.8 wheel이 드라이버와 맞지 않으면 CUDA 12.6 wheel로 설치할 수 있습니다.

Windows:

```powershell
.\scripts\install_ai.ps1 -TorchCuda cu126
```

Linux:

```bash
TORCH_CUDA=cu126 bash scripts/install_ai.sh
```

남는 수동 작업:

- `.env`에 Discord 봇 토큰 넣기
- AI 커버에 쓸 RVC 목소리 모델을 `voice_models/<이름>/`에 넣기

## AI 커버 목소리 모델 넣기

RVC 모델 파일을 아래처럼 넣으면 `/play`의 `target_voice` 자동완성에 뜹니다.

```text
voice_models/
  myvoice/
    myvoice.pth
    myvoice.index
```

`.index` 파일은 있으면 사용하고, 없으면 `.pth`만으로 시도합니다.

## 실행

```powershell
python bot.py
```

## 명령어 예시

```text
/play url:https://youtu.be/... mode:원본 재생
/play url:https://youtu.be/... mode:MR만 재생
/play url:https://youtu.be/... mode:AI 커버 target_voice:myvoice pitch_shift:+6
/playlist url:https://youtube.com/playlist?... max_count:50
/loop mode:전체 반복
/normalizer enabled:True
/remove position:3
/lyrics
/lyrics query:artist song title
```

## 설계 메모

- `/play` 안에서 일반 재생과 AI 처리를 함께 다룹니다.
- 일반 재생은 스트리밍 방식이라 다운로드 파일을 남기지 않습니다.
- 보컬 분리는 Demucs를 `cuda` 디바이스로 실행합니다.
- AI 커버는 Applio 환경의 CUDA PyTorch를 사용합니다.
- AI 처리 결과물은 `data/processed`에 저장되고, 재생이 끝나거나 대기열에서 삭제되면 정리됩니다.
- 플레이리스트는 목록만 먼저 가져오고, 실제 스트림 URL은 재생 직전에 다시 가져와 만료 문제를 줄입니다.
- 노멀라이저는 현재 곡 다음부터 적용됩니다.
