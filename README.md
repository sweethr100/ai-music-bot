# AI Music Discord Bot

Discord 음성 채널에서 음악 재생, MR/보컬 분리, AI 커버, AI 듀엣, 목소리 녹음과 RVC 학습까지 할 수 있는 음악 봇입니다.

## 빠른 시작

처음 설치하는 사람은 아래 순서대로 진행하면 됩니다.

### 1. 준비물

- Windows 10/11 또는 Linux
- Python 3.11 이상
- NVIDIA GPU와 최신 NVIDIA 드라이버
- `git`
- Discord 봇 토큰

AI 기능은 CUDA GPU 기준입니다. 기본 설치는 CUDA 12.8 PyTorch wheel을 사용합니다.

### 2. 설치

Windows PowerShell:

```powershell
cd path\to\ai-music-bot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\scripts\install.ps1
```

Linux:

```bash
cd /path/to/ai-music-bot
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

CUDA 12.8 설치가 드라이버와 맞지 않으면 CUDA 12.6으로 다시 설치합니다.

```powershell
.\scripts\install.ps1 -TorchCuda cu126
```

Linux에서는 이렇게 실행합니다.

```bash
TORCH_CUDA=cu126 bash scripts/install.sh
```

기본 설치가 성공했다면 CUDA 12.6 명령은 실행하지 않아도 됩니다.

### 3. Discord 토큰 넣기

`.env.example`을 `.env`로 복사한 뒤 아래 값을 채웁니다.

```env
DISCORD_TOKEN=put-your-bot-token-here
GUILD_ID=123456789012345678
```

- `DISCORD_TOKEN`: Discord Developer Portal에서 만든 봇 토큰
- `GUILD_ID`: 테스트할 Discord 서버 ID. 넣으면 슬래시 명령어가 더 빨리 반영됩니다.

### 4. 실행

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python bot.py
```

Linux:

```bash
source .venv/bin/activate
python bot.py
```

이미 가상환경이 켜져 있으면 `python bot.py`만 실행하면 됩니다.

## 자주 쓰는 명령어

```text
/play query:노래 제목 mode:원본 재생
/play query:https://youtu.be/... mode:MR만 재생
/play query:노래 제목 target_voice:myvoice vocal_pitch_shift:+6
/play_file file:<업로드 파일> mode:원본 재생
/play_duet query:듀엣 노래 제목
/playlist url:https://youtube.com/playlist?... max_count:50
/queue
/skip
/pause
/resume
/loop mode:전체 반복
/normalizer enabled:True
/lyrics query:가수 노래 제목
```

## 주요 기능

- `/play`: 유튜브 URL이나 검색어로 원본 재생, MR/보컬 분리, AI 커버 생성
- `/play_file`: Discord에 업로드한 파일로 재생 또는 AI 처리
- `/play_duet`: 두 파트 AI 듀엣 커버 생성
- `/separate_singers`: 듀엣 곡의 가수별 보컬 stem 분리 테스트
- `/record_start`, `/record_stop`, `/record_status`: 유저별 학습용 음성 녹음
- `/train_voice`: 녹음 데이터로 RVC 목소리 모델 자동 학습
- `/playlist`, `/queue`, `/remove`, `/loop`, `/skip`, `/pause`, `/resume`, `/nowplaying`
- `/normalizer`: 음량 노멀라이저 켜기/끄기
- `/lyrics`: 가사 조회

봇은 대기열이 비어도, 음성방에 사람이 없어도 자동으로 나가지 않습니다. 내보내려면 Discord에서 직접 연결을 끊거나 봇을 이동/강퇴하면 됩니다.

## AI 목소리 모델 넣기

직접 준비한 RVC 모델은 아래처럼 넣습니다.

```text
voice_models/
  myvoice/
    myvoice.pth
    myvoice.index
```

`.index` 파일은 있으면 사용하고, 없으면 `.pth`만으로 시도합니다. 폴더 이름 `myvoice`가 `/play`의 `target_voice` 자동완성에 표시됩니다.

## 목소리 녹음과 학습

1. Discord 음성 채널에 들어갑니다.
2. `/record_start`로 녹음을 시작합니다.
3. 충분히 녹음한 뒤 `/record_stop`으로 끝냅니다.
4. `/train_voice dataset:<유저 데이터셋> model_name:<목소리이름>`을 실행합니다.

녹음 파일은 `data/recordings/<표시이름_유저ID>/`에 저장됩니다. 학습이 끝난 모델은 `voice_models/<목소리이름>/`에 저장되고, AI 커버와 AI 듀엣에서 바로 선택할 수 있습니다.

학습 기본값은 epoch 300, batch size 16입니다. 조절이 필요하면 `.env.example`의 `RVC_TRAIN_*` 항목을 참고하세요.

## AI 듀엣 참고

`/play_duet`은 먼저 곡에서 누가 어느 구간을 부르는지 분석하고, 1번/2번 보컬 미리듣기 파일을 Discord 채팅에 올립니다. 미리 들어본 뒤 설정 패널에서 각 파트의 목소리, 피치, 볼륨을 고르고 `듀엣 렌더`를 누르면 최종 파일을 만듭니다.

기본 듀엣 분석은 PyAnnote Audio를 사용합니다. 처음 사용할 때 Hugging Face 로그인이 필요할 수 있습니다.

```powershell
huggingface-cli login
huggingface-cli whoami
```

`huggingface-cli`가 없거나 동작하지 않으면 venv에 설치된 CLI를 사용합니다.

```powershell
.\.venv\Scripts\hf.exe auth login --force
.\.venv\Scripts\hf.exe auth whoami
```

PyAnnote 모델 접근 오류가 나면 Hugging Face에서 `pyannote/speaker-diarization-community-1` 모델 사용 조건에 동의한 뒤 다시 로그인합니다.

로그인 없이 로컬 방식만 쓰고 싶으면 `.env`에 아래 값을 넣습니다.

```env
MULTI_SINGER_SEPARATOR_BACKEND=local_diarization
```

듀엣 분리는 실제 보컬을 완전히 분리하는 기능이 아니라, “누가 부르는 구간인지”를 찾아 시간표대로 나누는 방식입니다. 두 사람이 동시에 부르는 화음이나 후렴은 완벽히 분리되지 않을 수 있습니다.

## 고급 설정

대부분은 기본값 그대로 쓰면 됩니다. 품질 튜닝이나 디버깅이 필요할 때만 `.env.example`의 주석을 보고 `.env`에 원하는 값을 추가하세요.

자주 만지는 값:

```env
VOCAL_SEPARATOR_BACKEND=auto
VOCAL_SEPARATOR_MODEL=mel_band_roformer_kim_ft_unwa.ckpt
RVC_INFER_F0_METHOD=rmvpe
RVC_INFER_INDEX_RATE=0.25
RVC_INFER_VOLUME_ENVELOPE=0.75
RVC_INFER_PROTECT=0.50
PYANNOTE_DEVICE=cuda
```

## 설치 스크립트가 하는 일

- `requirements.txt` 설치
- 선택한 CUDA PyTorch wheel 설치와 GPU 인식 확인
- 보컬 분리 엔진 설치
- `vendor/Applio`에 Applio RVC 엔진 준비
- Applio 전용 가상환경과 의존성 준비
- `voice_models`, `data/recordings` 폴더 준비

설치 후 사용자가 직접 해야 하는 일은 Discord 토큰 입력, 필요한 경우 RVC 모델 추가, 필요한 경우 Hugging Face 로그인입니다.

## 작동 방식 요약

- 일반 재생은 스트리밍 방식이라 다운로드 파일을 남기지 않습니다.
- AI 처리 결과물은 `data/processed`에 저장되고, 재생이 끝나거나 대기열에서 삭제되면 정리됩니다.
- 보컬 분리는 Roformer 기반 `audio-separator`를 사용합니다.
- AI 커버와 학습은 Applio RVC 엔진을 사용합니다.
- 플레이리스트는 목록만 먼저 가져오고, 실제 스트림 URL은 재생 직전에 다시 가져와 만료 문제를 줄입니다.
