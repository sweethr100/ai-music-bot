# AI Music Discord Bot

음악 재생과 AI 커버는 `/play`, AI 듀엣 커버는 `/play_duet`으로 쓰는 Discord 음악 봇입니다.

## 지원 기능

- `/play` 유튜브 URL/검색어 원본 재생
- `/play` MR만 재생, 보컬만 재생, 보컬 강조 믹스
- `/play` AI 커버 생성 및 재생
- `/play_file` 유저가 업로드한 파일로 원본/MR/보컬/AI 커버 생성 및 재생
- `/play_duet` AI 듀엣 생성 및 재생
- `/record_start`, `/record_stop`, `/record_status` 유저별 학습용 음성 녹음
- `/train_voice` 녹음 데이터로 RVC 목소리 모델 자동 학습
- `/playlist` 유튜브 재생목록 전체 추가
- `/queue` 현재 곡과 대기열 확인
- `/remove` 특정 대기열 곡 삭제
- `/loop` 반복재생: 꺼짐, 현재곡 반복, 전체 반복
- `/normalizer` 음량 노멀라이저 켜기/끄기
- `/lyrics` 현재 곡 또는 검색어 가사 조회
- `/skip`, `/pause`, `/resume`, `/nowplaying`

봇은 대기열이 비어도, 음성방에 사람이 없어도 자동으로 나가지 않습니다. 음성방에서 내보내고 싶으면 Discord에서 직접 연결을 끊거나 봇을 이동/강퇴하면 됩니다.

## 설치

이 봇은 NVIDIA GPU가 있는 컴퓨터를 기준으로 설치합니다. AI 기능은 CUDA로 실행되도록 고정되어 있고, 설치 스크립트는 CUDA PyTorch가 실제로 GPU를 잡는지 검사합니다. 기본 설치는 CUDA 12.8 PyTorch wheel을 사용하므로 RTX 30/40/50 계열을 목표로 합니다.

권장 환경:

- Windows 10/11 또는 Linux
- Python 3.11 이상
- NVIDIA GPU
- 최신 NVIDIA 드라이버
- `git`

### 1. 기본 설치

처음 설치할 때는 프로젝트 폴더로 이동한 뒤 가상환경을 만들고, 설치 스크립트를 실행합니다.

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

설치 스크립트가 음악 재생 패키지, AI 오디오 패키지, CUDA PyTorch, Applio RVC 엔진을 함께 준비합니다. FFmpeg는 시스템에 설치되어 있으면 그걸 쓰고, 없으면 `imageio-ffmpeg`의 번들 FFmpeg를 사용합니다.

### 2. CUDA 12.6으로 설치해야 할 때

기본 설치는 CUDA 12.8 PyTorch wheel을 사용합니다. CUDA 12.8 wheel이 드라이버와 맞지 않으면 아래 명령어로 CUDA 12.6 wheel을 설치할 수 있습니다.

Windows PowerShell:

```powershell
.\scripts\install.ps1 -TorchCuda cu126
```

Linux:

```bash
TORCH_CUDA=cu126 bash scripts/install.sh
```

이미 기본 설치 명령어 `.\scripts\install.ps1`이 성공했다면 이 명령어를 따로 실행할 필요는 없습니다.

### 3. Discord 토큰 설정

설치가 끝나면 `.env.example`을 `.env`로 복사하고 Discord 봇 토큰을 넣습니다.

```env
DISCORD_TOKEN=put-your-bot-token-here
GUILD_ID=123456789012345678
```

`GUILD_ID`를 넣으면 해당 서버에 슬래시 명령어가 빠르게 동기화됩니다.

### 설치 스크립트가 하는 일

스크립트가 하는 일:

- 봇 실행에 필요한 전체 `requirements.txt` 설치
- 선택한 CUDA PyTorch wheel 설치
- `torch.cuda.is_available()` 검증
- 봇 환경에 고품질 보컬 분리 엔진(audio-separator Roformer) 설치
- `vendor/Applio`에 Applio RVC 엔진 다운로드
- Applio 전용 가상환경 생성
- Applio 환경에도 선택한 CUDA PyTorch wheel 설치
- Applio 의존성 설치
- `voice_models` 폴더 준비
- `data/recordings` 녹음 데이터 폴더 준비

남는 수동 작업:

- `.env`에 Discord 봇 토큰 넣기
- 직접 준비한 RVC 목소리 모델을 쓰려면 `voice_models/<이름>/`에 넣기
- 직접 학습하려면 Discord 음성 채널에서 `/record_start`로 녹음 후 `/train_voice` 실행

## AI 커버 목소리 모델 넣기

RVC 모델 파일을 아래처럼 넣으면 `/play`의 `target_voice` 자동완성에 뜹니다.

```text
voice_models/
  myvoice/
    myvoice.pth
    myvoice.index
```

`.index` 파일은 있으면 사용하고, 없으면 `.pth`만으로 시도합니다.

## AI 커버 품질 튜닝

AI 커버는 기본적으로 `audio-separator`의 Roformer 보컬 특화 모델을 사용합니다. 이 모델이 실패하면 처리를 중단합니다. 분리 모델과 RVC 추론값은 `.env`에서 조절할 수 있습니다.

```env
VOCAL_SEPARATOR_BACKEND=auto
VOCAL_SEPARATOR_MODEL=mel_band_roformer_kim_ft_unwa.ckpt
RVC_INFER_F0_METHOD=rmvpe
RVC_INFER_INDEX_RATE=0.35
RVC_INFER_VOLUME_ENVELOPE=0.80
RVC_INFER_PROTECT=0.45
AI_COVER_INPUT_VOCAL_FILTER=aresample=48000,highpass=f=40,lowpass=f=19000,loudnorm=I=-17:TP=-2:LRA=10,aresample=48000
AI_COVER_OUTPUT_VOCAL_FILTER=aresample=48000,highpass=f=45,equalizer=f=4500:t=q:w=1.2:g=0.9,deesser=i=0.35:m=0.45:f=0.55,alimiter=limit=0.96
```

AI 듀엣은 `/play_duet`에서 `voice1`과 `voice2`를 고르면 실행됩니다. 두 옵션은 모두 필수이며, 원곡 가수를 그대로 둘 파트는 `원본 가수`를 선택합니다. 기본값은 추가 모델 설치 없이 내장 NMF 분리기를 사용합니다.

별도 multi-singer 모델을 쓰고 싶을 때만 아래 값을 `.env`에 넣습니다. Hugging Face/Asteroid 모델은 `asteroid`를 따로 설치해야 하며, Windows에서는 `pesq` 빌드를 위해 Microsoft C++ Build Tools가 필요할 수 있습니다.

```env
MULTI_SINGER_SEPARATOR_BACKEND=asteroid
MULTI_SINGER_SEPARATOR_MODEL=Cyru5/MedleyVox
MULTI_SINGER_SEPARATOR_DEVICE=cuda
```

`MULTI_SINGER_SEPARATOR_MODEL`에 `*.ckpt` 같은 audio-separator 호환 모델명을 넣으면 audio-separator 방식으로 시도합니다. 별도 연구 repo/스크립트를 쓰는 경우:

```env
MULTI_SINGER_SEPARATOR_COMMAND=python path/to/infer.py --input {input} --output_dir {output_dir}
```

외부 명령은 `{output_dir}` 안에 두 가수의 WAV stem을 만들어야 합니다. 예: `singer1.wav`, `singer2.wav`.

내장 자동 듀엣 분리기는 모델 실행이 실패해도 설정 없이 바로 쓸 수 있는 fallback입니다. 곡마다 두 가수 배정이 완벽하지 않을 수 있으므로, 전용 모델 없이는 듀엣 품질이 원곡의 믹싱 상태에 따라 달라집니다. 전용 모델 없이 실행되는 것을 막고 싶다면 `MULTI_SINGER_SEPARATOR_REQUIRE_MODEL=true`를 설정합니다.

## 목소리 녹음과 자동 학습

음성 채널에 들어간 뒤 `/record_start`를 실행하면 봇이 채널의 발화를 유저별 WAV 파일로 저장합니다. 저장 위치는 `data/recordings/<표시이름_유저ID>/`입니다.

녹음 파일은 48kHz stereo PCM WAV로 저장되고, 한 파일이 약 5분 분량에 도달하면 자동으로 다음 파일로 분할됩니다. 봇이나 음성 수신이 일시적으로 끊겨도 WAV 헤더를 주기적으로 갱신해 손상 가능성을 줄입니다.

녹음을 끝낼 때는 `/record_stop`을 실행합니다. 이후 `/train_voice dataset:<유저 데이터셋> model_name:<목소리이름>`을 실행하면 Applio RVC 학습 파이프라인이 돌아가고, 결과가 `voice_models/<목소리이름>/`에 저장됩니다. 학습이 끝난 모델은 `/play`의 `target_voice`와 `/play_duet`의 `voice1`, `voice2` 자동완성에 바로 표시됩니다.

학습 기본값:

- sample rate: 40000
- pitch/F0: rmvpe
- embedder: contentvec
- epoch: 300
- batch size: 16

필요하면 환경변수로 조절할 수 있습니다.

```env
RVC_TRAIN_EPOCHS=300
RVC_TRAIN_BATCH_SIZE=16
RVC_TRAIN_CPU_CORES=12
```

## 실행

가상환경이 켜진 상태에서 봇을 실행합니다.

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

이미 가상환경이 켜져 있다면 `python bot.py`만 실행하면 됩니다.

## 명령어 예시

```text
/play query:https://youtu.be/... mode:원본 재생
/play query:artist song title mode:원본 재생
/play query:artist song title mode:MR만 재생
/play query:https://youtu.be/... target_voice:myvoice vocal_pitch_shift:+6
/play_file file:<업로드 파일> mode:원본 재생
/play_file file:<업로드 파일> target_voice:myvoice vocal_pitch_shift:+6
/play_duet query:artist duet song voice1:원본 가수 voice2:voice2
/play_duet query:artist duet song voice1:voice1 voice2:voice2
/record_start
/record_status
/record_stop
/train_voice dataset:myname_123456789012345678 model_name:myvoice
/playlist url:https://youtube.com/playlist?... max_count:50
/loop mode:전체 반복
/normalizer enabled:True
/remove position:3
/lyrics
/lyrics query:artist song title
```

## 설계 메모

- `/play` 안에서 일반 재생, MR/보컬 분리, AI 커버를 다룹니다.
- `/play_file`은 유튜브 다운로드 대신 Discord 첨부 파일을 저장한 뒤 `/play`와 같은 오디오 처리 파이프라인을 사용합니다.
- `/play_duet`은 AI 듀엣 커버만 다룹니다.
- AI 커버는 `mode`가 아니라 `target_voice`에 목소리 모델을 고르면 자동으로 적용됩니다. `target_voice`의 `원본 가수`는 AI 변환을 하지 않는 선택지입니다.
- 일반 재생은 스트리밍 방식이라 다운로드 파일을 남기지 않습니다.
- 보컬 분리는 Roformer 기반 `audio-separator`를 사용하고, 실패하면 중단합니다.
- AI 커버는 Applio 환경의 CUDA PyTorch를 사용합니다.
- 목소리 녹음은 `discord-ext-voice-recv`로 수신하고, 음악 재생 연결과 같은 voice client를 공유합니다.
- 학습 데이터는 `data/recordings`에, 완성 모델은 기존 AI 커버와 같은 `voice_models`에 저장합니다.
- AI 듀엣은 기본 multi-singer separator 또는 fallback 분리기로 두 가수 stem을 자동 분리한 뒤 각각 변환합니다.
- AI 처리 결과물은 `data/processed`에 저장되고, 재생이 끝나거나 대기열에서 삭제되면 정리됩니다.
- 플레이리스트는 목록만 먼저 가져오고, 실제 스트림 URL은 재생 직전에 다시 가져와 만료 문제를 줄입니다.
- 노멀라이저는 현재 곡 다음부터 적용됩니다.
