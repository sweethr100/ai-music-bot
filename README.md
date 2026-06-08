# AI Music Discord Bot

음악 재생과 AI 커버는 `/play`, AI 듀엣 커버는 `/play_duet`으로 쓰는 Discord 음악 봇입니다.

## 지원 기능

- `/play` 유튜브 URL/검색어 원본 재생
- `/play` MR만 재생, 보컬만 재생, 보컬 강조 믹스
- `/play` AI 커버 생성 및 재생
- `/play_file` 유저가 업로드한 파일로 원본/MR/보컬/AI 커버 생성 및 재생
- `/play_duet` AI 듀엣 생성 및 재생
- `/separate_singers` 가수별 보컬 stem MP3 분리 테스트
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
RVC_INFER_INDEX_RATE=0.25
RVC_INFER_VOLUME_ENVELOPE=0.75
RVC_INFER_PROTECT=0.50
AI_COVER_INPUT_VOCAL_FILTER=none
AI_COVER_OUTPUT_VOCAL_FILTER=none
```

AI 듀엣은 `/play_duet`에서 곡만 넣으면 먼저 PyAnnote Audio로 “몇 분 몇 초에 누가 부르는지” 시간표를 만들고, 분리된 1번/2번 보컬 stem 미리듣기 파일을 채팅에 올립니다. 미리 들어본 뒤 설정 패널에서 각 파트의 목소리, 피치, 볼륨을 고르고 `듀엣 렌더`를 누르면 최종 파일을 만듭니다.

PyAnnote Audio는 기본 설치에 포함됩니다.

```powershell
.\scripts\install.ps1
```

PyAnnote 공식 모델을 쓰려면 Hugging Face에서 모델 사용 조건에 동의하고 로그인해야 합니다. 기본 모델은 Community-1입니다.

1. Hugging Face 계정을 만듭니다: https://huggingface.co/join
2. `pyannote/speaker-diarization-community-1` 페이지에서 사용 조건에 동의합니다: https://huggingface.co/pyannote/speaker-diarization-community-1
3. Hugging Face 토큰을 만듭니다: https://huggingface.co/settings/tokens
   - fine-grained token을 만들면 `Read access to contents of all public gated repos you can access` 권한을 켭니다.
   - 헷갈리면 classic `read` 토큰을 만들어도 됩니다.
4. PowerShell에서 로그인합니다.

```powershell
huggingface-cli login
huggingface-cli whoami
```

`login` 명령어가 토큰을 물어보면 3번에서 만든 토큰을 붙여넣습니다. `whoami`가 내 Hugging Face 계정을 출력하면 로그인 캐시가 준비된 상태입니다. 이미 봇이 켜져 있었다면 봇을 재시작해 주세요. 로그인 캐시를 쓰므로 `.env`에 토큰을 넣을 필요는 없습니다.

`huggingface-cli`가 없거나 deprecated 오류가 나면 프로젝트 venv의 최신 CLI를 사용합니다.

```powershell
.\.venv\Scripts\hf.exe auth login --force
.\.venv\Scripts\hf.exe auth whoami
```

403 또는 `not in the authorized list`가 뜨면 Community-1 모델 페이지에서 사용 조건 동의가 아직 처리되지 않은 상태입니다. 같은 계정으로 https://huggingface.co/pyannote/speaker-diarization-community-1 에 들어가 조건 동의/접근 요청을 완료한 뒤 다시 실행합니다.

`cannot find the requested files in the local cache` 또는 `Please check your connection`이 뜨면 모델이 아직 캐시에 없는데 Hugging Face 연결이 실패한 상태입니다. 먼저 오프라인 모드가 켜져 있지 않은지 확인하고, 모델 다운로드가 되는지 테스트합니다.

```powershell
Get-ChildItem Env:HF_HUB_OFFLINE
Remove-Item Env:HF_HUB_OFFLINE
huggingface-cli whoami
huggingface-cli download pyannote/speaker-diarization-community-1 config.yaml
```

`huggingface-cli download`가 동작하지 않으면 대체 명령을 사용합니다.

```powershell
.\.venv\Scripts\hf.exe auth whoami
.\.venv\Scripts\hf.exe download pyannote/speaker-diarization-community-1 config.yaml
```

`Invalid user token`이 뜨면 저장된 토큰이 만료되었거나 잘못된 상태입니다. 권한을 켠 새 토큰으로 `huggingface-cli login` 또는 `.\.venv\Scripts\hf.exe auth login --force`를 다시 실행합니다.

`Please enable access to public gated repositories in your fine-grained token settings`가 뜨면 토큰 권한 문제입니다. https://huggingface.co/settings/tokens 에서 현재 토큰의 gated repo read 권한을 켜거나, 권한을 켠 새 토큰으로 다시 로그인합니다.

기본 백엔드는 다음과 같습니다.

```env
MULTI_SINGER_SEPARATOR_BACKEND=pyannote
PYANNOTE_DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
PYANNOTE_DEVICE=cuda
```

PyAnnote 없이 로컬 방식만 쓰고 싶으면:

```env
MULTI_SINGER_SEPARATOR_BACKEND=local_diarization
```

이 방식들은 실제 source separation이 아니라 diarization 기반 시간 절단 방식입니다. 두 사람이 동시에 부르는 화음/후렴은 완전 분리하지 못하고, 파트가 번갈아 나오는 듀엣에서 가장 잘 맞습니다. 기본값은 PyAnnote가 구분한 가수 구간을 유지하고, 피치는 `1번/2번` 파트 순서를 정렬하는 데만 사용합니다.

```env
DUET_PITCH_REFINE=true
DUET_PITCH_REFINE_MODE=order
DUET_PITCH_PART_ORDER=low_first
DUET_PITCH_MIN_CLUSTER_SEMITONES=3.0
```

특정 곡에서 파트 순서를 반대로 쓰고 싶으면 `DUET_PITCH_PART_ORDER=high_first`로 바꿉니다. 남녀 듀엣처럼 음역 차이가 아주 크고 PyAnnote가 남녀를 섞어 잡는 곡에서는 `DUET_PITCH_REFINE_MODE=segment`를 켜면 구간 자체를 낮은 음역/높은 음역 기준으로 다시 나눕니다.

다른 multi-singer 모델을 실험하고 싶을 때만 아래 값을 `.env`에 넣습니다.

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

PyAnnote가 실패하면 듀엣 처리는 중단됩니다. 곡마다 두 가수 배정이 완벽하지 않을 수 있으므로, 듀엣 품질은 원곡의 믹싱 상태와 두 보컬의 음색 차이에 따라 달라집니다.

## 목소리 녹음과 자동 학습

음성 채널에 들어간 뒤 `/record_start`를 실행하면 봇이 채널의 발화를 유저별 WAV 파일로 저장합니다. 저장 위치는 `data/recordings/<표시이름_유저ID>/`입니다.

녹음 파일은 48kHz stereo PCM WAV로 저장되고, 한 파일이 약 5분 분량에 도달하면 자동으로 다음 파일로 분할됩니다. 봇이나 음성 수신이 일시적으로 끊겨도 WAV 헤더를 주기적으로 갱신해 손상 가능성을 줄입니다.

녹음을 끝낼 때는 `/record_stop`을 실행합니다. 이후 `/train_voice dataset:<유저 데이터셋> model_name:<목소리이름>`을 실행하면 Applio RVC 학습 파이프라인이 돌아가고, 결과가 `voice_models/<목소리이름>/`에 저장됩니다. 학습이 끝난 모델은 `/play`의 `target_voice`와 `/play_duet` 설정 패널의 목소리 선택에 바로 표시됩니다.

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
/play_duet query:artist duet song
/separate_singers query:artist duet song
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
- `/separate_singers`는 커버 변환 없이 가수별 보컬 stem만 각각 MP3로 올려 분리 품질을 확인합니다.
- AI 커버는 `mode`가 아니라 `target_voice`에 목소리 모델을 고르면 자동으로 적용됩니다. `target_voice`의 `원본 가수`는 AI 변환을 하지 않는 선택지입니다.
- 일반 재생은 스트리밍 방식이라 다운로드 파일을 남기지 않습니다.
- 보컬 분리는 Roformer 기반 `audio-separator`를 사용하고, 실패하면 중단합니다.
- AI 커버는 Applio 환경의 CUDA PyTorch를 사용합니다.
- 목소리 녹음은 `discord-ext-voice-recv`로 수신하고, 음악 재생 연결과 같은 voice client를 공유합니다.
- 학습 데이터는 `data/recordings`에, 완성 모델은 기존 AI 커버와 같은 `voice_models`에 저장합니다.
- AI 듀엣은 PyAnnote 또는 로컬 시간표 분리기로 두 가수 stem을 자동 분리한 뒤 각각 변환합니다.
- AI 처리 결과물은 `data/processed`에 저장되고, 재생이 끝나거나 대기열에서 삭제되면 정리됩니다.
- 플레이리스트는 목록만 먼저 가져오고, 실제 스트림 URL은 재생 직전에 다시 가져와 만료 문제를 줄입니다.
- 노멀라이저는 현재 곡 다음부터 적용됩니다.
