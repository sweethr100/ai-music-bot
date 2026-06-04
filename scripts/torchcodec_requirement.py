from __future__ import annotations

import sys

import torch


def main() -> None:
    version = torch.__version__.split("+", 1)[0].split(".")
    major, minor = int(version[0]), int(version[1])

    if sys.version_info >= (3, 13) and (major, minor) < (2, 8):
        raise SystemExit(
            "Python 3.13에서 TorchCodec을 쓰려면 torch 2.8 이상이 필요합니다. "
            f"현재 torch는 {torch.__version__}입니다. cu128로 다시 설치하거나 Python 3.12 venv를 사용해 주세요."
        )

    if (major, minor) >= (2, 11):
        print("torchcodec")
    elif (major, minor) == (2, 10):
        print("torchcodec==0.10.*")
    elif (major, minor) == (2, 9):
        print("torchcodec==0.9.*")
    elif (major, minor) == (2, 8):
        print("torchcodec==0.7.*")
    elif (major, minor) == (2, 7):
        print("torchcodec==0.5.*")
    else:
        print("torchcodec==0.2.*")


if __name__ == "__main__":
    main()
