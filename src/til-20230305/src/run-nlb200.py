"""NLLB200を利用した翻訳."""
import logging
import sys
from argparse import ArgumentParser
from enum import Enum
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_logger = logging.getLogger(__name__)


class _AvailableDeviceName(Enum):
    # 利用可能なデバイス名一覧.

    CPU: str = "cpu"
    CUDA: str = "cuda"
    MPS: str = "mps"


class _RunConfig(BaseModel):
    # スクリプト実行のためのオプション.

    device_name: str  # cpu, cuda, mps の選択肢

    data_dir: Path  # モデルファイルやログファイルなどの記録場所
    verbose: int  # ログレベル


def _get_device(selected_device: _AvailableDeviceName) -> torch.device:
    # 指定したデバイスが利用できるか判定して、利用できる場合のみデバイス情報を返す.
    if _AvailableDeviceName.CPU == selected_device:
        return torch.device("cpu")

    if _AvailableDeviceName.CUDA == selected_device:
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available.")
        return torch.device("cuda:0")

    if _AvailableDeviceName.MPS == selected_device:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                raise ValueError(
                    "MPS not available because the current PyTorch install was not"
                    " built with MPS enabled."
                )
            else:
                raise ValueError(
                    "MPS not available because the current MacOS version is not 12.3+"
                    " and/or you do not have an MPS-enabled device on this machine."
                )
        return torch.device("mps")

    raise ValueError(f"Unknown device name: {selected_device}")


def _main() -> None:
    # スクリプトのエントリポイント.

    # 実行時引数の読み込み
    config = _parse_args()
    print(config.data_dir)

    # 保存場所の初期化
    interim_dir = config.data_dir / "interim"
    interim_dir.mkdir(exist_ok=True)
    external_dir = config.data_dir / "external"
    external_dir.mkdir(exist_ok=True)

    # ログ設定
    loglevel = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }.get(config.verbose, logging.DEBUG)
    _setup_logger(filepath=(interim_dir / "log.txt"), loglevel=loglevel)
    _logger.info(config)

    # デバイスの設定
    device_info = _get_device(_AvailableDeviceName(config.device_name))

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", cache_dir=external_dir
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M", cache_dir=external_dir
    )
    model = model.to(device_info)

    article = """
    I watched a lot of interesting animation last week.
    I heard it was nice today, but it didn't rain!
    I saw a lot of interesting animation last week,
    but this week I couldn't watch any of it.
    Hopefully I'll be able to watch it next week...
    """
    _logger.info("input buffer: %s", article)
    inputs = tokenizer(article, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs.to(device_info),
        forced_bos_token_id=tokenizer.lang_code_to_id["jpn_Jpan"],
        max_length=100,
    )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"result: {result}")
    _logger.info("result: %s", result)


def _parse_args() -> _RunConfig:
    # スクリプト実行のための引数を読み込む.
    parser = ArgumentParser(description="Translation using NLLB200.")

    parser.add_argument(
        "--device-name",
        default=_AvailableDeviceName.CPU.value,
        choices=[v.value for v in _AvailableDeviceName],
        type=str,
        help="Select the device to be used.",
    )

    parser.add_argument(
        "--data-dir",
        default=(Path(__file__).parents[1] / "data").resolve(),
        help="Root path of where model files nad log files are saved.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Set the log level for detailed messages.",
    )

    args = parser.parse_args()
    config = _RunConfig(**vars(args))

    return config


def _setup_logger(
    filepath: Path | None,  # ログ出力するファイルパス. Noneの場合はファイル出力しない.
    loglevel: int,  # 出力するログレベル
) -> None:
    # ログ出力設定
    # ファイル出力とコンソール出力を行うように設定する。
    _logger.setLevel(loglevel)

    # consoleログ
    console_handler = StreamHandler(stream=sys.stdout)
    console_handler.setLevel(loglevel)
    console_handler.setFormatter(
        Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
    )
    _logger.addHandler(console_handler)

    # ファイル出力するログ
    # 基本的に大量に利用することを想定していないので、ログファイルは多くは残さない。
    if filepath is not None:
        file_handler = RotatingFileHandler(
            filepath,
            encoding="utf-8",
            mode="a",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=1,
        )
        # ファイル出力のログレベルは最低でもINFOとする。
        # debug出力の時はdebugレベルまで出力するようにする。
        file_loglevel = loglevel if loglevel <= logging.INFO else logging.INFO
        file_handler.setLevel(file_loglevel)
        file_handler.setFormatter(
            Formatter("[%(levelname)7s] %(asctime)s (%(name)s) %(message)s")
        )
        _logger.addHandler(file_handler)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        _logger.exception(e)
        sys.exit(1)
