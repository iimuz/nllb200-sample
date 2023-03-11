"""NLLB200を利用した翻訳."""
import logging
import sys
from argparse import ArgumentParser
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
from pathlib import Path

import torch
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_logger = logging.getLogger(__name__)


class _RunConfig(BaseModel):
    # スクリプト実行のためのオプション.

    data_dir: Path  # モデルファイルやログファイルなどの記録場所
    verbose: int  # ログレベル


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

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", cache_dir=external_dir
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M", cache_dir=external_dir
    )
    if torch.cuda.is_available():
        # model = model.to("mps")
        model = model.to("cuda:0")

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
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.lang_code_to_id["jpn_Jpan"],
        max_length=100,
    )
    result = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"result: {result}")
    _logger.info("result: %s", result)


def _parse_args() -> _RunConfig:
    # スクリプト実行のための引数を読み込む.
    parser = ArgumentParser(description="NLLB200を利用した日英翻訳.")

    parser.add_argument(
        "--data-dir",
        default=(Path(__file__).parents[1] / "data").resolve(),
        help="モデルファイルやログファイルの記録場所のルートパス.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="詳細メッセージのレベルを設定."
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
