"""救援GIFから時間軸を均等抽出した監査用接触表を作成する。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_contact_sheet(
    gif_path: Path,
    output_path: Path,
    *,
    sample_count: int = 12,
    columns: int = 4,
) -> None:
    """GIF全体から代表フレームを選びPNGへ整列する。"""
    if sample_count < 1 or columns < 1:
        raise ValueError("抽出数と列数は一以上でなければならない。")
    with Image.open(gif_path) as source:
        frame_count = int(getattr(source, "n_frames", 1))
        indices = np.linspace(
            0,
            frame_count - 1,
            num=min(sample_count, frame_count),
            dtype=int,
        )
        frames = []
        for index in indices:
            source.seek(int(index))
            frames.append(source.convert("RGB").copy())
    width, height = frames[0].size
    label_height = 28
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new(
        "RGB",
        (columns * width, rows * (height + label_height)),
        color="white",
    )
    draw = ImageDraw.Draw(sheet)
    for slot, (index, frame) in enumerate(zip(indices, frames)):
        x = (slot % columns) * width
        y = (slot // columns) * (height + label_height)
        sheet.paste(frame, (x, y))
        draw.text((x + 8, y + height + 6), f"frame {int(index)}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    """指定GIFを一枚の監査用PNGへ変換する。"""
    parser = argparse.ArgumentParser(description="救援GIFの接触表を作成する。")
    parser.add_argument("--gif", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    make_contact_sheet(
        Path(args.gif),
        Path(args.output),
        sample_count=args.samples,
        columns=args.columns,
    )


if __name__ == "__main__":
    main()
