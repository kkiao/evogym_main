"""GIFから等間隔フレームを抽出して視覚検査用一覧画像を作る。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageSequence


def main():
    parser = argparse.ArgumentParser(description="生成GIF视觉验收联系表。")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    source = Image.open(args.input)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(source)]
    indices = [
        round(index * (len(frames) - 1) / max(args.samples - 1, 1))
        for index in range(args.samples)
    ]
    selected = [frames[index] for index in indices]
    width, height = selected[0].size
    label_height = 24
    rows = math.ceil(len(selected) / args.columns)
    sheet = Image.new(
        "RGB",
        (args.columns * width, rows * (height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for slot, (index, frame) in enumerate(zip(indices, selected)):
        x = (slot % args.columns) * width
        y = (slot // args.columns) * (height + label_height)
        sheet.paste(frame, (x, y + label_height))
        draw.text((x + 6, y + 4), f"frame {index}/{len(frames) - 1}", fill="black")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
