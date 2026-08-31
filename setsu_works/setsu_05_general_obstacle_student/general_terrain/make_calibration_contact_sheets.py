"""既存GIFから目視較正用の等間隔コンタクトシートを作成する。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "calibration" / "contact_sheets"
SOURCES = {
    "upright_success": REPOSITORY_ROOT
    / "setsu_04_long_legged_hierarchical"
    / "media"
    / "level2_composite_success.gif",
    "side_fall_false_positive": REPOSITORY_ROOT
    / "setsu_03_fixed_obstacle_curriculum"
    / "media"
    / "08_step_045000_success.gif",
    "roll_then_recover": REPOSITORY_ROOT
    / "setsu_03_fixed_obstacle_curriculum"
    / "media"
    / "09_step_050000_regression.gif",
}


def make_sheet(name: str, source: Path, sample_count: int = 12) -> Path:
    """一GIFから十二時点を抽出し四列の一覧画像を保存する。"""
    if not source.exists():
        raise FileNotFoundError(f"再生GIFが見つからない：{source}")
    with Image.open(source) as gif:
        frame_count = int(getattr(gif, "n_frames", 1))
        indices = [
            round(index * (frame_count - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
        frames = []
        for frame_index in indices:
            gif.seek(frame_index)
            frame = gif.convert("RGB")
            frame.thumbnail((320, 220))
            canvas = Image.new("RGB", (330, 250), "white")
            canvas.paste(frame, ((330 - frame.width) // 2, 22))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), f"{name} | frame {frame_index}/{frame_count - 1}", fill="black")
            frames.append(canvas)
    sheet = Image.new("RGB", (330 * 4, 250 * 3), (235, 235, 235))
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % 4) * 330, (index // 4) * 250))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{name}.png"
    sheet.save(output)
    return output


def main() -> None:
    """三種類の目視比較画像をまとめて生成する。"""
    for name, source in SOURCES.items():
        print(make_sheet(name, source))


if __name__ == "__main__":
    main()
