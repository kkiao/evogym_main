"""訓練、検証、留保評価の固定乱数種目録を読み取り検査する。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_MANIFEST = PROJECT_ROOT / "config" / "seed_manifest_v1.json"
REQUIRED_SPLITS = ("train", "validation", "holdout")


@dataclass(frozen=True)
class SeedManifest:
    """相互に重ならない固定乱数種集合と対象段階を保持する。"""

    version: str
    stage: str
    seeds: dict[str, tuple[int, ...]]
    source_path: Path
    sha256: str

    def for_split(self, split: str) -> tuple[int, ...]:
        """指定区分の固定乱数種を返す。"""
        try:
            return self.seeds[split]
        except KeyError as error:
            raise ValueError(f"未知の乱数種区分: {split}") from error

    def as_dict(self) -> dict[str, object]:
        """監査記録へ保存できる辞書形式を返す。"""
        return {
            "version": self.version,
            "stage": self.stage,
            "source_path": str(self.source_path.resolve()),
            "sha256": self.sha256,
            "splits": {
                split: list(values) for split, values in self.seeds.items()
            },
        }


def load_seed_manifest(path: Path = DEFAULT_SEED_MANIFEST) -> SeedManifest:
    """JSON目録を読み、重複、欠落、型違反を拒否する。"""
    resolved = Path(path).resolve()
    encoded = resolved.read_bytes()
    raw = json.loads(encoded.decode("utf-8"))
    if not bool(raw.get("frozen", False)):
        raise ValueError("乱数種目録は凍結済みでなければならない。")
    version = str(raw["version"])
    stage = str(raw["stage"])
    raw_splits = raw["splits"]
    seeds: dict[str, tuple[int, ...]] = {}
    for split in REQUIRED_SPLITS:
        if split not in raw_splits:
            raise ValueError(f"乱数種目録に区分がない: {split}")
        values = tuple(int(value) for value in raw_splits[split]["seeds"])
        if len(values) != 11:
            raise ValueError(f"区分{split}の乱数種数は11でなければならない。")
        if len(set(values)) != len(values):
            raise ValueError(f"区分{split}内に重複乱数種がある。")
        seeds[split] = values
    all_values = [value for split in REQUIRED_SPLITS for value in seeds[split]]
    if len(set(all_values)) != len(all_values):
        raise ValueError("訓練、検証、留保区分の乱数種が重複している。")
    return SeedManifest(
        version=version,
        stage=stage,
        seeds=seeds,
        source_path=resolved,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
