"""学生プレフィックス救援リセット地点の凍結目録を読み込み検査する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from general_terrain.seed_manifest import load_seed_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESCUE_RESET_MANIFEST = (
    PROJECT_ROOT / "config" / "m2_2_rescue_reset_manifest_v1.json"
)


@dataclass(frozen=True)
class RescueResetSpec:
    """一つの訓練コースと凍結済みの引き継ぎ状態を保持する。"""

    seed: int
    start_runway_voxels: int
    prefix_steps: int
    trigger_reason: str
    x_position: float
    orientation_error: float
    angular_velocity: float
    stall_steps: int
    raw_clearances: int
    recovered_obstacles: int


@dataclass(frozen=True)
class RescueResetManifest:
    """凍結済みの救援リセット地点と出典検証情報を保持する。"""

    version: str
    stage: str
    split: str
    source_profile: str
    source_summary: Path
    source_summary_sha256: str
    student_model_sha256: str
    states: tuple[RescueResetSpec, ...]
    source_path: Path
    sha256: str

    def state_for_seed(self, seed: int) -> RescueResetSpec:
        """指定した訓練用乱数シードのリセット地点を返す。"""
        for state in self.states:
            if state.seed == seed:
                return state
        raise ValueError(f"救援重置目録にない乱数種: {seed}")

    def as_dict(self) -> dict[str, object]:
        """監査結果へ保存できる辞書を返す。"""
        return {
            "version": self.version,
            "stage": self.stage,
            "split": self.split,
            "source_profile": self.source_profile,
            "source_summary": str(self.source_summary.resolve()),
            "source_summary_sha256": self.source_summary_sha256,
            "student_model_sha256": self.student_model_sha256,
            "source_path": str(self.source_path.resolve()),
            "sha256": self.sha256,
            "states": [asdict(state) for state in self.states],
        }


def _sha256(path: Path) -> str:
    """ファイル全体のSHA-256を小文字で返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rescue_reset_manifest(
    path: Path = DEFAULT_RESCUE_RESET_MANIFEST,
) -> RescueResetManifest:
    """救援リセット目録と出典を読み込み、件数、区分、ハッシュを検査する。"""
    resolved = Path(path).resolve()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not bool(raw.get("frozen", False)):
        raise ValueError("救援重置目録は凍結済みでなければならない。")
    if str(raw["split"]) != "train":
        raise ValueError("救援重置目録は訓練区分だけを使用できる。")
    source_summary = PROJECT_ROOT / str(raw["source_summary"])
    expected_source_hash = str(raw["source_summary_sha256"]).lower()
    if _sha256(source_summary) != expected_source_hash:
        raise ValueError("救援重置目録の出典要約ハッシュが一致しない。")
    states = tuple(RescueResetSpec(**item) for item in raw["states"])
    frozen_seeds = load_seed_manifest().for_split("train")
    if tuple(state.seed for state in states) != frozen_seeds:
        raise ValueError("救援重置点は凍結訓練乱数種順と一致しなければならない。")
    if len(states) != 11:
        raise ValueError("救援重置点は十一個でなければならない。")
    if {state.start_runway_voxels for state in states} != set(range(20, 31)):
        raise ValueError("救援重置点は開始位置二十から三十を各一回含まねばならない。")
    if any(state.prefix_steps < 1 for state in states):
        raise ValueError("学生前缀歩数は一以上でなければならない。")
    return RescueResetManifest(
        version=str(raw["version"]),
        stage=str(raw["stage"]),
        split=str(raw["split"]),
        source_profile=str(raw["source_profile"]),
        source_summary=source_summary,
        source_summary_sha256=expected_source_hash,
        student_model_sha256=str(raw["student_model_sha256"]).lower(),
        states=states,
        source_path=resolved,
        sha256=_sha256(resolved),
    )
