# 自動試験の実行記録

## 実行条件

- 実行日：2026年9月1日（日本標準時）
- Python：3.10.11
- EvoGym：2.0.0
- Gymnasium：1.3.0
- Stable-Baselines3：2.9.0
- pytest：9.1.1
- 対象：`setsu_05_general_obstacle_student/tests`
- 学習ステップ：0
- 重み更新：0
- 教師読込：なし
- 教師介入：0

## 実行方法

`setsu_05_general_obstacle_student` を作業ディレクトリとし、同階層の `setsu_04_long_legged_hierarchical` と現在のプロジェクトをPython検索パスの先頭へ追加した上で、次と同等のコマンドを実行した。

```powershell
$longLegPath = (Resolve-Path ..\setsu_04_long_legged_hierarchical).Path
$env:PYTHONPATH = "$(Get-Location);$longLegPath"
python -m pytest -q --disable-warnings
```

## 実際の出力

```text
........................................................................ [100%]Using Evolution Gym Simulator v2.2.5

72 passed, 1 warning in 2.37s
```

警告一件はEvoGym依存関係の非推奨APIに関するものであり、試験失敗ではない。`tests/conftest.py` は、提出物から意図的に除外した大規模中間軌跡と全チェックポイントを必要とする履歴試験だけを収集対象外にする。上記72件は、提出物だけで実際に収集、実行、合格した独立試験である。
