"""Compare complete Stage 03 A/B runs and write the required result table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.presence.labels import CLASS_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--conditioned-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/results/03_presence_conditioning_results.md"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh report path")
    config_a = json.loads((args.control_dir / "config.json").read_text(encoding="utf-8"))
    config_b = json.loads((args.conditioned_dir / "config.json").read_text(encoding="utf-8"))
    ignored = {"arm", "output_dir", "pretrained_dir", "pretrained_path", "cache_dir", "hf_endpoint"}
    for key in config_a.keys() | config_b.keys():
        if key not in ignored and config_a.get(key) != config_b.get(key):
            raise ValueError(f"A/B training protocol differs at {key}")
    if (config_a["arm"], config_b["arm"]) != ("control", "conditioned"):
        raise ValueError("Expected control and conditioned run directories")
    metrics_a = json.loads((args.control_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics_b = json.loads((args.conditioned_dir / "metrics.json").read_text(encoding="utf-8"))

    def row(label: str, a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return f"| {label} | {a} | {b} | n/a |"
        return f"| {label} | {a:.4f} | {b:.4f} | {b-a:+.4f} |"

    lines = [
        "# 03｜Presence Conditioning 结果", "",
        f"Control: `{args.control_dir}`；Conditioned: `{args.conditioned_dir}`。", "",
        "两组仅差 Presence Conditioning 开关；checkpoint 只按 val_stratified mIoU 选择。",
        "Train Presence probability is in-sample prediction. 两组条件输入均为阶段 02 预测概率。", "",
    ]
    for split in ("val_stratified", "val_domain"):
        a, b = metrics_a[split], metrics_b[split]
        lines += [f"## {split}", "", "| Metric | Clean B3 | B3 + Presence | Delta |", "| --- | ---: | ---: | ---: |",
                  row("mIoU", a["miou"], b["miou"])]
        lines += [row(f"{name} IoU", a["class_iou"][name], b["class_iou"][name]) for name in CLASS_NAMES]
        lines.append("")
    delta_strat = metrics_b["val_stratified"]["miou"] - metrics_a["val_stratified"]["miou"]
    delta_domain = metrics_b["val_domain"]["miou"] - metrics_a["val_domain"]["miou"]
    if delta_strat > 0 and delta_domain > 0:
        direction = "两个验证集均为正向差值"
    elif delta_strat <= 0 and delta_domain < 0:
        direction = "两个验证集均未支持提升"
    else:
        direction = "两个验证集方向不一致或均接近零"
    lines += ["## 判断", "", f"{direction}。Δ val_stratified={delta_strat:+.4f}；Δ val_domain={delta_domain:+.4f}。", "",
              "最终结论（提升 / 无提升 / 证据不足）：待结合差值大小、逐类 IoU 和训练波动填写。", "",
              "历史 SegFormer-B3 成绩属于其他训练协议，不用于本消融判断。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
