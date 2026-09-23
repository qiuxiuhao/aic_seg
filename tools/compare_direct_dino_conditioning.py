"""Write the Stage 04 A/B/C report from existing Stage 03 and Direct DINO outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.presence.labels import CLASS_NAMES


def percent(value: float | None) -> str:
    return "待正式训练" if value is None else f"{100 * value:.4f}"


def delta(value: float | None, reference: float) -> str:
    return "待正式训练" if value is None else f"{100 * (value - reference):+.4f} pp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-dir", type=Path,
        default=Path("outputs/presence_conditioning/control_b3_bs4"),
    )
    parser.add_argument(
        "--presence-dir", type=Path,
        default=Path("outputs/presence_conditioning/conditioned_b3_bs4"),
    )
    parser.add_argument(
        "--direct-dir", type=Path,
        default=Path("outputs/direct_dino_conditioning/direct_dino_b3_bs4"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/04_direct_dino_conditioning_results.md"),
    )
    args = parser.parse_args()
    clean = json.loads((args.control_dir / "metrics.json").read_text(encoding="utf-8"))
    presence = json.loads((args.presence_dir / "metrics.json").read_text(encoding="utf-8"))
    if (clean.get("arm"), presence.get("arm")) != ("control", "conditioned"):
        raise ValueError("Expected Stage 03 control and conditioned metrics")

    direct_metrics_path = args.direct_dir / "metrics.json"
    direct_config_path = args.direct_dir / "config.json"
    direct_history_path = args.direct_dir / "history.json"
    direct = None
    direct_config = None
    history = None
    if direct_metrics_path.is_file():
        if not direct_config_path.is_file() or not direct_history_path.is_file():
            raise FileNotFoundError("Direct DINO formal result is missing config.json or history.json")
        direct = json.loads(direct_metrics_path.read_text(encoding="utf-8"))
        direct_config = json.loads(direct_config_path.read_text(encoding="utf-8"))
        history = json.loads(direct_history_path.read_text(encoding="utf-8"))
        if direct.get("arm") != "direct_dino" or direct_config.get("arm") != "direct_dino":
            raise ValueError("Direct result directory is not arm=direct_dino")
        expected = {
            "seed": 42, "max_steps": 30000, "eval_every": 2000, "batch_size": 4,
            "grad_accum": 1, "learning_rate": 6e-5, "weight_decay": 0.01,
            "amp": True, "input_size": [1024, 1024], "num_classes": 8,
            "ignore_index": 255, "conditioning_type": "direct_dino",
            "dino_input_dimension": 1536, "film_channels": 768,
        }
        for key, expected_value in expected.items():
            if direct_config.get(key) != expected_value:
                raise ValueError(
                    f"Direct DINO formal protocol differs at {key}: "
                    f"{direct_config.get(key)!r} != {expected_value!r}"
                )

    lines = [
        "# 04｜Direct DINO Conditioning 结果", "",
        f"- Clean B3：`{args.control_dir}`",
        f"- Presence Conditioning：`{args.presence_dir}`",
        f"- Direct DINO Conditioning：`{args.direct_dir}`", "",
    ]
    if direct is None:
        lines += [
            "> 状态：实现与 smoke 已完成；当前环境无 CUDA，Direct DINO 30,000-step 正式训练尚未执行。以下 C 列与 Delta 保留为待正式训练，不填入推测值。",
            "",
        ]
    lines += [
        "## 汇总", "",
        "| Metric | Clean B3 | + Presence | + Direct DINO |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("val_stratified", "val_domain"):
        direct_miou = None if direct is None else direct[split]["miou"]
        lines.append(
            f"| {split} mIoU | {percent(clean[split]['miou'])} | "
            f"{percent(presence[split]['miou'])} | {percent(direct_miou)} |"
        )
    lines += [
        "", "## 8 类 IoU", "",
        "| Split | Class | Clean B3 | + Presence | + Direct DINO | Direct - Clean | Direct - Presence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("val_stratified", "val_domain"):
        for name in CLASS_NAMES:
            a = clean[split]["class_iou"][name]
            b = presence[split]["class_iou"][name]
            c = None if direct is None else direct[split]["class_iou"][name]
            lines.append(
                f"| {split} | {name} | {percent(a)} | {percent(b)} | {percent(c)} | "
                f"{delta(c, a)} | {delta(c, b)} |"
            )

    lines += ["", "## Delta", ""]
    for split in ("val_stratified", "val_domain"):
        c = None if direct is None else direct[split]["miou"]
        lines += [
            f"- {split}，Direct DINO - Clean B3：{delta(c, clean[split]['miou'])}",
            f"- {split}，Direct DINO - Presence：{delta(c, presence[split]['miou'])}",
        ]
    lines += [
        "", "## 结构与参数量", "",
        "```text",
        "Presence Conditioning:    7 → 64 → 1536",
        "Direct DINO Conditioning: 1536 → 256 → 1536",
        "```", "",
        "- Presence Conditioning FiLM 参数量：100,352。",
        "- Direct DINO Conditioning FiLM 参数量：791,296。",
        "- Direct DINO 与 Presence 的差异同时包含更丰富的输入表示和更大的 Conditioning 容量。",
        "- 参数量差异不用于否定比赛收益；本对比不能解释为严格的表示优劣因果证明。",
        "",
        "## Best step 与 validation history", "",
        f"- Clean B3 best step：{clean['best_step']}。",
        f"- Presence best step：{presence['best_step']}。",
        f"- Direct DINO best step：{'待正式训练' if direct is None else direct['best_step']}。",
        "",
    ]
    if history is not None:
        lines += [
            "| Step | Train loss | val_stratified mIoU |",
            "| ---: | ---: | ---: |",
        ]
        for item in history:
            lines.append(
                f"| {item['step']} | {item['train_loss']:.6f} | "
                f"{100 * item['val_stratified_miou']:.4f} |"
            )
        lines.append("")
    else:
        lines += ["Direct DINO validation history：待正式训练。", ""]

    lines += ["## Direct DINO confusion matrix", ""]
    if direct is None:
        lines += ["`val_stratified` 与 `val_domain` confusion matrix：待正式训练。", ""]
    else:
        for split in ("val_stratified", "val_domain"):
            lines += [f"### {split}", "", "行是真值，列是预测；类别顺序与 `data/Label.txt` 一致。", "", "```text"]
            lines += [str(row) for row in direct[split]["confusion"]]
            lines += ["```", ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}; direct_status={'complete' if direct is not None else 'pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
