"""Write the Stage 05 v1/v2/v3/v4 comparison and Router statistics report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.presence.labels import CLASS_NAMES


def percentage(value: float | None) -> str:
    return "待正式训练" if value is None else f"{100 * value:.4f}%"


def delta(value: float | None, reference: float) -> str:
    return "待正式训练" if value is None else f"{100 * (value - reference):+.4f} pp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dir", type=Path,
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
        "--moe-dir", type=Path,
        default=Path("outputs/dino_moe/dino_soft_moe_b3_bs4"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/results/05_dino_conditioned_moe_results.md"),
    )
    args = parser.parse_args()

    clean = json.loads((args.clean_dir / "metrics.json").read_text(encoding="utf-8"))
    presence = json.loads((args.presence_dir / "metrics.json").read_text(encoding="utf-8"))
    direct = json.loads((args.direct_dir / "metrics.json").read_text(encoding="utf-8"))
    if (clean.get("arm"), presence.get("arm"), direct.get("arm")) != (
        "control", "conditioned", "direct_dino"
    ):
        raise ValueError("Expected v1 control, v2 conditioned and v3 direct_dino metrics")

    moe_metrics_path = args.moe_dir / "metrics.json"
    moe_config_path = args.moe_dir / "config.json"
    routing_path = args.moe_dir / "routing_statistics.json"
    moe = config = routing = None
    if moe_metrics_path.is_file():
        if not moe_config_path.is_file() or not routing_path.is_file():
            raise FileNotFoundError("Formal v4 output lacks config.json or routing_statistics.json")
        moe = json.loads(moe_metrics_path.read_text(encoding="utf-8"))
        config = json.loads(moe_config_path.read_text(encoding="utf-8"))
        routing = json.loads(routing_path.read_text(encoding="utf-8"))
        if moe.get("arm") != "dino_soft_moe" or config.get("arm") != "dino_soft_moe":
            raise ValueError("v4 result is not arm=dino_soft_moe")
        expected = {
            "seed": 42, "max_steps": 30000, "eval_every": 2000,
            "batch_size": 4, "grad_accum": 1, "learning_rate": 6e-5,
            "weight_decay": 0.01, "amp": True, "input_size": [1024, 1024],
            "num_classes": 8, "ignore_index": 255, "num_experts": 4,
            "router_regularization": None,
            "initialization_strategy": "nvidia/mit-b3 start; Stage 04 best.pt is not loaded",
        }
        for key, expected_value in expected.items():
            if config.get(key) != expected_value:
                raise ValueError(f"v4 protocol differs at {key}: {config.get(key)!r}")
        if routing.get("final_best") is None:
            raise ValueError("v4 routing statistics have no final best-checkpoint summary")

    versions = (
        ("v1", "Clean B3", clean),
        ("v2", "+ Presence FiLM", presence),
        ("v3", "+ Direct DINO FiLM", direct),
        ("v4", "+ Direct DINO FiLM + Soft MoE", moe),
    )
    lines = [
        "# 05｜DINO-Conditioned Soft MoE 结果", "",
        "## 实验定义", "",
        "```text",
        "v3: SegFormer-B3 + Direct DINO FiLM",
        "v4: SegFormer-B3 + Direct DINO FiLM + DINO-Conditioned 4-Expert Soft MoE",
        "```", "",
        "初始化策略：v4 从 `nvidia/mit-b3` 重新训练 30,000 optimizer steps，不加载 v3 `best.pt`。", "",
    ]
    if moe is None:
        lines += [
            "> 状态：实现与 smoke 已完成，v4 CUDA 30,000-step 正式训练尚未执行。",
            "", 
        ]
    lines += [
        "## 汇总", "",
        "| Version | Model | val_stratified mIoU | val_domain mIoU |",
        "| --- | --- | ---: | ---: |",
    ]
    for version, label, metrics in versions:
        stratified = None if metrics is None else metrics["val_stratified"]["miou"]
        domain = None if metrics is None else metrics["val_domain"]["miou"]
        lines.append(f"| {version} | {label} | {percentage(stratified)} | {percentage(domain)} |")

    lines += ["", "## v4 - v3", ""]
    for split in ("val_stratified", "val_domain"):
        value = None if moe is None else moe[split]["miou"]
        lines.append(f"- {split} ΔmIoU：{delta(value, direct[split]['miou'])}")

    lines += [
        "", "## 8 类 IoU", "",
        "| Split | Class | v3 | v4 | v4 - v3 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for split in ("val_stratified", "val_domain"):
        for name in CLASS_NAMES:
            v3 = direct[split]["class_iou"][name]
            v4 = None if moe is None else moe[split]["class_iou"][name]
            lines.append(
                f"| {split} | {name} | {percentage(v3)} | {percentage(v4)} | {delta(v4, v3)} |"
            )

    lines += [
        "", "## 结构与参数", "",
        "```text",
        "Router: LayerNorm → 1536→256 → GELU → 256→4 → Softmax",
        "Expert ×4: 768→192 → GELU → 192→768",
        "F_moe = F_film + Σ w_i E_i(F_film)",
        "```", "",
        "- Router 参数量：397,572。",
        "- 每个 Expert 参数量：295,872。",
        "- 四个 Expert 参数量：1,183,488。",
        "- Soft MoE 总参数量：1,581,060。",
        "- 不使用 load balancing loss、Top-k、hard routing 或 Presence Router。",
        "", "## Router statistics", "",
    ]
    if routing is None:
        lines += ["训练过程和 best checkpoint 的 Router statistics：待正式训练。", ""]
    else:
        lines += [
            "| Split | Samples | Mean weight | Top-1 fraction | Mean entropy | Collapse |",
            "| --- | ---: | --- | --- | ---: | --- |",
        ]
        for split in ("train", "val_stratified", "val_domain"):
            item = routing["final_best"][split]
            mean_weight = ", ".join(f"{value:.6f}" for value in item["mean_weight"])
            top1 = ", ".join(f"{value:.6f}" for value in item["top1_fraction"])
            lines.append(
                f"| {split} | {item['sample_count']} | {mean_weight} | {top1} | "
                f"{item['mean_entropy']:.6f} | {item['router_collapsed']} |"
            )
        lines += ["", "完整的逐验证点 Router history 保存在 `routing_statistics.json`。", ""]

    lines += ["## Best step 与 validation history", ""]
    if moe is None:
        lines += ["- v4 best step：待正式训练。", "", "v4 validation history：待正式训练。", ""]
    else:
        history = json.loads((args.moe_dir / "history.json").read_text(encoding="utf-8"))
        lines += [
            f"- v4 best step：{moe['best_step']}。", "",
            "| Step | Train loss | val_stratified mIoU |",
            "| ---: | ---: | ---: |",
        ]
        for item in history:
            lines.append(
                f"| {item['step']} | {item['train_loss']:.6f} | "
                f"{100 * item['val_stratified_miou']:.4f}% |"
            )
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}; v4_status={'complete' if moe is not None else 'pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
