# 2026AIC·无人机低空航拍图像语义分割

## 一、当前版本迭代情况与提升

> 注1：当前为进行 baseline 级别的框架尝试，训练采用同一配置：AdamW、相同初始学习率、Weight decay、Scheduler、optimizer steps = 30000、常规多类别 Cross Entropy Loss。

> 注2: 数据增强仅采用 水平翻转、垂直翻转、随机 0°/90°/180°/270° 旋转 这几种最简单的数据增强方式

> 注3: 因时间和资源有限，不保证所有训练均已完全收敛稳定，旨在统一设置下进行架构调整效果对比。

当前完成的 Stage 03 实验包含两个版本：

| 版本 | 模型 | Presence Conditioning | 最佳 step | val_stratified mIoU | val_domain mIoU |
| --- | --- | --- | ---: | ---: | ---: |
| v1 | SegFormer-B3 | 关闭 | 28,000 | 74.6640% | 68.9888% |
| v2 | SegFormer-B3 + Presence Conditioning | 开启 | 28,000 | 74.6940% | 69.6476% |
| v2 - v1 | — | — | — | +0.0299 pp | +0.6588 pp |

两个版本的 checkpoint 均只按照 `val_stratified` mIoU 选择。`val_domain` 不参与 checkpoint 选择。

各类别 IoU：

| 数据集 | 类别 | v1 | v2 | v2 - v1 |
| --- | --- | ---: | ---: | ---: |
| val_stratified | Background | 66.7450% | 66.8950% | +0.1500 pp |
| val_stratified | Building | 85.6661% | 85.9595% | +0.2934 pp |
| val_stratified | Road | 78.8383% | 79.1101% | +0.2718 pp |
| val_stratified | Water | 85.3687% | 85.2992% | -0.0695 pp |
| val_stratified | Barren | 39.3018% | 39.1525% | -0.1493 pp |
| val_stratified | Vegetation | 85.6111% | 85.5968% | -0.0143 pp |
| val_stratified | Agricultural | 78.5599% | 78.3958% | -0.1641 pp |
| val_stratified | Vehicle | 77.2215% | 77.1428% | -0.0786 pp |
| val_domain | Background | 69.4301% | 69.8004% | +0.3704 pp |
| val_domain | Building | 82.9688% | 82.8927% | -0.0761 pp |
| val_domain | Road | 75.5821% | 75.2794% | -0.3027 pp |
| val_domain | Water | 70.2525% | 71.6820% | +1.4294 pp |
| val_domain | Barren | 58.8264% | 60.4940% | +1.6677 pp |
| val_domain | Vegetation | 83.0794% | 83.4941% | +0.4147 pp |
| val_domain | Agricultural | 35.7729% | 37.4464% | +1.6736 pp |
| val_domain | Vehicle | 75.9981% | 76.0915% | +0.0934 pp |

正式输出目录：

```text
outputs/presence_conditioning/control_b3_bs4/
outputs/presence_conditioning/conditioned_b3_bs4/
```

每个目录包含：

```text
best.pt
config.json
history.json
metrics.json
```

使用当前训练入口新建的训练还会在每次验证后生成可续训的 `last.pt`。

## 二、部署运行流程

### 1. 创建环境

```bash
conda create -n aic_seg python=3.11 -y
conda activate aic_seg
```

根据运行平台安装 PyTorch。当前正式训练环境使用：

```text
PyTorch 2.7.0+cu128
CUDA
```

安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

### 2. 准备固定数据与 Presence probability

训练图片和标注放在仓库根目录的 `data/train/` 下。项目文件结构如下：

```text
aic_seg/
├── README.md
├── requirements.txt
├── src/
├── tools/
├── data/
│   ├── Label.txt
│   ├── splits/
│   │   ├── train.txt
│   │   ├── val_stratified.txt
│   │   └── val_domain.txt
│   └── train/
│       ├── images/
│       │   ├── 0000.png
│       │   ├── 0001.png
│       │   └── ...
│       └── masks/
│           ├── 0000.png
│           ├── 0001.png
│           └── ...
└── outputs/
```

图片和 mask 使用相同文件名并一一对应：

```text
data/train/images/<image_id>.png
data/train/masks/<image_id>.png
```

三个 split 文件每行保存一个不带 `.png` 后缀的 `image_id`。例如 split 中的 `0000` 对应：

```text
data/train/images/0000.png
data/train/masks/0000.png
```

`data/train/` 和 `outputs/` 不上传到 Git 仓库，由运行环境本地准备或生成。

### 3. 执行完整数据审计

```bash
python -m tools.audit_data \
  --data-dir data \
  --output-dir outputs/data_audit
```

完整审计通过后生成：

```text
outputs/data_audit/audit.json
outputs/data_audit/class_statistics.csv
```

### 4. 提取 DINOv2 全量 embedding

该步骤读取三个固定 split 中的全部 6,996 张图片，将完整的 1024×1024 图片直接 Resize 到 518×518，再使用冻结的 DINOv2 ViT-B/14 提取 CLS、Mean Patch 和 Combined embedding。

```bash
python -m tools.extract_dinov2_embeddings \
  --data-dir data \
  --audit-file outputs/data_audit/audit.json \
  --device cuda \
  --batch-size 1 \
  --output-dir outputs/dinov2_518/full_cuda
```

生成文件：

```text
outputs/dinov2_518/full_cuda/cls.npy
outputs/dinov2_518/full_cuda/mean_patch.npy
outputs/dinov2_518/full_cuda/combined.npy
outputs/dinov2_518/full_cuda/manifest.csv
outputs/dinov2_518/full_cuda/metadata.json
```

### 5. 训练 Presence Head 并导出 probability

该步骤使用 `train.txt` 训练线性 Presence Head，使用 `val_stratified.txt` 选择特征、checkpoint 和分类阈值，随后导出三个 split 的预测 probability。

```bash
python -m tools.run_presence_experiment \
  --data-dir data \
  --embedding-dir outputs/dinov2_518/full_cuda \
  --features all \
  --device cuda \
  --batch-size 256 \
  --max-epochs 100 \
  --patience 10 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --seed 42 \
  --output-dir outputs/class_presence/linear_518
```

Stage 03 使用以下三个预测 probability 文件：

```text
outputs/class_presence/linear_518/predictions_train.csv
outputs/class_presence/linear_518/predictions_val_stratified.csv
outputs/class_presence/linear_518/predictions_val_domain.csv
```

每个 CSV 的列顺序为：

```text
image_id, split, Background, Building, Road, Water,
Barren, Vegetation, Agricultural, Vehicle
```

Stage 03 Conditioning 从中读取七个前景类别的连续 probability，不读取 Background probability。

### 6. CUDA smoke test

```bash
python -m tools.smoke_presence_conditioning \
  --device cuda \
  --output-dir outputs/presence_conditioning/smoke_cuda
```

### 7. 训练 v1

```bash
python -m tools.train_presence_conditioning \
  --arm control \
  --device cuda \
  --amp \
  --max-steps 30000 \
  --eval-every 2000 \
  --batch-size 4 \
  --grad-accum 1 \
  --num-workers 0 \
  --learning-rate 6e-5 \
  --weight-decay 0.01 \
  --seed 42 \
  --output-dir outputs/presence_conditioning/control_b3_bs4
```

### 8. 训练 v2

```bash
python -m tools.train_presence_conditioning \
  --arm conditioned \
  --device cuda \
  --amp \
  --max-steps 30000 \
  --eval-every 2000 \
  --batch-size 4 \
  --grad-accum 1 \
  --num-workers 0 \
  --learning-rate 6e-5 \
  --weight-decay 0.01 \
  --seed 42 \
  --output-dir outputs/presence_conditioning/conditioned_b3_bs4
```

### 9. Stage 03 断点续训

训练会在每次验证完成后更新输出目录中的 `last.pt`。该文件保存：

```text
模型参数
优化器状态
Scheduler 状态
AMP GradScaler 状态
当前 optimizer step
最佳 mIoU 与最佳 step
训练历史
随机数状态
DataLoader shuffle 状态与当前迭代位置
```

断点续训必须继续使用原输出目录和完全相同的训练参数。`best.pt` 只保存最佳模型参数，不用于断点续训。该方式只适用于由当前训练入口生成了 `last.pt` 的训练任务。

v1 从最近一次完整验证点继续：

```bash
python -m tools.train_presence_conditioning \
  --arm control \
  --device cuda \
  --amp \
  --max-steps 30000 \
  --eval-every 2000 \
  --batch-size 4 \
  --grad-accum 1 \
  --num-workers 0 \
  --learning-rate 6e-5 \
  --weight-decay 0.01 \
  --seed 42 \
  --output-dir outputs/presence_conditioning/control_b3_bs4 \
  --resume-from outputs/presence_conditioning/control_b3_bs4/last.pt
```

v2 从最近一次完整验证点继续：

```bash
python -m tools.train_presence_conditioning \
  --arm conditioned \
  --device cuda \
  --amp \
  --max-steps 30000 \
  --eval-every 2000 \
  --batch-size 4 \
  --grad-accum 1 \
  --num-workers 0 \
  --learning-rate 6e-5 \
  --weight-decay 0.01 \
  --seed 42 \
  --output-dir outputs/presence_conditioning/conditioned_b3_bs4 \
  --resume-from outputs/presence_conditioning/conditioned_b3_bs4/last.pt
```

### 10. 生成对比结果

```bash
python -m tools.compare_presence_conditioning \
  --control-dir outputs/presence_conditioning/control_b3_bs4 \
  --conditioned-dir outputs/presence_conditioning/conditioned_b3_bs4 \
  --output docs/results/03_presence_conditioning_results.md
```

## 三、baseline版本具体情况

### v1: SegFormer-B3

#### 训练配置

| 配置项 | 数值 |
| --- | --- |
| 模型 | SegFormer-B3 |
| 预训练模型 | `nvidia/mit-b3` |
| Presence Conditioning | 关闭 |
| 训练集 | `data/splits/train.txt`，5,597 张 |
| 主验证集 | `data/splits/val_stratified.txt`，700 张 |
| 泛化验证集 | `data/splits/val_domain.txt`，699 张 |
| 分割输入 | 完整 RGB 图像，1024×1024 |
| 分割类别 | 8 |
| Ignore index | 255 |
| Loss | Cross Entropy |
| 优化器 | AdamW |
| 初始学习率 | `6e-5` |
| Weight decay | `0.01` |
| Scheduler | CosineAnnealingLR，`T_max=30000` |
| Optimizer steps | 30,000 |
| 验证间隔 | 2,000 steps |
| Batch size | 4 |
| Gradient accumulation | 1 |
| 有效 batch size | 4 |
| AMP | CUDA FP16 |
| Gradient checkpointing | 开启 |
| 随机种子 | 42 |
| 数据增强 | 水平翻转、垂直翻转、随机 0°/90°/180°/270° 旋转 |
| 最佳 checkpoint | `best.pt`，step 28,000 |

#### 训练 loss：常规 Cross Entropy Loss（无类别权重、无 label smoothing，`ignore_index=255`）

| Step | Train loss | val_stratified mIoU |
| ---: | ---: | ---: |
| 2,000 | 1.170976 | 62.4248% |
| 4,000 | 0.451629 | 62.8257% |
| 6,000 | 0.220474 | 68.9211% |
| 8,000 | 0.215823 | 69.3572% |
| 10,000 | 0.190246 | 70.5502% |
| 12,000 | 0.109108 | 71.1500% |
| 14,000 | 0.241082 | 72.4969% |
| 16,000 | 0.088612 | 73.0877% |
| 18,000 | 0.215156 | 73.9522% |
| 20,000 | 0.143724 | 73.8162% |
| 22,000 | 0.118651 | 74.2963% |
| 24,000 | 0.166164 | 74.3050% |
| 26,000 | 0.116383 | 74.6631% |
| 28,000 | 0.325926 | 74.6640% |
| 30,000 | 0.133420 | 74.6437% |

### v2: SegFormer-B3 + Presence Conditioning

#### 训练配置

| 配置项 | 数值 |
| --- | --- |
| 模型 | SegFormer-B3 |
| 预训练模型 | `nvidia/mit-b3` |
| Presence Conditioning | 开启 |
| Presence 来源 | Stage 02 预测 probability |
| Presence 输入 | 7 维前景连续概率 |
| Presence 类别 | Building、Road、Water、Barren、Vegetation、Agricultural、Vehicle |
| DINOv2 模型 | DINOv2 ViT-B/14，冻结，离线提取 |
| DINOv2 输入 | 完整 1024×1024 图像直接 Resize 到 518×518 |
| DINOv2 CLS 维度 | 768 |
| DINOv2 Mean Patch 维度 | 768 |
| DINOv2 Combined 维度 | 1,536 |
| Presence Head 输出 | 8 维 probability，Stage 03 去除 Background 后使用 7 维 |
| Conditioning | FiLM，`7 → 64 → 1536` |
| FiLM 输出 | 768 维 gamma + 768 维 beta |
| FiLM 注入位置 | SegFormer decoder fused feature 与 classifier 之间 |
| 训练集 | `data/splits/train.txt`，5,597 张 |
| 主验证集 | `data/splits/val_stratified.txt`，700 张 |
| 泛化验证集 | `data/splits/val_domain.txt`，699 张 |
| 分割输入 | 完整 RGB 图像，1024×1024 |
| 分割类别 | 8 |
| Ignore index | 255 |
| Loss | Cross Entropy |
| 优化器 | AdamW |
| 初始学习率 | `6e-5` |
| Weight decay | `0.01` |
| Scheduler | CosineAnnealingLR，`T_max=30000` |
| Optimizer steps | 30,000 |
| 验证间隔 | 2,000 steps |
| Batch size | 4 |
| Gradient accumulation | 1 |
| 有效 batch size | 4 |
| AMP | CUDA FP16 |
| Gradient checkpointing | 开启 |
| 随机种子 | 42 |
| 数据增强 | 水平翻转、垂直翻转、随机 0°/90°/180°/270° 旋转 |
| 最佳 checkpoint | `best.pt`，step 28,000 |

#### 训练 loss：常规 Cross Entropy Loss（无类别权重、无 label smoothing，`ignore_index=255`）

| Step | Train loss | val_stratified mIoU |
| ---: | ---: | ---: |
| 2,000 | 1.069569 | 63.6212% |
| 4,000 | 0.413735 | 64.5040% |
| 6,000 | 0.197554 | 69.5345% |
| 8,000 | 0.214148 | 69.7909% |
| 10,000 | 0.188977 | 70.8947% |
| 12,000 | 0.093350 | 71.6653% |
| 14,000 | 0.194700 | 72.1463% |
| 16,000 | 0.089087 | 72.9523% |
| 18,000 | 0.210829 | 73.6792% |
| 20,000 | 0.130409 | 73.7160% |
| 22,000 | 0.112528 | 74.2773% |
| 24,000 | 0.171479 | 74.1524% |
| 26,000 | 0.113745 | 74.6554% |
| 28,000 | 0.307231 | 74.6940% |
| 30,000 | 0.130804 | 74.6766% |
