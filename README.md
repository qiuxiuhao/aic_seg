# 2026AIC·无人机低空航拍图像语义分割

## 一、当前版本迭代情况与提升

> 注1：当前为进行 baseline 级别的框架尝试，训练采用同一配置：AdamW、相同初始学习率、Weight decay、Scheduler、optimizer steps = 30000、常规多类别 Cross Entropy Loss。

> 注2: 数据增强仅采用 水平翻转、垂直翻转、随机 0°/90°/180°/270° 旋转 这几种最简单的数据增强方式

> 注3: 因时间和资源有限，不保证所有训练均已完全收敛稳定，旨在统一设置下进行架构调整效果对比。

当前完成的 Stage 03、Stage 04 和 Stage 05 实验包含四个版本：

| 版本 | 模型 | Conditioning | 最佳 step | val_stratified mIoU | val_domain mIoU |
| --- | --- | --- | ---: | ---: | ---: |
| v1 | SegFormer-B3 | 无 | 28,000 | 74.6640% | 68.9888% |
| v2 | SegFormer-B3 + Presence Conditioning | Presence probability | 28,000 | 74.6940% | 69.6476% |
| v3 | SegFormer-B3 + Direct DINO Conditioning | DINO Combined embedding | 28,000 | 75.1535% | 69.9629% |
| v4 | SegFormer-B3 + Direct DINO Conditioning + Soft MoE | DINO Combined embedding | 26,000 | 75.0706% | 70.0225% |
| v2 - v1 | — | — | — | +0.0299 pp | +0.6588 pp |
| v3 - v1 | — | — | — | +0.4894 pp | +0.9742 pp |
| v3 - v2 | — | — | — | +0.4595 pp | +0.3154 pp |
| v4 - v1 | — | — | — | +0.4066 pp | +1.0337 pp |
| v4 - v2 | — | — | — | +0.3766 pp | +0.3749 pp |
| v4 - v3 | — | — | — | -0.0829 pp | +0.0596 pp |

四个版本的 checkpoint 均只按照 `val_stratified` mIoU 选择。`val_domain` 不参与 checkpoint 选择。

各类别 IoU：

| 数据集 | 类别 | v1 | v2 | v3 | v4 | v2 - v1 | v3 - v1 | v3 - v2 | v4 - v1 | v4 - v2 | v4 - v3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val_stratified | Background | 66.7450% | 66.8950% | 67.3464% | 67.3517% | +0.1500 pp | +0.6014 pp | +0.4514 pp | +0.6067 pp | +0.4567 pp | +0.0053 pp |
| val_stratified | Building | 85.6661% | 85.9595% | 85.6580% | 85.7198% | +0.2934 pp | -0.0081 pp | -0.3015 pp | +0.0537 pp | -0.2397 pp | +0.0618 pp |
| val_stratified | Road | 78.8383% | 79.1101% | 79.2987% | 79.2248% | +0.2718 pp | +0.4604 pp | +0.1886 pp | +0.3865 pp | +0.1146 pp | -0.0740 pp |
| val_stratified | Water | 85.3687% | 85.2992% | 85.2983% | 85.0610% | -0.0695 pp | -0.0703 pp | -0.0009 pp | -0.3077 pp | -0.2382 pp | -0.2373 pp |
| val_stratified | Barren | 39.3018% | 39.1525% | 40.7432% | 40.2672% | -0.1493 pp | +1.4415 pp | +1.5908 pp | +0.9655 pp | +1.1147 pp | -0.4760 pp |
| val_stratified | Vegetation | 85.6111% | 85.5968% | 86.0248% | 86.0883% | -0.0143 pp | +0.4137 pp | +0.4280 pp | +0.4772 pp | +0.4915 pp | +0.0635 pp |
| val_stratified | Agricultural | 78.5599% | 78.3958% | 78.6291% | 78.6249% | -0.1641 pp | +0.0692 pp | +0.2333 pp | +0.0650 pp | +0.2291 pp | -0.0042 pp |
| val_stratified | Vehicle | 77.2215% | 77.1428% | 78.2290% | 78.2272% | -0.0786 pp | +1.0076 pp | +1.0862 pp | +1.0057 pp | +1.0844 pp | -0.0018 pp |
| val_domain | Background | 69.4301% | 69.8004% | 69.3696% | 69.6994% | +0.3704 pp | -0.0605 pp | -0.4309 pp | +0.2693 pp | -0.1011 pp | +0.3298 pp |
| val_domain | Building | 82.9688% | 82.8927% | 83.0501% | 83.0953% | -0.0761 pp | +0.0813 pp | +0.1574 pp | +0.1265 pp | +0.2027 pp | +0.0453 pp |
| val_domain | Road | 75.5821% | 75.2794% | 76.4606% | 76.5869% | -0.3027 pp | +0.8785 pp | +1.1812 pp | +1.0049 pp | +1.3076 pp | +0.1264 pp |
| val_domain | Water | 70.2525% | 71.6820% | 73.3733% | 74.7837% | +1.4294 pp | +3.1208 pp | +1.6914 pp | +4.5312 pp | +3.1017 pp | +1.4104 pp |
| val_domain | Barren | 58.8264% | 60.4940% | 60.2623% | 60.3862% | +1.6677 pp | +1.4360 pp | -0.2317 pp | +1.5598 pp | -0.1078 pp | +0.1238 pp |
| val_domain | Vegetation | 83.0794% | 83.4941% | 83.1754% | 83.3840% | +0.4147 pp | +0.0960 pp | -0.3187 pp | +0.3046 pp | -0.1101 pp | +0.2086 pp |
| val_domain | Agricultural | 35.7729% | 37.4464% | 36.8916% | 35.4625% | +1.6736 pp | +1.1187 pp | -0.5549 pp | -0.3104 pp | -1.9839 pp | -1.4291 pp |
| val_domain | Vehicle | 75.9981% | 76.0915% | 77.1206% | 76.7818% | +0.0934 pp | +1.1225 pp | +1.0291 pp | +0.7837 pp | +0.6903 pp | -0.3388 pp |

正式输出目录：

```text
outputs/presence_conditioning/control_b3_bs4/
outputs/presence_conditioning/conditioned_b3_bs4/
outputs/direct_dino_conditioning/direct_dino_b3_bs4/
outputs/dino_moe/dino_soft_moe_b3_bs4/
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

### 2. 准备固定数据

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

### 10. 生成 Stage 04 几何增强 DINO cache

Stage 04 固定使用 Stage 01 的 Combined embedding（CLS 768 + Mean Patch 768，共 1536 维）。训练集为每个 image ID 保存 8 个唯一几何状态，`r0` 直接复用 Stage 01 embedding，其余 7 个状态使用冻结的 DINOv2 ViT-B/14 离线提取。

```bash
python -m tools.cache_augmented_dino_embeddings \
  --data-dir data \
  --base-embedding-dir outputs/dinov2_518/full_mps \
  --cache-dir outputs/hf_cache \
  --device cuda \
  --batch-size 16 \
  --output-dir outputs/dinov2_518/augmented_train
```

完整输出的 Combined 数组 shape 为：

```text
[5597, 8, 1536]
```

8 个状态为：

```text
r0
r90
r180
r270
flip_r0
flip_r90
flip_r180
flip_r270
```

### 11. Stage 04 smoke test

```bash
python -m tools.smoke_direct_dino_conditioning \
  --data-dir data \
  --embedding-dir outputs/dinov2_518/full_mps \
  --augmented-dino-dir outputs/dinov2_518/augmented_train \
  --device cuda \
  --output-dir outputs/direct_dino_conditioning/smoke_cuda
```

### 12. 训练 v3 Direct DINO Conditioning

```bash
python -m tools.train_presence_conditioning \
  --arm direct_dino \
  --data-dir data \
  --embedding-dir outputs/dinov2_518/full_mps \
  --augmented-dino-dir outputs/dinov2_518/augmented_train \
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
  --output-dir outputs/direct_dino_conditioning/direct_dino_b3_bs4
```

### 13. Stage 04 断点续训

```bash
python -m tools.train_presence_conditioning \
  --arm direct_dino \
  --data-dir data \
  --embedding-dir outputs/dinov2_518/full_mps \
  --augmented-dino-dir outputs/dinov2_518/augmented_train \
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
  --output-dir outputs/direct_dino_conditioning/direct_dino_b3_bs4 \
  --resume-from outputs/direct_dino_conditioning/direct_dino_b3_bs4/last.pt
```

### 14. Stage 05 smoke test

```bash
python -m tools.smoke_dino_soft_moe \
  --data-dir data \
  --embedding-dir outputs/dinov2_518/full_mps \
  --augmented-dino-dir outputs/dinov2_518/augmented_train \
  --device cuda \
  --output-dir outputs/dino_moe/smoke_cuda
```

### 15. 训练 v4 DINO-Conditioned Soft MoE

```bash
python -m tools.train_presence_conditioning \
  --arm dino_soft_moe \
  --data-dir data \
  --embedding-dir outputs/dinov2_518/full_mps \
  --augmented-dino-dir outputs/dinov2_518/augmented_train \
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
  --output-dir outputs/dino_moe/dino_soft_moe_b3_bs4
```

### 16. Stage 05 断点续训

```bash
python -m tools.train_presence_conditioning \
  --arm dino_soft_moe \
  --data-dir data \
  --embedding-dir outputs/dinov2_518/full_mps \
  --augmented-dino-dir outputs/dinov2_518/augmented_train \
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
  --output-dir outputs/dino_moe/dino_soft_moe_b3_bs4 \
  --resume-from outputs/dino_moe/dino_soft_moe_b3_bs4/last.pt
```

### 17. 使用 v1、v2、v3、v4 生成比赛提交文件

test 图像目录：

```text
data/test/images/
├── test2_1.png
├── test2_2.png
└── ...                     # 共 1,300 张 1024×1024 RGB PNG
```

v1：

```bash
python -m tools.infer_submission \
  --version v1 \
  --test-image-dir data/test/images \
  --device cuda \
  --amp \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/submissions/v1_control
```

v2：

```bash
python -m tools.infer_submission \
  --version v2 \
  --test-image-dir data/test/images \
  --presence-dir outputs/class_presence/linear_518 \
  --embedding-dir outputs/dinov2_518/full_mps \
  --device cuda \
  --amp \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/submissions/v2_presence
```

v3：

```bash
python -m tools.infer_submission \
  --version v3 \
  --test-image-dir data/test/images \
  --embedding-dir outputs/dinov2_518/full_mps \
  --device cuda \
  --amp \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/submissions/v3_direct_dino
```

v4：

```bash
python -m tools.infer_submission \
  --version v4 \
  --test-image-dir data/test/images \
  --embedding-dir outputs/dinov2_518/full_mps \
  --device cuda \
  --amp \
  --batch-size 1 \
  --num-workers 0 \
  --output-dir outputs/submissions/v4_dino_soft_moe
```

每个版本分别生成：

```text
outputs/submissions/<version>/
├── predictions/            # 1,300 张同名、单通道 1024×1024 PNG
├── submission.zip          # 比赛平台上传文件
└── submission_report.json  # checkpoint、标签映射、文件数量和 ZIP 校验记录
```

v2 在线使用 Stage 02 Combined Presence Head 生成7维前景连续 probability。v3 和 v4 在线生成冻结 DINOv2 Combined 1,536维特征。v4 的 `submission_report.json` 额外记录 test routing statistics。

脚本会先核对 `data/Label.txt` 与内部类别顺序。模型内部类别 `0..7` 会在保存时显式转换为比赛标签 `1..8`。`submission.zip` 中的 PNG 位于压缩包根目录，不包含额外目录层级。所有版本均为 single pass，不使用 TTA 和后处理。

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

### v3: SegFormer-B3 + Direct DINO Conditioning

#### 训练配置

| 配置项 | 数值 |
| --- | --- |
| 模型 | SegFormer-B3 |
| 预训练模型 | `nvidia/mit-b3` |
| Direct DINO Conditioning | 开启 |
| DINO 特征来源 | Stage 01 DINOv2 Combined embedding |
| DINOv2 模型 | `facebook/dinov2-base`，ViT-B/14，冻结，离线提取 |
| DINOv2 输入 | 完整 1024×1024 图像直接 Resize 到 518×518 |
| DINOv2 CLS 维度 | 768 |
| DINOv2 Mean Patch 维度 | 768 |
| DINOv2 Combined 维度 | 1,536 |
| 训练集 DINO cache | 每个 image ID 对应 8 个几何 transform state |
| 训练集 transform state | `r0`、`r90`、`r180`、`r270`、`flip_r0`、`flip_r90`、`flip_r180`、`flip_r270` |
| 验证集 DINO embedding | 原始完整图像 `r0` embedding |
| Conditioning | LayerNorm，`1536 → 256 → 1536` |
| Conditioning 激活函数 | GELU |
| Conditioning 参数量 | 791,296 |
| 模型可训练参数量 | 48,019,912 |
| FiLM 输出 | 768 维 gamma + 768 维 beta |
| FiLM 计算 | `F' = (1 + gamma) * F + beta` |
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
| 2,000 | 0.896250 | 63.2269% |
| 4,000 | 0.416938 | 66.9397% |
| 6,000 | 0.156465 | 68.8650% |
| 8,000 | 0.203638 | 69.8081% |
| 10,000 | 0.159549 | 71.3690% |
| 12,000 | 0.083413 | 71.8376% |
| 14,000 | 0.213179 | 72.7105% |
| 16,000 | 0.075330 | 73.6717% |
| 18,000 | 0.194961 | 74.2285% |
| 20,000 | 0.162203 | 74.7492% |
| 22,000 | 0.108852 | 74.7240% |
| 24,000 | 0.140542 | 74.9821% |
| 26,000 | 0.097219 | 75.0984% |
| 28,000 | 0.263647 | 75.1535% |
| 30,000 | 0.110888 | 75.1082% |

### v4: SegFormer-B3 + Direct DINO Conditioning + DINO-Conditioned Soft MoE

#### 训练配置

| 配置项 | 数值 |
| --- | --- |
| 模型 | SegFormer-B3 |
| 预训练模型 | `nvidia/mit-b3` |
| 初始化策略 | 从 `nvidia/mit-b3` 重新训练，不加载 v3 `best.pt` |
| Direct DINO Conditioning | 开启 |
| DINOv2 模型 | `facebook/dinov2-base`，ViT-B/14，冻结，离线提取 |
| DINOv2 输入 | 完整 1024×1024 图像直接 Resize 到 518×518 |
| DINOv2 Combined 维度 | 1,536 |
| Direct DINO Conditioning | LayerNorm，`1536 → 256 → 1536` |
| FiLM 输出 | 768维 gamma + 768维 beta |
| MoE 类型 | DINO-Conditioned dense soft routing |
| Router | LayerNorm，`1536 → 256 → 4 → Softmax` |
| Expert 数量 | 4 |
| Expert Adapter | `Conv1×1 768 → 192 → GELU → Conv1×1 192 → 768` |
| MoE residual | `F_moe = F_film + Σ w_i × E_i(F_film)` |
| Routing | 4个 Expert 全部执行，不使用 Top-k 或 hard routing |
| Router regularization | 无 |
| Conditioning 参数量 | 791,296 |
| MoE 参数量 | 1,581,060 |
| Router 参数量 | 397,572 |
| 单个 Expert 参数量 | 295,872 |
| 模型可训练参数量 | 49,600,972 |
| 训练集 | `data/splits/train.txt`，5,597张 |
| 主验证集 | `data/splits/val_stratified.txt`，700张 |
| 泛化验证集 | `data/splits/val_domain.txt`，699张 |
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
| 最佳 checkpoint | `best.pt`，step 26,000 |

#### 训练 loss：常规 Cross Entropy Loss（无类别权重、无 label smoothing，`ignore_index=255`）

| Step | Train loss | val_stratified mIoU |
| ---: | ---: | ---: |
| 2,000 | 0.940607 | 63.7223% |
| 4,000 | 0.413852 | 66.9594% |
| 6,000 | 0.173694 | 69.1442% |
| 8,000 | 0.203543 | 70.3494% |
| 10,000 | 0.151696 | 70.8545% |
| 12,000 | 0.078295 | 71.3304% |
| 14,000 | 0.216636 | 72.6119% |
| 16,000 | 0.072812 | 73.6422% |
| 18,000 | 0.200159 | 73.8148% |
| 20,000 | 0.217447 | 74.4132% |
| 22,000 | 0.110374 | 74.7316% |
| 24,000 | 0.152786 | 74.8646% |
| 26,000 | 0.092007 | 75.0706% |
| 28,000 | 0.251027 | 75.0633% |
| 30,000 | 0.121742 | 75.0456% |

#### Best checkpoint routing statistics

| 数据集 | Expert 0 mean weight | Expert 1 mean weight | Expert 2 mean weight | Expert 3 mean weight | Mean entropy | Router collapsed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 46.8680% | 21.9809% | 24.2149% | 6.9362% | 0.529263 | False |
| val_stratified | 46.8500% | 36.6447% | 12.6275% | 3.8778% | 0.435112 | False |
| val_domain | 46.4493% | 34.7151% | 15.4703% | 3.3653% | 0.519925 | False |
