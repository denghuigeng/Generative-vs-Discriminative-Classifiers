# 服务器复现流程：Figure 2 样本效率实验与扩展指标

本文档面向服务器路径：

```bash
/data/gdh/Generative-vs-Discriminative-Classifiers
```

目标是复现论文 **Generative or Discriminative? Revisiting Text Classification in the Era of Transformers** 中 Figure 2。

本文档提供两个档位：

1. **核心版**：`AG News + SST-5`，`AR + ENC`，固定 `12-layer`。适合先跑通流程。
2. **扩展版**：尽量复现 Figure 2 主图结构，覆盖 `5 个数据集 x 3 个模型 x 3 个模型规模`。这是更推荐的最终课程复现版本。

核心版设置：

- 模型：`AR` 真实生成式分类器 vs `ENC` 判别式编码器分类器
- 数据集：`AG News` 与 `SST-5`
- 横轴：训练样本量 `128, 256, 512, 1024, 2048, 4096`
- 纵轴：完整测试集 `weighted-F1`
- 扩展指标：
  - `ECE`、`MCE`：AG News 和 SST-5 都计算
  - `MAE`、`MSE`、`UM`：只建议在 SST-5 上解释，因为 SST-5 标签有自然顺序

扩展版设置：

- 模型：`ENC`、`AR`、`MLM`
- 数据集：`AG News`、`Emotion`、`Rotten Tomatoes`、`SST-5`、`Twitter Financial News`
- 模型规模：`1-layer`、`6-layer`、`12-layer`
- 样本量：`128, 256, 512, 1024, 2048, 4096`
- 随机种子：先用 1 个 seed 生成完整图；最终用 3 个 seed 画均值和标准差

说明：原论文 Figure 2 还包含 `DIFF`。DIFF 需要单独环境、训练和采样成本都更高，建议作为超额实验；课程复现优先做 `ENC / AR / MLM` 三条线，已经能形成完整的 `3 行 x 5 列` 主图结构。

本文档中的所有命令都假设仓库已经位于：

```bash
ROOT=/data/gdh/Generative-vs-Discriminative-Classifiers
```

## 1. 实验范围与计算量

核心版主实验固定 `12-layer / 12-head`，先复现 Figure 2 中“样本量变化”这条主线。

```text
2 datasets * 2 models * 6 sample sizes * 3 seeds = 72 个训练/评估任务
```

可选模型规模消融：

```text
layers/heads = 1/1, 6/6, 12/12
```

如果把模型规模也完整加入，任务数会变为：

```text
72 * 3 = 216 个任务
```

因此建议：

1. 必做：12-layer 下的样本效率曲线。
2. 可选：只挑代表性样本量，例如 `128, 1024, 4096`，补做 `1/6/12` 层消融。

如果做扩展版，任务数如下：

```text
5 datasets * 3 models * 3 layer sizes * 6 sample sizes * 1 seed = 270 个任务
5 datasets * 3 models * 3 layer sizes * 6 sample sizes * 3 seeds = 810 个任务
```

推荐执行顺序：

1. 先跑 `--one_seed` 扩展版，快速得到一张完整 Figure-2-like 图。
2. 如果趋势正常，再补 3 seeds，画均值和阴影。
3. DIFF 不作为必做项，除非服务器资源很充足。

## 2. 服务器目录准备

如果代码还在本地电脑，先把仓库同步到服务器。示例：

```bash
rsync -av \
  --exclude ".git" \
  --exclude "outputs" \
  /本地路径/Generative-vs-Discriminative-Classifiers-main/ \
  user@server:/data/gdh/Generative-vs-Discriminative-Classifiers/
```

进入服务器目录：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
chmod +x repro_fig2/*.py repro_fig2/*.sh
```

建议目录结构：

```text
/data/gdh/Generative-vs-Discriminative-Classifiers
├── ar/
├── encoder_mlm/
├── repro_fig2/
│   ├── download_assets.py
│   ├── train_one.py
│   ├── aggregate_and_plot.py
│   ├── make_jobs.py
│   ├── make_jobs_expanded.py
│   ├── run_all_12layer.sh
│   └── slurm_train_array.sh
├── environment.yml
└── outputs/
```

## 3. 创建运行环境

原仓库的 `environment.yml` 末尾带了作者机器上的 `prefix`，服务器上不建议直接使用。先去掉 `prefix`：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
grep -v "^prefix:" environment.yml > environment.server.yml
```

创建 conda 环境：

```bash
conda env create -n gendisc-transformers -f environment.server.yml
conda activate gendisc-transformers
```

如果服务器 CUDA / PyTorch 版本冲突，使用更保守的安装方式：

```bash
conda create -n gendisc-transformers python=3.10 -y
conda activate gendisc-transformers

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 datasets==3.5.1 accelerate==1.6.0
pip install scikit-learn pandas numpy matplotlib tqdm tensorboard
```

检查 GPU：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 4. 设置 Hugging Face 缓存和模型下载

建议把 Hugging Face 缓存放到 `/data/gdh`，避免写到 home 目录：

```bash
export HF_HOME=/data/gdh/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p $HF_HOME
```

可以把这几行写进 `~/.bashrc` 或 SLURM 脚本。

本复现实验从零训练模型权重，但仍需要下载：

- `bert-base-uncased` 的 tokenizer/config，用于 ENC 架构初始化
- `gpt2` 的 tokenizer/config，用于 AR 架构初始化
- 数据集 `ag_news`
- 数据集 `SetFit/sst5`
- 扩展版还会缓存 `emotion`、`cornell-movie-review-data/rotten_tomatoes`、`zeroshot/twitter-financial-news-sentiment`

执行预下载：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
conda activate gendisc-transformers
python repro_fig2/download_assets.py
```

如果服务器不能联网，可以在能联网的机器上提前缓存 Hugging Face 文件，再把 `$HF_HOME` 目录同步到服务器，并在服务器上设置同样的 `HF_HOME`。

## 5. 补充脚本说明

为避免原仓库脚本中的硬编码和不便批量运行的问题，新增了 `repro_fig2/` 目录。它不修改原作者代码，只作为课程复现的统一入口。

### 5.1 `download_assets.py`

作用：下载并缓存模型 tokenizer/config 与数据集。

```bash
python repro_fig2/download_assets.py
```

### 5.2 `train_one.py`

作用：跑一个具体实验，包括训练、完整测试集预测、保存指标。

支持模型：

```text
enc, ar, mlm
```

支持数据集：

```text
agnews, emotion, rottentomatoes, sst5, twitter
```

输出目录：

```text
outputs/fig2_repro/
└── <model>/
    └── <dataset>/
        └── layers_<L>/
            └── samples_<N>/
                └── seed_<S>/
                    ├── args.json
                    ├── model/
                    ├── predictions.csv
                    └── metrics.json
```

单次 ENC 示例：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
conda activate gendisc-transformers

python repro_fig2/train_one.py \
  --model enc \
  --dataset agnews \
  --sample_size 128 \
  --seed 79140 \
  --layers 12 \
  --heads 12 \
  --epochs 50 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_len 256 \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro
```

单次 AR 示例：

```bash
python repro_fig2/train_one.py \
  --model ar \
  --dataset sst5 \
  --sample_size 128 \
  --seed 79140 \
  --layers 12 \
  --heads 12 \
  --epochs 50 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_len 256 \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro
```

单次 MLM 示例：

```bash
python repro_fig2/train_one.py \
  --model mlm \
  --dataset emotion \
  --sample_size 512 \
  --seed 79140 \
  --layers 6 \
  --heads 6 \
  --epochs 50 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_len 256 \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro
```

参数解释：

| 参数 | 含义 |
| --- | --- |
| `--model` | `ar`、`enc` 或 `mlm` |
| `--dataset` | `agnews`、`emotion`、`rottentomatoes`、`sst5`、`twitter` |
| `--sample_size` | 训练样本量，主实验用 `128,256,512,1024,2048,4096` |
| `--seed` | 随机种子，建议 `79140,24561,54641` |
| `--layers` / `--heads` | 模型规模，主实验为 `12/12` |
| `--epochs` | 最大训练轮数，建议先用 `50`，如欠拟合可提高 |
| `--max_len` | 最大输入长度；AG News 可用 `256`，长文本任务才需要 `512` |

### 5.3 `aggregate_and_plot.py`

作用：读取所有 `predictions.csv`，计算指标并画图。

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro \
  --layers 12
```

核心版输出：

```text
outputs/fig2_repro/
├── all_run_metrics.csv
├── summary_layers_12.csv
└── figures/
    ├── weighted_f1_layers_12.png
    ├── ece_layers_12.png
    ├── mce_layers_12.png
    ├── sst5_mae_layers_12.png
    ├── sst5_mse_layers_12.png
    └── sst5_um_layers_12.png
```

扩展版聚合命令：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro_expanded \
  --layers 1 6 12 \
  --datasets agnews emotion rottentomatoes sst5 twitter \
  --full_grid
```

扩展版输出重点看：

```text
outputs/fig2_repro_expanded/figures/
├── weighted_f1_grid_layers_1_6_12.png
├── ece_grid_layers_1_6_12.png
├── mce_grid_layers_1_6_12.png
├── ordinal_mae_grid_layers_1_6_12.png
├── ordinal_mse_grid_layers_1_6_12.png
└── ordinal_um_grid_layers_1_6_12.png
```

其中 `weighted_f1_grid_layers_1_6_12.png` 就是最接近论文 Figure 2 主图结构的复现图：行是模型层数，列是数据集，曲线是不同模型。

指标含义：

| 指标 | 数据集 | 越高/越低越好 | 说明 |
| --- | --- | --- | --- |
| `weighted-F1` | AG News, SST-5 | 越高越好 | 主复现指标 |
| `ECE` | AG News, SST-5 | 越低越好 | 平均校准误差 |
| `MCE` | AG News, SST-5 | 越低越好 | 最大校准误差 |
| `MAE` | SST-5 | 越低越好 | 有序标签平均绝对误差 |
| `MSE` | SST-5 | 越低越好 | 有序标签均方误差 |
| `UM` | SST-5 | 越高越好 | 预测概率分布是否单峰 |

注意：`MAE/MSE/UM` 不对 AG News 做语义解释。AG News 的标签 `World/Sports/Business/Sci-Tech` 没有自然顺序，强行编码成 `0/1/2/3` 会引入人为距离。

## 6. 快速 smoke test

正式跑 72 个任务前，先用 1 层、1 epoch 检查环境、数据和输出目录。

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
conda activate gendisc-transformers

python repro_fig2/train_one.py \
  --model enc \
  --dataset sst5 \
  --sample_size 128 \
  --seed 79140 \
  --layers 1 \
  --heads 1 \
  --epochs 1 \
  --batch_size 4 \
  --eval_batch_size 8 \
  --max_len 128 \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/smoke_test
```

检查是否生成：

```bash
find /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/smoke_test -maxdepth 8 -type f
```

如果 smoke test 能生成 `predictions.csv` 和 `metrics.json`，再提交正式任务。

## 7. 跑完整 12-layer 主实验

### 7.1 顺序运行

如果没有任务调度系统，可以直接顺序跑：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
conda activate gendisc-transformers
bash repro_fig2/run_all_12layer.sh
```

这个方式最简单，但会非常慢，因为 72 个任务会串行执行。

### 7.2 SLURM 数组运行

如果服务器使用 SLURM，推荐数组任务。

生成 job 列表：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
conda activate gendisc-transformers
python repro_fig2/make_jobs.py
```

输出会提示类似：

```text
Wrote 72 jobs to /data/gdh/Generative-vs-Discriminative-Classifiers/repro_fig2/jobs_12layer.tsv
Use array range: 0-71
```

提交任务：

```bash
mkdir -p /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm
sbatch --array=0-71 repro_fig2/slurm_train_array.sh
```

如果服务器限制同时运行任务数，例如最多 8 个：

```bash
sbatch --array=0-71%8 repro_fig2/slurm_train_array.sh
```

查看进度：

```bash
squeue -u $USER
tail -f /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm/gendisc_fig2_<JOBID>_<TASKID>.out
```

所有任务完成后聚合：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro \
  --layers 12
```

## 8. 扩展版：尽量复现 Figure 2 主图

扩展版覆盖：

```text
datasets = agnews, emotion, rottentomatoes, sst5, twitter
models = enc, ar, mlm
layers = 1, 6, 12
sample sizes = 128, 256, 512, 1024, 2048, 4096
```

### 8.1 先跑 1 个 seed 的完整图

先生成较便宜的 1-seed 任务列表：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
conda activate gendisc-transformers

python repro_fig2/make_jobs_expanded.py --one_seed \
  --out /data/gdh/Generative-vs-Discriminative-Classifiers/repro_fig2/jobs_expanded_1seed.tsv
```

提交 SLURM 数组任务：

```bash
mkdir -p /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/slurm

JOBS=/data/gdh/Generative-vs-Discriminative-Classifiers/repro_fig2/jobs_expanded_1seed.tsv \
OUT=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro_expanded \
sbatch --array=0-269%12 repro_fig2/slurm_train_array.sh
```

聚合并画完整网格图：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro_expanded \
  --layers 1 6 12 \
  --datasets agnews emotion rottentomatoes sst5 twitter \
  --full_grid
```

### 8.2 再补 3 seeds 的均值和标准差

如果 1 seed 结果正常，再生成完整 3-seed 任务：

```bash
python repro_fig2/make_jobs_expanded.py \
  --out /data/gdh/Generative-vs-Discriminative-Classifiers/repro_fig2/jobs_expanded_3seed.tsv
```

任务数是 810。提交时建议限制并发，比如 `%12` 或 `%16`：

```bash
JOBS=/data/gdh/Generative-vs-Discriminative-Classifiers/repro_fig2/jobs_expanded_3seed.tsv \
OUT=/data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro_expanded \
sbatch --array=0-809%12 repro_fig2/slurm_train_array.sh
```

完成后重新聚合：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/fig2_repro_expanded \
  --layers 1 6 12 \
  --datasets agnews emotion rottentomatoes sst5 twitter \
  --full_grid
```

### 8.3 扩展实验产出

扩展版会同时得到：

1. `weighted-F1` 主图：复现 Figure 2 的核心趋势。
2. `ECE/MCE` 校准图：看模型准确率和概率可靠性是否一致。
3. `MAE/MSE/UM` 有序性图：默认只对 SST-5 解释，因为 SST-5 标签顺序最明确。
4. 模型规模分析：比较 `1/6/12` 层下 AR、ENC、MLM 的变化。

## 9. 可选：小规模模型规模消融

论文 Figure 2 同时画了 `1/6/12` 层。完整做会很贵，不建议一开始就全跑。

建议的小消融：

```text
datasets: agnews, sst5
models: ar, enc
sample sizes: 128, 1024, 4096
seeds: 79140, 24561, 54641
model sizes: 1/1, 6/6, 12/12
```

单个 1-layer 命令示例：

```bash
python repro_fig2/train_one.py \
  --model enc \
  --dataset agnews \
  --sample_size 1024 \
  --seed 79140 \
  --layers 1 \
  --heads 1 \
  --epochs 50 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_len 256 \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/layer_ablation
```

单个 6-layer 命令示例：

```bash
python repro_fig2/train_one.py \
  --model ar \
  --dataset sst5 \
  --sample_size 1024 \
  --seed 79140 \
  --layers 6 \
  --heads 6 \
  --epochs 50 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --max_len 256 \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/layer_ablation
```

聚合某个层数：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/layer_ablation \
  --layers 6
```

## 10. 原仓库脚本与本复现实验的关系

原仓库已有代码入口：

| 方法 | 原代码 | 问题 |
| --- | --- | --- |
| AR | `ar/train_gpt.py`, `ar/infer_gpt.py` | 有 `--n_tr_sub`，但不是严格分层采样；验证集截断到 480；批量跑不方便 |
| ENC/MLM | `encoder_mlm/mlm_classif_seed_fixed.py`, `encoder_mlm/inference.py` | 数据集、样本量、模型规模写成全局常量；默认策略不方便按单任务提交 |
| DIFF | `diff/` | 环境单独且训练/推理成本高，不在本课程主复现范围 |

因此本文档推荐使用 `repro_fig2/` 中的补充脚本，它们做了这些统一：

- 分层采样，保持类别比例
- 每个任务单独输出 `args.json / predictions.csv / metrics.json`
- 统一计算 `weighted-F1 / ECE / MCE`，并在 SST-5 上计算 `MAE / MSE / UM`
- 统一聚合并画图
- 所有路径默认基于 `/data/gdh/Generative-vs-Discriminative-Classifiers`

## 11. 常见问题

### 11.1 为什么模型还要下载 `bert-base-uncased` 和 `gpt2`？

本实验从零初始化权重，不使用预训练权重。但仍需要下载 tokenizer 和 config：

- ENC 使用 BERT tokenizer/config 来定义词表和 Transformer 结构。
- AR 使用 GPT-2 tokenizer/config 来定义词表和 Transformer 结构。

脚本中模型权重使用 `from_config(...)` 初始化，不是 `from_pretrained(...)` 加载预训练权重。

### 11.2 为什么 AG News 不算 MAE/MSE/UM？

AG News 标签是主题类别：

```text
World / Sports / Business / Sci-Tech
```

它们没有自然顺序。把它们编码为 `0/1/2/3` 后计算 MAE/MSE，会暗含“World 比 Sci-Tech 离 Sports 更远/更近”之类的人为假设，因此没有合理语义。

SST-5 标签有自然顺序：

```text
Very Negative < Negative < Neutral < Positive < Very Positive
```

所以 SST-5 可以解释 MAE、MSE 和 UM。

### 11.3 AR 为什么慢？

ENC 一次前向传播直接输出 `P(y|x)`。

AR 分类时需要对每个候选标签计算一次文本似然：

```text
y_hat = argmax_y log P(text | label)
```

因此测试时大约要多跑 `类别数` 倍前向传播。AG News 有 4 类，SST-5 有 5 类，AR 推理会明显慢于 ENC。

### 11.4 如果显存不够怎么办？

优先调整：

```bash
--batch_size 8
--eval_batch_size 16
--max_len 128
```

如果还不够，先跑 `layers=6, heads=6` 或 `layers=1, heads=1` 做流程验证。

### 11.5 结果和论文不完全一致怎么办？

可能原因：

- 论文完整实验可能使用了更细的训练轮数、checkpoint selection 和随机种子设置。
- 原仓库代码部分脚本使用 test split 做验证，严格复现需要统一验证集策略。
- 小样本实验波动大，必须看 3 个 seed 的均值与方差。
- 本文档推荐的是课程复现范围，不包含 DIFF、ARpseudo 和全部 9 个数据集；扩展版已经包含 `MLM`。

报告时建议强调：

```text
我们复现 Figure 2 的核心趋势，而不是完整复现论文全部 2835 个实验。
```
