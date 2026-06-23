# 论文完整服务器复现手册

本文档对应论文：

> **Generative or Discriminative? Revisiting Text Classification in the Era of Transformers**

所有命令默认在仓库根目录执行，路径统一使用当前目录 `.`：

```bash
cd .
```

目标不是要求一次性跑完所有实验，而是把论文中能够复现的实验矩阵、运行命令、GPU 分配和扩展分析全部准备好，之后根据服务器资源决定最终运行范围。

---

## 1. 论文实验矩阵

### 1.1 五种方法

| 论文缩写 | 本仓库入口 | 作用 |
| --- | --- | --- |
| `ENC` | `repro_fig2/train_one.py --model enc` | 判别式 BERT Encoder 分类 |
| `AR` | `repro_fig2/train_one.py --model ar` | 真实生成式 `P(text | label)` 分类 |
| `ARpseudo` | `repro_fig2/train_one.py --model arpseudo` | 文本后预测标签 token |
| `MLM` | `repro_fig2/train_one.py --model mlm` | `[MASK]` 位置预测标签 |
| `DIFF` | `diff/` + `repro_fig2/slurm_diff_*.sh` | 离散扩散分类 |

### 1.2 九个数据集

| 简写 | Hugging Face 路径 | 类别数 | 论文训练/测试规模 | 有序 |
| --- | --- | ---: | --- | --- |
| `imdb` | `imdb` | 2 | 25,000 / 25,000 | 否 |
| `agnews` | `ag_news` | 4 | 120,000 / 7,600 | 否 |
| `emotion` | `emotion` | 6 | 16,000 / 2,000 | 否 |
| `hatespeech` | `SetFit/hate_speech_offensive` | 3 | 22,783 / 2,000 | 是 |
| `multiclasssentiment` | `Sp1786/multiclass-sentiment-analysis-dataset` | 3 | 31,232 / 5,205 | 是 |
| `rottentomatoes` | `cornell-movie-review-data/rotten_tomatoes` | 2 | 8,530 / 1,066 | 否 |
| `sst2` | `SetFit/sst2` | 2 | 6,920 / 872 | 否 |
| `sst5` | `SetFit/sst5` | 5 | 8,544 / 1,101 | 是 |
| `twitter` | `zeroshot/twitter-financial-news-sentiment` | 3 | 9,543 / 2,388 | 是 |

标签编号和论文训练集比例：

| 数据集 | 标签编号 | 论文训练集比例 |
| --- | --- | --- |
| IMDb | `0=negative, 1=positive` | 50.0 / 50.0 |
| AG News | `0=World, 1=Sports, 2=Business, 3=Sci/Tech` | 每类 25.0 |
| Emotion | `0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise` | 29.2 / 33.5 / 8.2 / 13.5 / 12.1 / 3.6 |
| Hate Speech | `0=hate, 1=offensive, 2=neither` | 5.8 / 77.5 / 16.7 |
| Multiclass Sentiment | `0=negative, 1=neutral, 2=positive` | 29.2 / 37.3 / 33.6 |
| Rotten Tomatoes | `0=negative, 1=positive` | 50.0 / 50.0 |
| SST-2 | `0=negative, 1=positive` | 47.8 / 52.2 |
| SST-5 | `0=very negative, 1=negative, 2=neutral, 3=positive, 4=very positive` | 12.8 / 26.0 / 19.0 / 27.2 / 15.1 |
| Twitter Financial | `0=Bearish, 1=Bullish, 2=Neutral` | 15.1 / 20.2 / 64.7 |

脚本默认使用 `--test_split paper`，按照论文表格选择 SST-2、SST-5、Twitter
的 `validation` split 作为最终测试集。checkpoint 验证集优先使用不与最终测试集
重合的官方 split；没有可用 split 时再从训练集分层留出，不使用最终测试集。

所有小样本设置都按标签比例分层抽样。同一组 `dataset + sample size + seed`
在不同模型之间使用相同的抽样规则，减少模型比较中由样本组成造成的额外波动。

### 1.3 样本量、模型规模和随机种子

```text
sample sizes = 128, 256, 512, 1024, 2048, 4096, full
model sizes  = 1 layer/1 head, 6 layers/6 heads, 12 layers/12 heads
seeds        = 79140, 24561, 54641
```

论文总实验量：

```text
9 datasets * 7 sample sizes * 3 model sizes * 3 seeds * 5 methods
= 2835 experiments
```

### 1.4 论文主要图表与对应任务

| 论文内容 | 复现范围 |
| --- | --- |
| Figure 2 | 5 个主图数据集，ENC/DIFF/AR/MLM，1/6/12 层，样本量曲线 |
| Figure 3 | ARpseudo vs AR |
| Figure 5/6/7 | ECE、MCE、MAE、MSE、UM |
| Figure 8 | Figure 2 扩展到全部 9 个数据集 |
| Figure 9 | ARpseudo vs AR 扩展到全部 9 个数据集 |
| Figure 10 | 预训练 BERT-base ENC vs GPT-2 AR |
| 鲁棒性表格 | full-data、6/12 层、Token Drop/Substitution |

---

## 2. 从 GitHub 拉取代码

首次下载：

```bash
git clone https://github.com/denghuigeng/Generative-vs-Discriminative-Classifiers.git
cd Generative-vs-Discriminative-Classifiers
```

以后更新：

```bash
cd .
git pull origin main
```

确认版本：

```bash
git log --oneline -5
git status
```

---

## 3. 建立环境

### 3.1 ENC、AR、ARpseudo、MLM 环境

推荐先单独安装 PyTorch，再安装项目依赖，避免服务器 CUDA 版本不匹配。

```bash
conda create -n gendisc-transformers python=3.10 -y
conda activate gendisc-transformers
```

以下 CUDA 12.1 命令只是示例，应根据服务器驱动选择：

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r ./requirements-server.txt
```

检查：

```bash
python - <<'PY'
import torch
import transformers
import datasets
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

### 3.2 DIFF 独立环境

```bash
cd ./diff
grep -v "^prefix:" environment.yml > environment.server.yml
conda env create -n sedd -f environment.server.yml
conda activate sedd
```

DIFF 依赖更敏感。如果环境文件安装失败，应优先按照 `diff/README.md` 调整 CUDA、PyTorch、Flash Attention 等版本。

---

## 4. Hugging Face 缓存与模型下载

项目中的主要训练、推理和下载入口已经固定使用：

```text
https://hf-mirror.com
```

代码会在导入 Hugging Face 相关库之前设置
`HF_ENDPOINT=https://hf-mirror.com`，服务器运行时不需要再手动配置 endpoint。
缓存目录建议放在当前仓库下：

```bash
export HF_HOME=./hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
```

可以加入 `~/.bashrc`：

```bash
echo 'export HF_HOME=./hf_cache' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=$HF_HOME/datasets' >> ~/.bashrc
echo 'export TOKENIZERS_PARALLELISM=false' >> ~/.bashrc
```

### 4.1 下载全部九个数据集、Tokenizer 和 Config

```bash
cd .
conda activate gendisc-transformers
python repro_fig2/download_assets.py
```

### 4.2 同时下载预训练扩展所需权重

```bash
python repro_fig2/download_assets.py --include_pretrained_weights
```

下载内容：

- `bert-base-uncased` tokenizer/config
- `gpt2` tokenizer/config
- 可选 BERT-base、GPT-2 预训练权重
- 全部 9 个数据集

说明：论文主实验是**从随机权重训练**。下载 tokenizer/config 不等于使用预训练权重。脚本只有指定：

```bash
--initialization pretrained
```

才会加载预训练权重。

### 4.3 DIFF 可选额外下载

原 DIFF 代码在生成样本时可能用 `gpt2-large` 评估 perplexity：

```bash
python repro_fig2/download_assets.py \
  --include_pretrained_weights \
  --include_diffusion_eval_model
```

---

## 5. 如何指定 GPU

### 5.1 在普通服务器上指定一张 GPU

例如使用物理 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 python repro_fig2/train_one.py ...
```

程序内部看到的设备编号会重新从 `cuda:0` 开始，这是正常现象。

查看空闲显卡：

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

### 5.2 同时使用多张 GPU 跑不同任务

本项目任务之间彼此独立。相比用多卡训练一个小样本模型，更推荐“一张 GPU 跑一个实验”。

```bash
bash repro_fig2/run_job_file_local.sh \
  repro_fig2/jobs_paper.tsv \
  ./outputs/paper_repro \
  0,1,2,3
```

这会在 GPU `0,1,2,3` 上分别启动一个 worker，每张卡顺序处理分配给它的任务。

### 5.3 SLURM 指定 GPU

单任务脚本默认：

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
```

可以提交时覆盖：

```bash
sbatch \
  --partition=a100 \
  --gres=gpu:a100:1 \
  --array=0-99%8 \
  repro_fig2/slurm_train_array.sh
```

含义：

- `--array=0-99`：100 个任务
- `%8`：最多同时运行 8 个
- 每个任务申请 1 张 A100

DIFF 如果需要 8 卡：

```bash
sbatch \
  --partition=a100 \
  --gres=gpu:a100:8 \
  --export=ALL,NGPUS=8,JOBS=/绝对路径/jobs_diff.tsv \
  --array=0-10%2 \
  repro_fig2/slurm_diff_array.sh
```

---

## 6. 训练脚本的论文对齐设置

统一训练入口：

```bash
python repro_fig2/train_one.py --help
```

`repro_fig2/train_one.py` 是实际实现论文式训练流程的唯一入口。为了兼容原仓库命令，
`ar/train_gpt.py` 现在是一个轻量转发器，会把原来的参数翻译后交给统一入口：

```text
ar/train_gpt.py
  -> repro_fig2/train_one.py --model ar
```

原始 Lightning 实现保留在 `ar/train_gpt_legacy.py`，仅用于读取旧 checkpoint；
新实验不要直接运行 legacy 文件。可以先检查参数翻译而不启动训练：

```bash
python ar/train_gpt.py \
  --data_key SetFit/sst2 \
  --ckpt_dir "./outputs/paper_repro" \
  --model_size small \
  --seed 42 \
  --n_tr_sub 128 \
  --max_epochs 100 \
  --bsz 8 \
  --n_devices 1 \
  --max_len 512 \
  --dry_run
```

新的 AR 训练顺序是：

```text
按类别比例分层采样
-> 选择不与最终测试集重合的验证集
-> 随机初始化 GPT-2 架构
-> 训练语言模型目标
-> 每轮按 P(text|label) 计算 validation weighted-F1
-> 连续 10 轮无提升则早停
-> 恢复最佳 checkpoint
-> 在完整最终测试集输出 predictions.csv 和 metrics.json
```

### 6.1 自动使用的论文式默认值

| 模型 | 最大 epoch | 有效 batch size | 默认学习率 | checkpoint |
| --- | ---: | ---: | ---: | --- |
| ENC | 30 | 32 | `5e-5` | 最低 validation loss |
| MLM | 200 | 32 | `5e-5` | 最低 validation loss |
| AR | 100 | 32 | `5e-5` | 最高 validation weighted-F1 |
| ARpseudo | 100 | 32 | `5e-5` | 最高 validation weighted-F1 |

论文给出的搜索空间：

```text
learning rate = 1e-5, 2e-5, 3e-5, 4e-5, 5e-5
batch size    = 32, 64, 128, 256
sequence len  = 512
```

本脚本默认通过梯度累积得到有效 batch size 32。例如 AR 的：

```text
per-device batch 8 * accumulation 4 = effective batch 32
```

如果要严格做超参数搜索，可以单独指定：

```bash
--lr 2e-5 \
--batch_size 8 \
--gradient_accumulation_steps 8 \
--run_suffix lr2e-5_global64
```

论文没有公开每一个数据集/模型设置最终选中的完整超参数表，因此报告中需要说明：我们使用论文搜索空间，并在验证集上选择设置。

### 6.2 模型规模

```text
--layers 1  --heads 1
--layers 6  --heads 6
--layers 12 --heads 12
```

脚本按论文缩放 hidden size：

```text
1 head  -> hidden size 64
6 heads -> hidden size 384
12 heads -> hidden size 768
```

### 6.3 混合精度

A100/H100 推荐：

```bash
--precision bf16
```

V100、RTX 3090/4090 等可尝试：

```bash
--precision fp16
```

排错时：

```bash
--precision fp32
```

---

## 7. 必须先做的 Smoke Test

每个方法先跑 `1 layer + 128 samples + 1 epoch`。

### ENC

```bash
cd .
conda activate gendisc-transformers

CUDA_VISIBLE_DEVICES=0 python repro_fig2/train_one.py \
  --model enc \
  --dataset sst5 \
  --sample_size 128 \
  --seed 79140 \
  --layers 1 \
  --heads 1 \
  --epochs 1 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --max_len 128 \
  --precision fp32 \
  --output_root "./outputs/smoke"
```

把 `--model` 分别替换为：

```text
ar
arpseudo
mlm
```

检查输出：

```bash
find "./outputs/smoke" -name metrics.json -o -name predictions.csv
```

每次成功运行应生成：

```text
args.json
dataset_manifest.json
model/
predictions.csv
metrics.json
```

---

## 8. 单个正式实验示例

### 8.1 ENC：AG News，12 层，样本量 1024

```bash
CUDA_VISIBLE_DEVICES=0 python repro_fig2/train_one.py \
  --model enc \
  --dataset agnews \
  --sample_size 1024 \
  --seed 79140 \
  --layers 12 \
  --heads 12 \
  --precision bf16 \
  --output_root "./outputs/paper_repro"
```

### 8.2 AR：SST-5，12 层，样本量 128

```bash
CUDA_VISIBLE_DEVICES=1 python repro_fig2/train_one.py \
  --model ar \
  --dataset sst5 \
  --sample_size 128 \
  --seed 79140 \
  --layers 12 \
  --heads 12 \
  --precision bf16 \
  --inference_batch_size 16 \
  --output_root "./outputs/paper_repro"
```

### 8.3 Full-data

用 `-1` 表示 full：

```bash
CUDA_VISIBLE_DEVICES=2 python repro_fig2/train_one.py \
  --model mlm \
  --dataset emotion \
  --sample_size -1 \
  --seed 79140 \
  --layers 6 \
  --heads 6 \
  --precision bf16 \
  --output_root "./outputs/paper_repro"
```

程序会自动跳过已经存在 `predictions.csv + metrics.json` 的完整任务。重新运行时使用：

```bash
--overwrite
```

---

## 9. 生成批量任务列表

统一生成器：

```bash
python repro_fig2/make_jobs_paper.py --help
```

### 9.1 Figure 2 主图

5 个数据集、ENC/AR/MLM、1/6/12 层、7 个样本量：

```bash
python repro_fig2/make_jobs_paper.py \
  --preset figure2 \
  --one_seed \
  --out "./repro_fig2/jobs_figure2_1seed.tsv"
```

单 seed 非 DIFF 部分：

```text
5 * 3 * 3 * 7 = 315 tasks
```

三 seeds：

```bash
python repro_fig2/make_jobs_paper.py \
  --preset figure2 \
  --out "./repro_fig2/jobs_figure2_3seed.tsv"
```

### 9.2 Figure 8：全部九个数据集

```bash
python repro_fig2/make_jobs_paper.py \
  --preset figure8 \
  --one_seed \
  --out "./repro_fig2/jobs_figure8_1seed.tsv"
```

### 9.3 Figure 3/9：AR vs ARpseudo

```bash
python repro_fig2/make_jobs_paper.py \
  --preset non_diff_full \
  --models ar,arpseudo \
  --one_seed \
  --out "./repro_fig2/jobs_arpseudo_1seed.tsv"
```

### 9.4 全部非扩散实验

9 数据集 × ENC/AR/ARpseudo/MLM：

```bash
python repro_fig2/make_jobs_paper.py \
  --preset non_diff_full \
  --out "./repro_fig2/jobs_non_diff_full.tsv"
```

任务数：

```text
9 * 4 * 3 * 7 * 3 = 2268
```

---

## 10. 运行非扩散批量实验

### 10.1 本地多 GPU

```bash
bash repro_fig2/run_job_file_local.sh \
  "./repro_fig2/jobs_figure2_1seed.tsv" \
  "./outputs/paper_repro" \
  0,1,2,3
```

日志：

```text
outputs/paper_repro/local_logs/
```

### 10.2 SLURM 数组

先查看任务数：

```bash
wc -l "./repro_fig2/jobs_figure2_1seed.tsv"
```

假设为 315 行：

```bash
sbatch \
  --array=0-314%12 \
  --export=ALL,JOBS="./repro_fig2/jobs_figure2_1seed.tsv",OUT="./outputs/paper_repro",PRECISION=bf16,MAX_LEN=512 \
  repro_fig2/slurm_train_array.sh
```

查看：

```bash
squeue -u "$USER"
tail -f "./outputs/slurm/gendisc_fig2_<JOBID>_<TASKID>.out"
```

失败任务可以直接重新提交。已经完成的任务会自动跳过。

---

## 11. DIFF 完整流程

### 11.1 论文设置

```text
batch size: 64
learning rate: 3e-4
iterations: 200,000
noise: log-linear/geometric, sigma 1e-4 -> 20
transition: absorbing/masking
paper hardware: 8 * NVIDIA A100
```

### 11.2 生成任务

全部九个数据集：

```bash
conda activate sedd
python repro_fig2/make_diff_jobs.py \
  --one_seed \
  --out "./repro_fig2/jobs_diff_1seed.tsv"
```

只生成 Figure 2 的代表性 DIFF 任务：

```bash
python repro_fig2/make_diff_jobs.py \
  --datasets agnews emotion rottentomatoes sst5 twitter \
  --models small medium large \
  --samples 128 1024 4096 -1 \
  --one_seed \
  --out "./repro_fig2/jobs_diff_figure2_representative.tsv"
```

这里 `small/medium/large` 分别对应 1/6/12 层。

### 11.3 单卡先跑

```bash
JOBS="./repro_fig2/jobs_diff_figure2_representative.tsv"
N=$(wc -l < "$JOBS")

sbatch \
  --gres=gpu:1 \
  --array=0-$((N-1))%2 \
  --export=ALL,JOBS="$JOBS",DIFF_OUT="./outputs/diff_repro",NGPUS=1,N_ITERS=200000 \
  repro_fig2/slurm_diff_array.sh
```

### 11.4 八卡论文式运行

```bash
JOBS="./repro_fig2/jobs_diff_figure2_representative.tsv"
N=$(wc -l < "$JOBS")

sbatch \
  --gres=gpu:a100:8 \
  --array=0-$((N-1))%1 \
  --export=ALL,JOBS="$JOBS",DIFF_OUT="./outputs/diff_repro",NGPUS=8,N_ITERS=200000 \
  repro_fig2/slurm_diff_array.sh
```

DIFF 单任务很重，不建议直接提交全部 567 个任务。先跑：

```text
5 个 Figure 2 数据集
1 seed
代表性 sample sizes: 128, 1024, 4096, full
```

### 11.5 DIFF 推理

训练完成后，用相同任务列表提交推理：

```bash
JOBS="./repro_fig2/jobs_diff_figure2_representative.tsv"
N=$(wc -l < "$JOBS")

sbatch \
  --array=0-$((N-1))%4 \
  --export=ALL,JOBS="$JOBS",DIFF_OUT="./outputs/diff_repro",STEPS=128,BATCH_SIZE=64,MAX_LENGTH=128 \
  repro_fig2/slurm_diff_infer_array.sh
```

输出：

```text
outputs/diff_repro/<dataset>/<model-size>/samples_<N>/seed_<S>/predictions.csv
```

DIFF 默认 absorbing diffusion 不提供自然的完整类别概率，因此只纳入 weighted-F1，不纳入 ECE/MCE/UM。
推理脚本会检查是否为最终测试集的每一条样本都成功生成了合法标签；若有漏项，
任务会失败并写出 `inference_manifest.json`，不会悄悄用不完整样本计算结果。

---

## 12. 汇总并绘制 Figure 2 / Figure 8

### 12.1 只汇总非扩散结果

```bash
conda activate gendisc-transformers

python repro_fig2/aggregate_and_plot.py \
  --output_root "./outputs/paper_repro" \
  --initialization scratch \
  --layers 1 6 12 \
  --datasets agnews emotion rottentomatoes sst5 twitter \
  --full_grid
```

### 12.2 合并 DIFF

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root "./outputs/paper_repro" \
  --additional_output_roots "./outputs/diff_repro" \
  --initialization scratch \
  --layers 1 6 12 \
  --datasets agnews emotion rottentomatoes sst5 twitter \
  --full_grid
```

主要输出：

```text
outputs/paper_repro/
├── all_run_metrics.csv
├── summary_layers_1_6_12.csv
└── figures/
    ├── weighted_f1_grid_layers_1_6_12.png
    ├── ece_grid_layers_1_6_12.png
    ├── mce_grid_layers_1_6_12.png
    ├── nll_grid_layers_1_6_12.png
    ├── brier_grid_layers_1_6_12.png
    ├── ordinal_mae_grid_layers_1_6_12.png
    ├── ordinal_mse_grid_layers_1_6_12.png
    └── ordinal_um_grid_layers_1_6_12.png
```

`weighted_f1_grid_layers_1_6_12.png` 对应 Figure 2 风格：

- 行：1、6、12 层
- 列：数据集
- 横轴：128 到 4096 和 full
- 曲线：ENC、DIFF、AR、MLM；如果输出目录包含 ARpseudo，也会额外绘制
- 阴影：3 seeds 标准差

全部 9 数据集 Figure 8：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root "./outputs/paper_repro" \
  --additional_output_roots "./outputs/diff_repro" \
  --initialization scratch \
  --layers 1 6 12 \
  --datasets imdb agnews emotion hatespeech multiclasssentiment rottentomatoes sst2 sst5 twitter \
  --full_grid
```

---

## 13. 校准度和有序性实验

聚合脚本会从 `predictions.csv` 自动计算：

### 论文指标

| 指标 | 范围 | 方向 |
| --- | --- | --- |
| ECE | 所有有概率输出的模型/数据集 | 越低越好 |
| MCE | 所有有概率输出的模型/数据集 | 越低越好 |
| MAE | 四个有序数据集 | 越低越好 |
| MSE | 四个有序数据集 | 越低越好 |
| UM | 四个有序数据集 | 越高越好 |

四个有序数据集：

```text
hatespeech
multiclasssentiment
sst5
twitter
```

有序指标先把原始标签映射到语义顺序。前三个数据集的编号本身已经按顺序排列；
Twitter 使用 `Bearish -> Neutral -> Bullish`，即原始编号 `[0, 2, 1]`。
ECE/MCE 默认采用 15 个等宽 confidence bins。

### 本项目额外扩展

| 指标 | 作用 |
| --- | --- |
| NLL | 真标签概率的负对数似然 |
| Brier Score | 完整概率分布的平方误差 |
| Expected MAE/MSE | 用预测分布期望类别，而不是 argmax 类别计算 |
| Reliability Diagram | 直观看 confidence 与 accuracy 是否对齐 |

画可靠性图：

```bash
python repro_fig2/plot_reliability.py \
  --output_root "./outputs/paper_repro" \
  --dataset sst5 \
  --layers 12 \
  --sample_size 1024 \
  --models enc ar arpseudo mlm
```

---

## 14. Q2：Token Drop / Substitution 鲁棒性

论文主要在：

```text
full-data checkpoints
6-layer and 12-layer
```

上做：

- Random Token Drop
- Random Token Substitution

### 14.1 找出已完成 checkpoint

```bash
python repro_fig2/make_robustness_jobs.py \
  --output_root "./outputs/paper_repro" \
  --layers 6 12 \
  --out "./repro_fig2/jobs_robustness.tsv"
```

### 14.2 运行

```bash
N=$(wc -l < "./repro_fig2/jobs_robustness.tsv")

sbatch \
  --array=0-$((N-1))%8 \
  --export=ALL,JOBS="./repro_fig2/jobs_robustness.tsv" \
  repro_fig2/slurm_robustness_array.sh
```

噪声比例：

```text
0, 5%, 10%, 15%, 20%, 30%, 40%, 50%
```

每个 checkpoint 会输出：

```text
robustness/drop/summary.csv
robustness/drop/drop_thresholds.json
robustness/substitute/summary.csv
robustness/substitute/drop_thresholds.json
```

`drop_thresholds.json` 表示 weighted-F1 相对下降 5%、10%、15%、20%、30% 时所需的最低噪声比例。

聚合：

```bash
python repro_fig2/aggregate_robustness.py \
  --output_root "./outputs/paper_repro"
```

---

## 15. Figure 10：预训练 ENC vs AR

论文预训练补充实验只比较：

```text
BERT-base ENC
GPT-2 base AR
```

都使用 12 层模型。

生成任务：

```bash
python repro_fig2/make_jobs_paper.py \
  --preset figure8 \
  --models enc,ar \
  --initialization pretrained \
  --one_seed \
  --out "./repro_fig2/jobs_pretrained_1seed.tsv"
```

运行：

```bash
bash repro_fig2/run_job_file_local.sh \
  "./repro_fig2/jobs_pretrained_1seed.tsv" \
  "./outputs/pretrained_repro" \
  0,1,2,3
```

聚合：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root "./outputs/pretrained_repro" \
  --initialization pretrained \
  --layers 12 \
  --datasets imdb agnews emotion hatespeech multiclasssentiment rottentomatoes sst2 sst5 twitter
```

---

## 16. 推荐的实际执行顺序

不要一上来提交 2835 个任务。建议：

### 阶段 A：验证流程

```text
4 个非扩散模型
SST-5
1 layer
128 samples
1 epoch
```

### 阶段 B：得到一张完整 Figure 2

```text
5 个 Figure 2 数据集
ENC/AR/MLM
1/6/12 layers
7 sample sizes
1 seed
```

然后给 DIFF 补代表性设置。

### 阶段 C：补统计稳定性

对最重要的结果补齐 3 seeds：

```text
AG News + SST-5
ENC/AR/MLM
1/6/12 layers
all sample sizes
```

### 阶段 D：扩展实验

```text
ECE/MCE/MAE/MSE/UM
NLL/Brier/Reliability
full-data 6/12-layer robustness
pretrained ENC vs AR
```

### 阶段 E：资源有余时

```text
Figure 8 全 9 数据集
Figure 3/9 ARpseudo
更多 DIFF 设置
全部 3 seeds
```

---

## 17. 训练监控、恢复和排错

### 查看 GPU

```bash
nvidia-smi
watch -n 2 nvidia-smi
```

### 查看 SLURM

```bash
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

### 查看 TensorBoard

```bash
tensorboard \
  --logdir "./outputs" \
  --host 0.0.0.0 \
  --port 6006
```

本地建立 SSH 转发：

```bash
ssh -L 6006:localhost:6006 user@server
```

浏览器访问：

```text
http://localhost:6006
```

### 显存不足

依次降低：

```bash
--batch_size 8
--eval_batch_size 16
--inference_batch_size 4
--gradient_accumulation_steps 4
--max_len 256
--precision fp16
```

### bf16 不支持

```bash
PRECISION=fp16
```

或：

```bash
--precision fp32
```

### AR 推理慢

AR 对每个类别都要计算一次 `P(text | label)`：

- AG News：约 4 倍候选计算
- SST-5：约 5 倍候选计算
- Emotion：约 6 倍候选计算

可以提高：

```bash
--inference_batch_size 32
```

如果显存允许。

---

## 18. 结果完整性检查

统计完成任务：

```bash
find "./outputs" -name metrics.json | wc -l
find "./outputs" -name predictions.csv | wc -l
```

查找有参数但没有预测结果的失败任务：

```bash
python - <<'PY'
from pathlib import Path
root = Path("./outputs")
for args_file in root.rglob("args.json"):
    if not (args_file.parent / "predictions.csv").exists():
        print(args_file.parent)
PY
```

输出路径示例：

```text
outputs/paper_repro/
└── init_scratch/
    └── ar/
        └── sst5/
            └── layers_12/
                └── samples_128/
                    └── seed_79140/
                        ├── args.json
                        ├── dataset_manifest.json
                        ├── model/
                        ├── predictions.csv
                        └── metrics.json
```

---

## 19. 复现报告中需要说明的差异

即使严格按上述流程运行，也应在报告中明确：

1. 论文只公开了超参数搜索空间，没有公开每组实验最终选中的完整超参数。
2. 原始仓库部分脚本曾把最终测试集用于验证；补充脚本优先使用独立官方 split，
   否则从训练集留出验证集，避免测试泄漏。
3. AR 按论文正文使用 validation weighted-F1 选择 checkpoint。
4. DIFF absorbing 模式只提供 hard label，不报告 ECE/MCE/UM。
5. 论文使用 8 张 A100；如果服务器硬件不同，训练时间和可用 batch size 会变化。
6. 小样本结果波动很大，最终结论应以 3 seeds 的均值和标准差为准。

---

## 20. 最推荐的课程交付组合

如果最终无法跑完整 2835 组，建议至少交付：

1. Figure 2 风格的 3×5 主图，ENC/AR/MLM 至少 1 seed。
2. AG News 和 SST-5 的 3 seeds 完整曲线。
3. SST-5 的 ECE/MCE/MAE/MSE/UM。
4. 一组 full-data 6/12 层鲁棒性实验。
5. 预训练 ENC vs AR 或 Reliability/NLL/Brier 中任选一个扩展。
6. 清楚报告计算预算、失败任务和与原论文的实现差异。

这个组合已经同时包含：

- 主实验复现
- 模型规模分析
- 校准和有序性
- 鲁棒性或预训练扩展
- 自己新增的概率质量分析
