# 一步步跑完整复现实验

本文档只写“怎么一步步跑出结果”。更完整的背景、实验矩阵和可选扩展见
`REPRODUCE_FIGURE2_SERVER.md`。

默认服务器路径：

```bash
export ROOT=/data/gdh/Generative-vs-Discriminative-Classifiers
cd "$ROOT"
```

最终目标：

1. 跑出 Figure 2 风格主图：`ENC / AR / MLM`，5 个数据集，1/6/12 层，7 个样本量。
2. 对 `AG News + SST-5` 补 3 seeds，重点分析 AR 和 ENC 的样本效率。
3. 做扩展分析：ECE/MCE；SST-5 上做 MAE/MSE/UM；可选 Reliability 或鲁棒性。

不建议一开始跑 DIFF。DIFF 成本高，主复现先用非扩散模型完成。

---

## 0. 更新代码

如果之前已经克隆过仓库：

```bash
cd /data/gdh/Generative-vs-Discriminative-Classifiers
git pull origin main
git log -1 --oneline
```

确认代码里已经包含新的统一训练入口和本地多 GPU 进度日志。

现在 `ar/train_gpt.py` 已经不是旧训练逻辑，而是转发到统一入口：

```text
ar/train_gpt.py -> repro_fig2/train_one.py --model ar
```

---

## 1. 激活环境并设置缓存

如果你已经有可用环境，例如 `llm-26-gpu`：

```bash
conda activate llm-26-gpu
```

建议统一 Hugging Face 缓存目录：

```bash
export HF_HOME=/data/gdh/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
```

代码里已经写死镜像：

```text
https://hf-mirror.com
```

所以一般不需要手动设置 `HF_ENDPOINT`。

检查环境：

```bash
python - <<'PY'
import torch
import transformers
import datasets

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

如果缺包：

```bash
pip install -r requirements-server.txt
```

---

## 2. 测试数据集和 GPT-2 能否下载

先测 SST-2 和 GPT-2：

```bash
python - <<'PY'
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config

ds = load_dataset("SetFit/sst2")
print(ds)
print(ds["train"][0])

tok = GPT2Tokenizer.from_pretrained("gpt2")
cfg = GPT2Config.from_pretrained("gpt2")
print("GPT-2 tokenizer/config ok:", len(tok), cfg.n_layer)
PY
```

如果这里卡住或报网络错误，先不要训练，先解决网络或缓存。

---

## 3. 先做 dry run，确认命令会跑什么

这一步不训练，只看参数是否被翻译正确：

```bash
CUDA_VISIBLE_DEVICES=0 python ar/train_gpt.py \
  --data_key SetFit/sst2 \
  --ckpt_dir "$ROOT/outputs/paper_repro" \
  --model_size small \
  --seed 42 \
  --n_tr_sub 128 \
  --max_epochs 100 \
  --patience 10 \
  --bsz 8 \
  --n_devices 1 \
  --max_len 512 \
  --dry_run
```

应看到类似：

```text
Using the canonical paper-style AR training pipeline.
Model=small (1L/1H), sample_size=128, patience=10, effective_batch≈32
Command: ... repro_fig2/train_one.py --model ar ...
Run output directory: ...
```

这说明现在走的是新训练流程。

---

## 4. Smoke Test：真正训练一个很小的任务

先跑 AR：

```bash
CUDA_VISIBLE_DEVICES=0 python repro_fig2/train_one.py \
  --model ar \
  --dataset sst5 \
  --sample_size 128 \
  --seed 42 \
  --layers 1 \
  --heads 1 \
  --epochs 1 \
  --patience 1 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 4 \
  --inference_batch_size 4 \
  --max_len 128 \
  --precision fp32 \
  --output_root "$ROOT/outputs/smoke" \
  --overwrite
```

再跑 ENC：

```bash
CUDA_VISIBLE_DEVICES=0 python repro_fig2/train_one.py \
  --model enc \
  --dataset sst5 \
  --sample_size 128 \
  --seed 42 \
  --layers 1 \
  --heads 1 \
  --epochs 1 \
  --batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_batch_size 4 \
  --max_len 128 \
  --precision fp32 \
  --output_root "$ROOT/outputs/smoke" \
  --overwrite
```

检查输出：

```bash
find "$ROOT/outputs/smoke" -name metrics.json -o -name predictions.csv
```

如果两个任务都能生成 `metrics.json` 和 `predictions.csv`，说明环境、数据、模型和显卡都通了。

---

## 5. 生成主实验任务：Figure 2 非扩散完整图

主实验范围：

```text
数据集：agnews, emotion, rottentomatoes, sst5, twitter
模型：ENC, AR, MLM
模型规模：1, 6, 12 层
样本量：128, 256, 512, 1024, 2048, 4096, full
随机种子：先跑 1 个 seed
任务数：5 * 3 * 3 * 7 = 315
```

生成任务：

```bash
python repro_fig2/make_jobs_paper.py \
  --preset figure2 \
  --one_seed \
  --out "$ROOT/repro_fig2/jobs_figure2_1seed.tsv"
```

确认任务数：

```bash
wc -l "$ROOT/repro_fig2/jobs_figure2_1seed.tsv"
head "$ROOT/repro_fig2/jobs_figure2_1seed.tsv"
```

应为：

```text
315
```

---

## 6. 指定显卡并运行主实验

假设你用 4 张卡：`0,1,2,3`。

如果是 A100/H100，优先用 `bf16`：

```bash
PRECISION=bf16 bash repro_fig2/run_job_file_local.sh \
  "$ROOT/repro_fig2/jobs_figure2_1seed.tsv" \
  "$ROOT/outputs/paper_repro" \
  0,1,2,3
```

如果是 3090/4090/V100，`bf16` 不一定稳定，可以用：

```bash
PRECISION=fp16 bash repro_fig2/run_job_file_local.sh \
  "$ROOT/repro_fig2/jobs_figure2_1seed.tsv" \
  "$ROOT/outputs/paper_repro" \
  0,1,2,3
```

脚本启动时会先打印：

```text
Total jobs: 315
Logs: /data/gdh/Generative-vs-Discriminative-Classifiers/outputs/paper_repro/local_logs
```

每个任务结束后会打印类似：

```text
[2026-06-23 20:15:31] [GPU 0] DONE 12/315 enc agnews sample=2048 seed=79140 layers=1 job=00:03:18 elapsed=00:42:10 avg/job=00:03:30 ETA=17:40:30
```

其中 `job` 是当前任务耗时，`elapsed` 是本轮脚本已经运行的时间，`avg/job`
是已完成任务的平均耗时，`ETA` 是粗略预计剩余时间。后面的 12 层和 full-data
任务通常更慢，所以 ETA 只是参考，不是精确倒计时。

后台运行：

```bash
nohup bash -c 'PRECISION=fp16 bash repro_fig2/run_job_file_local.sh \
  "$ROOT/repro_fig2/jobs_figure2_1seed.tsv" \
  "$ROOT/outputs/paper_repro" \
  0,1,2,3' > "$ROOT/outputs/run_figure2_1seed.log" 2>&1 &
```

监控：

```bash
watch -n 2 nvidia-smi
tail -f "$ROOT/outputs/run_figure2_1seed.log"
find "$ROOT/outputs/paper_repro" -name metrics.json | wc -l
```

如果你已经用旧脚本启动了任务，旧日志不会显示 ETA。可以先用下面两个命令估计进度：

```bash
find "$ROOT/outputs/paper_repro" -name metrics.json | wc -l
ps -eo pid,etime,cmd | grep train_one.py | grep -v grep
```

跑满主实验后，`metrics.json` 数量应接近 `315`。

---

## 7. 对 AG News + SST-5 补 3 seeds

这一步用于课程报告重点分析。

生成任务：

```bash
python repro_fig2/make_jobs_paper.py \
  --preset figure2 \
  --datasets agnews,sst5 \
  --models enc,ar,mlm \
  --out "$ROOT/repro_fig2/jobs_agnews_sst5_3seed.tsv"
```

任务数：

```text
2 datasets * 3 models * 3 layers * 7 sample sizes * 3 seeds = 378
```

运行：

```bash
PRECISION=fp16 bash repro_fig2/run_job_file_local.sh \
  "$ROOT/repro_fig2/jobs_agnews_sst5_3seed.tsv" \
  "$ROOT/outputs/paper_repro" \
  0,1,2,3
```

已经跑过的任务会自动跳过。

---

## 8. 汇总结果并画图

主图：

```bash
python repro_fig2/aggregate_and_plot.py \
  --output_root "$ROOT/outputs/paper_repro" \
  --initialization scratch \
  --layers 1 6 12 \
  --datasets agnews emotion rottentomatoes sst5 twitter \
  --full_grid
```

主要输出：

```text
outputs/paper_repro/summary_layers_1_6_12.csv
outputs/paper_repro/all_run_metrics.csv
outputs/paper_repro/figures/weighted_f1_grid_layers_1_6_12.png
outputs/paper_repro/figures/ece_grid_layers_1_6_12.png
outputs/paper_repro/figures/mce_grid_layers_1_6_12.png
outputs/paper_repro/figures/ordinal_mae_grid_layers_1_6_12.png
outputs/paper_repro/figures/ordinal_mse_grid_layers_1_6_12.png
outputs/paper_repro/figures/ordinal_um_grid_layers_1_6_12.png
```

你论文报告中的 Figure 2 复现图主要用：

```text
weighted_f1_grid_layers_1_6_12.png
```

---

## 9. 快速查看 AG News + SST-5 的数值结果

```bash
python - <<'PY'
import pandas as pd

path = "outputs/paper_repro/summary_layers_1_6_12.csv"
df = pd.read_csv(path)
df = df[df["dataset"].isin(["agnews", "sst5"])]
df = df[df["layers"] == 12]
df = df.sort_values(["dataset", "model", "sample_size"])

cols = [
    "dataset",
    "model",
    "sample_size",
    "weighted_f1_mean",
    "weighted_f1_std",
]
for extra in ["ece_mean", "mce_mean", "mae_mean", "mse_mean", "um_mean"]:
    if extra in df.columns:
        cols.append(extra)

print(df[cols].to_string(index=False))
PY
```

如果某些列是空的，通常说明对应实验还没跑完。

---

## 10. 画 Reliability Diagram 扩展图

建议先画 SST-5，12 层，full-data。如果 full-data 没跑完，把 `--sample_size -1`
改成 `4096`。

```bash
python repro_fig2/plot_reliability.py \
  --output_root "$ROOT/outputs/paper_repro" \
  --dataset sst5 \
  --layers 12 \
  --sample_size -1 \
  --models enc ar mlm
```

输出：

```text
outputs/paper_repro/figures/reliability_sst5_12L_-1.png
```

---

## 11. 最后怎么分析

报告里按这个顺序写。

### 11.1 主实验：weighted-F1 样本效率

看：

```text
weighted_f1_grid_layers_1_6_12.png
summary_layers_1_6_12.csv
```

分析重点：

1. 小样本区间：`128, 256, 512`
   - 比较 AR 和 ENC 谁更高。
   - 如果 AR 更稳，可以说生成式 `P(text|label)` 在小数据下更能利用类别条件建模。

2. 中大样本区间：`1024, 2048, 4096, full`
   - 看 ENC 是否随样本增加上升更快。
   - 如果 ENC 在大样本超过 AR，可以对应论文的“两制度”现象。

3. 模型规模：
   - 比较 1/6/12 层三行。
   - 看 ENC 是否更吃模型规模，AR 是否在小模型或小数据下更稳定。

### 11.2 AG News vs SST-5

AG News：

```text
4 类主题分类，类别均衡，标签无顺序
```

分析重点：

```text
更适合看普通分类性能和大样本下 ENC 是否反超
```

SST-5：

```text
5 类细粒度情感，标签有自然顺序
```

分析重点：

```text
既能看 weighted-F1，也能看 MAE/MSE/UM 等有序性指标
```

### 11.3 校准度：ECE/MCE

看：

```text
ece_grid_layers_1_6_12.png
mce_grid_layers_1_6_12.png
```

结论写法：

```text
ECE/MCE 越低，说明模型置信度越接近真实正确率。
如果某模型 weighted-F1 高但 ECE 也高，说明它虽然分类准，但概率不一定可靠。
```

### 11.4 有序性：只重点解释 SST-5

看：

```text
ordinal_mae_grid_layers_1_6_12.png
ordinal_mse_grid_layers_1_6_12.png
ordinal_um_grid_layers_1_6_12.png
```

解释：

```text
MAE/MSE 越低，预测标签离真实标签越近。
UM 越高，预测概率越符合单峰结构。
SST-5 标签从 very negative 到 very positive 有自然顺序，所以这些指标有意义。
AG News 标签没有顺序，因此不解释 MAE/MSE/UM。
```

### 11.5 扩展分析

可以选一个写：

1. Reliability Diagram：
   - 看 confidence 和 accuracy 是否贴近对角线。
   - 越贴近，校准越好。

2. 鲁棒性：
   - 对 full-data checkpoint 做 token drop/substitution。
   - 看噪声增加时 weighted-F1 掉得快不快。

---

## 12. 最终应该交哪些文件

建议课程报告至少包含：

```text
1. weighted_f1_grid_layers_1_6_12.png
2. AG News + SST-5 的 12 层数值表
3. ece_grid_layers_1_6_12.png 或 mce_grid_layers_1_6_12.png
4. SST-5 的 ordinal_mae/mse/um 图
5. reliability_sst5_12L_-1.png 或鲁棒性图
6. summary_layers_1_6_12.csv
7. 说明哪些任务跑完，哪些因为时间/资源没有跑
```

如果时间不够，最低交付：

```text
Figure 2 风格 weighted-F1 图
AG News + SST-5 的 3 seeds 结果
SST-5 的 ECE/MCE/MAE/MSE/UM 扩展分析
```
