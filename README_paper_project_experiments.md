# 论文与项目详细分析：Generative or Discriminative?

本文档对应本仓库根目录下的论文 `Generative or Discriminative.pdf` 和项目代码，目标有两个：

1. 说明论文和项目分别在做什么，以及每个方法、数据集、指标、训练设置的关键细节。
2. 将论文中的实验与项目中的代码入口逐项对应，给出可执行的运行方式，并标明哪些实验需要补充脚本才能完整复现。

## 一、论文在做什么

论文题目是 **Generative or Discriminative? Revisiting Text Classification in the Era of Transformers**。核心问题是：在 Transformer 时代，文本分类到底应该优先选择判别式模型，还是生成式模型？

传统理论认为生成式分类器在低数据量下样本效率更高，但数据充足后判别式模型渐近误差更低。论文把这个经典问题放到现代 Transformer 架构里重新评估，比较了 5 类文本分类范式：

- `ENC`：判别式 Encoder 分类器，直接学习 `P(y|X)`。
- `AR`：自回归生成式分类器，学习带标签条件的序列似然，通过 `argmax_y log P(X|y)` 分类。
- `ARpseudo`：伪自回归生成式分类器，把标签放在文本末尾，只需一次前向传播预测标签 token。
- `MLM`：Masked Language Modeling 伪生成式分类器，在文本后拼接标签模板，用 `[MASK]` 预测标签。
- `DIFF`：离散文本扩散分类器，把标签 token 作为待恢复位置，通过离散去噪预测类别。

### 1.1 论文方法细节

#### ENC：判别式 Encoder 分类

Encoder 用 Transformer 编码输入文本 `X`，取句向量后接线性分类头：

```text
h = f_theta(X)
y_hat = softmax(W h)
L_enc = - log P(y | X)
```

这是标准监督分类训练方式，优点是推理快、分类目标直接、概率分数通常更稳定；缺点是低样本时可能不如生成式模型充分利用文本分布。

#### MLM：掩码语言模型式分类

训练时把文本和标签拼在一起，例如：

```text
X [SEP] The label is y
```

再随机 mask 15% token，用 MLM 损失训练。推理时使用：

```text
X [SEP] The label is [MASK]
```

只取 `[MASK]` 位置上合法标签 token 的概率并重新归一化作为类别概率。论文把它视为伪生成式模型，因为 MLM 近似 pseudo-likelihood，而不是直接学习 `P(y|X)`。

#### AR：自回归生成式分类

训练时把标签放在文本前面，学习完整序列的 next-token likelihood：

```text
Label:y, Text:X
```

推理时对每个候选标签都跑一次前向传播，计算 `log P(X|y)`，选择似然最高的标签：

```text
y_hat = argmax_y log P(X | y)
```

优点是低数据场景更接近经典生成式分类器；缺点是推理要跑 `类别数` 次前向传播，延迟和计算更高。

#### ARpseudo：伪自回归分类

训练时把标签放到文本末尾：

```text
text: X, label: y
```

推理时输入：

```text
text: X, label:
```

模型直接在最后一个位置预测标签 token。它牺牲了严格的 `P(X|y)` 建模，但推理只需要一次前向传播。论文发现它在 full-data 场景可能更快、更强，但在低数据下通常不如 AR 稳定。

#### DIFF：离散文本扩散分类

论文采用 SEDD 风格的离散扩散模型。训练时使用：

```text
X [SEP] The label is y
```

扩散前向过程逐步把 token 腐蚀为 `[MASK]` 或吸收态；反向过程学习从噪声恢复真实 token。分类时固定文本部分，只恢复标签 token，并限制输出为合法标签集合。

DIFF 的优势是天然训练在噪声恢复任务上，因此对 token drop/substitution 的鲁棒性较好；缺点是推理要多步采样，延迟远高于 ENC/MLM/AR。

## 二、论文实验设计

### 2.1 主实验 Q1：从零训练时，不同范式表现如何

论文比较 5 个方法：

```text
ENC, AR, ARpseudo, MLM, DIFF
```

使用 9 个文本分类数据集：

| 论文名称 | Hugging Face / 项目键 | 类别数 | 类型 |
| --- | --- | ---: | --- |
| AG News | `ag_news` | 4 | 新闻主题 |
| Emotion | `emotion` | 6 | 情绪分类 |
| SST-2 | `SetFit/sst2` | 2 | 二分类情感 |
| SST-5 | `SetFit/sst5` | 5 | 五分类情感，有序 |
| Multiclass Sentiment | `Sp1786/multiclass-sentiment-analysis-dataset` | 3 | 情感，有序 |
| Twitter Financial News | `zeroshot/twitter-financial-news-sentiment` | 3 | 金融推文情感，有序 |
| IMDb | `imdb` | 2 | 长文本情感 |
| Rotten Tomatoes | `cornell-movie-review-data/rotten_tomatoes` | 2 | 短句影评 |
| Hate Speech Offensive | `SetFit/hate_speech_offensive` | 3 | 冒犯/仇恨言论，有序 |

训练样本规模：

```text
128, 256, 512, 1024, 2048, 4096, full_data
```

模型规模：

```text
small  = 1 layer, 1 head
medium = 6 layers, 6 heads
large  = 12 layers, 12 heads
```

每组实验重复 3 个随机种子，总量为：

```text
9 datasets * 7 sample sizes * 3 model sizes * 3 seeds * 5 methods = 2835 experiments
```

主指标是 `weighted-F1`，图 2、图 3 和附录图 8、图 9 展示主要结果。

### 2.2 Q2：输入噪声鲁棒性

论文在 full-data 设置下比较 6 层和 12 层模型的鲁棒性，不包含 1 层模型，因为多数 1 层模型性能接近随机。

两类噪声：

- Random Token Drop：随机删除 `X%` token。
- Random Token Substitution：随机把 `X%` token 替换为词表中的随机 token，排除 `[PAD]`、`[MASK]` 等特殊符号。

论文报告的是 “达到某个 F1 降幅所需要的最低噪声比例”，例如性能下降 5%、10%、15%、20%、30% 时所需的噪声比例。这个数越高，模型越鲁棒。

主要结论：

- 所有模型对 substitution 都比对 dropping 更敏感。
- DIFF 整体最抗噪，因为训练本身就是噪声恢复。
- ENC 鲁棒性稳定。
- AR 和 ARpseudo 对噪声更敏感，尤其 ARpseudo 容易受输入中垃圾 token 影响。

### 2.3 Q3：校准与有序性

论文关注真实部署中常被忽略的概率质量：

- `ECE`：Expected Calibration Error，越低越好。
- `MCE`：Maximum Calibration Error，越低越好。
- `MAE`：有序标签平均绝对误差，越低越好。
- `MSE`：有序标签均方误差，越低越好。
- `UM`：Unimodality，预测分布是否单峰，越高越好。

有序任务主要包括：

```text
SST-5, Multiclass Sentiment, Hate Speech Offensive, Twitter Financial News Sentiment
```

关键结论：

- ENC 的校准最稳定。
- MLM 在大模型和高数据量下既有高 F1，又有较好校准和有序性。
- DIFF 在 absorbing noise 设置下不能自然输出完整 soft probability，因此论文不对它报告 ECE/MCE/UM。
- 大模型 F1 变高不代表校准一定更好，校准有时会持平甚至变差。

### 2.4 预训练模型补充实验

论文还比较了预训练初始化下的 ENC 和 AR：

- ENC 使用 BERT-base。
- AR 使用 GPT-2 base。

结论是：预训练后经典 “two regimes” 现象基本消失，ENC 在大多数数据集和数据规模下持续优于 AR。论文解释为预训练已经提供了近似 “海量数据” 的先验表示，削弱了生成式模型在低数据下的样本效率优势。

## 三、论文主要结论

论文最终给出面向实践的建议：

| 场景 | 推荐 |
| --- | --- |
| 延迟敏感、模型很小 | 优先 `ENC`，尤其 1 层模型 |
| 低资源、低样本、可接受较大模型 | `AR` 或 `DIFF` |
| 低样本且重视噪声鲁棒性 | `DIFF` 更合适 |
| 低样本且必须要概率校准 | `AR` 比 DIFF 更合适 |
| full-data 且离线训练/推理 | 12 层 `MLM` 表现很强 |
| 需要概率可作为下游排序特征 | 优先 `ENC` 或高数据下的 `MLM` |
| 直接使用预训练模型 | `ENC` 通常更稳 |

一句话概括：**Transformer 时代的生成式 vs 判别式不再是简单的低样本/高样本二分，而是同时受模型规模、推理延迟、噪声鲁棒性、概率校准和是否预训练影响。**

## 四、项目在做什么

本项目是论文的实验代码集合，按模型范式分成 4 个主要目录：

```text
.
├── ar/                  # AR 自回归生成式分类
├── ar_pseudo/           # ARpseudo 伪自回归分类
├── encoder_mlm/         # ENC 和 MLM 的 BERT 系实验
├── diff/                # DIFF 离散文本扩散实验
├── examples/            # 综合运行脚本
├── environment.yml      # AR / ARpseudo / ENC / MLM 共用环境
├── diff/environment.yml # diffusion 单独环境
├── Readme.md            # 原项目 README
└── Generative or Discriminative.pdf
```

### 4.1 环境

AR、ARpseudo、ENC、MLM 共用根目录环境：

```bash
conda env create -f environment.yml
conda activate gendisc-transformers
```

DIFF 使用单独环境：

```bash
cd diff
conda env create -f environment.yml
conda activate sedd
```

注意：

- 代码依赖 Hugging Face `datasets` 和 `transformers`，首次运行需要联网下载数据和 tokenizer/model config。
- 训练脚本默认使用 GPU。AR/ARpseudo 使用 PyTorch Lightning DDP；DIFF 使用 `torch.multiprocessing` + NCCL。
- `diff/configs/config.yaml` 默认 `ngpus: 8`，更接近论文的 8 张 A100 设置；单卡运行必须覆盖 `ngpus=1`。
- 当前仓库只有根目录 `environment.yml` 和 `diff/environment.yml`。`validate_setup.py` 里检查的 `ar/environment.yml`、`encoder_mlm/environment.yml` 等文件在当前仓库并不存在，因此该校验脚本有过时项。

## 五、各模块代码细节

### 5.1 `ar/`：AR 自回归生成式分类

入口：

- `ar/train_gpt.py`
- `ar/infer_gpt.py`

训练数据格式：

```text
Label:{label},Text:{text}
```

训练目标：

- 使用 GPT-2 tokenizer。
- 从 GPT-2 config 初始化模型，但不加载 GPT-2 预训练权重。
- 训练完整序列 next-token prediction loss。

模型规模：

| 参数 | 实际配置 |
| --- | --- |
| `--model_size small` | 1 layer, 1 head, embedding 64 |
| `--model_size medium` | 6 layers, 6 heads, embedding 384 |
| `--model_size full` | GPT-2 base 默认配置，12 layers, 12 heads, embedding 768 |
| `--model_size large` | 代码未显式判断，但会保留 GPT-2 base 默认配置，等价于 12 layers |

验证与 checkpoint：

- 训练时将 test split 或 validation split 前 480 条用于验证。
- 每个 epoch 用 AR 的多标签似然推理方式计算 weighted-F1。
- checkpoint 文件名中包含 `macro_f1`、`weighted_f1`、`accuracy`。
- checkpoint selection 监控 `weighted_f1`，与论文中对 AR 使用下游分类指标选 checkpoint 的描述一致。

推理：

- 对每个候选标签构造 `Label:{label},Text:{text}`。
- 分别计算语言模型 loss，取负 loss 作为 log-likelihood。
- 对所有标签 softmax，输出 `predictions_format.csv`：

```text
ground_truth,predicted_label,scores
```

### 5.2 `ar_pseudo/`：伪自回归生成式分类

入口：

- `ar_pseudo/train_gpt.py`
- `ar_pseudo/infer_gpt.py`

训练输入：

```text
text: {text}, label:{label}
```

推理输入：

```text
text: {text}, label:
```

核心差异：

- 标签 token 被追加到文本末尾。
- 推理时只需要一次前向传播，取最后位置上合法 label token 的 softmax。
- 代码同样从 GPT-2 config 随机初始化，而不是加载预训练 GPT-2 权重。

模型规模和 checkpoint 逻辑基本与 `ar/` 相同。

### 5.3 `encoder_mlm/`：ENC 与 MLM

入口：

- `encoder_mlm/mlm_classif_seed_fixed.py`
- `encoder_mlm/inference.py`

`mlm_classif_seed_fixed.py` 同时跑两种策略：

```python
training_strategy in ["mlm", "classification"]
```

其中：

- `classification` 对应论文 `ENC`。
- `mlm` 对应论文 `MLM`。

模型配置：

```python
AutoConfig.from_pretrained("bert-base-uncased")
config.num_hidden_layers = num_layers
config.num_attention_heads = num_heads
config.hidden_size = int((768 * num_heads) // 12)
```

因此：

| `(layers, heads)` | 论文规模 |
| --- | --- |
| `(1, 1)` | small |
| `(6, 6)` | medium |
| `(12, 12)` | large |

MLM 数据格式：

```text
{text} Label:{label}
```

ENC 数据格式：

```text
{text}
```

输出目录：

```text
encoder_mlm/models/
encoder_mlm/models-trained/
encoder_mlm/logs/
```

推理脚本 `inference.py` 会根据 checkpoint 目录名解析 dataset 和 strategy，例如：

```text
bert_dataset=hatespeech_strategy=classification_samples=128_layers=6_heads=6_seed=79140
```

然后：

- `classification` 使用 `AutoModelForSequenceClassification`。
- `mlm` 使用 `AutoModelForMaskedLM`，构造 `text Label:[MASK]`，只在合法标签 token 集合内选择。
- 输出 `results.csv`，并打印 `classification_report`。

当前代码注意点：

- `DATASET_PATH` 当前只打开了 `hatespeech`，其他数据集被注释。要复现论文 9 个数据集，需要编辑 `DATASET_PATH`。
- `MODEL_CONFIGS` 当前是 `[(1,1),(6,6)]`，缺少论文的 `(12,12)`，需要打开。
- `SAMPLE_SIZES` 当前含 `[-1,4096,2048,1024,512,256,128]`，其中 `-1` 表示 full data。
- 原生脚本在 `sample_size == -1` 后会 `break`，因此 full-data 当前只跑第一个 seed；论文主实验要求 full-data 也跑 3 个 seed，严格复现时需要移除这段 `break` 或改写循环。
- 训练 epoch 当前是 200；论文附录中 ENC 说 30 epochs、MLM 说 200 epochs。当前统一 200，若严格复现，需要按 strategy 拆开设置。
- IMDb 数据集在 Hugging Face 上通常使用小写 `imdb`。如果在 encoder 脚本中用键名 `"IMDb"`，建议把 `encoder_mlm/mlm_classif_seed_fixed.py` 和 `encoder_mlm/inference.py` 里的映射都统一成 `"IMDb": "imdb"`，否则训练和推理阶段可能不一致。

### 5.4 `diff/`：离散扩散分类

入口：

- `diff/train.py`
- `diff/run_train.py`
- `diff/data.py`
- `diff/parallel_inference.py`

配置：

- 主配置：`diff/configs/config.yaml`
- 模型配置：
  - `diff/configs/model/small.yaml`
  - `diff/configs/model/medium.yaml`
  - `diff/configs/model/large.yaml`
- 数据集配置：`diff/dataset_config.yaml`

模型规模：

| Hydra 参数 | 层数/头数 |
| --- | --- |
| `model=small` | 1 block, 1 head |
| `model=medium` | 6 blocks, 6 heads |
| `model=large` | 12 blocks, 12 heads |

训练数据格式在 `diff/data.py`：

```text
{text}. Label:{label}
```

扩散图：

- 默认 `graph.type: absorb`，即吸收/掩码扩散。
- 默认 `noise.type: loglinear`。
- `tokens: 50257`，使用 GPT-2 tokenizer 词表。
- `pad_token_id: 50256`。

训练目录：

```text
diff/exp_local/{dataset}_6/{date}/{time}_size{train_size}/
├── .hydra/
├── checkpoints/
├── checkpoints-meta/checkpoint.pth
├── samples/
└── logs
```

推理：

- `parallel_inference.py` 加载训练目录。
- 构造 `text. Label:` 模板。
- 固定 label 前的位置，使用扩散采样生成 label。
- 从生成文本中的 `Label:` 后提取第一个数字作为预测类别。
- 打印 `classification_report`。

当前代码注意点：

- 默认 `ngpus: 8`，单卡运行需要覆盖。
- `parallel_inference.py` 使用 `torch.cuda.device_count()` 作为 world size；没有 GPU 时无法正常跑。
- 当前推理脚本打印报告，但不像 AR/ENC 那样稳定保存预测 CSV。

## 六、论文实验与项目代码对应

### 6.1 Q1 主实验：weighted-F1 vs 样本规模/模型规模

#### 6.1.1 AR 对应论文 `AR`

单个实验训练：

```bash
cd ar

python train_gpt.py \
  --data_key "SetFit/sst2" \
  --ckpt_dir "../experiment_results/ar/sst2_samples128_small_seed42" \
  --model_size "small" \
  --seed 42 \
  --n_tr_sub 128 \
  --max_epochs 100 \
  --bsz 8 \
  --n_devices 1 \
  --max_len 512
```

full-data 时去掉 `--n_tr_sub` 或设为 `-1`：

```bash
python train_gpt.py \
  --data_key "SetFit/sst2" \
  --ckpt_dir "../experiment_results/ar/sst2_full_large_seed42" \
  --model_size "large" \
  --seed 42 \
  --max_epochs 100 \
  --bsz 8 \
  --n_devices 1 \
  --max_len 512
```

推理：

```bash
python infer_gpt.py \
  --data_key "SetFit/sst2" \
  --data_name "sst2_samples128_small_seed42" \
  --base_dpath "../experiment_results/ar" \
  --bsz 16 \
  --max_len 512
```

说明：

- `--data_name` 必须等于 checkpoint 所在子目录名。
- `infer_gpt.py` 会在该目录下寻找 `gpt2-*` checkpoint。
- 输出文件在 checkpoint 目录下：`predictions_format.csv`。

要复现论文 AR 的 Q1，循环：

```text
dataset in 9 datasets
sample_size in 128,256,512,1024,2048,4096,-1
model_size in small,medium,large
seed in 3 seeds
```

#### 6.1.2 ARpseudo 对应论文 `ARpseudo`

训练：

```bash
cd ar_pseudo

python train_gpt.py \
  --data_key "SetFit/sst2" \
  --ckpt_dir "../experiment_results/ar_pseudo/sst2_samples128_small_seed42" \
  --model_size "small" \
  --seed 42 \
  --n_tr_sub 128 \
  --max_epochs 100 \
  --bsz 8 \
  --n_devices 1 \
  --max_len 512
```

推理：

```bash
python infer_gpt.py \
  --data_key "SetFit/sst2" \
  --data_name "sst2_samples128_small_seed42" \
  --base_dpath "../experiment_results/ar_pseudo" \
  --bsz 1024 \
  --max_len 512
```

输出同样是 checkpoint 目录下的 `predictions_format.csv`。

#### 6.1.3 ENC 对应 `encoder_mlm/mlm_classif_seed_fixed.py` 的 `classification`

当前脚本是硬编码批量实验。严格复现论文前，先编辑文件顶部：

```python
SEEDS = [42, 123, 456]
SAMPLE_SIZES = [-1, 4096, 2048, 1024, 512, 256, 128]
MODEL_CONFIGS = [(1,1), (6,6), (12,12)]
DATASET_PATH = {
    "IMDb": "imdb",
    "agnews": "ag_news",
    "emotion": "emotion",
    "hatespeech": "SetFit/hate_speech_offensive",
    "multiclasssentiment": "Sp1786/multiclass-sentiment-analysis-dataset",
    "rottentomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "sst2": "SetFit/sst2",
    "sst5": "SetFit/sst5",
    "twitter": "zeroshot/twitter-financial-news-sentiment",
}
```

如果只跑 ENC，把循环里的策略改为：

```python
for training_strategy in ["classification"]:
```

然后运行：

```bash
cd encoder_mlm
python mlm_classif_seed_fixed.py
```

推理：

```bash
python inference.py \
  --checkpoint_path "./models-trained/bert_dataset=sst2_strategy=classification_samples=128_layers=6_heads=6_seed=42"
```

输出：

```text
encoder_mlm/results.csv
```

#### 6.1.4 MLM 对应 `encoder_mlm/mlm_classif_seed_fixed.py` 的 `mlm`

配置同 ENC，只是策略改为：

```python
for training_strategy in ["mlm"]:
```

训练：

```bash
cd encoder_mlm
python mlm_classif_seed_fixed.py
```

推理：

```bash
python inference.py \
  --checkpoint_path "./models-trained/bert_dataset=sst2_strategy=mlm_samples=128_layers=6_heads=6_seed=42"
```

说明：

- MLM 推理输出的 `scores` 是合法标签 token 的概率字典。
- 论文中 MLM 用 `[MASK]` 预测标签；代码中模板是 `"{text} Label:{mask}"`，与论文思想一致，但文字模板略有差异。

#### 6.1.5 DIFF 对应 `diff/train.py`

单个实验训练：

```bash
cd diff

DATASET_NAME="SetFit/sst2" \
TRAIN_SIZE="128" \
N_ITERS="50000" \
python train.py model=small ngpus=1 training.batch_size=8 eval.batch_size=8
```

full-data：

```bash
DATASET_NAME="SetFit/sst2" \
N_ITERS="150000" \
python train.py model=large ngpus=1 training.batch_size=8 eval.batch_size=8
```

如果使用论文类似 8 卡设置，可使用默认 `ngpus: 8`，或显式：

```bash
DATASET_NAME="SetFit/sst2" \
TRAIN_SIZE="128" \
N_ITERS="150000" \
python train.py model=medium ngpus=8
```

推理：

```bash
python parallel_inference.py \
  --model_path "exp_local/SetFit/sst2_6/2026.05.18/123456_size128" \
  --dataset "SetFit/sst2" \
  --batch_size 256 \
  --steps 16
```

说明：

- `--model_path` 指向一次训练生成的 Hydra 运行目录，目录下应包含 `.hydra/` 和 `checkpoints-meta/checkpoint.pth`。
- `--steps` 默认 16，注释中写到原始可用 1024，但 16 更快。
- `parallel_inference.py` 当前只打印中间和最终分类报告，不稳定保存 CSV。

### 6.2 一键脚本与论文实验矩阵

项目提供：

```bash
./examples/run_comprehensive_experiments.sh demo
./examples/run_comprehensive_experiments.sh ar
./examples/run_comprehensive_experiments.sh diffusion
./examples/run_comprehensive_experiments.sh encoder
./examples/run_comprehensive_experiments.sh full
```

它做了三件事：

- 对 AR 循环数据集、样本规模、模型规模、seed。
- 对 DIFF 循环数据集、样本规模、模型规模、seed。
- 对 encoder/MLM 临时生成一个 `comprehensive_experiments.py` 再运行。

但要注意：

- `demo` 只跑 AR 和 DIFF 的小样本示例，不跑 encoder/MLM。
- `encoder` 默认只跑 `sst2 emotion ag_news`，不是论文的 9 个数据集。
- `AR_MODEL_SIZES=("small" "medium" "large")` 中 `large` 在代码里等价于 GPT-2 base 默认 12 层。
- `DIFF_MODEL_SIZES` 包含 small/medium/large，与配置文件完全对应。
- 论文完整 2835 组实验计算量非常大，不建议在普通单卡机器上直接跑 `full`。

## 七、Q2 鲁棒性实验如何对应项目

论文 Q2 没有在当前仓库中提供独立的鲁棒性复现实验脚本。当前可复用的是训练和推理入口，需要补一个 perturbation wrapper。

### 7.1 论文需要的流程

对每个方法、每个 full-data checkpoint、每个噪声比例：

1. 读取 test/validation split。
2. 对文本做随机 token drop 或 random token substitution。
3. 调用对应模型推理。
4. 计算 weighted-F1。
5. 和该模型无噪声 peak weighted-F1 比较，找到下降 5%、10%、15%、20%、30% 所需的最小噪声比例。

### 7.2 各方法可插入扰动的位置

| 方法 | 建议插入位置 |
| --- | --- |
| AR | `ar/infer_gpt.py` 中 `compute_log_likelihoods()` 调用前，对 `text` 做扰动 |
| ARpseudo | `ar_pseudo/infer_gpt.py` 中构造 `val_dataset` 前，对 `val_texts` 做扰动 |
| ENC/MLM | `encoder_mlm/inference.py` 中遍历 `dataset_test` 后，对 `text` 做扰动 |
| DIFF | `diff/parallel_inference.py` 的 `get_dataset_agnews()` / `preprocess_and_tokenize()` 里，在 tokenization 前扰动文本 |

### 7.3 噪声实现建议

Token drop：

```python
def token_drop(tokens, pct, rng):
    keep = [tok for tok in tokens if rng.random() > pct]
    return keep if keep else tokens[:1]
```

Token substitution：

```python
def token_substitute(token_ids, pct, vocab_ids, special_ids, rng):
    out = []
    valid_vocab = [x for x in vocab_ids if x not in special_ids]
    for tok in token_ids:
        if rng.random() < pct and tok not in special_ids:
            out.append(rng.choice(valid_vocab))
        else:
            out.append(tok)
    return out
```

为了与论文更接近，建议在 tokenizer token 级别扰动，而不是空格词级别扰动。对 BERT 系模型使用 BERT tokenizer；对 GPT/DIFF 使用 GPT-2 tokenizer。

## 八、Q3 校准与有序性实验如何对应项目

当前仓库没有独立的 ECE/MCE/UM/MAE/MSE 脚本，但 AR、ARpseudo、ENC/MLM 的推理输出已经包含计算指标所需的大部分信息。

### 8.1 可用输出

| 方法 | 输出 |
| --- | --- |
| AR | `predictions_format.csv`，含 `ground_truth,predicted_label,scores` |
| ARpseudo | `predictions_format.csv`，含 `ground_truth,predicted_label,scores` |
| ENC | `results.csv`，含 `ground_truth,predicted_label,scores` |
| MLM | `results.csv`，含 `ground_truth,predicted_label,scores`，scores 是标签到概率的字典 |
| DIFF | 当前主要打印 hard-label `classification_report` |

### 8.2 指标计算方式

ECE：

1. 对每条样本取 `confidence = max(class_probs)`。
2. 判断 `correct = predicted_label == ground_truth`。
3. 按 confidence 分桶。
4. 对所有桶计算 `|acc(bucket) - conf(bucket)|` 的加权平均。

MCE：

```text
max_bucket |acc(bucket) - conf(bucket)|
```

MAE / MSE：

```text
MAE = mean(abs(predicted_label - ground_truth))
MSE = mean((predicted_label - ground_truth)^2)
```

如果使用完整概率分布，也可先计算期望标签：

```text
expected_label = sum_k k * P(k)
```

再与真实标签计算 MAE/MSE。

UM：

检查类别概率序列是否单峰。对于有序标签 `0,1,2,...,K-1`，理想分布应先非降后非增，不能在两端同时高置信。

### 8.3 DIFF 的特殊情况

论文指出，默认 absorbing diffusion 设置下 DIFF 不自然输出完整类别概率，因此不报告 ECE、MCE、UM。当前项目也主要输出生成出的 hard label。如果要给 DIFF 做概率类指标，需要改用 uniform noise 或额外统计多次采样频率，但这已经不是当前代码的原始复现路径。

## 九、预训练实验如何对应项目

论文补充实验使用预训练 BERT-base 和 GPT-2 base 对比 ENC 与 AR。当前代码默认都是从 config 随机初始化，不加载预训练权重：

- `ar/train_gpt.py` 使用 `GPT2LMHeadModel(self.config)`，不是 `from_pretrained("gpt2")`。
- `encoder_mlm/mlm_classif_seed_fixed.py` 使用 `AutoModelForSequenceClassification.from_config(config)` 和 `AutoModelForMaskedLM.from_config(config)`。

因此，预训练实验当前没有直接脚本。若要实现：

AR 需要把模型初始化改为：

```python
GPT2LMHeadModel.from_pretrained("gpt2")
```

ENC 需要把模型初始化改为：

```python
AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)
```

并保持相同数据规模和 seed 循环。

## 十、推荐复现实验顺序

### 10.1 先跑最小 smoke test

```bash
./examples/run_comprehensive_experiments.sh demo
```

如果只想手动确认 AR：

```bash
cd ar
python train_gpt.py \
  --data_key "SetFit/sst2" \
  --ckpt_dir "../experiment_results/smoke/ar_sst2" \
  --model_size "small" \
  --seed 42 \
  --n_tr_sub 100 \
  --max_epochs 3 \
  --bsz 4 \
  --n_devices 1 \
  --max_len 128
```

### 10.2 再跑单数据集完整 Q1

建议先选 `SetFit/sst2` 或 `emotion`，跑：

```text
5 methods * 7 sample sizes * 3 model sizes * 3 seeds
```

确认日志、checkpoint、推理 CSV、weighted-F1 聚合都正常后，再扩展到 9 个数据集。

### 10.3 最后补 Q2/Q3

Q2/Q3 需要在现有推理输出上补指标脚本：

- Q2：增加 test-time perturbation。
- Q3：从 `scores` 解析概率，计算 ECE/MCE/UM/MAE/MSE。

## 十一、当前项目与论文的覆盖关系总结

| 论文内容 | 当前项目是否直接覆盖 | 说明 |
| --- | --- | --- |
| AR 从零训练 | 是 | `ar/train_gpt.py` |
| AR 推理与 weighted-F1 | 是 | `ar/infer_gpt.py` |
| ARpseudo 从零训练 | 是 | `ar_pseudo/train_gpt.py` |
| ARpseudo 推理与 weighted-F1 | 是 | `ar_pseudo/infer_gpt.py` |
| ENC 从零训练 | 部分覆盖 | `encoder_mlm/mlm_classif_seed_fixed.py`，但需打开全部数据集和 12 层配置 |
| MLM 从零训练 | 部分覆盖 | 同上 |
| DIFF 从零训练 | 是 | `diff/train.py` + Hydra 配置 |
| DIFF 推理 | 部分覆盖 | `parallel_inference.py` 可打印报告，但输出文件不完善 |
| 9 数据集主实验 | 部分覆盖 | 数据集键基本齐全，但 encoder 原脚本默认只开 hatespeech |
| 7 个样本规模 | 部分覆盖 | AR/DIFF 支持；encoder 脚本支持但硬编码 |
| 3 模型规模 | 部分覆盖 | AR/DIFF 支持；encoder 需打开 `(12,12)` |
| 3 seeds | 是 | 各脚本支持或硬编码 |
| Q2 噪声鲁棒性 | 需要补脚本 | 当前无独立 perturbation/evaluation pipeline |
| Q3 校准/有序性 | 需要补脚本 | 当前有概率输出，但无指标计算脚本 |
| 预训练 ENC vs AR | 需要改代码 | 当前默认从零初始化 |

## 十二、常见坑

1. `validate_setup.py` 当前会检查不存在的环境文件，不能完全作为项目健康检查依据。
2. `ar/infer_gpt.py` 和 `ar_pseudo/infer_gpt.py` 通过 `base_dpath/data_name/gpt2-*` 找 checkpoint，训练目录命名要配合。
3. AR 和 ARpseudo 的 `large` 没有显式分支，但会保留 GPT-2 base 默认 12 层配置，因此可视为 large/full。
4. DIFF 默认 8 GPU，单 GPU 必须覆盖 `ngpus`，并确保 batch size 能被 `ngpus * accum` 整除。
5. DIFF 的 `DATASET_NAME` 必须是 `diff/dataset_config.yaml` 里的键。
6. Hugging Face 数据集字段大多假设存在 `text` 和 `label`，新数据集需要改 loader。
7. 论文完整实验量非常大，普通机器应先跑单数据集/单模型规模验证。
8. 当前代码多处使用 test split 做验证或 checkpoint selection，论文也在部分代码注释中承认这只是实验实现方式；若做严格研究复现，最好拆分 validation 和 test。
9. `encoder_mlm/mlm_classif_seed_fixed.py` 对 full-data 的 `-1` 样本规模默认只跑第一个 seed，和论文 “所有实验 3 seeds” 不完全一致。
10. encoder/MLM 的 IMDb 映射要同时改训练和推理脚本，保持 checkpoint 目录名中的 `dataset=...` 能被 `inference.py` 正确解析。

## 十三、最小命令速查

AR：

```bash
cd ar
python train_gpt.py --data_key "SetFit/sst2" --ckpt_dir "../runs/ar_sst2" --model_size small --seed 42 --n_tr_sub 128 --max_epochs 10 --bsz 8 --n_devices 1
python infer_gpt.py --data_key "SetFit/sst2" --data_name "ar_sst2" --base_dpath "../runs"
```

ARpseudo：

```bash
cd ar_pseudo
python train_gpt.py --data_key "SetFit/sst2" --ckpt_dir "../runs/arp_sst2" --model_size small --seed 42 --n_tr_sub 128 --max_epochs 10 --bsz 8 --n_devices 1
python infer_gpt.py --data_key "SetFit/sst2" --data_name "arp_sst2" --base_dpath "../runs"
```

ENC/MLM：

```bash
cd encoder_mlm
python mlm_classif_seed_fixed.py
python inference.py --checkpoint_path "./models-trained/bert_dataset=hatespeech_strategy=classification_samples=128_layers=6_heads=6_seed=79140"
```

DIFF：

```bash
cd diff
DATASET_NAME="SetFit/sst2" TRAIN_SIZE="128" N_ITERS="50000" python train.py model=small ngpus=1 training.batch_size=8 eval.batch_size=8
python parallel_inference.py --model_path "exp_local/SetFit/sst2_6/YYYY.MM.DD/HHMMSS_size128" --dataset "SetFit/sst2" --batch_size 256 --steps 16
```
