# 128–1024 样本量实验结果分析

## 1. 实验设置

结果目录：

```text
outputs/paper_repro/init_scratch/
```

每次实验保留 `args.json`、`metrics.json` 和 `predictions.csv` 三类结果文件：

| 样本量 | 实验数 |
|---:|---:|
| 128 | 45 |
| 256 | 45 |
| 512 | 45 |
| 1024 | 45 |

主实验共有：

```text
5 datasets × 3 models × 3 layer sizes × 4 sample sizes = 180 runs
```

五个数据集为 AG News、Emotion、Rotten Tomatoes、SST-5、Twitter；模型为 ENC、AR、MLM；规模为 1、6、12 层。所有实验只包含种子 `79140`。

## 2. Figure 2 受控复现的主要结果

跨五个数据集和三种模型规模平均：

| 模型 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|
| ENC | 0.311 | 0.305 | 0.386 | **0.401** |
| AR | **0.357** | **0.359** | 0.358 | 0.380 |
| MLM | 0.317 | 0.325 | 0.335 | 0.399 |

结果呈现出较弱的“两阶段”趋势：

- 极低样本量下 AR 平均 weighted-F1 最高。
- ENC 和 MLM 随数据量增加提升更快。
- 到 1024 样本时，ENC 和 MLM 均超过 AR。
- 单个数据集和层数上的曲线波动很大，不能只用平均曲线做模型选择。

在 60 组“数据集 × 层数 × 样本量”横向比较中，有 4 组 ENC 与 MLM 并列。剩余 56 组的独立最优次数为：

| 模型 | 独立最优次数 |
|---|---:|
| AR | 27 |
| ENC | 20 |
| MLM | 9 |

## 3. 模型规模效应

按层数对所有数据集和样本量取平均：

| 模型 | 1 层 | 6 层 | 12 层 |
|---|---:|---:|---:|
| ENC | 0.287 | **0.423** | 0.342 |
| AR | 0.288 | 0.387 | **0.416** |
| MLM | 0.269 | 0.347 | **0.416** |

AR 和 MLM 基本随规模增大而提升；ENC 在 6 层最好，12 层反而下降。主要原因不是“大模型必然更差”，而是随机初始化的大模型在小数据下容易优化失败。

典型案例是 12 层 ENC 在 SST-5 上四档样本量均得到约 0.109 weighted-F1，且测试集只预测一个类别。

## 4. 多数类塌缩

将“超过 95% 的测试样本被预测为同一类别”定义为预测塌缩：

| 模型 | 塌缩配置数 |
|---|---:|
| ENC | 30 |
| MLM | 18 |
| AR | 0 |

总计 48/180 个实验发生严重塌缩。

Twitter 的多数类占测试集约 65.6%，始终预测多数类即可得到约 0.519 weighted-F1。因此，多条完全水平的 0.519 曲线实际上是多数类基线，而不是有效学习结果。对应 Macro-F1 只有约 0.264。

这说明 weighted-F1 必须和 Macro-F1、混淆矩阵及预测类别覆盖一起分析。

## 5. 校准结果

180 个实验的平均概率指标：

| 模型 | weighted-F1 | Macro-F1 | ECE ↓ | MCE ↓ | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ENC | 0.351 | 0.275 | **0.042** | 0.126 | **1.175** | **0.634** |
| AR | **0.364** | **0.328** | 0.076 | **0.116** | 1.312 | 0.708 |
| MLM | 0.344 | 0.269 | 0.108 | 0.304 | 1.285 | 0.682 |

ENC 平均校准最好，但低 ECE 可能是“无效校准”：例如 12 层 ENC 在 SST-5 的 1024 样本实验中 ECE 只有 0.004，但 weighted-F1 约 0.109、UM 为 0，且预测完全塌缩。

因此，ECE 低只能说明置信度和经验正确率接近，不能说明分类器有足够的区分能力。

## 6. SST-5 有序指标

12 层模型在 SST-5 上：

- AR 的 MAE 从 1.458 降至 1.332。
- MLM 的 MAE 从 1.423 降至 1.240。
- ENC 的 MAE 固定为 1.291，MSE 固定为 2.697，原因是持续预测同一类别。
- AR 和 MLM 在 512 样本时 UM 分别达到 0.328 和 0.284。
- ENC 的 UM 始终为 0。

当前单峰率低于原论文报告的水平，说明在所研究的样本量范围内，随机初始化的 12 层模型仍难以稳定学习五级情感顺序。

## 7. 论文与可复现文件

ACL 格式论文：

```text
Association_for_Computational_Linguistics__ACL__conference/latex/course_paper.tex
Association_for_Computational_Linguistics__ACL__conference/latex/course_paper.pdf
```

统计与绘图脚本：

```text
Association_for_Computational_Linguistics__ACL__conference/latex/analyze_course_paper.py
```

生成的图表和 CSV：

```text
Association_for_Computational_Linguistics__ACL__conference/latex/course_paper_artifacts/
```

重新生成：

```bash
cd Association_for_Computational_Linguistics__ACL__conference/latex
python analyze_course_paper.py
xelatex course_paper.tex
bibtex course_paper
xelatex course_paper.tex
xelatex course_paper.tex
```

## 8. 当前结论的边界

- 只有一个随机种子，不能报告可靠方差或显著性。
- 本实验研究范围为 128、256、512 和 1024 四档训练样本量。
- 未包含 DIFF 和 AR-pseudo。
- 全部模型从零初始化，不代表预训练模型微调结果。
- 当前结果适合形成课程论文分析；增加随机种子后可进一步检验结论稳定性。
