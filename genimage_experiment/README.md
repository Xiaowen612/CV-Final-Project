# ARIN5201：GenImage 跨生成器检测实验

本项目研究 AI 图像检测模型对未见生成器及图像退化的泛化能力，是 Effort 在本项目 GenImage 子集上的适配研究，不声称复现原论文表格。

## 当前实验状态

已完成小样本实验：七种方法各运行 seed 42/43/44，另有三组 seed 42 的约束消融，共 24 个训练模型。训练集 512 张、内部验证集 128 张、测试集 512 张，每次最多训练 3 epochs。

- Seen：SD1.4、BigGAN；Unseen：ADM、GLIDE，仅用于最终测试。
- 核心对照：Frozen CLIP + Linear、Effort、Effort + 退化增强。
- 附加基线：ResNet-50、DeiT-S、Swin-T、CNNSpot 协议适配。
- 测试条件：clean；JPEG 90/70/50；Gaussian blur 0.5/1/2；缩放 0.75/0.5/0.25。
- 结果见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。当前清晰图像结果未显示 Effort 相比 Frozen CLIP 的实质优势，增强方法在部分强退化下改善表现。这些结果不支持全量 benchmark 或 SOTA 声明。

## 1. 仓库提供什么

GitHub 仓库提供代码、配置、测试、实验报告和历史运行记录；数据图片、预训练权重及训练后的大型 checkpoint 通过独立附件提供。

| 文件或目录 | 用途 |
|---|---|
| `experiment/` | 数据处理、模型、训练、评估与报告生成 |
| `run.py` | prepare/train/evaluate/suite/report/predict 入口 |
| `app.py` | Streamlit 单图演示 |
| `config_pilot.yaml` | 单 seed 小样本配置，适合作为新机器配置起点 |
| `config_reproduce.yaml` | 新机器复现用的三 seed 配置，结果写入 runs_reproduce |
| `config_server_repeated.yaml` | 本次三 seed 实验配置，含原服务器路径，使用前必须修改 |
| `config.yaml`、`config_server_full.yaml` | 较大样本实验配置，本次未运行 |
| `tests/`、`pytest.ini` | 自动化测试 |
| `EXPERIMENT_REPORT.md` | 已完成实验的总结 |
| `logs/`、各 verification/COMPLETE JSON | 历史运行与检查记录，不代表当前机器已经完成训练 |

仓库包含固定划分 `manifests_pilot/`、历史结果 `runs_pilot/`（不含 best.pt）以及 `data/GenImage/download_provenance.json`。原始图片和训练模型仍需另外获取。仅下载仓库可以查看报告与预测结果，不能直接完成模型评估或演示。

## 2. 数据、划分与模型附件

**图片与 checkpoint 下载链接尚待维护者补充；“待补充”不是可用下载地址。划分清单和历史结果已在仓库中。**

| 附件 | 下载位置 | 解压后应有的路径 | 用途 |
|---|---|---|---|
| 本次完整图片子集及来源记录 | 待补充 | `data/GenImage/` | 重新训练与测试 |
| 原始数据划分清单 | 已随仓库提供 | `manifests_pilot/train.csv`、`dev.csv`、`test.csv`、`audit.json`、`rejected.csv` | 保持与原实验相同的样本划分 |
| 训练 checkpoint | 待补充 | `runs_pilot/<method>/seed_<N>/best.pt` | 直接评估与单图演示 |
| 历史预测、汇总与图表（不含权重） | 已随仓库提供 | `runs_pilot/report/` 及逐方法 CSV、run.json 等 | 查看汇总、曲线与逐图预测 |

建议分别打包上述目录，附上文件大小和 SHA256，写明附件版本。不要将 `.venv/`、缓存或重复项目备份一并打包。

现有 `experiment_results.tar.gz` 包含划分清单和结果文件，但不含 `best.pt`；它不能替代模型附件。该包也含旧版代码和 README，建议先解压到临时目录，再按需复制 `manifests_pilot/`、`runs_pilot/`，避免覆盖仓库代码。

本地 `seen_data.tar` 在整理时检测到截断错误，需要重新打包并验证后才能分享。不要把目前的 `seen_data.tar` 和 `final_data.tar` 直接当成完整图片子集。

建议补齐附件后的目录：

```text
<项目目录>/
  README.md
  run.py
  app.py
  experiment/
  tests/
  data/GenImage/
    Stable Diffusion V1.4/{train,val}/{ai,nature}/...
    BigGAN/{train,val}/{ai,nature}/...
    ADM/val/{ai,nature}/...
    GLIDE/val/{ai,nature}/...
    download_provenance.json
  manifests_pilot/
    train.csv
    dev.csv
    test.csv
    audit.json
    rejected.csv
  runs_pilot/
    <method>/seed_<N>/best.pt
    report/
```

## 3. 安装环境

推荐 Python 3.11 和 NVIDIA GPU。先下载仓库并进入包含 `run.py` 的目录，下面所有命令都在该目录运行。新建自己的环境，不复制原机器的 `.venv`。

```bash
python -m venv .venv
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装与检查：

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
```

`requirements.txt` 固定的是 PyTorch 2.7.1 / torchvision 0.22.1；原实验服务器实际使用 PyTorch 2.9.0+cu128 / torchvision 0.24.0+cu128，详见 `environment_server.txt` 和各 run.json。二者不是同一套环境。严格对照原结果时需记录并尽量对齐原环境；CUDA 安装版本还需与本机驱动兼容。历史环境列表不应直接替代适合本机的安装步骤。

## 4. 在自己的机器上配置

仓库提供 `config_reproduce.yaml`：基于 pilot 配置，沿用原始划分，使用三个 seed 和新的结果目录。请检查或修改以下字段：

```yaml
data_root: data/GenImage
manifest_dir: manifests_pilot
run_dir: runs_reproduce
clip_id: openai/clip-vit-large-patch14
clip_revision: 8d052a0f05efbaefbc9e8786ba291cfdf93e5bff
seeds: [42, 43, 44]
```

- `data_root` 指向完整图片子集，`manifest_dir` 指向原始划分清单。
- `run_dir` 使用新目录，避免与下载的历史结果混用。
- CLIP 训练初始化需要下载预训练权重；网络不可用时，将 `clip_id` 改为本机完整模型目录。该 revision 来自原服务器记录的 CLIP snapshot。
- 其他基线训练初始化也需要 timm 对应预训练权重，来源记录见 `baseline_pretrained_sources.json`；离线运行需要提前缓存。
- 不要直接使用 `config_server_*.yaml` 中的 `/home/smbu/...` 路径。
- Windows 数据加载遇到进程问题时，可设置 `workers: 0`，并记录这一变更。
- 默认使用 CUDA；CPU 上可以尝试 `device: cpu`、`amp: false`，但不适合按原规模训练。

## 5. 选择复现方式

### A. 只查看历史结果

阅读 `EXPERIMENT_REPORT.md`，查看仓库中的 `runs_pilot/report/`。无需安装训练环境或重新训练。历史 JSON 是原运行记录，不是本机验证结果。

### B. 使用已有模型评估

需要完整图片子集、**原始划分清单**和对应 `best.pt`。例如：

```bash
python run.py evaluate --config config_reproduce.yaml --checkpoint runs_pilot/effort/seed_42/best.pt
```

评估会核对 train/dev/test 三份 CSV 的 SHA256 是否与 checkpoint 一致。不要重新导出、编辑或重新生成这三份文件，否则可能被拒绝评估。评估输出写入 checkpoint 所在目录；如需保留下载的原始结果，请先复制 checkpoint 到单独工作目录再评估。

### C. 重新训练七种方法和三组消融

获取与原实验一致的完整图片子集，使用仓库自带的原始划分清单，检查 `config_reproduce.yaml` 后运行：

```bash
python run.py suite --config config_reproduce.yaml
```

该命令根据配置运行七种方法的三个 seed，共 21 个模型，不会自动加入消融。继续运行：

```bash
python run.py train --config config_reproduce.yaml --method effort_no_orth --seed 42
python run.py evaluate --config config_reproduce.yaml --checkpoint runs_reproduce/effort_no_orth/seed_42/best.pt
python run.py train --config config_reproduce.yaml --method effort_no_energy --seed 42
python run.py evaluate --config config_reproduce.yaml --checkpoint runs_reproduce/effort_no_energy/seed_42/best.pt
python run.py train --config config_reproduce.yaml --method effort_no_constraints --seed 42
python run.py evaluate --config config_reproduce.yaml --checkpoint runs_reproduce/effort_no_constraints/seed_42/best.pt
python run.py report --config config_reproduce.yaml
```

训练只使用 seen train，选择最佳 epoch 和分类阈值只使用 seen dev，unseen 及所有测试条件使用同一阈值。不要根据 test 表现调整超参数。

已有原始清单时**不要执行 prepare**。已有 checkpoint 的运行目录不要重新训练覆盖；失败重跑请使用新的 `run_dir`。当前未实现优化器断点恢复。跨环境和硬件不保证逐位一致，需记录配置、环境、seed 和数据版本。

### D. 单图推理与演示

需要对应的训练 checkpoint；完整训练集不是单图推理的必要条件。将 `example.jpg` 换成自己的图片：

```bash
python run.py predict --config config_reproduce.yaml --checkpoint runs_pilot/effort_aug/seed_42/best.pt --image example.jpg
streamlit run app.py
```

Streamlit 页面中的 Checkpoint 填写实际模型路径，默认尝试 `runs_pilot/effort_aug/seed_42/best.pt`。没有模型文件时仅启动页面不能完成检测。模型分数未做概率校准，不是图片来源的确定证据。

## 6. 重新采样或扩大实验

如果无法取得原数据附件，可尝试仓库的官方子集下载脚本（先备份已有的 download_provenance.json，避免覆盖原实验来源记录）：

```bash
python -m pip install requests beautifulsoup4
python download_subset.py --root data/GenImage --train-pool 220 --test-pool 100 --workers 8
```

官方来源：[GenImage](https://github.com/GenImage-Dataset/GenImage)。脚本依赖访问官方 Google Drive 分卷及 HTTP Range 支持，下载来源和 CRC32 记录于 `download_provenance.json`。

下载候选池不等于恢复原实验。要重评原 checkpoint，仍需原始 CSV 中的全部图片与三份原始清单。若重新采样，请使用新的 `manifest_dir` 和 `run_dir`，再运行：

```bash
python run.py prepare --config config_reproduce.yaml
python run.py suite --config config_reproduce.yaml
```

重新采样的结果应标记为新实验。`prepare` 遇到已有 audit.json 会拒绝覆盖。

扩大实验可参考 `config.yaml`：每个 seen 生成器每类 train=2000、dev=500，每个测试生成器每类 test=1000，最多 10 epochs、三个 seed。需先扩大下载池，例如 `--train-pool 3500 --test-pool 1600`。大样本配置尚未运行，不要将它描述为本次已完成结果。

后续可补充多 seed 消融、不同 residual rank、CLIP 全参数微调（`clip_full`）及更充分的偏差分析。额外合成样本拓展尚未实现。

## 7. 数据协议

官方 train 仅用于训练及内部验证；官方 val 是封存测试集。每个生成器、每类固定样本数，AI=1、Real=0。所有方法使用相同 CSV 清单、split seed。选中图像做 RGB 像素 SHA256 与已识别 ImageNet ID 的全局去重：测试优先占位，重复内容不能进入 train/dev，测试不同生成器组之间也不重复真实图像。

清单包含相对路径、标签、生成器、官方 split、内部 split、像素 hash、原图尺寸。`audit.json` 与 `rejected.csv` 保存审计记录。审计只覆盖遍历到的候选/最终选中图像，不宣称全库查重，也不检测未知重编码/近重复图像。真实图像的 generator 字段表示其来源目录和评估组，不表示真实图像由生成器生成。

采样是每生成器、每真伪类均匀随机，不保证 ImageNet 的 1000 语义类别在小样本中均匀覆盖。小样本泛化结果应谨慎解释。


## 8. 模型和训练细节

Effort 使用 CLIP 视觉编码器 1024 维 pooler_output，替换每层 attention 的 Q/K/V/output 四个线性投影。对原始 W 做 SVD，冻结前 d-k 个方向，训练尾部 U、S、V；初始模型重建原始 W。默认尾部 rank k=1。视觉编码器其余部分冻结，只训练尾部因子和二分类头。分类器输出一个 logit，损失为 BCEWithLogits。

总损失：BCE + lambda_orth × 平均正交误差 + lambda_energy × 平均权重能量误差。正交误差是拼接主/残余基的 Gram 矩阵与单位矩阵差的 Frobenius 范数；能量误差为 `abs(||W_current||_F² - ||W_initial||_F²)`。实现用低秩计算避免存储完整额外基，并用单元测试验证与直接矩阵计算一致。

参考 [Effort 官方代码](https://github.com/YZY-stack/Effort-AIGI-Detection/blob/main/DeepfakeBench/training/detectors/effort_detector.py)：所检查版本定义了两个正则函数，但当前激活的训练 loss 未使用它们。本项目按照 proposal 对约束的研究要求显式启用，并提供关闭消融；不能把该配置称为官方原样复现。BCE 单 logit、统一图像预处理和 GenImage 双生成器训练也属于本项目适配。

Frozen CLIP 与 Effort 用同一视觉骨干及分类头维度。传统基线使用 timm ImageNet 预训练；CNNSpot 是 ResNet-50 加 JPEG/blur 训练增强的统一协议适配，未完整照搬 [CNNDetection 原仓库](https://github.com/PeterWang512/CNNDetection) 的全部训练超参数/图像处理选项。传统模型与 CLIP 预训练数据不同，所以跨骨干对比不能单独归因为适配方式。

训练统一 bicubic 缩短边至 256、随机 crop 224、随机水平翻转；测试 center crop。CLIP 与 ImageNet 模型使用各自 normalization。effort_aug 在训练 resize 后独立以 0.5 概率施加 blur sigma U(0,3)、JPEG quality U整数(30,100)、downsample U(0.25,1)，对两类一视同仁。测试退化在原图 RGB 上实施，再 resize/crop，不随机，不组合，所有方法完全一致。

AdamW、cosine 学习率、梯度累积、AMP 和非重入 gradient checkpointing；只按内部验证 ROC-AUC 选 best epoch，阈值仅在该 epoch 的内部验证上最大化 Balanced Accuracy，测试所有生成器和退化共用同一阈值。没有在 test 调阈值或挑模型。验证选择和阈值共享 dev，可能产生选择乐观性，较大研究可另设 calibration split。


## 9. 输出与报告

每个 `run_dir/method/seed_N/` 含 best.pt、history.csv、run.json、dev_predictions.csv、predictions/*.csv、metrics.csv。报告目录含清洁/退化汇总 CSV、各 seed 诊断图、错误案例和 RESULTS.md。单 seed std 为空；多 seed 报告 mean/std，不等同于置信区间。seen/unseen/all 指标是对应样本池化；逐生成器结果另列。

预训练权重、训练后的模型及图片仍遵循其原作者条款；本课程研究不应把模型分数作为某张图片真伪的确定证据。报告应讨论 JPEG/分辨率偏差、CLIP 预训练污染无法核验、子集抽样误差和有限生成器覆盖。


