# ARIN5201：GenImage 跨生成器检测实验

本项目根据 proposal 实现统一数据协议下的 Python 实验。它是 **Effort 在本项目 GenImage 子集上的适配研究**，不声称复现原论文表格。实际结果由训练、预测文件自动生成，代码不预置任何性能数字。

## GitHub 精简发布与复现说明

仓库提供代码、固定数据划分和精简历史结果。原始图片与训练模型通过单独附件提供，下载地址尚待维护者补充。

### 精简结果保留范围

`runs_pilot/` 仅保留以下文件，总大小约 1 MB：

| 文件 | 用途 |
|---|---|
| `report/RESULTS.md` | 结果概览 |
| `report/clean_comparison.csv` | 清晰图像结果 |
| `report/summary_mean_std.csv` | 各方法多 seed 均值和标准差 |
| `report/all_metrics.csv` | 各方法、seed、生成器及退化条件的指标 |
| `report/compute_cost.csv` | 参数量、耗时和显存记录 |
| `report/robustness_seen.png`、`robustness_unseen.png` | 鲁棒性对比 |
| `report/ablation_robustness_seen.png`、`ablation_robustness_unseen.png` | 消融对比 |
| `<method>/seed_<N>/run.json` | 24 次训练的配置、环境和数据划分指纹 |

未上传：`best.pt`、逐图预测、dev_predictions.csv、逐次 history.csv/metrics.csv、诊断图及错误案例图。下文描述这些文件时，指完整运行生成的输出，不代表仓库已包含它们。

`manifests_pilot/` 中的 train.csv、dev.csv、test.csv、audit.json、rejected.csv 已提供，请保留原始文件。`data/GenImage/download_provenance.json` 仅是来源记录，不包含图片。

### 资源获取状态

| 资源 | 状态 |
|---|---|
| 固定划分、汇总结果、关键图、运行元数据 | 随仓库提供 |
| 与本次实验一致的完整图片子集 | 下载地址：待维护者补充 |
| 训练好的模型 best.pt | 下载地址：待维护者补充 |
| 完整逐图预测及其他诊断文件 | 可另行索取或重新运行生成 |

历史 verification/COMPLETE JSON 和日志记录的是原实验。精简仓库省略了部分原始产物，不能仅凭这些记录在本机重新核验所有历史指标。

### 新机器复现步骤

1. 按下文“环境”创建全新 Python 3.11 环境；不要复制原服务器的 `.venv`。
2. 获取完整图片子集，放到 `data/GenImage/`，保持原始目录和文件名。当前旧附件 `seen_data.tar` 曾检测到截断错误，应重新打包验证后分享。
3. 检查新增的 `config_reproduce.yaml`：它保留原始 pilot 样本规模，使用 seed 42/43/44，使用新的 `runs_reproduce/` 结果目录。CLIP revision 固定为原服务器 snapshot。训练初始化仍需联网下载 CLIP/timm 预训练权重，或提前准备本机缓存。不要沿用 `config_server_*.yaml` 中的服务器绝对路径。
4. 已有仓库提供的原始清单时，不要执行 prepare。测试后运行七种方法：

```bash
python -m pytest -q
python run.py suite --config config_reproduce.yaml
```

七方法 × 三 seed 为 21 个模型。另三组约束消融使用同一配置和 seed 42：

```bash
python run.py train --config config_reproduce.yaml --method effort_no_orth --seed 42
python run.py evaluate --config config_reproduce.yaml --checkpoint runs_reproduce/effort_no_orth/seed_42/best.pt
python run.py train --config config_reproduce.yaml --method effort_no_energy --seed 42
python run.py evaluate --config config_reproduce.yaml --checkpoint runs_reproduce/effort_no_energy/seed_42/best.pt
python run.py train --config config_reproduce.yaml --method effort_no_constraints --seed 42
python run.py evaluate --config config_reproduce.yaml --checkpoint runs_reproduce/effort_no_constraints/seed_42/best.pt
python run.py report --config config_reproduce.yaml
```

只查看结果不需要数据或模型。单图演示需要 best.pt 和输入图片；完整评估还需完整测试图片及与 checkpoint 指纹一致的三份原始 CSV。下载的 checkpoint 可放在 `runs_pilot/<method>/seed_<N>/best.pt`，然后用 config_reproduce.yaml 运行 evaluate/predict。评估输出写到 checkpoint 所在目录。

仓库的 run.json 不是模型文件。不要对精简的历史 runs_pilot 直接运行 suite；请使用新的 runs_reproduce，以免软件因已有 run.json 而尝试读取不存在的 checkpoint。

本次实测为小样本、3 epochs：七方法各三个 seed，三组消融各一个 seed。结果见 `EXPERIMENT_REPORT.md`。`config.yaml` 的较大实验尚未运行。

requirements 固定 PyTorch 2.7.1 / torchvision 0.22.1，原服务器实际使用 2.9.0+cu128 / 0.24.0+cu128，详见 environment_server.txt 和 run.json。复现时需记录环境差异，不保证跨硬件逐位一致。Windows 数据加载遇到进程问题可设置 workers: 0。

以下为方法及原始流程说明；新机器复现优先使用上面的步骤。

## 1. 实验与 proposal 对应

| 目标 | 实现 |
|---|---|
| Seen 生成器 | SD1.4、BigGAN |
| Unseen 生成器 | ADM、GLIDE，仅用于最终测试 |
| 三个核心对照 | `clip_linear`、`effort`、`effort_aug`，统一 CLIP ViT-L/14 |
| 四个附加基线 | `resnet50`、`deit_small`、`swin_tiny`、`cnnspot` |
| 可选完整微调 | `clip_full`（未包含在默认七方法中，资源允许时单独运行） |
| 消融 | `effort_no_orth`、`effort_no_energy`、`effort_no_constraints`；可修改 residual_rank |
| 退化 | JPEG 90/70/50；Gaussian sigma 0.5/1/2；downsample 0.75/0.5/0.25 |
| 指标 | Accuracy、Balanced Accuracy、Precision、Recall、F1、ROC-AUC、AP、真实图像 FPR |
| 诊断 | 按生成器统计、混淆矩阵、ROC、鲁棒性曲线、错误样本、时间/显存/可训练参数 |
| Demo | 单图命令行推理及 Streamlit 界面 |

## 2. 环境

推荐 Python 3.11，NVIDIA GPU。全新环境安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Windows 激活使用 `.venv\Scripts\Activate.ps1`。若安装到 CPU 版，请依据 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装与你驱动兼容的 CUDA wheel。requirements 提供一套固定核心版本；实际服务器可采用已验证的兼容版本，版本写入各 run.json，服务器也保存 environment_server.txt。

当前服务器部署位置：`/home/smbu/lxw/genimage_experiment`。为避免重复安装 CUDA 大依赖，服务器的 `.venv` 以已有 faceanything Python 3.11 环境为只读基础、启用 system-site-packages，仅在项目 `.venv` 内安装额外依赖，不修改原环境。该部署依赖基础环境持续存在；需要完全独立复现时用上面的新环境安装方式。

## 3. 下载真实 GenImage 子集

官方来源：[GenImage repository](https://github.com/GenImage-Dataset/GenImage)、该仓库链接的 [Google Drive](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS)。只下载 SD1.4、BigGAN 的 train/val，以及 ADM、GLIDE 的 val。

```bash
python -m pip install requests beautifulsoup4
python download_subset.py --root data/GenImage --train-pool 220 --test-pool 100 --workers 8
```

脚本读取官方分卷 ZIP 中央目录，再用 HTTP Range 提取随机选中的原始图片，不重新编码；每张图片检查 ZIP CRC32 和长度，已有正确文件可断点复用。下载池比最终选样多，供去重、移除损坏文件。`download_provenance.json` 记录官方压缩包成员及 CRC32；缓存保存官方卷 ID 和成员索引。网络必须能访问 Google Drive。若服务器无法连接，在可连接的电脑下载，再传 `data/GenImage/` 到服务器即可。

默认 `config.yaml` 的较大实验需要扩大下载池，例如 `--train-pool 3500 --test-pool 1600`。切换样本规模时必须使用新的 manifest_dir 和 run_dir；不要覆盖已有测试结果。

目录格式：

```text
data/GenImage/
  Stable Diffusion V1.4/{train,val}/{ai,nature}/...
  BigGAN/{train,val}/{ai,nature}/...
  ADM/val/{ai,nature}/...
  GLIDE/val/{ai,nature}/...
```

## 4. 数据协议

官方 train 仅用于训练及内部验证；官方 val 是封存测试集。每个生成器、每类固定样本数，AI=1、Real=0。所有方法使用相同 CSV 清单、split seed。选中图像做 RGB 像素 SHA256 与已识别 ImageNet ID 的全局去重：测试优先占位，重复内容不能进入 train/dev，测试不同生成器组之间也不重复真实图像。

清单包含相对路径、标签、生成器、官方 split、内部 split、像素 hash、原图尺寸。`audit.json` 与 `rejected.csv` 保存审计记录。审计只覆盖遍历到的候选/最终选中图像，不宣称全库查重，也不检测未知重编码/近重复图像。真实图像的 generator 字段表示其来源目录和评估组，不表示真实图像由生成器生成。

采样是每生成器、每真伪类均匀随机，不保证 ImageNet 的 1000 语义类别在小样本中均匀覆盖。小样本泛化结果应谨慎解释。

## 5. 运行

以下为重新采样的示例。已有原始 manifests_pilot 时不要运行 prepare；复现历史实验使用前文 config_reproduce.yaml。先修改 data_root；所有相对路径以启动命令时的工作目录为基准。

```bash
python -m pytest -q
python run.py prepare --config config_pilot.yaml
python run.py suite --config config_pilot.yaml
```

服务器使用 `config_server_pilot.yaml`（指向已有 CLIP 本地权重），通过 `.venv/bin/python` 启动。

服务器两卡后台启动（先完成数据传输；不要重复启动同一个实验）：

```bash
cd /home/smbu/lxw/genimage_experiment
nohup bash start_server.sh > experiment.log 2>&1 < /dev/null &
tail -f experiment.log
cat logs/gpu0_status.json logs/gpu1_status.json
```

详细日志在 `logs/<method>_<seed>.log`。全部七方法完成时才生成 `EXPERIMENT_COMPLETE.json`；任一组失败会写 `state=failed`，不会标成成功。GPU0 运行前三个核心方法，GPU1 运行四个基线。该调度只是任务并行，不是多卡分布式训练，所有单模型仍用单 GPU。

| 配置 | 训练每生成器每类 | 内部验证每生成器每类 | 测试每生成器每类 | epochs | seeds |
|---|---:|---:|---:|---:|---|
| pilot / server_pilot | 128 | 32 | 64 | 3 | 42 |
| config.yaml | 2000 | 500 | 1000 | 10 | 42/43/44 |

本次服务器进一步执行了 `config_server_repeated.yaml`：沿用 pilot 的样本数和 3 epochs，七方法各跑 42/43/44；另外三种约束消融仅 seed=42。最终实测数字以 `EXPERIMENT_REPORT.md` 为准。`config_server_full.yaml` 是已适配服务器本地 CLIP 路径的较大样本配置，当前未运行。

Pilot 总计 train 512、dev 128、test 512。主配置总计 train 8000、dev 2000、test 8000。Pilot 是真实数据的端到端试验，不足以支持最终论文级结论。主配置同样是 GenImage 子集，不是全库 benchmark。

单独运行：

```bash
python run.py train --config config.yaml --method effort --seed 42
python run.py evaluate --config config.yaml --checkpoint runs/effort/seed_42/best.pt
python run.py report --config config.yaml
python run.py predict --config config.yaml --checkpoint runs/effort/seed_42/best.pt --image example.jpg
streamlit run app.py
```

消融使用同一 manifest，修改 run_dir 后运行 `--method effort_no_orth` 等；若使用不同 rank，必须记录配置并单独存放结果。`suite` 顺序运行方法与 seeds，已完成的 run 会复用并重做评估；存在未完成 checkpoint 时明确报错，不会悄悄覆盖。断点恢复优化器尚未实现，失败的训练请使用新的 run_dir 重新启动。

## 6. 模型和训练细节

Effort 使用 CLIP 视觉编码器 1024 维 pooler_output，替换每层 attention 的 Q/K/V/output 四个线性投影。对原始 W 做 SVD，冻结前 d-k 个方向，训练尾部 U、S、V；初始模型重建原始 W。默认尾部 rank k=1。视觉编码器其余部分冻结，只训练尾部因子和二分类头。分类器输出一个 logit，损失为 BCEWithLogits。

总损失：BCE + lambda_orth × 平均正交误差 + lambda_energy × 平均权重能量误差。正交误差是拼接主/残余基的 Gram 矩阵与单位矩阵差的 Frobenius 范数；能量误差为 `abs(||W_current||_F² - ||W_initial||_F²)`。实现用低秩计算避免存储完整额外基，并用单元测试验证与直接矩阵计算一致。

参考 [Effort 官方代码](https://github.com/YZY-stack/Effort-AIGI-Detection/blob/main/DeepfakeBench/training/detectors/effort_detector.py)：所检查版本定义了两个正则函数，但当前激活的训练 loss 未使用它们。本项目按照 proposal 对约束的研究要求显式启用，并提供关闭消融；不能把该配置称为官方原样复现。BCE 单 logit、统一图像预处理和 GenImage 双生成器训练也属于本项目适配。

Frozen CLIP 与 Effort 用同一视觉骨干及分类头维度。传统基线使用 timm ImageNet 预训练；CNNSpot 是 ResNet-50 加 JPEG/blur 训练增强的统一协议适配，未完整照搬 [CNNDetection 原仓库](https://github.com/PeterWang512/CNNDetection) 的全部训练超参数/图像处理选项。传统模型与 CLIP 预训练数据不同，所以跨骨干对比不能单独归因为适配方式。

训练统一 bicubic 缩短边至 256、随机 crop 224、随机水平翻转；测试 center crop。CLIP 与 ImageNet 模型使用各自 normalization。effort_aug 在训练 resize 后独立以 0.5 概率施加 blur sigma U(0,3)、JPEG quality U整数(30,100)、downsample U(0.25,1)，对两类一视同仁。测试退化在原图 RGB 上实施，再 resize/crop，不随机，不组合，所有方法完全一致。

AdamW、cosine 学习率、梯度累积、AMP 和非重入 gradient checkpointing；只按内部验证 ROC-AUC 选 best epoch，阈值仅在该 epoch 的内部验证上最大化 Balanced Accuracy，测试所有生成器和退化共用同一阈值。没有在 test 调阈值或挑模型。验证选择和阈值共享 dev，可能产生选择乐观性，较大研究可另设 calibration split。

## 7. 输出与报告

每个 `run_dir/method/seed_N/` 含 best.pt、history.csv、run.json、dev_predictions.csv、predictions/*.csv、metrics.csv。报告目录含清洁/退化汇总 CSV、各 seed 诊断图、错误案例和 RESULTS.md。单 seed std 为空；多 seed 报告 mean/std，不等同于置信区间。seen/unseen/all 指标是对应样本池化；逐生成器结果另列。

预训练权重、训练后的模型及图片仍遵循其原作者条款；本课程研究不应把模型分数作为某张图片真伪的确定证据。报告应讨论 JPEG/分辨率偏差、CLIP 预训练污染无法核验、子集抽样误差和有限生成器覆盖。
