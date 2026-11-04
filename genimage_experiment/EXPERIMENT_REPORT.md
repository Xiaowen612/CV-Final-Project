# ARIN5201 GenImage 实验报告

> 状态：小样本实测实验完成。七种方法 × 三个随机种子，加三组单 seed 约束消融，共 24 个训练模型；不是 GenImage 全量 benchmark。

## 1. 研究问题

在相同 GenImage 数据划分下，比较冻结 CLIP、Effort 参数高效适配和退化增强，观察对未见生成器与图像退化的泛化；补充四种常用基线。

## 2. 实际执行配置

- 数据来自 GenImage 官方 Google Drive 分卷 ZIP，HTTP Range 提取原始图片，逐成员 CRC32 校验；下载 1,680 张候选图片。
- SD1.4、BigGAN 为 seen；ADM、GLIDE 为 unseen。官方 train 划分为 train/dev，官方 val 仅测试。
- 最终 train=512、dev=128、test=512；每生成器每类 train=128、dev=32、test=64，unseen 无 train/dev。
- 对选中图片做像素 SHA256 和可识别 ImageNet ID 全局去重；所有方法共用 manifest，禁止跨 split 重复。
- 七方法各运行 seed=42/43/44；约束消融仅 seed=42。每次最多 3 epochs；最佳 epoch 依 seen dev AUC 选择。
- 输入 224×224；batch=8、梯度累积=4、有效 batch=32；AdamW 骨干 lr=2e-4、分类头 lr=1e-3；余弦调度。
- CLIP ViT-L/14 使用已有 OpenAI 预训练权重；Effort 对每层 Q/K/V/output 投影的尾部 rank=1 因子训练，分类头为单 logit BCE。
- Effort 启用正交/能量约束，系数各 1；官方公开实现存在默认不启用正则的差异，本项目明确作为适配与消融研究。
- 两张 RTX 4090 各运行独立模型任务；PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、transformers 4.51.3、timm 1.0.15。
- 分类阈值仅在最佳 epoch 的 seen dev 上最大化 Balanced Accuracy，随后所有测试生成器/退化使用同一阈值。

## 3. 清洁图像主结果

以下均为百分比，格式为三个训练 seed 的均值 ± 样本标准差；不是置信区间。Seen/Unseen 指标分别池化两组生成器。

| 方法 | Seen BA | Unseen BA | Unseen AUC | Unseen AP | Unseen real FPR |
|---|---:|---:|---:|---:|---:|
| Frozen CLIP + Linear | 90.62 ± 0.39 | 79.82 ± 2.60 | 88.43 ± 1.49 | 90.24 ± 1.07 | 10.42 ± 3.93 |
| Effort 适配 | 90.62 ± 0.39 | 79.82 ± 2.60 | 88.44 ± 1.49 | 90.25 ± 1.06 | 10.42 ± 3.93 |
| Effort + 退化增强 | 81.12 ± 5.39 | 73.18 ± 2.39 | 82.93 ± 3.21 | 81.64 ± 3.55 | 21.61 ± 7.26 |
| ResNet-50 | 70.70 ± 3.38 | 63.67 ± 1.95 | 70.74 ± 1.43 | 73.06 ± 0.64 | 17.97 ± 4.13 |
| DeiT-S | 84.11 ± 1.85 | 66.67 ± 2.54 | 71.92 ± 2.56 | 76.14 ± 3.19 | 17.97 ± 4.13 |
| Swin-T | 93.23 ± 0.60 | 73.31 ± 0.23 | 74.68 ± 1.74 | 80.68 ± 1.31 | 7.81 ± 1.35 |
| CNNSpot 协议适配 | 62.89 ± 1.35 | 57.03 ± 1.79 | 61.85 ± 2.52 | 61.12 ± 1.87 | 38.80 ± 18.36 |

本轮清洁图像结果未显示 Effort 相比 Frozen CLIP 的实质优势：两者的 mean BA 相同，AUC 只有极小差异。因此本实验不能支持“Effort 明显更好”的结论。小样本、仅 3 epochs 和尾部 rank=1 都可能限制可观察到的收益，但需要进一步受控实验验证，不能直接作因果解释。

## 4. 核心方法的逐生成器结果

| 方法 | SD1.4 BA | BigGAN BA | ADM BA | GLIDE BA |
|---|---:|---:|---:|---:|
| Frozen CLIP + Linear | 84.90 ± 1.19 | 96.35 ± 0.90 | 66.93 ± 5.76 | 92.71 ± 1.63 |
| Effort 适配 | 84.90 ± 1.19 | 96.35 ± 0.90 | 66.93 ± 5.76 | 92.71 ± 1.63 |
| Effort + 退化增强 | 76.04 ± 3.16 | 86.20 ± 8.28 | 63.54 ± 4.01 | 82.81 ± 0.78 |

## 5. 未见生成器上的退化测试

原始 RGB 图片先施加退化，再统一 resize/crop。每个条件独立测试，不组合；所有模型使用同一批原图。下表为三 seed 均值 BA，Δ 为增强 Effort 减标准 Effort，单位百分点。

| 条件 | Frozen CLIP | Effort | Effort + 增强 | Δ增强 |
|---|---:|---:|---:|---:|
| clean | 79.82 | 79.82 | 73.18 | -6.64 |
| jpeg90 | 73.44 | 73.44 | 69.79 | -3.65 |
| jpeg70 | 69.27 | 69.27 | 68.36 | -0.91 |
| jpeg50 | 63.93 | 64.06 | 66.80 | +2.73 |
| blur0.5 | 79.95 | 79.82 | 75.91 | -3.91 |
| blur1 | 71.48 | 71.61 | 75.13 | +3.52 |
| blur2 | 58.72 | 58.72 | 72.66 | +13.93 |
| scale0.75 | 85.42 | 85.42 | 74.61 | -10.81 |
| scale0.5 | 77.34 | 77.34 | 75.52 | -1.82 |
| scale0.25 | 54.04 | 54.04 | 70.05 | +16.02 |

在九种退化条件中，增强模型的均值 BA 高于标准 Effort 的条件数为 4/9。该比较是描述性结果，不代表统计显著性，也不能推断对所有退化均有效。

## 6. 约束消融（seed=42）

与同一个 seed=42 的标准 Effort 配对比较；不能把单 seed 消融当作稳定性结论。

| 配置 | Unseen BA | Unseen AUC | Unseen AP |
|---|---:|---:|---:|
| Effort 适配 | 82.81 | 89.96 | 91.32 |
| 关闭正交约束 | 82.81 | 89.96 | 91.31 |
| 关闭范数约束 | 82.81 | 89.96 | 91.31 |
| 关闭两项约束 | 82.81 | 89.96 | 91.32 |

这组短训练、单 seed 消融在清洁 Unseen BA 上没有区分出约束收益；不能据此断言正则项无效。独立检查确认标准 Effort 的 96 个 attention 投影残余 U 因子均发生了更新，详见 adaptation_verification.json。

## 7. 计算开销

耗时包括初始化、训练、验证及保存 checkpoint，单卡任务墙钟时间；不包含全测试与作图，也不等同于 FLOPs。显存为训练进程峰值 allocated。

| 方法 | 可训练参数 | 平均单次秒数 | 最大峰值显存 MiB |
|---|---:|---:|---:|
| Frozen CLIP + Linear | 1,025 | 9.6 | 1349.6 |
| Effort 适配 | 197,729 | 35.3 | 1518.4 |
| Effort + 退化增强 | 197,729 | 34.6 | 1518.4 |
| ResNet-50 | 23,510,081 | 11.2 | 782.3 |
| DeiT-S | 21,666,049 | 11.2 | 682.5 |
| Swin-T | 27,520,123 | 8.2 | 1061.6 |
| CNNSpot 协议适配 | 23,510,081 | 7.5 | 782.3 |

## 8. 可靠性检查与限制

- 7 项自动化测试通过，覆盖 SVD 重建/梯度、正则与直接公式一致性、阈值/指标、退化确定性、数据隔离、CLIP 保存恢复及训练到报告闭环。
- 独立重新读取逐图预测，验证所有 24 个模型共用 512 张测试图，重算 1,680 组指标、检查 122,880 行预测，并验证阈值未随测试条件改变；详见 results_verification.json。
- 单图真实 checkpoint 推理和 Streamlit 页面启动均验证通过。demo_prediction.json 选取 manifest 第一张图，包含误分类示例，未筛选“好看”的样本。
- 当前是小样本、短训练的完整实验流程；三 seed 只反映训练随机性，不能消除抽样偏差。每个生成器测试只有 64 真 + 64 假，不支持 SOTA 或普适检测声明。
- 1000 个 ImageNet 语义类没有强制均匀覆盖，近重复图像未被穷尽检测，CLIP 的预训练污染无法核验。JPEG/分辨率等数据偏差仍可能影响性能。
- ResNet/DeiT/Swin 与 CLIP 的预训练语料不同；CNNSpot 是统一协议适配，并非原论文全部训练细节的逐项复现。
- 超参数和 epoch/阈值选择严格使用 dev，未使用 test 调参；测试表现不应反过来决定正式实验超参数。
- 完整 CLIP 全参数微调、额外合成样本拓展尚未运行；代码支持 clip_full，合成拓展未实现。

## 9. 如何扩展为正式报告实验

使用 config.yaml 扩展为每 seen 生成器每类 train=2000、dev=500、每测试生成器每类 test=1000，最多 10 epochs、三 seed；先扩大官方图片下载池，再新建 manifest_dir/run_dir。当前小样本结果不得与扩展实验混成同一表。完整复现步骤与方法差异见 README.md。

## 10. 文件与来源

- runs_pilot/report/：全部指标、三 seed 汇总、鲁棒性曲线、ROC/混淆矩阵、训练曲线与失败案例。
- runs_pilot/<method>/seed_<N>/：best.pt、逐图预测、历史、环境与配置；导出包不含大型 best.pt，权重保留在服务器。
- manifests_pilot/：三个固定清单与 audit.json；logs/：逐方法训练/测试日志。
- 官方数据与代码来源：[GenImage](https://github.com/GenImage-Dataset/GenImage)、[Effort](https://github.com/YZY-stack/Effort-AIGI-Detection)、[CNNDetection](https://github.com/PeterWang512/CNNDetection)。
- 服务器项目：/home/smbu/lxw/genimage_experiment/。
