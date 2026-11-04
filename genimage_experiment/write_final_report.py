"""Write a Chinese report from measured outputs only, including limitations."""
import json
from pathlib import Path
import pandas as pd
import yaml

cfg = yaml.safe_load(Path('config_server_repeated.yaml').read_text())
verification = json.loads(Path('results_verification.json').read_text())
assert verification['status'] == 'passed' and verification['trained_models'] == 24
assert Path('REPEATS_AND_ABLATIONS_COMPLETE.json').exists()
root = Path(cfg['run_dir'])
df = pd.read_csv(root/'report/all_metrics.csv')
cost = pd.read_csv(root/'report/compute_cost.csv')
methods = cfg['methods']
names = dict(clip_linear='Frozen CLIP + Linear', effort='Effort 适配', effort_aug='Effort + 退化增强',
             resnet50='ResNet-50', deit_small='DeiT-S', swin_tiny='Swin-T', cnnspot='CNNSpot 协议适配',
             effort_no_orth='关闭正交约束',effort_no_energy='关闭范数约束',effort_no_constraints='关闭两项约束')


def summary(method, group, metric, condition='clean'):
    values = df[(df.method==method)&(df.group==group)&(df.condition==condition)][metric]
    return f'{values.mean()*100:.2f} ± {values.std()*100:.2f}'


lines = ['# ARIN5201 GenImage 实验报告', '',
         '> 状态：小样本实测实验完成。七种方法 × 三个随机种子，加三组单 seed 约束消融，共 24 个训练模型；不是 GenImage 全量 benchmark。', '',
         '## 1. 研究问题', '',
         '在相同 GenImage 数据划分下，比较冻结 CLIP、Effort 参数高效适配和退化增强，观察对未见生成器与图像退化的泛化；补充四种常用基线。', '',
         '## 2. 实际执行配置', '',
         '- 数据来自 GenImage 官方 Google Drive 分卷 ZIP，HTTP Range 提取原始图片，逐成员 CRC32 校验；下载 1,680 张候选图片。',
         '- SD1.4、BigGAN 为 seen；ADM、GLIDE 为 unseen。官方 train 划分为 train/dev，官方 val 仅测试。',
         '- 最终 train=512、dev=128、test=512；每生成器每类 train=128、dev=32、test=64，unseen 无 train/dev。',
         '- 对选中图片做像素 SHA256 和可识别 ImageNet ID 全局去重；所有方法共用 manifest，禁止跨 split 重复。',
         '- 七方法各运行 seed=42/43/44；约束消融仅 seed=42。每次最多 3 epochs；最佳 epoch 依 seen dev AUC 选择。',
         '- 输入 224×224；batch=8、梯度累积=4、有效 batch=32；AdamW 骨干 lr=2e-4、分类头 lr=1e-3；余弦调度。',
         '- CLIP ViT-L/14 使用已有 OpenAI 预训练权重；Effort 对每层 Q/K/V/output 投影的尾部 rank=1 因子训练，分类头为单 logit BCE。',
         '- Effort 启用正交/能量约束，系数各 1；官方公开实现存在默认不启用正则的差异，本项目明确作为适配与消融研究。',
         '- 两张 RTX 4090 各运行独立模型任务；PyTorch 2.9.0+cu128、torchvision 0.24.0+cu128、transformers 4.51.3、timm 1.0.15。',
         '- 分类阈值仅在最佳 epoch 的 seen dev 上最大化 Balanced Accuracy，随后所有测试生成器/退化使用同一阈值。', '',
         '## 3. 清洁图像主结果', '',
         '以下均为百分比，格式为三个训练 seed 的均值 ± 样本标准差；不是置信区间。Seen/Unseen 指标分别池化两组生成器。', '',
         '| 方法 | Seen BA | Unseen BA | Unseen AUC | Unseen AP | Unseen real FPR |',
         '|---|---:|---:|---:|---:|---:|']
for m in methods:
    values = [summary(m,'seen','balanced_accuracy'),summary(m,'unseen','balanced_accuracy'),
              summary(m,'unseen','roc_auc'),summary(m,'unseen','average_precision'),summary(m,'unseen','real_fpr')]
    lines.append('| '+names[m]+' | '+' | '.join(values)+' |')
lines += ['', '本轮清洁图像结果未显示 Effort 相比 Frozen CLIP 的实质优势：两者的 mean BA 相同，AUC 只有极小差异。因此本实验不能支持“Effort 明显更好”的结论。小样本、仅 3 epochs 和尾部 rank=1 都可能限制可观察到的收益，但需要进一步受控实验验证，不能直接作因果解释。']
lines += ['', '## 4. 核心方法的逐生成器结果', '', '| 方法 | SD1.4 BA | BigGAN BA | ADM BA | GLIDE BA |', '|---|---:|---:|---:|---:|']
for m in methods[:3]:
    lines.append('| '+names[m]+' | '+' | '.join(summary(m,g,'balanced_accuracy') for g in ['sd14','biggan','adm','glide'])+' |')
lines += ['', '## 5. 未见生成器上的退化测试', '',
          '原始 RGB 图片先施加退化，再统一 resize/crop。每个条件独立测试，不组合；所有模型使用同一批原图。下表为三 seed 均值 BA，Δ 为增强 Effort 减标准 Effort，单位百分点。', '',
          '| 条件 | Frozen CLIP | Effort | Effort + 增强 | Δ增强 |', '|---|---:|---:|---:|---:|']
deltas=[]
for c in cfg['conditions']:
    values=[df[(df.method==m)&(df.group=='unseen')&(df.condition==c)].balanced_accuracy.mean()*100 for m in methods[:3]]
    delta=values[2]-values[1]
    if c!='clean': deltas.append(delta)
    lines.append(f'| {c} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {delta:+.2f} |')
lines += ['', f'在九种退化条件中，增强模型的均值 BA 高于标准 Effort 的条件数为 {sum(x>1e-10 for x in deltas)}/9。该比较是描述性结果，不代表统计显著性，也不能推断对所有退化均有效。', '',
          '## 6. 约束消融（seed=42）', '',
          '与同一个 seed=42 的标准 Effort 配对比较；不能把单 seed 消融当作稳定性结论。', '',
          '| 配置 | Unseen BA | Unseen AUC | Unseen AP |', '|---|---:|---:|---:|']
for m in ['effort','effort_no_orth','effort_no_energy','effort_no_constraints']:
    r=df[(df.method==m)&(df.seed==42)&(df.group=='unseen')&(df.condition=='clean')].iloc[0]
    lines.append(f'| {names[m]} | {100*r.balanced_accuracy:.2f} | {100*r.roc_auc:.2f} | {100*r.average_precision:.2f} |')
lines += ['', '这组短训练、单 seed 消融在清洁 Unseen BA 上没有区分出约束收益；不能据此断言正则项无效。独立检查确认标准 Effort 的 96 个 attention 投影残余 U 因子均发生了更新，详见 adaptation_verification.json。']
lines += ['', '## 7. 计算开销', '',
          '耗时包括初始化、训练、验证及保存 checkpoint，单卡任务墙钟时间；不包含全测试与作图，也不等同于 FLOPs。显存为训练进程峰值 allocated。', '',
          '| 方法 | 可训练参数 | 平均单次秒数 | 最大峰值显存 MiB |', '|---|---:|---:|---:|']
for m in methods:
    r=cost[cost.method==m]
    lines.append(f'| {names[m]} | {int(r.trainable_parameters.iloc[0]):,} | {r.total_seconds.mean():.1f} | {r.peak_gpu_mb.max():.1f} |')
lines += ['', '## 8. 可靠性检查与限制', '',
          '- 7 项自动化测试通过，覆盖 SVD 重建/梯度、正则与直接公式一致性、阈值/指标、退化确定性、数据隔离、CLIP 保存恢复及训练到报告闭环。',
          '- 独立重新读取逐图预测，验证所有 24 个模型共用 512 张测试图，重算 1,680 组指标、检查 122,880 行预测，并验证阈值未随测试条件改变；详见 results_verification.json。',
          '- 单图真实 checkpoint 推理和 Streamlit 页面启动均验证通过。demo_prediction.json 选取 manifest 第一张图，包含误分类示例，未筛选“好看”的样本。',
          '- 当前是小样本、短训练的完整实验流程；三 seed 只反映训练随机性，不能消除抽样偏差。每个生成器测试只有 64 真 + 64 假，不支持 SOTA 或普适检测声明。',
          '- 1000 个 ImageNet 语义类没有强制均匀覆盖，近重复图像未被穷尽检测，CLIP 的预训练污染无法核验。JPEG/分辨率等数据偏差仍可能影响性能。',
          '- ResNet/DeiT/Swin 与 CLIP 的预训练语料不同；CNNSpot 是统一协议适配，并非原论文全部训练细节的逐项复现。',
          '- 超参数和 epoch/阈值选择严格使用 dev，未使用 test 调参；测试表现不应反过来决定正式实验超参数。',
          '- 完整 CLIP 全参数微调、额外合成样本拓展尚未运行；代码支持 clip_full，合成拓展未实现。', '',
          '## 9. 如何扩展为正式报告实验', '',
          '使用 config.yaml 扩展为每 seen 生成器每类 train=2000、dev=500、每测试生成器每类 test=1000，最多 10 epochs、三 seed；先扩大官方图片下载池，再新建 manifest_dir/run_dir。当前小样本结果不得与扩展实验混成同一表。完整复现步骤与方法差异见 README.md。', '',
          '## 10. 文件与来源', '',
          '- runs_pilot/report/：全部指标、三 seed 汇总、鲁棒性曲线、ROC/混淆矩阵、训练曲线与失败案例。',
          '- runs_pilot/<method>/seed_<N>/：best.pt、逐图预测、历史、环境与配置；导出包不含大型 best.pt，权重保留在服务器。',
          '- manifests_pilot/：三个固定清单与 audit.json；logs/：逐方法训练/测试日志。',
          '- 官方数据与代码来源：[GenImage](https://github.com/GenImage-Dataset/GenImage)、[Effort](https://github.com/YZY-stack/Effort-AIGI-Detection)、[CNNDetection](https://github.com/PeterWang512/CNNDetection)。',
          '- 服务器项目：/home/smbu/lxw/genimage_experiment/。']
Path('EXPERIMENT_REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Chinese experiment report generated.')
