import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import roc_curve


def report(cfg):
    root = Path(cfg['run_dir'])
    files = sorted(root.glob('*/seed_*/metrics.csv'))
    if not files:
        raise FileNotFoundError('No evaluated runs found. Train and evaluate first.')
    out = root / 'report'
    out.mkdir(exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out / 'all_metrics.csv', index=False)
    measures = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision', 'real_fpr']
    summary = df.groupby(['method', 'condition', 'group'])[measures].agg(['mean', 'std', 'count'])
    summary.to_csv(out / 'summary_mean_std.csv')
    clean = df[df.condition == 'clean'].groupby(['method', 'group'])[measures].mean()
    clean.to_csv(out / 'clean_comparison.csv')
    for group in ['seen', 'unseen']:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, (prefix, label) in zip(axes, [('jpeg', 'JPEG quality'), ('blur', 'Blur sigma'), ('scale', 'Downsample ratio')]):
            main_data = df[(df.group == group) & df.method.isin(cfg.get('methods', list(df.method.unique()))) ]
            for method, values in main_data.groupby('method'):
                conditions = [c for c in cfg['conditions'] if c.startswith(prefix)]
                x = [float(c[len(prefix):]) for c in conditions]
                y = [values[values.condition == c].balanced_accuracy.mean() for c in conditions]
                ax.plot(x, y, 'o-', label=method)
            ax.set_xlabel(label)
            ax.set_ylim(0, 1)
            ax.grid(alpha=.3)
        axes[0].set_ylabel('Balanced accuracy')
        axes[-1].legend(fontsize=7)
        fig.suptitle(f'{group.capitalize()} generators: main comparison (mean across training seeds)')
        fig.tight_layout()
        fig.savefig(out / f'robustness_{group}.png', dpi=160)
        plt.close(fig)
        ablation_methods = ['effort', 'effort_no_orth', 'effort_no_energy', 'effort_no_constraints']
        ablation_data = df[(df.group == group) & (df.seed == 42) & df.method.isin(ablation_methods)]
        if ablation_data.method.nunique() > 1:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
            for ax, (prefix, label) in zip(axes, [('jpeg', 'JPEG quality'), ('blur', 'Blur sigma'), ('scale', 'Downsample ratio')]):
                for method, values in ablation_data.groupby('method'):
                    conditions = [c for c in cfg['conditions'] if c.startswith(prefix)]
                    ax.plot([float(c[len(prefix):]) for c in conditions],
                            [values[values.condition == c].balanced_accuracy.mean() for c in conditions], 'o-', label=method)
                ax.set_xlabel(label)
                ax.set_ylim(0, 1)
                ax.grid(alpha=.3)
            axes[0].set_ylabel('Balanced accuracy')
            axes[-1].legend(fontsize=7)
            fig.suptitle(f'{group.capitalize()} generators: constraint ablation (seed 42 only)')
            fig.tight_layout()
            fig.savefig(out / f'ablation_robustness_{group}.png', dpi=160)
            plt.close(fig)
    # Individual runs are plotted separately; no synthetic pooled ROC across seeds.
    for file in files:
        run = file.parent
        meta = json.loads((run / 'run.json').read_text())
        predictions = pd.read_csv(run / 'predictions/clean.csv')
        tag = f"{meta['method']}_seed{meta['seed']}"
        history = pd.read_csv(run / 'history.csv')
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].plot(history.epoch, history.train_loss, 'o-')
        axes[0].set(xlabel='Epoch', ylabel='Training loss')
        axes[1].plot(history.epoch, history.dev_roc_auc, 'o-', label='Dev AUC')
        axes[1].plot(history.epoch, history.dev_balanced_accuracy, 'o-', label='Dev BA')
        axes[1].set(xlabel='Epoch', ylim=(0, 1))
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out / f'{tag}_learning.png', dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        subset = df[(df.method == meta['method']) & (df.seed == meta['seed']) & (df.condition == 'clean') & (df.group == 'all')].iloc[0]
        matrix = np.array([[subset.tn, subset.fp], [subset.fn, subset.tp]], dtype=int)
        axes[0].imshow(matrix, cmap='Blues')
        for (i, j), n in np.ndenumerate(matrix):
            axes[0].text(j, i, str(n), ha='center', va='center')
        axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'], xlabel='Predicted', ylabel='Actual', title='Clean test confusion matrix')
        for generator, part in predictions.groupby('generator'):
            fpr, tpr, _ = roc_curve(part.label, part.probability)
            axes[1].plot(fpr, tpr, label=generator)
        axes[1].plot([0, 1], [0, 1], '--', color='gray')
        axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='Clean test ROC')
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f'{tag}_diagnostics.png', dpi=150)
        plt.close(fig)
        failures = predictions[predictions.label != predictions.prediction].copy()
        failures['error_confidence'] = np.where(failures.label == 0, failures.probability, 1-failures.probability)
        failures = failures.sort_values('error_confidence', ascending=False).head(12)
        failures.to_csv(out / f'{tag}_failures.csv', index=False)
        if len(failures):
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for ax in axes.flat:
                ax.axis('off')
            for ax, (_, row) in zip(axes.flat, failures.iterrows()):
                with Image.open(Path(cfg['data_root']) / row.path) as image:
                    ax.imshow(ImageOps.exif_transpose(image).convert('RGB'))
                ax.set_title(f'{row.generator} / true={row.label}\np(AI)={row.probability:.3f}', fontsize=9)
            fig.tight_layout()
            fig.savefig(out / f'{tag}_failures.png', dpi=130)
            plt.close(fig)
    costs = [json.loads((f.parent / 'run.json').read_text()) for f in files]
    pd.DataFrame([{k: r[k] for k in ['method', 'seed', 'trainable_parameters', 'total_seconds', 'setup_seconds', 'peak_gpu_mb', 'best_dev_auc']} for r in costs]).to_csv(out / 'compute_cost.csv', index=False)
    clean_unseen = df[(df.condition == 'clean') & (df.group == 'unseen')]
    lines = ['# GenImage 实验结果', '', '此报告仅汇总实际生成的预测文件，不包含论文原表数字或模拟结果。', '',
             '| Method | Seed | Unseen Acc | Unseen BA | AUC | AP | Real FPR |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in clean_unseen.itertuples():
        lines.append(f'| {row.method} | {row.seed} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | {row.roc_auc:.4f} | {row.average_precision:.4f} | {row.real_fpr:.4f} |')
    lines += ['', '## 解读范围', '', f"训练样本设定：每个 seen generator、每类 {cfg['train_per_class']} 张；验证 {cfg['dev_per_class']} 张；测试每 generator、每类 {cfg['test_per_class']} 张。",
              'AI 为正类。阈值和最佳 epoch 仅由 seen 验证集选择。seen/unseen 使用样本池化统计，逐生成器结果见 CSV。',
              '单 seed 不能估计训练随机性；summary 中单次运行的标准差为空，不是零。小样本不能支持 SOTA 声明。',
              '测试结果不可用来选超参数。模型分数未做概率校准。训练耗时含验证及 checkpoint 写盘，不等同于 FLOPs。',
              '查看 robustness_*.png、*_diagnostics.png、*_failures.png 及 compute_cost.csv。',
              'GenImage 存在原始分辨率/JPEG 编码等潜在混杂因素；退化测试本身不能证明完全消除此类偏差。']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report saved: {out.resolve()}')
