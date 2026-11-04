import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description='ARIN5201 controlled GenImage experiment')
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'suite', 'report', 'predict'])
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--method', default='clip_linear')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint')
    parser.add_argument('--image')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if cfg['resize_short'] < cfg['image_size']:
        parser.error('resize_short must be >= image_size')
    if args.command == 'prepare':
        from experiment.data import prepare
        prepare(cfg)
    elif args.command == 'report':
        from experiment.report import report
        report(cfg)
    elif args.command == 'predict':
        from experiment.inference import infer
        if not args.checkpoint or not args.image:
            parser.error('predict requires --checkpoint and --image')
        print(infer(args.checkpoint, args.image, cfg['device']))
    else:
        from experiment.engine import train, evaluate
        if args.command == 'train':
            train(cfg, args.method, args.seed)
        elif args.command == 'evaluate':
            if not args.checkpoint:
                parser.error('evaluate requires --checkpoint')
            evaluate(cfg, args.checkpoint)
        else:
            for method in cfg['methods']:
                for seed in cfg['seeds']:
                    path = Path(cfg['run_dir']) / method / f'seed_{seed}' / 'best.pt'
                    if not (path.parent / 'run.json').exists():
                        path = train(cfg, method, seed)
                    evaluate(cfg, path)
            from experiment.report import report
            report(cfg)


if __name__ == '__main__':
    main()
