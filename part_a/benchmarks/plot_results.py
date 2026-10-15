"""Plot measured benchmark results"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def plot(summary, output):
    import matplotlib.pyplot as plt
    with summary.open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('No measured rows')
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ('B','H','d','mode','variant','tile','softmax_threads'))
        groups[key].append(row)
    output.mkdir(parents=True, exist_ok=True)
    for metric, label in [('forward_median_ms', 'Forward latency (ms)'),
                          ('speedup_vs_naive', 'Speedup vs matched naive baseline'),
                          ('peak_workspace_allocated_bytes', 'Allocated workspace (MiB)')]:
        for mode in ('dense','causal'):
            fig, ax = plt.subplots(figsize=(9, 5))
            drawn = False
            for key, series in sorted(groups.items()):
                if key[3] != mode:
                    continue
                series = sorted((r for r in series if r[metric]), key=lambda r: int(r['N']))
                if not series:
                    continue
                x = [int(r['N']) for r in series]
                divisor = 2**20 if metric.endswith('_bytes') else 1
                y = [float(r[metric])/divisor for r in series]
                legend = f'B={key[0]}, H={key[1]}, d={key[2]}, {key[4]}, tile={key[5]}, softmax={key[6]}'
                if metric == 'forward_median_ms':
                    lower = [float(r[metric])-float(r['forward_q1_ms']) for r in series]
                    upper = [float(r['forward_q3_ms'])-float(r[metric]) for r in series]
                    ax.errorbar(x, y, yerr=[lower, upper], marker='o', label=legend, capsize=3)
                else:
                    ax.plot(x, y, marker='o', label=legend)
                drawn = True
            if drawn:
                ax.set_xlabel('Sequence length N'); ax.set_ylabel(label)
                ax.set_title(f'Materialized attention: {mode}')
                ax.grid(alpha=.25); ax.legend(fontsize=7)
                fig.tight_layout()
                fig.savefig(output / f'{mode}_{metric}.png', dpi=160)
                svg = output / f'{mode}_{metric}.svg'
                fig.savefig(svg)
                svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)


if __name__ == '__main__':
    main()
