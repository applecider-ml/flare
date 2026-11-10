"""Plot the complete Mondrian macro-yield comparison from benchmark results."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(results, out):
    path = Path(results)
    payload = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    data = json.loads(payload)
    records = data['budget']['pooled']
    policies = ['round_robin', 'entropy', 'candidate_count', 'rare_mass', 'confidence',
                'aggregate_output', 'macro_output', 'macro_quality',
                'random_seed0', 'random_seed1', 'random_seed2']
    labels = ['Round robin', 'Entropy', 'Candidate count', 'Rare mass', 'Confidence',
              'Learned aggregate', 'Learned macro', 'Learned macro + quality',
              'Random seeds 0–2', None, None]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharey=True)
    for ax, alpha in zip(axes, [.1, .2]):
        for index, (policy, label) in enumerate(zip(policies, labels)):
            selected = sorted((r for r in records if r['regime'] == 'mondrian' and r['alpha'] == alpha
                               and r['policy'] == policy), key=lambda r: r['budget_multiplier'])
            x = [r['metrics']['resources']['consumed_extra_tokens'] / r['metrics']['overall']['n_objects'] for r in selected]
            y = [100 * r['metrics']['macro_true_class_correct_automatic_rate'] for r in selected]
            ax.plot(x, y, marker='.', linewidth=1.4, label=label,
                    linestyle='-' if index in (5, 6, 7) else '--',
                    color='0.65' if index >= 8 else f'C{index}')
        ax.set_title(f'Mondrian alpha = {alpha:g}')
        ax.set_xlabel('Consumed extra checkpoint queries / object')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Macro true-class correct / all (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=5, fontsize=7.5)
    figure.suptitle('Exploratory development replay; random seeds show algorithmic variation', fontsize=10)
    figure.tight_layout(rect=(0, .15, 1, .94))
    figure.savefig(out, dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.results, args.out)
