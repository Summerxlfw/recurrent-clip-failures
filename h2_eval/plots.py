"""Native vector plots of the recorded extension outputs; no estimator fitting."""
from pathlib import Path
import csv, hashlib, json, os


def render_all(output_dir):
    out = Path(output_dir)
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#9DA8B0', 'axes.linewidth': .5, 'axes.titlepad': 7,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'H2-v6-extension', 'savefig.facecolor': 'white'})
    B, T, O, G, INK = '#285E8C', '#21877D', '#B66942', '#8C99A4', '#243747'
    summary = json.loads((out / 'summary.json').read_text())
    sun = list(csv.DictReader((out / 'sun_before_after.csv').open()))
    pg = list(csv.DictReader((out / 'pg_retained.csv').open()))
    names = {'uac_base': 'UACANet', 'pranet_plain': 'PraNet', 'community_pooled': 'Combined'}
    receipts = []

    def truth(x): return x == 'True'
    def tidy(ax):
        ax.grid(axis='y', color='#E9EEF2', lw=.45)
        ax.set_axisbelow(True)
    def title(ax, letter, text):
        ax.set_title(f'{letter}  {text}', loc='left', fontweight='bold', color=INK)
    def sun_panel(ax, host, letter, heading):
        rows = [r for r in sun if r['host'] == host]
        stats = summary['sun']['groups'][host]
        valid = [r for r in rows if r['retained_risk'] != '']
        for reference, color in [(False, B), (True, O)]:
            rr = [r for r in valid if truth(r['reference_failure']) == reference]
            ax.scatter([float(r['before_risk']) for r in rr], [float(r['retained_risk']) for r in rr],
                       s=17, c=color, alpha=.78, edgecolors='white', linewidths=.35, zorder=3)
        ax.plot([0, 1], [0, 1], color=G, lw=.65, ls=':')
        ax.axhline(.15, color=O, lw=.75, ls='--')
        ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), xticks=[0, .5, 1], yticks=[0, .5, 1],
               xlabel='Failure fraction before selection', ylabel='Retained failure fraction')
        title(ax, letter, heading)
        rho = stats['spearman']
        ann = f"{len(rows)} clips; Spearman {rho:.3f}" if rho is not None else f'{len(rows)} clips; correlation undefined'
        ax.text(.025, .96, ann, transform=ax.transAxes, va='top', fontsize=7.2)
        if len(valid) != len(rows):
            ax.text(.025, .87, f'{len(rows)-len(valid)} with no retained observations', transform=ax.transAxes, fontsize=7)
        tidy(ax)
        return len(valid)
    def pg_panels(ax, count_ax, host, letter, compact=False):
        rows = sorted([r for r in pg if r['host'] == host and r['rule'] == 'prefix'], key=lambda r: r['clip'])
        stats = summary['pg']['groups'][host]['prefix']['crossfold']
        xx = np.arange(len(rows))
        colors = [B if r['fold'] == 'primary' else T for r in rows]
        for fold, marker, color in [('primary', 'o', B), ('reverse', '^', T)]:
            keep = [(i, r) for i, r in enumerate(rows) if r['fold'] == fold and r['retained_risk'] != '']
            ax.scatter([i for i, r in keep], [float(r['retained_risk']) for i, r in keep],
                       s=25 if not compact else 18, marker=marker, color=color, edgecolors='white', linewidths=.3, zorder=4)
        missing = [i for i, r in enumerate(rows) if r['retained_risk'] == '']
        if missing:
            ax.scatter(missing, [0] * len(missing), marker='x', s=20, color=G, zorder=4)
        ax.axhline(.15, color=O, lw=.75, ls='--')
        pooled = stats['pooled_risk']
        if pooled is not None: ax.axhline(pooled, color=INK, lw=.7)
        ann = f"Pooled {pooled:.3f}; {stats['compliant_clips']}/{stats['admitted_clips']} meet 0.15" if pooled is not None else 'No retained observations'
        if not compact:
            folds = summary['pg']['groups'][host]['prefix']['folds']
            a, b = folds['primary']['pooled_risk'], folds['reverse']['pooled_risk']
            if a is not None and b is not None:
                ann = f'Direction A / B: {a:.3f} / {b:.3f}\n' + ann
        ax.text(.02, .96, ann, transform=ax.transAxes, va='top', fontsize=7 if compact else 7.2)
        ax.set(ylim=(-.025, 1.025), xlim=(-.8, len(rows)-.2), yticks=[0, .5, 1], ylabel='Retained failure fraction')
        ax.tick_params(axis='x', labelbottom=False)
        title(ax, letter, 'PolypGen: ' + (names[host] if compact else 'each sequence evaluated once'))
        tidy(ax)
        count_ax.bar(xx, [int(r['retained_count']) for r in rows], color=colors, width=.60, linewidth=0)
        count_ax.set(xlim=(-.8, len(rows)-.2), ylabel='Retained\nobservations', xlabel='Sequence ID')
        labels = [r['clip'].removeprefix('seq') for r in rows]
        step = max(1, len(rows)//5) if compact else 1
        count_ax.set_xticks(xx[::step], labels[::step])
        count_ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
        count_ax.yaxis.get_offset_text().set_fontsize(6.5)
        tidy(count_ax)
        return len(rows)
    def save(fig, name, counts, note):
        if summary.get('synthetic_demo'):
            fig.text(.5, .992, 'SYNTHETIC DEMONSTRATION — NOT STUDY RESULTS', ha='center', va='top', fontsize=7, color=O)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside = []
        for tx in fig.findobj(Text):
            if tx.get_visible() and tx.get_text().strip():
                bb = tx.get_window_extent(renderer)
                if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1+1 or bb.y1 > fig.bbox.y1+1:
                    outside.append(tx.get_text())
        if outside: raise ValueError(f'{name}: text outside canvas: {outside}')
        fig.savefig(out / f'{name}.svg', metadata={'Date': None})
        fig.savefig(out / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(out / f'{name}.png', dpi=600)
        svg = (out / f'{name}.svg').read_text()
        assert '<image' not in svg
        receipts.append({'figure': name, 'counts': counts, 'inches': list(fig.get_size_inches()),
            'png_dpi': 600, 'raster_in_vector': False, 'canvas_bounds': 'PASS', 'note': note,
            'sources': {p: hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ['sun_before_after.csv', 'pg_retained.csv', 'summary.json']},
            'outputs': {ext: hashlib.sha256((out/f'{name}.{ext}').read_bytes()).hexdigest() for ext in ['svg','pdf','png']}})
        plt.close(fig)

    fig = plt.figure(figsize=(7.2, 3.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.48], height_ratios=[2.6, 1],
                         left=.085, right=.98, bottom=.16, top=.89, wspace=.36, hspace=.18)
    ax = fig.add_subplot(gs[:, 0])
    n_sun = sun_panel(ax, 'community_pooled', 'a', 'SUN-SEG: paired clip risks')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=B, label='Other clips', markersize=4),
        Line2D([], [], marker='o', ls='', color=O, label='Reference failing clips', markersize=4)],
        loc='lower right', frameon=False, fontsize=6.8, handletextpad=.3)
    risk_ax, count_ax = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    n_pg = pg_panels(risk_ax, count_ax, 'community_pooled', 'b')
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=B, label='Direction A', markersize=4),
        Line2D([], [], marker='^', ls='', color=T, label='Direction B', markersize=4),
        Line2D([], [], color=INK, label='Pooled risk', lw=.7),
        Line2D([], [], color=O, label='0.15 criterion', lw=.75, ls='--')],
        loc='lower center', bbox_to_anchor=(.55, .001), ncol=4, frameon=False, fontsize=6.8, columnspacing=1, handlelength=1.3)
    save(fig, 'Figure_9_extension', {'sun_points': n_sun, 'pg_sequences': n_pg},
         'SUN uses frozen thresholds. PolypGen displays both prefix-scan directions, each sequence once. Dashed diagonal is identity, not a fitted line; risk values and counts are unmodified.')

    fig = plt.figure(figsize=(7.2, 5.6))
    outer = fig.add_gridspec(2, 3, left=.13, right=.98, bottom=.11, top=.935, hspace=.55, wspace=.52)
    point_counts = {}
    for j, host in enumerate(names):
        ax = fig.add_subplot(outer[0, j]); point_counts[host+'_sun'] = sun_panel(ax, host, chr(97+j), names[host])
        ax.set_xlabel('Failure fraction before selection', fontsize=7)
        ax.set_ylabel('Retained failure fraction', fontsize=7)
        inner = outer[1, j].subgridspec(2, 1, height_ratios=[2.5, 1], hspace=.12)
        a, b = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        point_counts[host+'_pg'] = pg_panels(a, b, host, chr(100+j), compact=True)
        a.set_ylabel('Retained risk', fontsize=7); b.set_ylabel('Retained count', fontsize=7)
    fig.text(.5, .015, 'SUN-SEG: orange points mark reference failing clips. PolypGen: circles = Direction A, triangles = Direction B.',
             ha='center', fontsize=6.7, color=INK)
    save(fig, 'Supplement_extension', point_counts, 'All model groups, all SUN evaluation clips and all PolypGen evaluation sequences; prefix scan only. Full-tie-group results are in the accompanying tables and CSV.')
    (out / 'plot_receipt.json').write_text(json.dumps({'synthetic_demo': summary.get('synthetic_demo', False), 'figures': receipts}, indent=2)+'\n')
