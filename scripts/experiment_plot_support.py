"""Shared rendering functions used by the v0.46 experiment plots."""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
GREY = '#A5ADB4'
PALETTE = ['#245E88', '#B56525', '#447A60', '#795A91', '#508B9A', '#5B626B']
SOURCES = ['Chen2014', 'Zhang2017']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})

def shown_rows(observed, source):
    return [r for r in observed if r['study_id'] == source and r['displayed']]

def condition_label(row):
    if row['study_id'] == 'Chen2014':
        return row['treatment_id'].split('_D')[1] + '% Met'
    return f"{row['temperature_c']:g} °C"

def style_legend(fig, y):
    handles = [Line2D([0], [0], color=GREY, lw=1, ls='--', label='Uncalibrated'),
               Line2D([0], [0], color=PALETTE[0], lw=1.4, label='Source calibrated'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor='white',
                      markeredgecolor='#333333', markersize=4, label='Reported mean'),
               Line2D([0], [0], marker='D', color='none', markerfacecolor='white',
                      markeredgecolor='#333333', markersize=4, label='Completed control')]
    return fig.legend(handles=handles, loc='lower center', ncol=2,
                      bbox_to_anchor=(.55, y), columnspacing=2.4, handlelength=2.4)

def panel(ax, letter, title, horizon):
    ax.set_title(title, fontsize=8, pad=12)
    ax.annotate(letter, (0, 1), xycoords='axes fraction', xytext=(-15, 12),
                textcoords='offset points', weight='bold', fontsize=9, annotation_clip=False)
    ax.set_xlabel('Culture day')
    ax.set_xlim(-1, horizon + 2)
    ax.set_xticks(np.arange(0, horizon + 1, 14))
    ax.grid(axis='y', color='#E5E9EC', lw=.5)

def plot_series(ax, records, observed, source, key, day_midpoint=False, scatter=False):
    shown = shown_rows(observed, source)
    colors = {r['treatment_id']: PALETTE[i] for i, r in enumerate(shown)}
    for stage in ['loso', 'information_matched_anchor']:
        for r in shown:
            rr = [x for x in records if x['treatment_id'] == r['treatment_id'] and x['stage'] == stage and x[key] != '']
            xx = np.array([x['day'] for x in rr], dtype=float) + (.5 if day_midpoint else 0.)
            yy = [x[key] for x in rr]
            calibrated = stage == 'information_matched_anchor'
            ax.plot(xx, yy, color=colors[r['treatment_id']] if calibrated else GREY,
                    ls='-' if calibrated else '--', lw=1.35 if calibrated else .8,
                    zorder=2 if calibrated else 1)
    if scatter:
        for r in shown:
            marker = 'D' if r['is_calibration_control'] else 'o'
            if key == 'weight_g':
                for day, val, err in [(0, r['initial_weight_g'], r['initial_weight_se_g']),
                                      (r['days'], r['final_weight_g'], r['final_weight_se_g'])]:
                    ax.errorbar(day, val, yerr=err if err != '' else None, fmt=marker,
                                color=colors[r['treatment_id']], mfc='white', mec=colors[r['treatment_id']],
                                ms=4 if marker == 'o' else 5, mew=.85, capsize=2,
                                elinewidth=.65, zorder=4)
            else:
                ax.plot(r['days'], r['cumulative_feed_target_g'], marker=marker, ls='none',
                        mfc='white', mec=colors[r['treatment_id']], ms=4 if marker == 'o' else 5, zorder=4)
    return [Line2D([0], [0], color=colors[r['treatment_id']], lw=1.4,
                   label=condition_label(r)) for r in shown]

def dynamic(records, observed):
    fig, axes = plt.subplots(2, 2, figsize=(7.0866, 6.4961))
    fig.subplots_adjust(left=.10, right=.97, bottom=.17, top=.855, wspace=.35, hspace=.78)
    for row, source in enumerate(SOURCES):
        horizon = int(shown_rows(observed, source)[0]['days'])
        panel(axes[row, 0], 'ac'[row], 'Wet mass', horizon)
        panel(axes[row, 1], 'bd'[row], 'Daily consumption', horizon)
        handles = plot_series(axes[row, 0], records, observed, source, 'weight_g', scatter=True)
        plot_series(axes[row, 1], records, observed, source, 'daily_intake_g', day_midpoint=True)
        axes[row, 0].set_ylabel('Wet mass (g fish⁻¹)')
        axes[row, 1].set_ylabel('Daily intake (g fish⁻¹ d⁻¹)')
        axes[row, 1].set_ylim(bottom=0)
        title = 'Chen2014 · six dietary methionine levels · 28 °C' if row == 0 else 'Zhang2017 · five temperatures · medium initial size'
        fig.text(.535, .982 if row == 0 else .520, title, ha='center', fontsize=8, weight='bold')
        fig.legend(handles=handles, loc='upper center', ncol=3 if row == 0 else 5,
                   bbox_to_anchor=(.535, .963 if row == 0 else .500),
                   columnspacing=1.7, handlelength=1.4)
    style_legend(fig, .025)
    save(fig, 'fig4_dynamic_trajectories', multi=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0866, 3.6220))
    fig.subplots_adjust(left=.10, right=.97, bottom=.25, top=.85, wspace=.35)
    for ax, source, letter in zip(axes, SOURCES, ['a', 'b']):
        horizon = int(shown_rows(observed, source)[0]['days'])
        panel(ax, letter, source + ' · cumulative consumption', horizon)
        plot_series(ax, records, observed, source, 'cumulative_feed_g', scatter=True)
        ax.set_ylabel('Cumulative intake (g fish⁻¹)')
        ax.set_ylim(bottom=0)
    style_legend(fig, .015)
    save(fig, 'figS1_cumulative_intake', multi=True)
