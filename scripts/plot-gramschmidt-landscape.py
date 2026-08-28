"""Motivating figure: the locality landscape of Gram-Schmidt.

One symbolic derivation yields a miss-ratio polynomial in the program
parameters (M, N) and the cache size C.  Every point drawn here is an
evaluation of that one closed form -- no simulation, no re-derivation.
"""
import sys, numpy as np, matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

SP = sys.argv[1] if len(sys.argv) > 1 else '.'
d, q = np.load(f'{SP}/figdata.npz'), np.load(f'{SP}/cube.npz')
CG, A, B, Ms_a, Ms_b = d['CG'], d['A'], d['B'], d['Ms_a'], d['Ms_b']
Ns, Ms, cube = q['Ns'], q['Ms'], q['cube']
N_FIX = 128
MRLO, MRHI, MRTOP = 2e-3, 0.75, 2.6

BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
        '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('seq', BLUE)
INK, INK2, INK3, ORANGE = '#0b0b0b', '#52514e', '#8a8983', '#eb6834'

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'font.size': 8, 'axes.labelsize': 8.2,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2, 'legend.fontsize': 6.8,
    'axes.edgecolor': INK3, 'axes.linewidth': .6,
    'xtick.color': INK3, 'ytick.color': INK3,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.labelcolor': INK, 'text.color': INK,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

def cmin(v, tau):
    """Smallest cache size whose miss ratio stays at or below tau."""
    ok = np.maximum.accumulate(v[::-1])[::-1] <= tau
    return CG[np.argmax(ok)] if ok.any() else np.nan

fig = plt.figure(figsize=(7.0, 2.33))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.06, 1.14],
                      left=.080, right=.955, bottom=.185, top=.845, wspace=.52)

def title(ax, tag, txt, pad=5):
    ax.set_title(f'({tag})  {txt}', loc='left', pad=pad, fontsize=8.4)

# --------------------------------------------------- (a) the conventional MRC
ax = fig.add_subplot(gs[0])
shades = [BLUE[3], BLUE[6], BLUE[9], BLUE[12]]
for k, (M, y) in enumerate(zip(Ms_a, A)):
    ax.plot(CG, np.maximum(y, MRLO), color=shades[k], lw=1.6,
            solid_capstyle='round', label=f'$M={M}$', zorder=3 + k)
MH = Ms_a[-1]
for x, lab in ((2 * MH, '$2M$'), (MH * N_FIX / 8, '$MN/8$')):
    ax.axvline(x, color=ORANGE, lw=.9, ls=(0, (1.6, 1.6)), zorder=2)
    ax.text(x, 1.02, lab, color=ORANGE, fontsize=7, ha='center', va='bottom')
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlim(8, 1 << 17); ax.set_ylim(MRLO, MRTOP)
ax.set_xticks([1 << k for k in (4, 8, 12, 16)])
ax.set_yticks([1e-2, 1e-1, .67]); ax.set_yticklabels(['1%', '10%', '67%'])
ax.set_xlabel('cache size $C$ (blocks)'); ax.set_ylabel('miss ratio')
title(ax, 'a', 'what a curve shows')
ax.text(.965, .40, f'$N={N_FIX}$', transform=ax.transAxes, ha='right',
        va='center', fontsize=7.2, color=INK2)

leg = ax.legend(loc='lower left', frameon=False, handlelength=1.1,
                borderpad=.1, labelspacing=.22, handletextpad=.5)
for t, s in zip(leg.get_texts(), shades):
    t.set_color(INK2)
ax.grid(axis='y', color='#ebeae6', lw=.5); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# --------------------------------------------------------- (b) the landscape
ax = fig.add_subplot(gs[1])
Cedge = np.geomspace(CG[0], CG[-1], len(CG) + 1)
Medge = np.concatenate([[Ms_b[0] * .93], np.sqrt(Ms_b[1:] * Ms_b[:-1]), [Ms_b[-1] * 1.07]])
pc = ax.pcolormesh(Cedge, Medge, np.maximum(B, MRLO), cmap=SEQ,
                   norm=LogNorm(MRLO, MRHI), rasterized=True)
mm = np.geomspace(Ms_b[0], Ms_b[-1], 200)
ax.plot(2 * mm, mm, color=ORANGE, lw=1.3, zorder=5)
ax.plot(mm * N_FIX / 8, mm, color=ORANGE, lw=1.3, ls=(0, (3.2, 2)), zorder=5)
ax.annotate('$C=2M$', (2 * 30, 30), color=ORANGE, fontsize=7.2,
            textcoords='offset points', xytext=(3.5, -10), zorder=6)
ax.annotate('$C=MN/8$', (40 * N_FIX / 8, 40), color=ORANGE, fontsize=7.2,
            textcoords='offset points', xytext=(3.5, -10), zorder=6)
ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
ax.set_xlim(16, 1 << 16); ax.set_ylim(Ms_b[0] * .93, Ms_b[-1] * 1.07)
ax.set_xticks([1 << k for k in (4, 8, 12, 16)])
ax.set_yticks([32, 128, 512]); ax.set_yticklabels(['32', '128', '512'])
ax.set_xlabel('cache size $C$ (blocks)'); ax.set_ylabel('rows $M$')
title(ax, 'b', 'the landscape it cuts')
ax.text(.045, .90, f'$N={N_FIX}$', transform=ax.transAxes, ha='left',
        fontsize=7.2, color='white')
cb = fig.colorbar(pc, ax=ax, pad=.035, fraction=.05, ticks=[1e-2, 1e-1])
cb.ax.set_yticklabels(['1%', '10%'])
cb.outline.set_visible(False)
cb.ax.tick_params(length=0, labelsize=6.8, colors=INK2)
cb.set_label('miss ratio', fontsize=7.2, labelpad=3)

# ---------------------------------------------- (c) cache-data co-scaling
ax = fig.add_subplot(gs[2], projection='3d', computed_zorder=False)
X, Y = np.meshgrid(np.log2(Ns.astype(float)), np.log2(Ms.astype(float)))
for tau, col, edge, z in ((.02, BLUE[8], BLUE[11], 3), (.05, ORANGE, '#b8471f', 4)):
    S = np.array([[cmin(cube[i, j], tau) for j in range(len(Ns))]
                  for i in range(len(Ms))])
    ax.plot_surface(X, Y, np.log2(S), color=col, edgecolor=edge, lw=.2,
                    alpha=.94, rstride=1, cstride=1, shade=True, zorder=z)
ax.set_xticks([4, 6, 8]); ax.set_xticklabels(['16', '64', '256'])
ax.set_yticks([4, 6, 8]); ax.set_yticklabels(['16', '64', '256'])
zt = [5, 8, 11, 14]
ax.set_zticks(zt); ax.set_zticklabels([f'$2^{{{t}}}$' for t in zt])
ax.set_xlabel('columns $N$', labelpad=-7, fontsize=7.6)
ax.set_ylabel('rows $M$', labelpad=-7, fontsize=7.6)
ax.set_zlabel('min cache (blocks)', labelpad=-8, fontsize=7.6)
ax.tick_params(pad=-2.5, labelsize=6.4, colors=INK2)
ax.view_init(elev=19, azim=-130)
ax.set_box_aspect((1, 1, .80), zoom=1.02)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_facecolor('#fcfcfb'); a.pane.set_edgecolor('#e4e3de')
    a._axinfo['grid'].update(color='#ebeae6', linewidth=.4)
title(ax, 'c', 'what to provision', pad=-1)
BOX = dict(fc='white', ec='none', alpha=.82, pad=1.2)
ax.text2D(.99, .90, 'for $\\leq 2\\%$: scales with $MN$', color=BLUE[10],
          fontsize=6.9, ha='right', transform=ax.transAxes, bbox=BOX)
ax.text2D(.99, .795, 'for $\\leq 5\\%$: scales with $M$ only', color=ORANGE,
          fontsize=6.9, ha='right', transform=ax.transAxes, bbox=BOX)

fig.savefig(f'{SP}/gramschmidt-landscape.pdf', dpi=300)
fig.savefig(f'{SP}/gramschmidt-landscape.png', dpi=230)
print('written')
