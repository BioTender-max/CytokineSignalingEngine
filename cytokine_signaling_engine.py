"""
CytokineSignalingEngine: NF-kB and JAK-STAT Signaling ODE Models
- NF-kB activation ODE (IkB/IKK/NF-kB dynamics, Lipniacki model simplified)
- JAK-STAT signaling ODE (STAT phosphorylation/dephosphorylation)
- Cytokine dose-response modeling (TNF, IL-6, IFN-g, IL-1b)
- Signaling crosstalk network
- Inflammatory gene expression signature scoring
"""

import numpy as np
import scipy.stats as stats
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)

# ─── ODE Models ──────────────────────────────────────────────────────────────

def nfkb_ode(t, y, TNF, k1=0.5, k2=0.2, k3=0.3, k4=1.0, k5=0.8, k6=0.5):
    """NF-kB signaling ODE (simplified Lipniacki model)
    y = [IKK_neutral, IKK_active, IkBa, NFkB_nuclear]
    """
    IKK_n, IKK_a, IkBa, NFkB_n = y
    IKK_n  = max(IKK_n,  0)
    IKK_a  = max(IKK_a,  0)
    IkBa   = max(IkBa,   0)
    NFkB_n = np.clip(NFkB_n, 0, 1)

    dIKK_n  = -k1 * TNF * IKK_n + k2 * IKK_a
    dIKK_a  =  k1 * TNF * IKK_n - k2 * IKK_a
    dIkBa   =  k3 * NFkB_n - k4 * IKK_a * IkBa
    dNFkB_n =  k5 * IKK_a * (1 - NFkB_n) - k6 * IkBa * NFkB_n
    return [dIKK_n, dIKK_a, dIkBa, dNFkB_n]

def jakstat_ode(t, y, IL6, k_jak=0.4, k_phos=0.3, k_import=0.5, k_export=0.2):
    """JAK-STAT signaling ODE
    y = [STAT_inactive, STAT_pY, STAT_nuclear]
    """
    STAT_i, STAT_pY, STAT_n = y
    STAT_i  = max(STAT_i,  0)
    STAT_pY = max(STAT_pY, 0)
    STAT_n  = max(STAT_n,  0)

    dSTAT_i  = -k_jak * IL6 * STAT_i + k_phos * STAT_pY
    dSTAT_pY =  k_jak * IL6 * STAT_i - k_phos * STAT_pY - k_import * STAT_pY
    dSTAT_n  =  k_import * STAT_pY - k_export * STAT_n
    return [dSTAT_i, dSTAT_pY, dSTAT_n]

# ─── Time course simulations ──────────────────────────────────────────────────
t_span = (0, 60)
t_eval = np.linspace(0, 60, 300)

TNF_doses = [0.1, 0.5, 1.0, 5.0, 10.0]
IL6_doses  = [0.1, 0.5, 1.0, 5.0, 10.0]

nfkb_traces = {}
for dose in TNF_doses:
    y0 = [1.0, 0.0, 0.5, 0.0]
    sol = solve_ivp(nfkb_ode, t_span, y0, t_eval=t_eval,
                    args=(dose,), method='RK45', rtol=1e-6, atol=1e-8)
    nfkb_traces[dose] = sol.y[3]

jakstat_traces = {}
for dose in IL6_doses:
    y0 = [1.0, 0.0, 0.0]
    sol = solve_ivp(jakstat_ode, t_span, y0, t_eval=t_eval,
                    args=(dose,), method='RK45', rtol=1e-6, atol=1e-8)
    jakstat_traces[dose] = sol.y[2]

# ─── Dose-response curves ─────────────────────────────────────────────────────
conc_range = np.logspace(-3, 2, 50)

def peak_nfkb(conc):
    y0 = [1.0, 0.0, 0.5, 0.0]
    sol = solve_ivp(nfkb_ode, t_span, y0, t_eval=t_eval,
                    args=(conc,), method='RK45', rtol=1e-5, atol=1e-7)
    return np.max(sol.y[3])

def peak_stat(conc):
    y0 = [1.0, 0.0, 0.0]
    sol = solve_ivp(jakstat_ode, t_span, y0, t_eval=t_eval,
                    args=(conc,), method='RK45', rtol=1e-5, atol=1e-7)
    return np.max(sol.y[2])

dr_tnf  = np.array([peak_nfkb(c) for c in conc_range])
dr_il6  = np.array([peak_stat(c) for c in conc_range])
dr_ifng = np.array([peak_stat(c * 0.7) for c in conc_range])
dr_il1b = np.array([peak_nfkb(c * 0.5) for c in conc_range])

# ─── Cell line simulation ─────────────────────────────────────────────────────
N_CELLS = 100
cell_tnf_sensitivity = np.random.lognormal(0, 0.5, N_CELLS)
cell_il6_sensitivity = np.random.lognormal(0, 0.5, N_CELLS)

peak_nfkb_cells = np.array([peak_nfkb(1.0 * s) for s in cell_tnf_sensitivity])
peak_stat_cells  = np.array([peak_stat(1.0 * s) for s in cell_il6_sensitivity])

# ─── Crosstalk: NF-kB suppresses JAK-STAT via SOCS ──────────────────────────
socs_factor = 1.0 / (1.0 + 0.5 * peak_nfkb_cells)
peak_stat_crosstalk = peak_stat_cells * socs_factor

tnf_grid = np.linspace(0.1, 5, 15)
il6_grid = np.linspace(0.1, 5, 15)
crosstalk_mat = np.zeros((15, 15))
for i, tnf in enumerate(tnf_grid):
    nfkb_val = peak_nfkb(tnf)
    socs = 1.0 / (1.0 + 0.5 * nfkb_val)
    for j, il6 in enumerate(il6_grid):
        stat_val = peak_stat(il6)
        crosstalk_mat[i, j] = stat_val * socs

# ─── Gene signature scoring ───────────────────────────────────────────────────
N_NFKB_GENES = 50
N_STAT_GENES  = 50

nfkb_gene_expr = np.outer(peak_nfkb_cells, np.random.uniform(0.5, 2.0, N_NFKB_GENES))
nfkb_gene_expr += np.random.normal(0, 0.2, nfkb_gene_expr.shape)

stat_gene_expr = np.outer(peak_stat_crosstalk, np.random.uniform(0.5, 2.0, N_STAT_GENES))
stat_gene_expr += np.random.normal(0, 0.2, stat_gene_expr.shape)

nfkb_score = np.mean(nfkb_gene_expr, axis=1)
stat_score  = np.mean(stat_gene_expr, axis=1)

# ─── Oscillation analysis ─────────────────────────────────────────────────────
trace_high = nfkb_traces[10.0]
peaks_idx, _ = find_peaks(trace_high, height=0.1, distance=10)
if len(peaks_idx) >= 2:
    peak_times = t_eval[peaks_idx]
    periods = np.diff(peak_times)
    mean_period = np.mean(periods)
    n_oscillations = len(peaks_idx)
else:
    mean_period = float('nan')
    n_oscillations = len(peaks_idx)

# ─── Signaling network adjacency ─────────────────────────────────────────────
node_names = ['TNF', 'IL-6', 'IFN-g', 'IL-1b', 'IKK', 'NF-kB', 'JAK', 'STAT', 'SOCS', 'Genes']
n_nodes = len(node_names)
net_adj = np.zeros((n_nodes, n_nodes))
edges = [(0,4,1),(3,4,1),(4,5,1),(1,6,1),(2,6,1),(6,7,1),(5,8,1),(8,7,-1),(7,9,1),(5,9,1)]
for s, t, w in edges:
    net_adj[s, t] = w

# ─── Dashboard ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

ACCENT = '#00d4ff'
RED    = '#ff4444'
GREEN  = '#44ff88'
YELLOW = '#ffdd44'
ORANGE = '#ff8844'
TEXT_COL = 'white'

def style_ax(ax, title=''):
    ax.set_facecolor('#111111')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.tick_params(colors=TEXT_COL, labelsize=7)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    if title:
        ax.set_title(title, color=TEXT_COL, fontsize=9, fontweight='bold', pad=4)

dose_colors = [ACCENT, GREEN, YELLOW, ORANGE, RED]

# Panel 1: NF-kB dynamics
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'NF-kB Dynamics (Multiple TNF Doses)')
for dose, col in zip(TNF_doses, dose_colors):
    ax1.plot(t_eval, nfkb_traces[dose], color=col, lw=1.5, label=f'TNF={dose}')
ax1.set_xlabel('Time (min)', color=TEXT_COL, fontsize=8)
ax1.set_ylabel('NF-kB Nuclear (a.u.)', color=TEXT_COL, fontsize=8)
ax1.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TEXT_COL)

# Panel 2: JAK-STAT dynamics
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'JAK-STAT Dynamics (Multiple IL-6 Doses)')
for dose, col in zip(IL6_doses, dose_colors):
    ax2.plot(t_eval, jakstat_traces[dose], color=col, lw=1.5, label=f'IL-6={dose}')
ax2.set_xlabel('Time (min)', color=TEXT_COL, fontsize=8)
ax2.set_ylabel('STAT Nuclear (a.u.)', color=TEXT_COL, fontsize=8)
ax2.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TEXT_COL)

# Panel 3: Dose-response curves
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'Cytokine Dose-Response Curves')
ax3.semilogx(conc_range, dr_tnf,  color=RED,    lw=2, label='TNF->NF-kB')
ax3.semilogx(conc_range, dr_il6,  color=ACCENT,  lw=2, label='IL-6->STAT')
ax3.semilogx(conc_range, dr_ifng, color=GREEN,   lw=2, label='IFN-g->STAT')
ax3.semilogx(conc_range, dr_il1b, color=ORANGE,  lw=2, label='IL-1b->NF-kB')
ax3.set_xlabel('Cytokine (ng/mL)', color=TEXT_COL, fontsize=8)
ax3.set_ylabel('Peak Activation (a.u.)', color=TEXT_COL, fontsize=8)
ax3.legend(fontsize=6, facecolor='#1a1a1a', labelcolor=TEXT_COL)

# Panel 4: Crosstalk heatmap
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4, 'Crosstalk Heatmap (NF-kB vs STAT)')
cmap_ct = LinearSegmentedColormap.from_list('ct', ['#0a0a0a', '#0044cc', '#00d4ff'])
im4 = ax4.imshow(crosstalk_mat, aspect='auto', cmap=cmap_ct, origin='lower',
                 extent=[il6_grid[0], il6_grid[-1], tnf_grid[0], tnf_grid[-1]])
ax4.set_xlabel('IL-6 (ng/mL)', color=TEXT_COL, fontsize=8)
ax4.set_ylabel('TNF (ng/mL)', color=TEXT_COL, fontsize=8)
cb4 = plt.colorbar(im4, ax=ax4, fraction=0.03, pad=0.02)
cb4.set_label('STAT Activity', color=TEXT_COL, fontsize=7)
cb4.ax.tick_params(colors=TEXT_COL, labelsize=6)

# Panel 5: Gene signature scores
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5, 'Gene Signature Scores (NF-kB vs STAT)')
sc = ax5.scatter(nfkb_score, stat_score, c=peak_nfkb_cells, cmap='plasma',
                 s=20, alpha=0.8)
ax5.set_xlabel('NF-kB Signature Score', color=TEXT_COL, fontsize=8)
ax5.set_ylabel('STAT Signature Score', color=TEXT_COL, fontsize=8)
cb5 = plt.colorbar(sc, ax=ax5, fraction=0.03, pad=0.02)
cb5.set_label('Peak NF-kB', color=TEXT_COL, fontsize=7)
cb5.ax.tick_params(colors=TEXT_COL, labelsize=6)

# Panel 6: Peak NF-kB by cell line
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6, 'Peak NF-kB by Cell Line')
sorted_idx = np.argsort(peak_nfkb_cells)[::-1]
ax6.bar(np.arange(N_CELLS), peak_nfkb_cells[sorted_idx],
        color=RED, alpha=0.8, width=1.0)
ax6.axhline(np.mean(peak_nfkb_cells), color=YELLOW, lw=1.5, ls='--',
            label=f'Mean={np.mean(peak_nfkb_cells):.3f}')
ax6.set_xlabel('Cell Line (ranked)', color=TEXT_COL, fontsize=8)
ax6.set_ylabel('Peak NF-kB (a.u.)', color=TEXT_COL, fontsize=8)
ax6.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COL)

# Panel 7: Signaling network adjacency
ax7 = fig.add_subplot(gs[2, 0])
style_ax(ax7, 'Signaling Network Adjacency')
cmap_net = LinearSegmentedColormap.from_list('net', ['#cc0000', '#111111', '#00cc44'])
im7 = ax7.imshow(net_adj, aspect='auto', cmap=cmap_net, vmin=-1, vmax=1,
                 interpolation='nearest')
ax7.set_xticks(np.arange(n_nodes))
ax7.set_xticklabels(node_names, fontsize=7, rotation=45, ha='right')
ax7.set_yticks(np.arange(n_nodes))
ax7.set_yticklabels(node_names, fontsize=7)
plt.colorbar(im7, ax=ax7, fraction=0.03, pad=0.02).ax.tick_params(colors=TEXT_COL, labelsize=6)

# Panel 8: Oscillation analysis
ax8 = fig.add_subplot(gs[2, 1])
style_ax(ax8, 'NF-kB Oscillation Analysis (TNF=10)')
ax8.plot(t_eval, trace_high, color=RED, lw=1.5, label='NF-kB nuclear')
if len(peaks_idx) > 0:
    ax8.scatter(t_eval[peaks_idx], trace_high[peaks_idx],
                color=YELLOW, s=40, zorder=5, label=f'{n_oscillations} peaks')
ax8.set_xlabel('Time (min)', color=TEXT_COL, fontsize=8)
ax8.set_ylabel('NF-kB Nuclear (a.u.)', color=TEXT_COL, fontsize=8)
period_str = f'{mean_period:.1f} min' if not np.isnan(mean_period) else 'N/A'
ax8.text(0.98, 0.95, f'Period: {period_str}', transform=ax8.transAxes,
         ha='right', va='top', color=TEXT_COL, fontsize=8,
         bbox=dict(facecolor='#1a1a1a', alpha=0.7, edgecolor='none'))
ax8.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COL)

# Panel 9: Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor('#111111')
for spine in ax9.spines.values():
    spine.set_edgecolor('#333333')
ax9.axis('off')

r_corr, r_p = stats.pearsonr(nfkb_score, stat_score)
summary_text = (
    f"CYTOKINE SIGNALING ENGINE SUMMARY\n"
    f"{'─'*36}\n"
    f"Cell lines:          {N_CELLS}\n"
    f"Cytokines modeled:   4 (TNF, IL-6, IFN-g, IL-1b)\n"
    f"ODE variables:       NF-kB(4) + JAK-STAT(3)\n\n"
    f"NF-kB DYNAMICS\n"
    f"Peak NF-kB (TNF=1):  {peak_nfkb(1.0):.4f}\n"
    f"Peak NF-kB (TNF=10): {peak_nfkb(10.0):.4f}\n"
    f"Oscillation period:  {period_str}\n"
    f"N oscillation peaks: {n_oscillations}\n\n"
    f"JAK-STAT DYNAMICS\n"
    f"Peak STAT (IL-6=1):  {peak_stat(1.0):.4f}\n"
    f"Peak STAT (IL-6=10): {peak_stat(10.0):.4f}\n\n"
    f"CELL LINE ANALYSIS\n"
    f"Mean peak NF-kB:     {np.mean(peak_nfkb_cells):.4f}\n"
    f"Mean peak STAT:      {np.mean(peak_stat_cells):.4f}\n"
    f"NF-kB/STAT corr:     r={r_corr:.3f} (p={r_p:.2e})\n\n"
    f"GENE SIGNATURES\n"
    f"NF-kB genes:         {N_NFKB_GENES}\n"
    f"STAT genes:          {N_STAT_GENES}\n"
)
ax9.text(0.04, 0.97, summary_text, transform=ax9.transAxes,
         va='top', ha='left', color=TEXT_COL, fontsize=7.5,
         fontfamily='monospace',
         bbox=dict(facecolor='#1a1a1a', alpha=0.6, edgecolor='#333333'))

fig.suptitle('CytokineSignalingEngine: NF-kB and JAK-STAT ODE Dashboard',
             color=TEXT_COL, fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/workspace/subagents/5c85659d/cytokine_signaling_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("Dashboard saved: /workspace/subagents/5c85659d/cytokine_signaling_dashboard.png")

# ─── Structured Summary ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("CYTOKINE SIGNALING ENGINE — STRUCTURED SUMMARY")
print("="*60)
print(f"Cell lines:               {N_CELLS}")
print(f"Cytokines modeled:        4 (TNF, IL-6, IFN-g, IL-1b)")
print(f"ODE variables:            NF-kB(4) + JAK-STAT(3)")
print(f"Simulation time:          0-60 min ({len(t_eval)} points)")
print(f"Peak NF-kB (TNF=1.0):    {peak_nfkb(1.0):.4f}")
print(f"Peak NF-kB (TNF=10.0):   {peak_nfkb(10.0):.4f}")
print(f"Peak STAT (IL-6=1.0):    {peak_stat(1.0):.4f}")
print(f"Peak STAT (IL-6=10.0):   {peak_stat(10.0):.4f}")
print(f"NF-kB oscillation peaks: {n_oscillations}  period={period_str}")
print(f"Mean peak NF-kB (cells): {np.mean(peak_nfkb_cells):.4f} +/- {np.std(peak_nfkb_cells):.4f}")
print(f"Mean peak STAT (cells):  {np.mean(peak_stat_cells):.4f} +/- {np.std(peak_stat_cells):.4f}")
print(f"NF-kB/STAT correlation:  r={r_corr:.3f}  p={r_p:.2e}")
print(f"SOCS suppression range:  {socs_factor.min():.3f} - {socs_factor.max():.3f}")
print("="*60)
