import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------
# Matplotlib: mathtext only (no external LaTeX; works on Windows without MiKTeX)
# --------------------------------------------------
mpl.rcParams.update({
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "axes.labelsize":     12,
    "font.size":          11,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
})

# --------------------------------------------------
# Piecewise functions
# --------------------------------------------------
def d_min_ar(ph):
    ph = np.asarray(ph)
    y = np.zeros_like(ph, dtype=float)

    m1 = (ph >= 6.0) & (ph < 6.5)
    m2 = (ph >= 5.5) & (ph < 6.0)
    m3 = (ph >= 5.0) & (ph < 5.5)
    m4 = ph < 5.0

    y[m1] = 4.5 * (6.5 - ph[m1]) / 0.5
    y[m2] = 4.5 + 4.5 * (6.0 - ph[m2]) / 0.5
    y[m3] = 9.0 + 4.0 * (5.5 - ph[m3]) / 0.5
    y[m4] = 13.0
    return y

def d_min_gr(ph):
    ph = np.asarray(ph)
    y = np.zeros_like(ph, dtype=float)

    m1 = (ph >= 5.0) & (ph < 5.5)
    m2 = (ph >= 4.5) & (ph < 5.0)
    m3 = ph < 4.5

    y[m1] = 3.5 * (5.5 - ph[m1]) / 0.5
    y[m2] = 3.5 + 3.0 * (5.0 - ph[m2]) / 0.5
    y[m3] = 6.5
    return y

def d_org_ar(ph):
    ph = np.asarray(ph)
    y = np.zeros_like(ph, dtype=float)

    m1 = (ph >= 5.5) & (ph < 6.0)
    m2 = (ph >= 5.0) & (ph < 5.5)
    m3 = ph < 5.0

    y[m1] = 8.5 * (6.0 - ph[m1]) / 0.5
    y[m2] = 8.5 + 7.0 * (5.5 - ph[m2]) / 0.5
    y[m3] = 15.5
    return y

def d_org_gr(ph):
    ph = np.asarray(ph)
    y = np.zeros_like(ph, dtype=float)

    m1 = (ph >= 5.0) & (ph < 5.5)
    m2 = (ph >= 4.5) & (ph < 5.0)
    m3 = ph < 4.5

    y[m1] = 1.5 * (5.5 - ph[m1]) / 0.5
    y[m2] = 1.5 + 5.0 * (5.0 - ph[m2]) / 0.5
    y[m3] = 6.5
    return y

# --------------------------------------------------
# Domain
# --------------------------------------------------
ph = np.linspace(4.0, 7.0, 600)

# --------------------------------------------------
# Figure  (GridSpec: curve panel + equation panel per subplot)
# --------------------------------------------------
from matplotlib.gridspec import GridSpec

curves = [
    {
        "title":   "Mineral soils \u2014 arable",
        "func":    d_min_ar,
        "color":   "tab:blue",
        "eq_lhs":  r"$d_{\mathrm{min,ar}}(\mathrm{pH}) =$",
        "eq_cases": [
            (r"$0$",
             r"if $\mathrm{pH} \geq 6.5$"),
            (r"$4.5\,\dfrac{6.5 - \mathrm{pH}}{0.5}$",
             r"if $6.0 \leq \mathrm{pH} < 6.5$"),
            (r"$4.5 + 4.5\,\dfrac{6.0 - \mathrm{pH}}{0.5}$",
             r"if $5.5 \leq \mathrm{pH} < 6.0$"),
            (r"$9.0 + 4.0\,\dfrac{5.5 - \mathrm{pH}}{0.5}$",
             r"if $5.0 \leq \mathrm{pH} < 5.5$"),
            (r"$13.0$",
             r"if $\mathrm{pH} < 5.0$"),
        ],
    },
    {
        "title":   "Mineral soils \u2014 grassland",
        "func":    d_min_gr,
        "color":   "tab:green",
        "eq_lhs":  r"$d_{\mathrm{min,gr}}(\mathrm{pH}) =$",
        "eq_cases": [
            (r"$0$",
             r"if $\mathrm{pH} \geq 5.5$"),
            (r"$3.5\,\dfrac{5.5 - \mathrm{pH}}{0.5}$",
             r"if $5.0 \leq \mathrm{pH} < 5.5$"),
            (r"$3.5 + 3.0\,\dfrac{5.0 - \mathrm{pH}}{0.5}$",
             r"if $4.5 \leq \mathrm{pH} < 5.0$"),
            (r"$6.5$",
             r"if $\mathrm{pH} < 4.5$"),
        ],
    },
    {
        "title":   "Organic/peaty soils \u2014 arable",
        "func":    d_org_ar,
        "color":   "tab:red",
        "eq_lhs":  r"$d_{\mathrm{org,ar}}(\mathrm{pH}) =$",
        "eq_cases": [
            (r"$0$",
             r"if $\mathrm{pH} \geq 6.0$"),
            (r"$8.5\,\dfrac{6.0 - \mathrm{pH}}{0.5}$",
             r"if $5.5 \leq \mathrm{pH} < 6.0$"),
            (r"$8.5 + 7.0\,\dfrac{5.5 - \mathrm{pH}}{0.5}$",
             r"if $5.0 \leq \mathrm{pH} < 5.5$"),
            (r"$15.5$",
             r"if $\mathrm{pH} < 5.0$"),
        ],
    },
    {
        "title":   "Organic/peaty soils \u2014 grassland",
        "func":    d_org_gr,
        "color":   "tab:purple",
        "eq_lhs":  r"$d_{\mathrm{org,gr}}(\mathrm{pH}) =$",
        "eq_cases": [
            (r"$0$",
             r"if $\mathrm{pH} \geq 5.5$"),
            (r"$1.5\,\dfrac{5.5 - \mathrm{pH}}{0.5}$",
             r"if $5.0 \leq \mathrm{pH} < 5.5$"),
            (r"$1.5 + 5.0\,\dfrac{5.0 - \mathrm{pH}}{0.5}$",
             r"if $4.5 \leq \mathrm{pH} < 5.0$"),
            (r"$6.5$",
             r"if $\mathrm{pH} < 4.5$"),
        ],
    },
]

fig = plt.figure(figsize=(14, 16))
gs = GridSpec(
    4, 2, figure=fig,
    height_ratios=[4.5, 3.4, 4.5, 3.6],
    hspace=0.10, wspace=0.38,
)

PLOT_ROWS = [0, 0, 2, 2]
EQ_ROWS   = [1, 1, 3, 3]
COLS      = [0, 1, 0, 1]
LABELS    = ["(a)", "(b)", "(c)", "(d)"]

VAL_X  = 0.52   # right edge of value column (axes fraction)
COND_X = 0.55   # left  edge of "if …" column (axes fraction)

for c, prow, erow, col, lbl in zip(curves, PLOT_ROWS, EQ_ROWS, COLS, LABELS):

    # ---- curve panel ----
    ax = fig.add_subplot(gs[prow, col])
    ax.plot(ph, c["func"](ph), color=c["color"], lw=2.0, zorder=3)

    for xv in [4.5, 5.0, 5.5, 6.0, 6.5]:
        ax.axvline(xv, color="0.82", lw=0.7, ls="--", zorder=1)

    ax.set_title(c["title"], fontsize=12, pad=6)
    ax.set_xlim(4.0, 7.0)
    ax.set_ylim(-0.4, 17.5)
    ax.set_yticks([0, 4, 8, 12, 16])

    if prow == 2:
        ax.set_xlabel(r"$\mathrm{pH}_{\mathrm{H_2O}}$", fontsize=12)
    else:
        ax.tick_params(labelbottom=False)

    if col == 0:
        ax.set_ylabel(r"$d(\mathrm{pH})$  [t CaCO$_3$ ha$^{-1}$]", fontsize=12)

    ax.text(-0.10, 1.04, lbl, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

    # ---- equation panel ----
    eqax = fig.add_subplot(gs[erow, col])
    eqax.axis("off")

    n = len(c["eq_cases"])
    # Bottom-row equation panels (c, d): shift content slightly lower in axes
    if erow == 3:
        ys = np.linspace(0.84, 0.06, n)
        lhs_y = 0.44
    else:
        ys = np.linspace(0.88, 0.12, n)
        lhs_y = 0.50

    eqax.text(0.02, lhs_y, c["eq_lhs"],
              transform=eqax.transAxes,
              ha="left", va="center", fontsize=13)

    for j, (val, cond) in enumerate(c["eq_cases"]):
        eqax.text(VAL_X, ys[j], val,
                  transform=eqax.transAxes,
                  ha="right", va="center", fontsize=12)
        eqax.text(COND_X, ys[j], cond,
                  transform=eqax.transAxes,
                  ha="left", va="center", fontsize=12)

    eqax.plot([0, 1], [1, 1], color="0.78", lw=0.7,
              transform=eqax.transAxes, clip_on=False)

fig.suptitle("RB209-based piecewise lime-demand functions", fontsize=14, y=0.998)
fig.subplots_adjust(top=0.958, bottom=0.04)

fig.savefig("rb209_lime_curves.pdf", bbox_inches="tight")
fig.savefig("rb209_lime_curves.png", dpi=300, bbox_inches="tight")
print("done")
plt.show()
