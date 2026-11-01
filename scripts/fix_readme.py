import os, sys, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.expanduser("~/analog/bgr-sky130")
BR = f"{ROOT}/scripts/build_readme.py"
IMG = f"{ROOT}/docs/img"

src = open(BR).read()

EDITS = [
 (r'''fig.savefig(f"{IMG}/architecture.png", dpi=150, bbox_inches="tight")''',
  r'''fig.savefig("/tmp/_old_architecture.png", dpi=40)'''),
 (r'''print("wrote docs/img/architecture.png")''',
  r'''print("old diagram skipped")'''),
 (r'''("step6c_supply.png", "Vref vs supply, tt (step 6c)."),''',
  r'''("step6c_supply.png", "Vref vs VDD at -40 / 27 / 125 C (step 6c). At -40 C Vref overshoots to about 1.33 V near VDD = 1.36 V before settling (read from the plot); all three curves are flat at 1.8 V."),'''),
 (r'''("step7b_cold_startup.png", "Cold (-40 C) 1 ms-ramp startup, showing the plateau and recovery (step 7b)."),''',
  r'''("step7b_cold_startup.png", "-40 C, 1 ms ramp, tt (left) and ss (right), clamp-diode width W = 2 / 4 / 8 (step 7b). Vref jumps to about 0.7 V, sits on a plateau, then recovers to 1.2 V; wider clamp diodes recover later."),'''),
 (r'''("step7e_pm_corners.png", "Loop-gain results over corners and temperatures, C1 = 20 pF, CL = 10 pF (step 7e)."),''',
  r'''("step7e_pm_corners.png", "Loop gain over corners and temperatures, C1 = 20 pF, CL = 10 pF (step 7e). 15 curves; the plot's color cycle repeats, so individual curves cannot be told apart, but they lie close together."),'''),
 (r'''("step8b_montecarlo.png", "Sensitivity-based Monte Carlo, tt, assumed AVT (step 8b)."),''',
  r'''("step8b_montecarlo.png", "Sensitivity-based Monte Carlo, tt, assumed AVT = 5 mV*um (step 8b). Untrimmed versus 4-bit and 6-bit ratio-only Vref trim: Vref tightens but TC spreads (p99 71.9 ppm/C for 6-bit), the failure that led to offset-then-ratio trim (strategy c, in the table below, not plotted)."),'''),
 (r'''| Min VDD, Vref within 0.5% | 1.26-1.30 V | tt (step 6c) |''',
  r'''| Min VDD, Vref within 0.5% | 1.26-1.30 V at 27 and 125 C; about 1.46 V at -40 C (read from the plot, estimate) | tt (step 6c); at -40 C Vref overshoots to about 1.33 V near VDD = 1.36 V |'''),
 (r'''- Load regulation slope grows strongly with temperature (-54.6 mV/mA at 125 C). Output impedance was not measured separately.''',
  r'''- Load regulation slope grows strongly with temperature (-54.6 mV/mA at 125 C). Output impedance was not measured separately.
- The DC supply sweep at -40 C overshoots to about 1.33 V near VDD = 1.36 V (read from the plot). Vref is correct at 1.8 V and above about 1.46 V, but the low-supply behaviour at cold is not a clean monotonic rise.'''),
]

for old, new in EDITS:
    if src.count(old) != 1:
        print("PATTERN NOT FOUND (nothing written):", old[:70])
        sys.exit(1)
    src = src.replace(old, new)
open(BR, "w").write(src)
print("patched build_readme.py:", len(EDITS), "edits")
def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="k"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

def arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", ls=ls))

def line(ax, xs, ys):
    ax.plot(xs, ys, "k", lw=1.2)

def lab(ax, x, y, t):
    ax.text(x, y, t, fontsize=8, color="darkred")

fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")
ax.plot([0.2, 11.8], [8.5, 8.5], "k", lw=3)
ax.text(0.2, 8.65, "VDD = 1.8 V", fontsize=9)

box(ax, 8.0, 6.4, 3.0, 1.0, "Pass PMOS XMP\nW/L = 100/2", "#fde7d9")
arrow(ax, 9.5, 8.5, 9.5, 7.4)
line(ax, [9.5, 9.5], [6.4, 5.7])
lab(ax, 9.65, 6.0, "Vref")
line(ax, [7.0, 10.4], [5.7, 5.7])
arrow(ax, 7.0, 5.7, 7.0, 5.0)
arrow(ax, 10.4, 5.7, 10.4, 5.0)

box(ax, 5.6, 3.9, 2.8, 1.1, "R2a (65 units)\nVref to node a", "#e8f0fe")
box(ax, 9.0, 3.9, 2.8, 1.1, "R2b (65 units)\nVref to node b", "#e8f0fe")
box(ax, 9.0, 2.5, 2.8, 0.9, "R1 (7 units)\nb to e2", "#e8f0fe")
box(ax, 5.6, 1.0, 2.8, 1.4, "Q1: 1x PNP\n(VEB, node a)", "#e6f4ea")
box(ax, 9.0, 0.4, 2.8, 1.4, "Q2: 8x PNP\n(VEB, node e2)", "#e6f4ea")
arrow(ax, 7.0, 3.9, 7.0, 2.4)
lab(ax, 7.1, 3.0, "a")
arrow(ax, 10.4, 3.9, 10.4, 3.4)
lab(ax, 10.5, 3.55, "b")
arrow(ax, 10.4, 2.5, 10.4, 1.8)

box(ax, 2.4, 3.0, 2.2, 2.2, "NMOS-input OTA\nM1 (b), M2 (a, inv.)\nPMOS load M3/M4\ntail XMT", "#fff4cc")
arrow(ax, 5.6, 4.45, 4.6, 4.45)
lab(ax, 4.7, 4.6, "a to M2")
ax.annotate("", xy=(4.6, 3.65), xytext=(10.4, 3.65),
            arrowprops=dict(arrowstyle="->", ls="--"))
lab(ax, 4.7, 3.78, "b to M1")

box(ax, 0.2, 3.0, 1.8, 2.2, "Self-bias\nXMB diode,\nXMBP mirror\n(~1/10 pass\ncurrent)\nvbn to tail", "#e6f4ea")
arrow(ax, 2.0, 4.1, 2.4, 4.1)

line(ax, [3.5, 3.5], [5.2, 6.9])
arrow(ax, 3.5, 6.9, 8.0, 6.9)
lab(ax, 3.65, 6.45, "g (pass gate)")
box(ax, 4.9, 7.5, 2.6, 0.6, "C1 = 20 pF", "#f3e8fd")
line(ax, [6.2, 6.2], [7.5, 6.9])
line(ax, [6.2, 6.2], [8.1, 8.5])
box(ax, 0.2, 6.2, 2.8, 1.4, "Startup (variant C)\nsenses Vref, pulls g\nthrough diode clamp", "#fde7d9")
arrow(ax, 3.0, 6.9, 3.45, 6.9)

ax.text(0.2, 0.15, "Dashed line = sense wire from node b (it crosses the a-branch, not connected). PNP collectors and bases are grounded.", fontsize=7)
ax.set_title("Self-biased CMOS bandgap, SKY130 1.8 V (block diagram, not a schematic)", fontsize=10)
fig.savefig(f"{IMG}/architecture.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote docs/img/architecture.png")

r = subprocess.run([sys.executable, BR], capture_output=True, text=True)
print(r.stdout, r.stderr)
