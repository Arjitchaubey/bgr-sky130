import os, re, json, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

des = json.load(open("results/design.json"))
R1, R2_0, VBE0, I0 = des["R1"], des["R2_best"], des["vbe0"], des["I"]
VREF_G = des["vref_ideal"]
VDD = 1.8

Q  = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")

NET = f"""* BGR with transistor-level PMOS-input OTA and PMOS pass device
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
VDD vdd 0 {VDD}
* bandgap core
XQ1 0 0 a {Q}
{q2}
R2a vref a @R2@
R2b vref b @R2@
R1  b e2 {R1}
* pass device
XMP vref g vdd vdd {PF} W=20 L=1
* OTA: M1 gate=b (+), M2 gate=a (-)
XMT tail vbp vdd vdd {PF} W=10 L=2
XM1 n1 b tail vdd {PF} W=20 L=2
XM2 g  a tail vdd {PF} W=20 L=2
XM3 n1 n1 0 0 {NF} W=5 L=2
XM4 g  n1 0 0 {NF} W=5 L=2
* bias (ideal for now)
XMB vbp vbp vdd vdd {PF} W=10 L=2
IB vbp 0 2u
C1 g vdd 2p
CL vref 0 1p
.nodeset v(vref)={VREF_G} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} v(g)=1.2 v(tail)=1.35 v(vbp)=1.2 v(n1)=0.55
"""

OP = NET + """.control
op
print v(vref) v(a) v(b) v(g) v(tail) v(n1) v(vbp) i(vdd)
quit
.endc
.end
"""

SWEEP = NET + """.control
dc temp 27 125 1
wrdata @HI@ vref a b g tail i(vdd)
dc temp 27 -40 -1
wrdata @LO@ vref a b g tail i(vdd)
quit
.endc
.end
"""

def run(text, name):
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    return subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)

os.makedirs("build", exist_ok=True)

# ---- stage 1: operating point sanity check ----
r = run(OP.replace("@R2@", f"{R2_0:.1f}"), "bgr_real_op")
print("---- DC operating point at 27C ----")
for l in r.stdout.splitlines():
    if "=" in l and re.match(r"\s*(v|i)\(", l, re.I):
        print(l.strip())
m = re.search(r"v\(vref\)\s*=\s*([-\d.e+]+)", r.stdout, re.I)
if not m or float(m.group(1)) < 1.0:
    print("\nOp point failed or stuck in the zero-current state. Last output:")
    print(r.stdout[-2000:])
    raise SystemExit(1)

# ---- stage 2: R2 sweep over temperature ----
factors = np.linspace(0.90, 1.10, 21)
rows, curves = [], {}
for f in factors:
    r2 = R2_0 * f
    tag = f"bgr_real_{f:.3f}"
    hi, lo = f"build/{tag}_hi.csv", f"build/{tag}_lo.csv"
    for p in (hi, lo):
        if os.path.exists(p):
            os.remove(p)
    run(SWEEP.replace("@R2@", f"{r2:.1f}").replace("@HI@", hi).replace("@LO@", lo), tag)
    if not (os.path.exists(hi) and os.path.exists(lo)):
        print(f"ngspice failed at R2={r2/1e3:.2f}k (no output), skipping")
        continue
    dh, dl = np.loadtxt(hi), np.loadtxt(lo)
    d = np.vstack([dl[::-1], dh[1:]])
    T = d[:, 0]
    vref, va, vb, vg, vtail, ivdd = (d[:, k] for k in (1, 3, 5, 7, 9, 11))
    v27 = np.interp(27, T, vref)
    i27 = (v27 - np.interp(27, T, va)) / r2
    if i27 < 1e-6:
        print(f"zero-current state at R2={r2/1e3:.2f}k, skipping")
        continue
    tc = (vref.max() - vref.min()) / (v27 * (T.max() - T.min())) * 1e6
    iq = -np.interp(27, T, ivdd)
    rows.append((r2, tc, v27, i27, iq, T[np.argmax(vref)],
                 VDD - vtail.max(), VDD - vref.max()))
    curves[r2] = (T, vref)

if not rows:
    print("No valid runs. Send me the first lines of build/bgr_real_*.spice output.")
    raise SystemExit(1)

print(f"\n{'R2 (kOhm)':>10} {'TC ppm/C':>9} {'Vref27 (V)':>11} {'Ibr (uA)':>9} {'Iq (uA)':>8} {'T@peak':>7}")
for r2, tc, v27, i27, iq, tpk, _, _ in rows:
    print(f"{r2/1e3:10.2f} {tc:9.1f} {v27:11.4f} {i27*1e6:9.2f} {iq*1e6:8.2f} {tpk:7.0f}")

best = min(rows, key=lambda x: x[1])
print(f"\nBest R2 = {best[0]/1e3:.2f} kOhm -> TC = {best[1]:.1f} ppm/C, "
      f"Vref(27C) = {best[2]:.4f} V, Iq = {best[4]*1e6:.1f} uA")
print(f"Min Vsd of tail source over temperature = {best[6]:.3f} V (want > ~0.15)")
print(f"Min Vsd of pass device over temperature = {best[7]:.3f} V (want > ~0.15)")

plt.figure(figsize=(7, 4.5))
for r2, (T, v) in curves.items():
    plt.plot(T, v * 1e3, lw=2 if r2 == best[0] else 0.7,
             label=f"{r2/1e3:.2f}k" if r2 == best[0] else None)
plt.xlabel("Temperature (C)"); plt.ylabel("Vref (mV)")
plt.title("BGR with transistor-level OTA: Vref vs T"); plt.grid(alpha=.3)
plt.legend(title="best R2"); plt.tight_layout()
plt.savefig("results/step4_real_sweep.png", dpi=150)

des.update(R2_real=best[0], tc_real_ppm=best[1], vref_real=best[2], iq_real=best[4])
json.dump(des, open("results/design.json", "w"), indent=2)
