import os, json, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

des = json.load(open("results/design.json"))
R1, R2_0, VG, VBE0 = des["R1"], des["R2"], des["vref_pred"], des["vbe0"]
I0 = des["I"]

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")

TEMPLATE = f"""* BGR core, ideal op-amp, ideal resistors
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
XQ1 0 0 a {Q}
{q2}
R2a vref a @R2@
R2b vref b @R2@
R1  b e2 {R1}
* ideal op-amp: negative feedback through both R2 branches
Eop vref 0 a b 1e5
* hints accurate at 27C so the solver picks the 5uA state, not zero-current
.nodeset v(vref)={VG} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1}
.control
dc temp 27 125 1
wrdata @HI@ vref a
dc temp 27 -40 -1
wrdata @LO@ vref a
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
factors = np.linspace(0.920, 0.940, 21)
rows, curves = [], {}

for f in factors:
    r2 = R2_0 * f
    tag = f"bgr_core_{f:.3f}"
    hi, lo = f"build/{tag}_hi.csv", f"build/{tag}_lo.csv"
    for p in (hi, lo):
        if os.path.exists(p):
            os.remove(p)
    net = TEMPLATE.replace("@R2@", f"{r2:.1f}").replace("@HI@", hi).replace("@LO@", lo)
    path = f"build/{tag}.spice"
    open(path, "w").write(os.path.expandvars(net))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    if not (os.path.exists(hi) and os.path.exists(lo)):
        print("ngspice failed for", tag)
        print(r.stdout[-1500:])
        raise SystemExit(1)
    dh, dl = np.loadtxt(hi), np.loadtxt(lo)
    d = np.vstack([dl[::-1], dh[1:]])          # -40 ... 27 ... 125
    T, vref, va = d[:, 0], d[:, 1], d[:, 3]
    i27 = (np.interp(27, T, vref) - np.interp(27, T, va)) / r2
    if i27 < 1e-6:
        print(f"STUCK in zero-current state at R2={r2/1e3:.1f}k (I={i27*1e6:.3f} uA)")
        raise SystemExit(1)
    tc = (vref.max() - vref.min()) / (np.interp(27, T, vref) * (T.max() - T.min())) * 1e6
    rows.append((r2, tc, np.interp(27, T, vref), i27, T[np.argmax(vref)]))
    curves[r2] = (T, vref)

print(f"{'R2 (kOhm)':>10} {'TC ppm/C':>9} {'Vref27 (V)':>11} {'I (uA)':>8} {'T@peak':>7}")
for r2, tc, v27, i27, tpk in rows:
    print(f"{r2/1e3:10.2f} {tc:9.1f} {v27:11.4f} {i27*1e6:8.2f} {tpk:7.0f}")

best = min(rows, key=lambda x: x[1])
print(f"\nBest R2 = {best[0]/1e3:.2f} kOhm  ->  TC = {best[1]:.1f} ppm/C, Vref = {best[2]:.4f} V")

plt.figure(figsize=(7, 4.5))
for r2, (T, v) in curves.items():
    plt.plot(T, v * 1e3, lw=2 if r2 == best[0] else 0.8,
             label=f"{r2/1e3:.1f}k" if r2 == best[0] else None)
plt.xlabel("Temperature (C)"); plt.ylabel("Vref (mV)")
plt.title("BGR core (ideal op-amp): Vref vs T for R2 sweep"); plt.grid(alpha=.3)
plt.legend(title="best R2"); plt.tight_layout()
plt.savefig("results/step3_r2_sweep.png", dpi=150)

des.update(R2_best=best[0], tc_ideal_ppm=best[1], vref_ideal=best[2])
json.dump(des, open("results/design.json", "w"), indent=2)
