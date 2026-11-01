import os, re, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = 5    # temperature step in C (set to 1 for final results)
des = json.load(open("results/design.json"))
R1, R2_0, VBE0, I0 = des["R1"], des["R2_best"], des["vbe0"], des["I"]
VG0 = des["vref_ideal"]
VDD = 1.8

Q  = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")

NET = f"""* BGR with NMOS-input OTA, PMOS mirror load, PMOS pass device
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
VDD vdd 0 {VDD}
XQ1 0 0 a {Q}
{q2}
R2a vref a @R2@
R2b vref b @R2@
R1  b e2 {R1}
* pass device: W/L = 10x load mirror, so its Vsg matches M3 (no systematic offset)
XMP vref g vdd vdd {PF} W=100 L=2
* OTA: M1 gate=b (+), M2 gate=a (-)
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 g  a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 g  n1 vdd vdd {PF} W=10 L=2
* bias (ideal current source for now)
XMB vbn vbn 0 0 {NF} W=10 L=2
IB vdd vbn 1u
C1 g vdd 2p
CL vref 0 1p
.nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} v(g)=0.68 v(n1)=0.68 v(tail)=0.17 v(vbn)=0.55
"""

OP = NET + """.control
op
print v(vref) v(a) v(b) v(g) v(n1) v(tail) v(vbn) i(vdd)
quit
.endc
.end
"""

SWEEP = NET + """.control
dc temp 25 125 @STEP@
wrdata @HI@ vref a b g tail n1 i(vdd)
dc temp 25 -40 -@STEP@
wrdata @LO@ vref a b g tail n1 i(vdd)
quit
.endc
.end
""".replace("@STEP@", str(STEP))

def run(text, name):
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    return subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)

os.makedirs("build", exist_ok=True)

# ---- stage 1: operating point ----
r = run(OP.replace("@R2@", f"{R2_0:.1f}"), "bgr_nmos_op")
print("---- DC operating point at 27C ----")
for l in r.stdout.splitlines():
    if "=" in l and re.match(r"\s*(v|i)\(", l, re.I):
        print(l.strip())
m = re.search(r"v\(vref\)\s*=\s*([-\d.e+]+)", r.stdout, re.I)
if not m or float(m.group(1)) < 1.0:
    print("\nOp point failed or stuck in the zero-current state. Last output:")
    print(r.stdout[-2000:])
    raise SystemExit(1)

# ---- stage 2: R2 sweep over temperature, in parallel ----
factors = np.linspace(0.96, 1.04, 9)

def one(f):
    r2 = R2_0 * f
    tag = f"bgr_nmos_{f:.3f}"
    hi, lo = f"build/{tag}_hi.csv", f"build/{tag}_lo.csv"
    for p in (hi, lo):
        if os.path.exists(p):
            os.remove(p)
    run(SWEEP.replace("@R2@", f"{r2:.1f}").replace("@HI@", hi).replace("@LO@", lo), tag)
    if not (os.path.exists(hi) and os.path.exists(lo)):
        return ("fail", r2)
    dh, dl = np.loadtxt(hi), np.loadtxt(lo)
    d = np.vstack([dl[::-1], dh[1:]])               # -40 ... 25 ... 125
    T = d[:, 0]
    vref, va, vb, vg, vtail, vn1, ivdd = (d[:, k] for k in (1, 3, 5, 7, 9, 11, 13))
    v27 = np.interp(27, T, vref)
    i27 = (v27 - np.interp(27, T, va)) / r2
    if i27 < 1e-6:
        return ("zero", r2)
    tc = (vref.max() - vref.min()) / (v27 * (T.max() - T.min())) * 1e6
    iq = -np.interp(27, T, ivdd)
    vab = (np.interp(27, T, va) - np.interp(27, T, vb)) * 1e3
    marg = dict(tail=vtail.min(), m2=(vg - vtail).min(),
                m1=(vn1 - vtail).min(), pass_=VDD - vref.max())
    return ("ok", (r2, tc, v27, i27, iq, T[np.argmax(vref)], vab, marg), (T, vref))

workers = 3
print(f"\nRunning {len(factors)} R2 points on {workers} worker(s), {STEP} C steps ...")
with ThreadPoolExecutor(max_workers=workers) as ex:
    results = list(ex.map(one, factors))

rows, curves = [], {}
for res in results:
    if res[0] == "ok":
        rows.append(res[1]); curves[res[1][0]] = res[2]
    else:
        print(f"{res[0]} at R2={res[1]/1e3:.2f}k, skipped")
if not rows:
    print("No valid runs. Send me the first lines of build/bgr_nmos_*.spice output.")
    raise SystemExit(1)

print(f"\n{'R2 (kOhm)':>10} {'TC ppm/C':>9} {'Vref27 (V)':>11} {'Ibr (uA)':>9} {'Iq (uA)':>8} {'a-b (mV)':>9} {'T@peak':>7}")
for r2, tc, v27, i27, iq, tpk, vab, _ in rows:
    print(f"{r2/1e3:10.2f} {tc:9.1f} {v27:11.4f} {i27*1e6:9.2f} {iq*1e6:8.2f} {vab:9.2f} {tpk:7.0f}")

best = min(rows, key=lambda x: x[1])
mg = best[7]
print(f"\nBest R2 = {best[0]/1e3:.2f} kOhm -> TC = {best[1]:.1f} ppm/C, "
      f"Vref(27C) = {best[2]:.4f} V, Iq = {best[4]*1e6:.1f} uA")
print("Headroom margins over temperature (want > ~0.10 V; pass > ~0.15 V):")
print(f"  tail source Vds   = {mg['tail']:.3f} V")
print(f"  M2 Vds (g-tail)   = {mg['m2']:.3f} V")
print(f"  M1 Vds (n1-tail)  = {mg['m1']:.3f} V")
print(f"  pass device Vsd   = {mg['pass_']:.3f} V")

plt.figure(figsize=(7, 4.5))
for r2, (T, v) in curves.items():
    plt.plot(T, v * 1e3, lw=2 if r2 == best[0] else 0.7,
             label=f"{r2/1e3:.2f}k" if r2 == best[0] else None)
plt.xlabel("Temperature (C)"); plt.ylabel("Vref (mV)")
plt.title("BGR with NMOS-input OTA: Vref vs T"); plt.grid(alpha=.3)
plt.legend(title="best R2"); plt.tight_layout()
plt.savefig("results/step4c_nmos_sweep.png", dpi=150)

des.update(R2_nmos=best[0], tc_nmos_ppm=best[1], vref_nmos=best[2], iq_nmos=best[4])
json.dump(des, open("results/design.json", "w"), indent=2)
