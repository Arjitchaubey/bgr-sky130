import os, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

des = json.load(open("results/design.json"))
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
R2, VG0 = des["R2_nmos"], des["vref_nmos"]

Q  = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")
STARTUP = (f"XMsp sn vref vdd vdd {PF} W=1 L=8\n"
           f"XMsn sn vref 0 0 {NF} W=2 L=1\n"
           f"XMst g sn 0 0 {NF} W=1 L=1\n")

def netlist(startup, temp_line, ns, tight, hi, lo):
    g, n1, tail = ns
    temp_txt = ".temp 27" if temp_line else ""
    tol_txt = ".options reltol=1e-6 vntol=1e-9 abstol=1e-14" if tight else ""
    su_txt = STARTUP if startup else ""
    ns_txt = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
              f"v(g)={g} v(n1)={n1} v(tail)={tail} v(vbn)=0.55"
              + (" v(sn)=0" if startup else ""))
    return f"""* set-up isolation test
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
{temp_txt}
{tol_txt}
VDD vdd 0 1.8
XQ1 0 0 a {Q}
{q2}
R2a vref a {R2:.1f}
R2b vref b {R2:.1f}
R1  b e2 {R1}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 g  a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 g  n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
IB vdd vbn 1u
C1 g vdd 2p
CL vref 0 1p
{su_txt}{ns_txt}
.control
dc temp 25 125 5
wrdata {hi} vref
dc temp 25 -40 -5
wrdata {lo} vref
quit
.endc
.end
"""

# label, startup, .temp line, nodesets (g, n1, tail), tight tolerances
VARIANTS = [
    ("A: as in Step 5b",              True,  True,  (0.72, 0.73, 0.20), False),
    ("B: A + tight tolerances",       True,  True,  (0.72, 0.73, 0.20), True),
    ("C: B + Step 4c nodesets",       True,  True,  (0.68, 0.68, 0.17), True),
    ("D: Step 4c-style netlist",      False, False, (0.68, 0.68, 0.17), False),
    ("E: D + tight tolerances",       False, False, (0.68, 0.68, 0.17), True),
]

os.makedirs("build", exist_ok=True)

def go(i):
    label, su, tl, ns, tight = VARIANTS[i]
    hi, lo = f"build/s5c_{i}_hi.csv", f"build/s5c_{i}_lo.csv"
    for p in (hi, lo):
        if os.path.exists(p):
            os.remove(p)
    path = f"build/s5c_{i}.spice"
    open(path, "w").write(os.path.expandvars(netlist(su, tl, ns, tight, hi, lo)))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return i, hi, lo, r.stdout[-600:]

print("Running 5 sweeps, 3 at a time ...")
with ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(go, range(len(VARIANTS))))

print(f"\n{'variant':<28} {'TC ppm/C':>8} {'V(-40C)':>9} {'V(25C)':>9} {'V(125C)':>9} {'Vmax':>9} {'T@max':>6}")
for i, hi, lo, tail in res:
    label = VARIANTS[i][0]
    if not (os.path.exists(hi) and os.path.exists(lo)):
        print(f"{label:<28} FAILED\n{tail}")
        continue
    dl, dh = np.loadtxt(lo), np.loadtxt(hi)
    d = np.vstack([dl[::-1], dh[1:]])
    T, v = d[:, 0], d[:, 1]
    v27 = np.interp(27, T, v)
    tc = (v.max() - v.min()) / (v27 * (T.max() - T.min())) * 1e6
    print(f"{label:<28} {tc:8.1f} {v[0]:9.5f} {np.interp(25, T, v):9.5f} {v[-1]:9.5f} "
          f"{v.max():9.5f} {T[np.argmax(v)]:6.0f}")

f = "build/s5b_zout.csv"
if os.path.exists(f):
    d = np.loadtxt(f)
    fr, z = d[:, 0], d[:, 1]
    print("\nZout(f) from the Step 5b run:")
    for x in (1e2, 1e3, 1e4, 1e5, 3e5, 1e6, 1.78e6, 3e6, 1e7, 1e8):
        zz = np.exp(np.interp(np.log(x), np.log(fr), np.log(z)))
        print(f"  {x:10.3g} Hz  {zz/1e3:8.2f} kOhm")
