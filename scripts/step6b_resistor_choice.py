import os, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

des = json.load(open("results/design.json"))
r6 = json.load(open("results/step6a_resistors.json"))
R1, VBE0, I0, VG0 = des["R1"], des["vbe0"], des["I"], des["vref_nmos"]

MODELS = ["sky130_fd_pr__res_high_po_1p41", "sky130_fd_pr__res_xhigh_po_2p85"]
SHORT = {MODELS[0]: "hp141", MODELS[1]: "xh285"}
N1, N2 = 7, 65
WORKERS = 4

Q  = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")

def chain(tag, na, nb, n, model, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {model} L={L:.4f}" for k in range(n))

def lengths(m, rho):
    p = r6[m]
    slope = p["sheet"] / p["W"]                    # ohm per um at 27C
    return p["L_unit"], (rho * R1 / N2 - p["rend"]) / slope

def netlist(m, rho, hi, lo):
    L1, L2 = lengths(m, rho)
    return f"""* BGR with series-unit poly resistors: {m}, R2/R1 = {rho:.4f}
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
VDD vdd 0 DC 1.8
XQ1 0 0 a {Q}
{q2}
{chain("r2a", "vref", "a", N2, m, L2)}
{chain("r2b", "vref", "b", N2, m, L2)}
{chain("r1", "b", "e2", N1, m, L1)}
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
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn 0 0 {NF} W=1 L=1
.nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} v(g)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0
.control
dc temp 25 125 5
wrdata {hi} vref i(vdd)
dc temp 25 -40 -5
wrdata {lo} vref i(vdd)
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)

def job(args):
    m, rho = args
    tag = f"{SHORT[m]}_{rho:.4f}"
    hi, lo = f"build/s6b_{tag}_hi.csv", f"build/s6b_{tag}_lo.csv"
    for p in (hi, lo):
        if os.path.exists(p):
            os.remove(p)
    path = f"build/s6b_{tag}.spice"
    open(path, "w").write(os.path.expandvars(netlist(m, rho, hi, lo)))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    if not (os.path.exists(hi) and os.path.exists(lo)):
        return m, rho, None, r.stdout[-700:]
    dl, dh = np.loadtxt(lo), np.loadtxt(hi)
    d = np.vstack([dl[::-1], dh[1:]])
    T, v, iv = d[:, 0], d[:, 1], -d[:, 3]
    v27 = float(np.interp(27, T, v))
    if v27 < 1.0:
        return m, rho, None, f"zero-current state (Vref(27C) = {v27:.3f} V)"
    tc = (v.max() - v.min()) / (v27 * (T.max() - T.min())) * 1e6
    return m, rho, dict(tc=float(tc), v27=v27, iq=float(np.interp(27, T, iv)) * 1e6,
                        tpk=float(T[np.argmax(v)]), T=T, v=v), ""

def batch(pairs):
    rows, shown = [], set()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for m, rho, res, tail in ex.map(job, pairs):
            if res is None:
                if m not in shown:
                    shown.add(m)
                    print(f"\nFAILED {SHORT[m]} at ratio {rho:.3f}:\n{tail}\n")
                continue
            rows.append((m, rho, res))
    return rows

data = {m: [] for m in MODELS}
coarse = np.linspace(8.65, 9.85, 9)
print(f"Coarse sweep: {len(MODELS) * len(coarse)} simulations, {WORKERS} at a time (several minutes) ...")
for m, rho, res in batch([(m, r) for m in MODELS for r in coarse]):
    data[m].append((rho, res))

pairs = []
for m in MODELS:
    if not data[m]:
        continue
    rows = sorted(data[m], key=lambda x: x[0])
    k = min(range(len(rows)), key=lambda i: rows[i][1]["tc"])
    if k in (0, len(rows) - 1):
        print(f"WARNING {SHORT[m]}: best coarse ratio {rows[k][0]:.2f} is at the edge of the range")
    for rho in rows[k][0] + np.linspace(-0.15, 0.15, 7):
        if all(abs(rho - r0) > 1e-6 for r0, _ in data[m]):
            pairs.append((m, float(rho)))

print(f"\nFine sweep: {len(pairs)} simulations ...")
for m, rho, res in batch(pairs):
    data[m].append((rho, res))

summary, best_curves = {}, {}
for m in MODELS:
    rows = sorted(data[m], key=lambda x: x[0])
    if not rows:
        continue
    print(f"\n==== {m} ====")
    print(f"{'R2/R1':>7} {'TC ppm/C':>9} {'Vref27 (V)':>11} {'Iq (uA)':>8} {'T@peak':>7}")
    for rho, r in rows:
        print(f"{rho:7.3f} {r['tc']:9.1f} {r['v27']:11.4f} {r['iq']:8.2f} {r['tpk']:7.0f}")
    rho, b = min(rows, key=lambda x: x[1]["tc"])
    L1, L2 = lengths(m, rho)
    summary[m] = dict(ratio=rho, tc=b["tc"], vref27=b["v27"], iq_uA=b["iq"],
                      L_unit_R1=L1, L_unit_R2=L2, n_R1=N1, n_R2=N2,
                      area_137_body=r6[m]["area_137"])
    best_curves[m] = (b["T"], b["v"], b["v27"])

print("\n==== Summary (tt, 1.8 V, ideal bias, 5 C steps) ====")
print(f"{'resistor':<34} {'best ratio':>10} {'TC ppm/C':>9} {'Vref27':>8} {'Iq uA':>6} {'area body um2':>14}")
for m, s in summary.items():
    print(f"{m:<34} {s['ratio']:10.3f} {s['tc']:9.1f} {s['vref27']:8.4f} {s['iq_uA']:6.2f} {s['area_137_body']:14.0f}")

plt.figure(figsize=(7, 4.5))
for m, (T, v, v27) in best_curves.items():
    plt.plot(T, (v - v27) * 1e3, label=f"{SHORT[m]}  {summary[m]['tc']:.1f} ppm/C")
plt.xlabel("Temperature (C)"); plt.ylabel("Vref - Vref(27C) (mV)"); plt.grid(alpha=.3)
plt.title("BGR with real poly resistors, best ratio each"); plt.legend(); plt.tight_layout()
plt.savefig("results/step6b_resistors.png", dpi=150)
json.dump(summary, open("results/step6b_resistors.json", "w"), indent=2)
