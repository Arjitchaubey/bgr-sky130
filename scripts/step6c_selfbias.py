import os, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

des = json.load(open("results/design.json"))
r6 = json.load(open("results/step6a_resistors.json"))
s6b = json.load(open("results/step6b_resistors.json"))
M = "sky130_fd_pr__res_xhigh_po_2p85"
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
RHO0, VG0 = s6b[M]["ratio"], s6b[M]["vref27"]
N1, N2, WORKERS = 7, 65, 4

Q  = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")
p = r6[M]
SLOPE = p["sheet"] / p["W"]

def lengths(rho):
    return p["L_unit"], (rho * R1 / N2 - p["rend"]) / SLOPE

def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}" for k in range(n))

def netlist(rho, vddline, nodeset, analysis, temp=27):
    L1, L2 = lengths(rho)
    ns = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
          "v(g)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0") if nodeset else ""
    return f"""* BGR step 6c: xhigh_po resistors, self-biased OTA, startup
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp {temp}
{vddline}
XQ1 0 0 a {Q}
{q2}
{chain("r2a", "vref", "a", N2, L2)}
{chain("r2b", "vref", "b", N2, L2)}
{chain("r1", "b", "e2", N1, L1)}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 g  a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 g  n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
XMBP vbn g vdd vdd {PF} W=10 L=2
C1 g vdd 2p
CL vref 0 1p
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn 0 0 {NF} W=1 L=1
{ns}
.control
{analysis}
quit
.endc
.end
"""

def sim(name, text, outs):
    for o in outs:
        if os.path.exists(o):
            os.remove(o)
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return all(os.path.exists(o) for o in outs), r.stdout[-700:]

os.makedirs("build", exist_ok=True)
VDC = "VDD vdd 0 DC 1.8"

# ---------- 0. unit-resistor check ----------
L1, L2 = lengths(RHO0)
o0 = "build/s6c_rcheck.csv"
rtext = f"""* unit resistor check
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
Vs1 n1 0 DC 0.1
X1 n1 0 0 {M} L={L1:.4f}
Vs2 n2 0 DC 0.1
X2 n2 0 0 {M} L={L2:.4f}
.control
dc temp 27 28 1
wrdata {o0} i(Vs1) i(Vs2)
quit
.endc
.end
"""
ok, tail = sim("s6c_rcheck", rtext, [o0])
print("---- Unit resistor check (27 C, direct simulation) ----")
if ok:
    d = np.loadtxt(o0)
    ra, rb = 0.1 / abs(d[0, 1]), 0.1 / abs(d[0, 3])
    print(f"R1 unit (L = {L1:.3f} um): {ra:.1f} Ohm   target {R1/N1:.1f}   ({(ra/(R1/N1)-1)*100:+.2f}%)")
    print(f"R2 unit (L = {L2:.3f} um): {rb:.1f} Ohm   target {RHO0*R1/N2:.1f}   ({(rb/(RHO0*R1/N2)-1)*100:+.2f}%)")
    print(f"Actual ratio R2/R1 = {N2*rb/(N1*ra):.4f}   (nominal {RHO0:.4f})")
else:
    print("check failed:", tail)

# ---------- 1. ratio sweep with the self-biased OTA ----------
def job_sweep(rho):
    tag = f"s6c_sw_{rho:.4f}"
    hi, lo = f"build/{tag}_hi.csv", f"build/{tag}_lo.csv"
    an = (f"dc temp 25 125 5\nwrdata {hi} vref i(vdd)\n"
          f"dc temp 25 -40 -5\nwrdata {lo} vref i(vdd)")
    ok, tail = sim(tag, netlist(rho, VDC, True, an), [hi, lo])
    if not ok:
        return rho, None, tail
    dl, dh = np.loadtxt(lo), np.loadtxt(hi)
    d = np.vstack([dl[::-1], dh[1:]])
    T, v, iv = d[:, 0], d[:, 1], -d[:, 3] * 1e6
    v27 = float(np.interp(27, T, v))
    if v27 < 1.0:
        return rho, None, f"zero-current state (Vref(27C) = {v27:.3f} V)"
    tc = (v.max() - v.min()) / (v27 * (T.max() - T.min())) * 1e6
    return rho, dict(tc=float(tc), v27=v27, iq27=float(np.interp(27, T, iv)),
                     iq125=float(np.interp(125, T, iv)), tpk=float(T[np.argmax(v)]), T=T, v=v), ""

rhos = RHO0 + np.array([-0.30, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.30])
print(f"\nRatio sweep: {len(rhos)} simulations, {WORKERS} at a time ...")
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    sw = list(ex.map(job_sweep, rhos))
rows = []
for rho, r, tail in sw:
    if r is None:
        print(f"FAILED at ratio {rho:.3f}: {tail}")
    else:
        rows.append((rho, r))
if not rows:
    raise SystemExit("no valid sweep points")

print(f"\n{'R2/R1':>7} {'TC ppm/C':>9} {'Vref27 (V)':>11} {'Iq27 (uA)':>10} {'Iq125 (uA)':>11} {'T@peak':>7}")
for rho, r in rows:
    print(f"{rho:7.3f} {r['tc']:9.1f} {r['v27']:11.4f} {r['iq27']:10.2f} {r['iq125']:11.2f} {r['tpk']:7.0f}")
best_rho, best = min(rows, key=lambda x: x[1]["tc"])
if best_rho in (rows[0][0], rows[-1][0]):
    print("WARNING: best ratio is at the edge of the sweep range")
Lb1, Lb2 = lengths(best_rho)
print(f"\nBest ratio {best_rho:.3f}: TC = {best['tc']:.1f} ppm/C, Vref(27C) = {best['v27']:.4f} V, "
      f"Iq = {best['iq27']:.2f} uA (27C) / {best['iq125']:.2f} uA (125C)")
print(f"Unit lengths: R1 {Lb1:.3f} um x {N1}, R2 {Lb2:.3f} um x {N2} (x2)")
print(f"For reference, ideal bias at ratio 9.10: 15.3 ppm/C, Iq 13.18 uA, Vref 1.2187 V")

# ---------- 2. startup ramps and supply sweeps at the best ratio ----------
def job_ramp(a):
    kind, T = a
    out = f"build/s6c_{kind}_{T}.csv"
    if kind == "fast":
        vdd, an = "VDD vdd 0 PWL(0 0 1u 1.8)", f"tran 20n 100u\nwrdata {out} vref"
    else:
        vdd, an = "VDD vdd 0 PWL(0 0 1m 1.8)", f"tran 2u 1.5m\nwrdata {out} vref"
    ok, tail = sim(f"s6c_{kind}_{T}", netlist(best_rho, vdd, False, an, temp=T), [out])
    return kind, T, ok, tail

def job_sup(T):
    lo, hi = f"build/s6c_sup_{T}_lo.csv", f"build/s6c_sup_{T}_hi.csv"
    an = (f"dc VDD 1.8 1.0 -0.02\nwrdata {lo} vref i(vdd)\n"
          f"dc VDD 1.8 2.0 0.02\nwrdata {hi} vref i(vdd)")
    ok, tail = sim(f"s6c_sup_{T}", netlist(best_rho, VDC, True, an, temp=T), [lo, hi])
    return T, ok, tail

print("\nRunning startup ramps and supply sweeps ...")
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    f1 = [ex.submit(job_ramp, (k, T)) for k in ("fast", "slow") for T in (-40, 125)]
    f2 = [ex.submit(job_sup, T) for T in (-40, 27, 125)]
    ramps = [f.result() for f in f1]
    sups = [f.result() for f in f2]

print("\n---- Startup (no nodesets), compared with the temperature-sweep value at the same T ----")
for kind, T, ok, tail in ramps:
    lab = "1 us ramp" if kind == "fast" else "1 ms ramp"
    if not ok:
        print(f"{lab:>9}, {T:>4}C: simulation FAILED\n{tail}"); continue
    d = np.loadtxt(f"build/s6c_{kind}_{T}.csv")
    t, vr = d[:, 0] * 1e6, d[:, 1]
    final = vr[-1]
    ref = float(np.interp(T, best["T"], best["v"]))
    if final < 1.0:
        print(f"{lab:>9}, {T:>4}C: FAILED TO START (Vref = {final:.4f} V)"); continue
    bad = np.where(np.abs(vr - final) > 0.01 * final)[0]
    ts = t[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(t) else t[0]
    print(f"{lab:>9}, {T:>4}C: final {final:.4f} V (sweep {ref:.4f}), 1% settle {ts:.1f} us, "
          f"peak {vr.max():.3f} V, overshoot {(vr.max()-final)*1e3:.0f} mV")

print("\n---- Supply sweep ----")
plt.figure(figsize=(7, 4.5))
sup_summary = {}
for T, ok, tail in sups:
    if not ok:
        print(f"T={T:>4}C: FAILED\n{tail}"); continue
    dl = np.loadtxt(f"build/s6c_sup_{T}_lo.csv"); dh = np.loadtxt(f"build/s6c_sup_{T}_hi.csv")
    d = np.vstack([dl[::-1], dh[1:]])
    vdd, vr = d[:, 0], d[:, 1]
    v18 = float(np.interp(1.8, vdd, vr))
    if v18 < 1.0:
        print(f"T={T:>4}C: zero-current state at 1.8 V"); continue
    lr = (np.interp(1.98, vdd, vr) - np.interp(1.62, vdd, vr)) / 0.36 * 1e3
    vmin = None
    for vv, x in zip(vdd[::-1], vr[::-1]):
        if vv > 1.8:
            continue
        if x >= 0.995 * v18:
            vmin = vv
        else:
            break
    print(f"T={T:>4}C: Vref(1.8V) = {v18:.4f} V, line reg (1.62-1.98 V) = {lr:.2f} mV/V, min VDD (0.5%) = {vmin:.2f} V")
    sup_summary[str(T)] = dict(vref=v18, line_reg_mV_per_V=float(lr), vmin=vmin)
    plt.plot(vdd, vr, label=f"{T} C")
plt.xlabel("VDD (V)"); plt.ylabel("Vref (V)"); plt.grid(alpha=.3); plt.legend()
plt.title("Vref vs supply, self-biased"); plt.tight_layout()
plt.savefig("results/step6c_supply.png", dpi=150)

plt.figure(figsize=(7, 4.5))
plt.plot(best["T"], (best["v"] - best["v27"]) * 1e3)
plt.xlabel("Temperature (C)"); plt.ylabel("Vref - Vref(27C) (mV)"); plt.grid(alpha=.3)
plt.title(f"Self-biased BGR, ratio {best_rho:.3f}, {best['tc']:.1f} ppm/C")
plt.tight_layout(); plt.savefig("results/step6c_temp.png", dpi=150)

json.dump(dict(model=M, ratio=best_rho, tc=best["tc"], vref27=best["v27"], iq27=best["iq27"],
               iq125=best["iq125"], L_R1_unit=Lb1, L_R2_unit=Lb2, n_R1=N1, n_R2=N2,
               supply=sup_summary), open("results/step6c_summary.json", "w"), indent=2)
