import os, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np

des = json.load(open("results/design.json"))
s6c = json.load(open("results/step6c_summary.json"))
M = s6c["model"]
L1, L2 = s6c["L_R1_unit"], s6c["L_R2_unit"]
N1, N2 = s6c["n_R1"], s6c["n_R2"]
WORKERS = 4
C1 = "20p"
TEMPS = (-40, 27, 125)
RAMPS = [("1u", "100u", "20n"), ("10u", "200u", "50n"), ("1m", "2m", "2u")]

Q  = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")

def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}" for k in range(n))

def head(T):
    return (".lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt\n"
            ".options reltol=1e-6 vntol=1e-9 abstol=1e-14\n"
            f".temp {T}")

SP = f"XMsp sn vref vdd vdd {PF} W=1 L=8\n"
STARTUPS = {
    "A": SP + f"XMsn sn vref 0 0 {NF} W=2 L=1\nXMst g sn 0 0 {NF} W=1 L=1\n",
    "B": SP + f"XMsn sn vref 0 0 {NF} W=2 L=1\nXMst g sn 0 0 {NF} W=1 L=4\n",
    "C": SP + (f"XMsn sn vref 0 0 {NF} W=2 L=1\nXMst g sn s2 0 {NF} W=1 L=1\n"
               f"XMd s2 s2 0 0 {NF} W=2 L=1\n"),
    "D": SP + f"XMsn sn vref 0 0 {NF} W=8 L=1\nXMst g sn 0 0 {NF} W=1 L=1\n",
}
LABELS = {"A": "A baseline", "B": "B Mst L=4", "C": "C diode in Mst source", "D": "D Msn W=8"}
STARTUPS = {"C": STARTUPS["C"]}

def ramp_net(T, k, ramp, tstop, step, out):
    return f"""* startup variant {k}, ramp {ramp}, C1 = {C1}
{head(T)}
VDD vdd 0 PWL(0 0 {ramp} 1.8)
XQ1 0 0 a {Q}
{q2}
{chain("r2a", "vref", "a", N2, L2)}
{chain("r2b", "vref", "b", N2, L2)}
{chain("r1", "b", "e2", N1, L1)}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 g a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 g n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
XMBP vbn g vdd vdd {PF} W=10 L=2
C1 g vdd {C1}
{STARTUPS[k]}CL vref 0 1p
.control
tran {step} {tstop}
wrdata {out} vdd vref g sn
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
jobs = []
for k in STARTUPS:
    for T in TEMPS:
        for ramp, tstop, step in RAMPS:
            o = f"build/s6i_{k}_{T}_{ramp}.csv"
            jobs.append((f"s6i_{k}_{T}_{ramp}", ramp_net(T, k, ramp, tstop, step, o), [o]))

def go(job):
    name, text, outs = job
    for o in outs:
        if os.path.exists(o):
            os.remove(o)
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return name, all(os.path.exists(o) for o in outs), r.stdout[-600:]

print(f"Running {len(jobs)} simulations, {WORKERS} at a time (about 7-9 min, estimate) ...")
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    res = list(ex.map(go, jobs))
for n, ok, tail in res:
    if not ok:
        print(f"\nFAILED {n}. Last ngspice output:\n{tail}")
summary = []
print(f"\n---- Startup variants, C1 = {C1}, no nodesets ----")
print("Reference, variant A at -40 C from 6g: +572.7 mV (1 us), +246.7 mV (10 us), 0.0 mV (1 ms)")
print(f"{'variant':>22} {'T (C)':>6} {'ramp':>5} {'final (V)':>10} {'peak (V)':>9} "
      f"{'over (mV)':>10} {'g min (V)':>10} {'settle (us)':>12}")
for k in STARTUPS:
    for T in TEMPS:
        for ramp, tstop, step in RAMPS:
            f = f"build/s6i_{k}_{T}_{ramp}.csv"
            lab = LABELS[k]
            if not os.path.exists(f):
                print(f"{lab:>22} {T:>6} {ramp:>5}   missing"); continue
            d = np.loadtxt(f)
            t, vr, g = d[:, 0] * 1e6, d[:, 3], d[:, 5]
            final = vr[-1]
            if final < 1.0:
                print(f"{lab:>22} {T:>6} {ramp:>5}   FAILED TO START (Vref = {final:.4f} V, "
                      f"g = {g[-1]:.3f} V, sn = {d[-1, 7]:.3f} V)")
                summary.append(dict(variant=k, T=T, ramp=ramp, started=False, final=float(final)))
                continue
            bad = np.where(np.abs(vr - final) > 0.01 * final)[0]
            ts = t[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(t) else t[0]
            flag = "" if abs(vr[-1] - vr[int(0.9 * len(vr))]) < 1e-3 else "  (not settled by end of run)"
            print(f"{lab:>22} {T:>6} {ramp:>5} {final:10.4f} {vr.max():9.4f} "
                  f"{(vr.max()-final)*1e3:10.1f} {g.min():10.3f} {ts:12.1f}{flag}")
            summary.append(dict(variant=k, T=T, ramp=ramp, started=True, final=float(final),
                                peak=float(vr.max()), overshoot_mV=float((vr.max() - final) * 1e3),
                                g_min=float(g.min()), settle_us=float(ts)))

json.dump(summary, open("results/step6i_summary.json", "w"), indent=2)
