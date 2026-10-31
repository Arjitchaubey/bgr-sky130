import os, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

des = json.load(open("results/design.json"))
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
R2, VG0 = des["R2_nmos"], des["vref_nmos"]

Q  = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")

STARTUP = f"""* startup: while vref is low, pull the pass-device gate down
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn 0 0 {NF} W=1 L=1
"""
NO_STARTUP = "Rdum sn 0 1e12\n"

def netlist(vddline, nodeset, startup, temp, analysis):
    ns = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
          "v(g)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0") if nodeset else ""
    return f"""* BGR step 5a
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.temp {temp}
{vddline}
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
{STARTUP if startup else NO_STARTUP}{ns}
.control
{analysis}
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
jobs = {}

def add(name, vddline, nodeset, startup, temp, analysis, outs):
    for o in outs:
        if os.path.exists(o):
            os.remove(o)
    jobs[name] = (netlist(vddline, nodeset, startup, temp, analysis), outs)

VDC, VRAMP = "VDD vdd 0 1.8", "VDD vdd 0 PWL(0 0 100u 1.8)"

for tag, su in (("on", True), ("off", False)):
    o = f"build/s5_tran_{tag}.csv"
    add(f"s5_tran_{tag}", VRAMP, False, su, 27,
        f"tran 0.5u 400u\nwrdata {o} vref g sn vdd i(vdd)", [o])

for T in (-40, 27, 125):
    lo, hi = f"build/s5_sup_{T}_lo.csv", f"build/s5_sup_{T}_hi.csv"
    add(f"s5_sup_{T}", VDC, True, True, T,
        f"dc VDD 1.8 1.0 -0.02\nwrdata {lo} vref i(vdd)\n"
        f"dc VDD 1.8 2.0 0.02\nwrdata {hi} vref i(vdd)", [lo, hi])

tlo, thi = "build/s5_temp_lo.csv", "build/s5_temp_hi.csv"
add("s5_temp", VDC, True, True, 27,
    f"dc temp 25 125 5\nwrdata {thi} vref a b i(vdd)\n"
    f"dc temp 25 -40 -5\nwrdata {tlo} vref a b i(vdd)", [tlo, thi])

def go(name):
    text, outs = jobs[name]
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return name, all(os.path.exists(o) for o in outs), r.stdout[-800:]

print(f"Running {len(jobs)} simulations, 4 at a time ...")
with ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(go, list(jobs)))
for n, ok, tail in res:
    if not ok:
        print(f"\nFAILED {n}. Last ngspice output:\n{tail}")

# ---------------- startup transient ----------------
print("\n---- Startup transient (VDD ramps 0 -> 1.8 V in 100 us, no nodesets) ----")
plt.figure(figsize=(7, 4.5))
for tag, lab in (("on", "with startup"), ("off", "no startup")):
    f = f"build/s5_tran_{tag}.csv"
    if not os.path.exists(f):
        continue
    d = np.loadtxt(f)
    t, vr, vdd = d[:, 0] * 1e6, d[:, 1], d[:, 7]
    final = vr[-1]
    line = f"{lab:>13}: Vref(400us) = {final:.4f} V, peak = {vr.max():.4f} V"
    if final > 1.0:
        out = np.where(abs(vr - final) > 0.01 * final)[0]
        ts = t[out[-1] + 1] if len(out) and out[-1] + 1 < len(t) else t[0]
        line += f", settles to 1% at {ts:.0f} us, overshoot = {(vr.max()-final)*1e3:.1f} mV"
    print(line)
    plt.plot(t, vr, label=lab)
plt.plot(t, vdd, "--", lw=0.8, label="VDD")
plt.xlabel("Time (us)"); plt.ylabel("Voltage (V)"); plt.grid(alpha=.3)
plt.title("BGR startup transient"); plt.legend(); plt.tight_layout()
plt.savefig("results/step5a_startup.png", dpi=150)

# ---------------- supply sweep ----------------
print("\n---- Supply sweep (startup circuit included) ----")
plt.figure(figsize=(7, 4.5))
for T in (-40, 27, 125):
    lo, hi = f"build/s5_sup_{T}_lo.csv", f"build/s5_sup_{T}_hi.csv"
    if not (os.path.exists(lo) and os.path.exists(hi)):
        continue
    dl, dh = np.loadtxt(lo), np.loadtxt(hi)
    d = np.vstack([dl[::-1], dh[1:]])
    vdd, vr = d[:, 0], d[:, 1]
    v18 = np.interp(1.8, vdd, vr)
    if v18 < 1.0:
        print(f"T={T:>4}C: stuck in zero-current state at 1.8 V"); continue
    lr = (np.interp(1.98, vdd, vr) - np.interp(1.62, vdd, vr)) / 0.36 * 1e3
    vmin = None
    for vv, x in zip(vdd[::-1], vr[::-1]):
        if vv > 1.8:
            continue
        if x >= 0.995 * v18:
            vmin = vv
        else:
            break
    print(f"T={T:>4}C: Vref(1.8V) = {v18:.4f} V, line reg (1.62-1.98 V) = {lr:.2f} mV/V, "
          f"min VDD for Vref within 0.5% = {vmin:.2f} V")
    plt.plot(vdd, vr, label=f"{T} C")
plt.xlabel("VDD (V)"); plt.ylabel("Vref (V)"); plt.grid(alpha=.3)
plt.title("Vref vs supply"); plt.legend(); plt.tight_layout()
plt.savefig("results/step5a_supply.png", dpi=150)

# ---------------- temperature sweep with startup ----------------
print("\n---- Temperature sweep with startup circuit ----")
if os.path.exists(tlo) and os.path.exists(thi):
    dl, dh = np.loadtxt(tlo), np.loadtxt(thi)
    d = np.vstack([dl[::-1], dh[1:]])
    T, vr, va, ivdd = d[:, 0], d[:, 1], d[:, 3], d[:, 7]
    v27 = np.interp(27, T, vr)
    tc = (vr.max() - vr.min()) / (v27 * (T.max() - T.min())) * 1e6
    iq = -np.interp(27, T, ivdd)
    ibr = (v27 - np.interp(27, T, va)) / R2
    print(f"TC = {tc:.1f} ppm/C (without startup: {des['tc_nmos_ppm']:.1f})")
    print(f"Iq(27C) = {iq*1e6:.2f} uA (without startup: {des['iq_nmos']*1e6:.2f}), "
          f"Iq(125C) = {-np.interp(125, T, ivdd)*1e6:.2f} uA, Ibr = {ibr*1e6:.2f} uA")
    des.update(tc_startup_ppm=tc, iq_startup=iq)
    json.dump(des, open("results/design.json", "w"), indent=2)
