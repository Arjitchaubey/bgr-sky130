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

STARTUP = f"""XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn 0 0 {NF} W=1 L=1
"""
NO_STARTUP = "Rdum sn 0 1e12\n"

def netlist(vddline, nodeset, startup, temp, analysis, extra=""):
    ns = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
          "v(g)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0") if nodeset else ""
    return f"""* BGR step 5b
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
{extra}
{STARTUP if startup else NO_STARTUP}{ns}
.control
{analysis}
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
jobs = {}

def add(name, vddline, nodeset, startup, temp, analysis, outs, extra=""):
    for o in outs:
        if os.path.exists(o):
            os.remove(o)
    jobs[name] = (netlist(vddline, nodeset, startup, temp, analysis, extra), outs)

VDC = "VDD vdd 0 DC 1.8"
TEMPS = (-40, 27, 125)

for T in TEMPS:
    o = f"build/s5b_psrr_{T}.csv"
    add(f"s5b_psrr_{T}", "VDD vdd 0 DC 1.8 AC 1", True, True, T,
        f"ac dec 20 1 1e8\nwrdata {o} vdb(vref)", [o])

o = "build/s5b_zout.csv"
add("s5b_zout", VDC, True, True, 27,
    f"ac dec 20 1 1e8\nwrdata {o} vm(vref)", [o], extra="ITEST 0 vref DC 0 AC 1")

for T in (-40, 125):
    o = f"build/s5b_fast_{T}.csv"
    add(f"s5b_fast_{T}", "VDD vdd 0 PWL(0 0 1u 1.8)", False, True, T,
        f"tran 20n 100u\nwrdata {o} vref vdd", [o])
    o = f"build/s5b_slow_{T}.csv"
    add(f"s5b_slow_{T}", "VDD vdd 0 PWL(0 0 1m 1.8)", False, True, T,
        f"tran 2u 1.5m\nwrdata {o} vref vdd", [o])

o = "build/s5b_vstep.csv"
add("s5b_vstep", "VDD vdd 0 PWL(0 1.8 10u 1.8 10.1u 1.98 30u 1.98 30.1u 1.62 50u 1.62 50.1u 1.8 70u 1.8)",
    True, True, 27, f"tran 20n 70u\nwrdata {o} vref vdd", [o])

for tag, su in (("on", True), ("off", False)):
    lo, hi = f"build/s5b_temp_{tag}_lo.csv", f"build/s5b_temp_{tag}_hi.csv"
    add(f"s5b_temp_{tag}", VDC, True, su, 27,
        f"dc temp 25 125 5\nwrdata {hi} vref i(vdd)\ndc temp 25 -40 -5\nwrdata {lo} vref i(vdd)", [lo, hi])

def go(name):
    text, outs = jobs[name]
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return name, all(os.path.exists(o) for o in outs), r.stdout[-800:]

print(f"Running {len(jobs)} simulations, 4 at a time (about 4-5 min) ...")
with ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(go, list(jobs)))
for n, ok, tail in res:
    if not ok:
        print(f"\nFAILED {n}. Last ngspice output:\n{tail}")

summary = {}

# ---------------- PSRR ----------------
print("\n---- PSRR (VDD ripple -> Vref), dB, higher is better ----")
plt.figure(figsize=(7, 4.5))
for T in TEMPS:
    f = f"build/s5b_psrr_{T}.csv"
    if not os.path.exists(f):
        continue
    d = np.loadtxt(f)
    fr, p = d[:, 0], -d[:, 1]
    pts = {x: float(np.interp(np.log10(x), np.log10(fr), p)) for x in (10, 1e3, 1e5, 1e6, 1e7)}
    print(f"T={T:>4}C: 10 Hz {pts[10]:.1f} | 1 kHz {pts[1e3]:.1f} | 100 kHz {pts[1e5]:.1f} "
          f"| 1 MHz {pts[1e6]:.1f} | 10 MHz {pts[1e7]:.1f}")
    summary[f"psrr_{T}C"] = {str(int(k)): v for k, v in pts.items()}
    plt.semilogx(fr, p, label=f"{T} C")
plt.xlabel("Frequency (Hz)"); plt.ylabel("PSRR (dB)"); plt.grid(alpha=.3, which="both")
plt.title("Vref PSRR"); plt.legend(); plt.tight_layout()
plt.savefig("results/step5b_psrr.png", dpi=150)

# ---------------- Zout ----------------
print("\n---- Output impedance at Vref (27 C) ----")
f = "build/s5b_zout.csv"
if os.path.exists(f):
    d = np.loadtxt(f)
    fr, z = d[:, 0], d[:, 1]
    k = int(np.argmax(z))
    print(f"Zout(1 Hz) = {z[0]/1e3:.1f} kOhm; max = {z[k]/1e3:.1f} kOhm at {fr[k]:.3g} Hz "
          f"({20*np.log10(z[k]/z[0]):+.1f} dB vs 1 Hz)")
    summary["zout_1Hz_ohm"] = float(z[0]); summary["zout_peak_ohm"] = float(z[k])
    plt.figure(figsize=(7, 4.5))
    plt.loglog(fr, z); plt.xlabel("Frequency (Hz)"); plt.ylabel("|Zout| (Ohm)")
    plt.grid(alpha=.3, which="both"); plt.title("Vref output impedance")
    plt.tight_layout(); plt.savefig("results/step5b_zout.png", dpi=150)

# ---------------- ramps ----------------
print("\n---- Startup vs ramp rate (no nodesets) ----")
target = VG0
for kind, lab in (("fast", "1 us ramp"), ("slow", "1 ms ramp")):
    for T in (-40, 125):
        f = f"build/s5b_{kind}_{T}.csv"
        if not os.path.exists(f):
            continue
        d = np.loadtxt(f)
        t, vr = d[:, 0] * 1e6, d[:, 1]
        final = vr[-1]
        if final < 1.0:
            print(f"{lab:>9}, {T:>4}C: FAILED TO START (Vref = {final:.4f} V)")
            continue
        bad = np.where(np.abs(vr - final) > 0.01 * final)[0]
        ts = t[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(t) else t[0]
        print(f"{lab:>9}, {T:>4}C: final {final:.4f} V (sweep value at 27C: {target:.4f}), "
              f"1% settle at {ts:.1f} us, overshoot {(vr.max()-final)*1e3:.1f} mV")

# ---------------- supply step ----------------
print("\n---- +-10% supply step at 27 C ----")
f = "build/s5b_vstep.csv"
if os.path.exists(f):
    d = np.loadtxt(f)
    t, vr = d[:, 0] * 1e6, d[:, 1]
    def mean(a, b):
        m = (t >= a) & (t <= b)
        return vr[m].mean()
    v180, v198, v162, v180b = mean(8, 10), mean(28, 29.9), mean(48, 49.9), mean(68, 70)
    print(f"Plateaus: 1.80 V -> {v180:.5f} | 1.98 V -> {v198:.5f} | 1.62 V -> {v162:.5f} | back to 1.80 V -> {v180b:.5f}")
    print(f"Line regulation from step test = {(v198 - v162)/0.36*1e3:.2f} mV/V")
    for te, lab in ((10.0, "1.80 -> 1.98 V"), (30.0, "1.98 -> 1.62 V"), (50.0, "1.62 -> 1.80 V")):
        w = (t > te + 0.1) & (t < te + 19.9)
        tw, vw = t[w], vr[w]
        final = vw[tw > te + 18].mean()
        dev = np.abs(vw - final)
        bad = np.where(dev > 20e-6)[0]
        ts = tw[bad[-1]] - te if len(bad) else 0.0
        print(f"  {lab}: peak excursion {dev.max()*1e6:.0f} uV, settles within 20 uV after {ts:.1f} us")

# ---------------- temperature: startup on vs off ----------------
print("\n---- Temperature sweep: startup circuit on vs off (same R2) ----")
def load_temp(tag):
    lo, hi = f"build/s5b_temp_{tag}_lo.csv", f"build/s5b_temp_{tag}_hi.csv"
    if not (os.path.exists(lo) and os.path.exists(hi)):
        return None
    dl, dh = np.loadtxt(lo), np.loadtxt(hi)
    d = np.vstack([dl[::-1], dh[1:]])
    T, vr, iv = d[:, 0], d[:, 1], -d[:, 3]
    v27 = np.interp(27, T, vr)
    return dict(tc=(vr.max() - vr.min()) / (v27 * (T.max() - T.min())) * 1e6,
                v=[float(np.interp(x, T, vr)) for x in (-40, 27, 125)],
                i=[float(np.interp(x, T, iv)) * 1e6 for x in (27, 125)])
on, off = load_temp("on"), load_temp("off")
if on and off:
    print(f"{'':>12} {'TC ppm/C':>9} {'Vref -40C':>10} {'Vref 27C':>10} {'Vref 125C':>10} {'Iq 27C':>8} {'Iq 125C':>8}")
    for lab, r in (("startup", on), ("no startup", off)):
        print(f"{lab:>12} {r['tc']:9.1f} {r['v'][0]:10.5f} {r['v'][1]:10.5f} {r['v'][2]:10.5f} "
              f"{r['i'][0]:8.2f} {r['i'][1]:8.2f}")
    dv = [(a - b) * 1e3 for a, b in zip(on["v"], off["v"])]
    print(f"{'difference':>12} {'':>9} {dv[0]:+10.3f} {dv[1]:+10.3f} {dv[2]:+10.3f}   (mV)")
    summary["tc_on"], summary["tc_off"] = on["tc"], off["tc"]

json.dump(summary, open("results/step5b_summary.json", "w"), indent=2)
