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
STARTUP = (f"XMsp sn vref vdd vdd {PF} W=1 L=8\n"
           f"XMsn sn vref 0 0 {NF} W=2 L=1\n"
           f"XMst g sn 0 0 {NF} W=1 L=1\n")

def netlist(temp, mode, out):
    inj = ("Vinj go g DC 0 AC 1" if mode == "v"
           else "Vinj go g DC 0\nIinj 0 g DC 0 AC 1")
    return f"""* Tian loop gain, {mode} injection, break between go (OTA out) and g (pass gate)
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp {temp}
VDD vdd 0 DC 1.8
XQ1 0 0 a {Q}
{q2}
R2a vref a {R2:.1f}
R2b vref b {R2:.1f}
R1  b e2 {R1}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 go a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 go n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
IB vdd vbn 1u
C1 go vdd 2p
CL vref 0 1p
{STARTUP}{inj}
.nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} v(g)=0.72 v(go)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0
.control
ac dec 30 1 1e9
wrdata {out} real(v(go)) imag(v(go)) real(v(g)) imag(v(g)) real(i(vinj)) imag(i(vinj))
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
TEMPS = (-40, 27, 125)
jobs = []
for T in TEMPS:
    for mode in ("v", "i"):
        out = f"build/s5d_{mode}_{T}.csv"
        if os.path.exists(out):
            os.remove(out)
        jobs.append((T, mode, out))

def go(job):
    T, mode, out = job
    path = f"build/s5d_{mode}_{T}.spice"
    open(path, "w").write(os.path.expandvars(netlist(T, mode, out)))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return job, os.path.exists(out), r.stdout[-800:]

print("Running 6 AC simulations, 3 at a time ...")
with ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(go, jobs))
bad = False
for job, ok, tail in res:
    if not ok:
        bad = True
        print(f"\nFAILED T={job[0]} mode={job[1]}. Last ngspice output:\n{tail}")
if bad:
    raise SystemExit(1)

def load(f):
    d = np.loadtxt(f)
    n = d.shape[1]
    if n == 12:
        step = 2
    elif n == 18:
        step = 3
    else:
        print(f"Unexpected column count {n} in {f}. First row:\n{d[0]}")
        raise SystemExit(1)
    c = lambda j: d[:, 1 + step * j]
    return d[:, 0], c(0) + 1j * c(1), c(2) + 1j * c(3), c(4) + 1j * c(5)

def cross_down(fr, y, level, interp_y=None):
    idx = np.where((y[:-1] > level) & (y[1:] <= level))[0]
    out = []
    for k in idx:
        w = (level - y[k]) / (y[k + 1] - y[k])
        f = 10 ** (np.log10(fr[k]) + w * (np.log10(fr[k + 1]) - np.log10(fr[k])))
        z = None if interp_y is None else interp_y[k] + w * (interp_y[k + 1] - interp_y[k])
        out.append((f, z))
    return out

fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
summary = {}
print("\nLoop gain T(f) from Tian two-injection:")
for T in TEMPS:
    fr, vgo, vg, _ = load(f"build/s5d_v_{T}.csv")
    _, _, _, ii = load(f"build/s5d_i_{T}.csv")
    Tv = -vgo / vg
    Ti = -ii / (ii + 1.0)
    Tl = (Tv * Ti - 1.0) / (Tv + Ti + 2.0)
    mag = 20 * np.log10(np.abs(Tl))
    ph = np.degrees(np.unwrap(np.angle(Tl)))
    ph -= 360 * np.round(ph[0] / 360)

    print(f"\nT = {T} C:  |Tv|(100 Hz) = {20*np.log10(abs(np.interp(100, fr, np.abs(Tv)))):.1f} dB, "
          f"|Ti|(100 Hz) = {20*np.log10(abs(np.interp(100, fr, np.abs(Ti)))):.1f} dB")
    print(f"  T(1 Hz) = {mag[0]:.1f} dB, phase {ph[0]:.0f} deg  (negative feedback should show a phase near 0)")
    for x in (1e2, 1e4, 1e5, 1e6, 1e7):
        print(f"  {x:8.0e} Hz: {np.interp(np.log10(x), np.log10(fr), mag):7.1f} dB, "
              f"{np.interp(np.log10(x), np.log10(fr), ph):8.1f} deg")
    xs = cross_down(fr, mag, 0.0, ph)
    if not xs:
        print("  no 0 dB crossing found")
    for f0, p0 in xs:
        print(f"  unity gain at {f0/1e6:.3f} MHz, phase margin = {180 + p0:.1f} deg")
    gx = cross_down(fr, ph, -180.0, mag)
    for f0, m0 in gx:
        print(f"  phase = -180 deg at {f0/1e6:.3f} MHz, gain margin = {-m0:.1f} dB")
    if not gx:
        print("  phase never reaches -180 deg up to 1 GHz")
    summary[str(T)] = dict(T0_dB=float(mag[0]),
                           unity=[(float(a), float(180 + b)) for a, b in xs])
    ax[0].semilogx(fr, mag, label=f"{T} C")
    ax[1].semilogx(fr, ph, label=f"{T} C")

ax[0].axhline(0, color="k", lw=0.5); ax[1].axhline(-180, color="k", lw=0.5)
ax[0].set_ylabel("|T| (dB)"); ax[1].set_ylabel("Phase (deg)"); ax[1].set_xlabel("Frequency (Hz)")
for a in ax:
    a.grid(alpha=.3, which="both"); a.legend()
ax[0].set_title("BGR loop gain (Tian), startup included")
plt.tight_layout(); plt.savefig("results/step5d_loopgain.png", dpi=150)
json.dump(summary, open("results/step5d_summary.json", "w"), indent=2)
