import os, json, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

des = json.load(open("results/design.json"))
s6c = json.load(open("results/step6c_summary.json"))
M = s6c["model"]
L1, L2 = s6c["L_R1_unit"], s6c["L_R2_unit"]
N1, N2 = s6c["n_R1"], s6c["n_R2"]
R1, VBE0, I0, VG0 = des["R1"], des["vbe0"], des["I"], s6c["vref27"]
WORKERS = 4

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

def core(out):
    return f"""XQ1 0 0 a {Q}
{q2}
{chain("r2a", "vref", "a", N2, L2)}
{chain("r2b", "vref", "b", N2, L2)}
{chain("r1", "b", "e2", N1, L1)}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 {out} a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 {out} n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
XMBP vbn g vdd vdd {PF} W=10 L=2
C1 {out} vdd 2p
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn 0 0 {NF} W=1 L=1
"""

NS = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
      "v(g)=0.72 v(go)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0")

def ac_net(T, mode, cl, out):
    inj = "Vinj go g DC 0 AC 1" if mode == "v" else "Vinj go g DC 0\nIinj 0 g DC 0 AC 1"
    return f"""* Tian loop gain, {mode} injection, CL = {cl}
{head(T)}
VDD vdd 0 DC 1.8
{core("go")}CL vref 0 {cl}
{inj}
{NS}
.control
ac dec 30 1 1e9
wrdata {out} real(v(go)) imag(v(go)) real(v(g)) imag(v(g)) real(i(vinj)) imag(i(vinj))
quit
.endc
.end
"""

def tran_net(T, vddline, tstop, step, out):
    return f"""* startup diagnostic
{head(T)}
{vddline}
{core("g")}CL vref 0 1p
.control
tran {step} {tstop}
wrdata {out} vdd vref g sn vbn tail
quit
.endc
.end
"""

os.makedirs("build", exist_ok=True)
TEMPS = (-40, 27, 125)
CLS = ("1p", "10p", "100p")
jobs = []
for T in TEMPS:
    for cl in CLS:
        for mode in ("v", "i"):
            o = f"build/s6d_{mode}_{T}_{cl}.csv"
            jobs.append((f"s6d_{mode}_{T}_{cl}", ac_net(T, mode, cl, o), [o]))
o = "build/s6d_fast_-40.csv"
jobs.append(("s6d_fast_-40", tran_net(-40, "VDD vdd 0 PWL(0 0 1u 1.8)", "20u", "20n", o), [o]))
o = "build/s6d_slow_-40.csv"
jobs.append(("s6d_slow_-40", tran_net(-40, "VDD vdd 0 PWL(0 0 1m 1.8)", "1.5m", "2u", o), [o]))

def go(job):
    name, text, outs = job
    for o in outs:
        if os.path.exists(o):
            os.remove(o)
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return name, all(os.path.exists(o) for o in outs), r.stdout[-600:]

print(f"Running {len(jobs)} simulations, {WORKERS} at a time (about 5-6 min) ...")
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    res = list(ex.map(go, jobs))
for n, ok, tail in res:
    if not ok:
        print(f"\nFAILED {n}. Last ngspice output:\n{tail}")

def load(f):
    d = np.loadtxt(f)
    n = d.shape[1]
    if n != 12:
        print(f"Unexpected column count {n} in {f}. First row:\n{d[0]}")
        raise SystemExit(1)
    c = lambda j: d[:, 1 + 2 * j]
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

fig, ax = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
colors = {"1p": "C0", "10p": "C1", "100p": "C3"}
styles = {-40: "-", 27: "--", 125: ":"}
summary = []
print(f"\n{'T (C)':>6} {'CL':>5} {'T(1Hz) dB':>10} {'ph(1Hz)':>8} {'fu (MHz)':>9} {'PM (deg)':>9} {'GM (dB)':>8}")
for T in TEMPS:
    for cl in CLS:
        fv, fi = f"build/s6d_v_{T}_{cl}.csv", f"build/s6d_i_{T}_{cl}.csv"
        if not (os.path.exists(fv) and os.path.exists(fi)):
            print(f"{T:>6} {cl:>5}   missing"); continue
        fr, vgo, vg, _ = load(fv)
        _, _, _, ii = load(fi)
        Tv = -vgo / vg
        Ti = -ii / (ii + 1.0)
        Tl = (Tv * Ti - 1.0) / (Tv + Ti + 2.0)
        mag = 20 * np.log10(np.abs(Tl))
        ph = np.degrees(np.unwrap(np.angle(Tl)))
        ph -= 360 * np.round(ph[0] / 360)
        xs = cross_down(fr, mag, 0.0, ph)
        gx = cross_down(fr, ph, -180.0, mag)
        fu = f"{xs[0][0]/1e6:9.3f}" if xs else "     none"
        pm = f"{180 + xs[0][1]:9.1f}" if xs else "     none"
        gm = f"{-gx[0][1]:8.1f}" if gx else "    none"
        print(f"{T:>6} {cl:>5} {mag[0]:10.1f} {ph[0]:8.1f} {fu} {pm} {gm}")
        if len(xs) > 1:
            print(f"        note: {len(xs)} unity-gain crossings, first one reported")
        summary.append(dict(T=T, CL=cl, T0_dB=float(mag[0]),
                            fu_MHz=float(xs[0][0] / 1e6) if xs else None,
                            PM_deg=float(180 + xs[0][1]) if xs else None,
                            GM_dB=float(-gx[0][1]) if gx else None))
        ax[0].semilogx(fr, mag, color=colors[cl], ls=styles[T], lw=1.2, label=f"{T}C {cl}")
        ax[1].semilogx(fr, ph, color=colors[cl], ls=styles[T], lw=1.2)
ax[0].axhline(0, color="k", lw=0.5); ax[1].axhline(-180, color="k", lw=0.5)
ax[0].set_ylabel("|T| (dB)"); ax[1].set_ylabel("Phase (deg)"); ax[1].set_xlabel("Frequency (Hz)")
for a in ax:
    a.grid(alpha=.3, which="both")
ax[0].legend(fontsize=7, ncol=3)
ax[0].set_title("Self-biased BGR loop gain (Tian) vs load capacitance")
plt.tight_layout(); plt.savefig("results/step6d_loopgain.png", dpi=150)

def show(f, label, times=None):
    if not os.path.exists(f):
        print(f"\n{label}: missing"); return
    d = np.loadtxt(f)
    t = d[:, 0] * 1e6
    names = ["vdd", "vref", "g", "sn", "vbn", "tail"]
    v = {n: d[:, 1 + 2 * k] for k, n in enumerate(names)}
    print(f"\n{label}")
    if times is None:
        tp = t[np.argmax(v["vref"])]
        print(f"(Vref peaks at {tp:.0f} us)")
        times = [min(max(tp + o, t[0]), t[-1]) for o in (-200, -100, -50, -20, -10, 0, 10, 20, 50, 100, 200)]
    print(f"{'t (us)':>9} " + " ".join(f"{n:>7}" for n in names))
    for x in times:
        print(f"{x:9.1f} " + " ".join(f"{np.interp(x, t, v[n]):7.3f}" for n in names))

show("build/s6d_fast_-40.csv", "Fast ramp (1 us), -40 C, node voltages in V",
     [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2, 2.5, 3, 4, 6, 10])
show("build/s6d_slow_-40.csv", "Slow ramp (1 ms), -40 C, node voltages in V around the peak")

json.dump(summary, open("results/step6d_summary.json", "w"), indent=2)
