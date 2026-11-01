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
C1, CC = "2p", "3p"
TEMPS = (-40, 27, 125)
CLS = ("1p", "10p", "100p")

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

def core(out, c1, cc):
    ccl = f"Ccomp vref g {cc}\n" if cc else ""
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
C1 {out} vdd {c1}
{ccl}XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn 0 0 {NF} W=1 L=1
"""

NS0 = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
       "v(g)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0")
NS = NS0 + " v(go)=0.72"

def ac_net(T, mode, cl, out):
    inj = "Vinj go g DC 0 AC 1" if mode == "v" else "Vinj go g DC 0\nIinj 0 g DC 0 AC 1"
    return f"""* Tian loop gain, {mode} injection, CL = {cl}, C1 = {C1}, Cc = {CC}
{head(T)}
VDD vdd 0 DC 1.8
{core("go", C1, CC)}CL vref 0 {cl}
{inj}
{NS}
.control
ac dec 30 1 1e9
wrdata {out} real(v(go)) imag(v(go)) real(v(g)) imag(v(g)) real(i(vinj)) imag(i(vinj))
quit
.endc
.end
"""

def psrr_net(T, cc, out):
    return f"""* PSRR, Cc = {cc}
{head(T)}
VDD vdd 0 DC 1.8 AC 1
{core("g", C1, cc)}CL vref 0 1p
{NS0}
.control
ac dec 20 1 1e8
wrdata {out} vdb(vref)
quit
.endc
.end
"""

def ramp_net(T, ramp, tstop, step, out):
    return f"""* startup ramp {ramp}, Cc = {CC}
{head(T)}
VDD vdd 0 PWL(0 0 {ramp} 1.8)
{core("g", C1, CC)}CL vref 0 1p
.control
tran {step} {tstop}
wrdata {out} vdd vref g sn
quit
.endc
.end
"""

RAMPS = [("1u", -40, "100u", "20n"), ("10u", -40, "200u", "50n"),
         ("1u", 125, "100u", "20n"), ("10u", 125, "200u", "50n"),
         ("1m", -40, "1.5m", "2u"), ("1m", 125, "1.5m", "2u")]

os.makedirs("build", exist_ok=True)
jobs = []
for T in TEMPS:
    for cl in CLS:
        for mode in ("v", "i"):
            o = f"build/s6f_{mode}_{T}_{cl}.csv"
            jobs.append((f"s6f_{mode}_{T}_{cl}", ac_net(T, mode, cl, o), [o]))
for T in TEMPS:
    o = f"build/s6f_psrr_{T}_cc.csv"
    jobs.append((f"s6f_psrr_{T}_cc", psrr_net(T, CC, o), [o]))
o = "build/s6f_psrr_27_base.csv"
jobs.append(("s6f_psrr_27_base", psrr_net(27, None, o), [o]))
for ramp, T, tstop, step in RAMPS:
    o = f"build/s6f_ramp_{ramp}_{T}.csv"
    jobs.append((f"s6f_ramp_{ramp}_{T}", ramp_net(T, ramp, tstop, step, o), [o]))

def go(job):
    name, text, outs = job
    for o in outs:
        if os.path.exists(o):
            os.remove(o)
    path = f"build/{name}.spice"
    open(path, "w").write(os.path.expandvars(text))
    r = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True)
    return name, all(os.path.exists(o) for o in outs), r.stdout[-600:]

print(f"Running {len(jobs)} simulations, {WORKERS} at a time (about 6-8 min, estimate) ...")
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
        z = None
        if interp_y is not None:
            z = interp_y[k] + w * (interp_y[k + 1] - interp_y[k])
        out.append((f, z))
    return out

summary = {"loop": [], "psrr": {}, "ramps": []}
pm_tab = {}
print(f"\nLoop gain, C1 = {C1}, Cc = {CC}")
print(f"{'T (C)':>6} {'CL':>5} {'T(1Hz) dB':>10} {'ph(1Hz)':>8} {'fu (MHz)':>9} {'PM (deg)':>9} {'GM (dB)':>8}")
for T in TEMPS:
    for cl in CLS:
        fv, fi = f"build/s6f_v_{T}_{cl}.csv", f"build/s6f_i_{T}_{cl}.csv"
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
        pm_tab.setdefault(T, {})[cl] = float(180 + xs[0][1]) if xs else np.nan
        summary["loop"].append(dict(T=T, CL=cl, T0_dB=float(mag[0]),
                                    fu_MHz=float(xs[0][0] / 1e6) if xs else None,
                                    PM_deg=float(180 + xs[0][1]) if xs else None,
                                    GM_dB=float(-gx[0][1]) if gx else None))

plt.figure(figsize=(7, 4.5))
for T in TEMPS:
    plt.semilogx([1, 10, 100], [pm_tab.get(T, {}).get(c, np.nan) for c in CLS], "o-", label=f"{T} C")
plt.axhline(45, color="k", lw=0.6, ls="--")
plt.xlabel("CL (pF)"); plt.ylabel("Phase margin (deg)"); plt.grid(alpha=.3, which="both")
plt.title("PM vs CL with Miller Cc = 3 pF"); plt.legend(); plt.tight_layout()
plt.savefig("results/step6f_pm_vs_cl.png", dpi=150)

FREQS = (10, 1e3, 1e5, 1e6, 1e7)
print("\n---- PSRR (VDD ripple -> Vref), dB, CL = 1 pF ----")
print(f"{'':>18} {'10 Hz':>7} {'1 kHz':>7} {'100 kHz':>8} {'1 MHz':>7} {'10 MHz':>7}")
rows = [(f"Cc=3p  {T} C", f"build/s6f_psrr_{T}_cc.csv") for T in TEMPS]
rows.append(("Cc=0 (base) 27 C", "build/s6f_psrr_27_base.csv"))
for lab, f in rows:
    if not os.path.exists(f):
        print(f"{lab:>18}   missing"); continue
    d = np.loadtxt(f)
    fr, p = d[:, 0], -d[:, 1]
    pts = [float(np.interp(np.log10(x), np.log10(fr), p)) for x in FREQS]
    print(f"{lab:>18} " + " ".join(f"{v:7.1f}" for v in pts))
    summary["psrr"][lab] = pts

print("\n---- Startup ramps, Cc = 3 pF, no nodesets ----")
print(f"{'ramp':>6} {'T (C)':>6} {'final (V)':>10} {'peak (V)':>9} {'overshoot (mV)':>15} {'1% settle (us)':>15}")
for ramp, T, tstop, step in RAMPS:
    f = f"build/s6f_ramp_{ramp}_{T}.csv"
    if not os.path.exists(f):
        print(f"{ramp:>6} {T:>6}   missing"); continue
    d = np.loadtxt(f)
    t, vr = d[:, 0] * 1e6, d[:, 3]
    final = vr[-1]
    if final < 1.0:
        print(f"{ramp:>6} {T:>6}   FAILED TO START (Vref = {final:.4f} V)"); continue
    bad = np.where(np.abs(vr - final) > 0.01 * final)[0]
    ts = t[bad[-1] + 1] if len(bad) and bad[-1] + 1 < len(t) else t[0]
    flag = "" if abs(vr[-1] - vr[int(0.9 * len(vr))]) < 1e-3 else "  (not settled by end of run)"
    print(f"{ramp:>6} {T:>6} {final:10.4f} {vr.max():9.4f} {(vr.max()-final)*1e3:15.1f} {ts:15.1f}{flag}")
    summary["ramps"].append(dict(ramp=ramp, T=T, final=float(final), peak=float(vr.max()),
                                 overshoot_mV=float((vr.max() - final) * 1e3), settle_us=float(ts)))

f = "build/s6f_ramp_1u_-40.csv"
if os.path.exists(f):
    d = np.loadtxt(f)
    t = d[:, 0] * 1e6
    names = ["vdd", "vref", "g", "sn"]
    v = {n: d[:, 1 + 2 * k] for k, n in enumerate(names)}
    print("\nFast ramp (1 us), -40 C, node voltages in V, Cc = 3 pF")
    print(f"{'t (us)':>9} " + " ".join(f"{n:>7}" for n in names))
    for x in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2, 3, 5, 10, 20, 50]:
        print(f"{x:9.1f} " + " ".join(f"{np.interp(x, t, v[n]):7.3f}" for n in names))

json.dump(summary, open("results/step6f_summary.json", "w"), indent=2)
