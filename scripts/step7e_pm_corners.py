import os, json, subprocess, time
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
CORNERS = ["tt", "ss", "ff", "sf", "fs"]
TEMPS = (-40, 27, 125)
CL = "10p"

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


NS = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
      "v(g)=0.72 v(go)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0 v(s2)=0")


def ac_net(corner, T, mode, out):
    inj = ("Vinj go g DC 0 AC 1" if mode == "v"
           else "Vinj go g DC 0\nIinj 0 g DC 0 AC 1")
    return f"""* Tian loop gain {corner} {T} C {mode} injection CL={CL}
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice {corner}
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp {T}
VDD vdd 0 DC 1.8
XQ1 0 0 a {Q}
{q2}
{chain("r2a", "vref", "a", N2, L2)}
{chain("r2b", "vref", "b", N2, L2)}
{chain("r1", "b", "e2", N1, L1)}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 go a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 go n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
XMBP vbn g vdd vdd {PF} W=10 L=2
C1 go vdd 20p
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn s2 0 {NF} W=1 L=1
XMd s2 s2 0 0 {NF} W=2 L=1
CL vref 0 {CL}
{inj}
{NS}
.control
ac dec 30 1 1e9
wrdata {out} real(v(go)) imag(v(go)) real(v(g)) imag(v(g)) real(i(vinj)) imag(i(vinj))
quit
.endc
.end
"""


def fname(c, T, m):
    return f"build/s7e_{c}_{T}_{m}.csv"


jobs = [(c, T, m) for c in CORNERS for T in TEMPS for m in ("v", "i")]


def go(job):
    c, T, m = job
    out = fname(c, T, m)
    if os.path.exists(out):
        os.remove(out)
    path = f"build/s7e_{c}_{T}_{m}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(ac_net(c, T, m, out)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(path.replace(".spice", ".log"), "w") as f:
        f.write(p.stdout + p.stderr)
    return job


def load(f):
    d = np.loadtxt(f)
    if d.shape[1] != 12:
        raise ValueError(f"unexpected column count {d.shape[1]}")
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
import os, json, subprocess, time
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
CORNERS = ["tt", "ss", "ff", "sf", "fs"]
TEMPS = (-40, 27, 125)
CL = "10p"

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


NS = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(e2)={VBE0 - I0*R1} "
      "v(g)=0.72 v(go)=0.72 v(n1)=0.73 v(tail)=0.2 v(vbn)=0.55 v(sn)=0 v(s2)=0")


def ac_net(corner, T, mode, out):
    inj = ("Vinj go g DC 0 AC 1" if mode == "v"
           else "Vinj go g DC 0\nIinj 0 g DC 0 AC 1")
    return f"""* Tian loop gain {corner} {T} C {mode} injection CL={CL}
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice {corner}
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp {T}
VDD vdd 0 DC 1.8
XQ1 0 0 a {Q}
{q2}
{chain("r2a", "vref", "a", N2, L2)}
{chain("r2b", "vref", "b", N2, L2)}
{chain("r1", "b", "e2", N1, L1)}
XMP vref g vdd vdd {PF} W=100 L=2
XMT tail vbn 0 0 {NF} W=20 L=2
XM1 n1 b tail 0 {NF} W=50 L=2
XM2 go a tail 0 {NF} W=50 L=2
XM3 n1 n1 vdd vdd {PF} W=10 L=2
XM4 go n1 vdd vdd {PF} W=10 L=2
XMB vbn vbn 0 0 {NF} W=10 L=2
XMBP vbn g vdd vdd {PF} W=10 L=2
C1 go vdd 20p
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn s2 0 {NF} W=1 L=1
XMd s2 s2 0 0 {NF} W=2 L=1
CL vref 0 {CL}
{inj}
{NS}
.control
ac dec 30 1 1e9
wrdata {out} real(v(go)) imag(v(go)) real(v(g)) imag(v(g)) real(i(vinj)) imag(i(vinj))
quit
.endc
.end
"""


def fname(c, T, m):
    return f"build/s7e_{c}_{T}_{m}.csv"


jobs = [(c, T, m) for c in CORNERS for T in TEMPS for m in ("v", "i")]


def go(job):
    c, T, m = job
    out = fname(c, T, m)
    if os.path.exists(out):
        os.remove(out)
    path = f"build/s7e_{c}_{T}_{m}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(ac_net(c, T, m, out)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(path.replace(".spice", ".log"), "w") as f:
        f.write(p.stdout + p.stderr)
    return job


def load(f):
    d = np.loadtxt(f)
    if d.shape[1] != 12:
        raise ValueError(f"unexpected column count {d.shape[1]}")
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


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("build", exist_ok=True)
    print(f"Running {len(jobs)} simulations")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(go, jobs))
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True)
    summary = []
    print(f"\n{'corner':7s}{'T':>5s}{'T(1Hz) dB':>11s}{'ph(1Hz)':>9s}"
          f"{'fu MHz':>9s}{'PM deg':>8s}{'GM dB':>8s}")
    for c in CORNERS:
        for T in TEMPS:
            try:
                fr, vgo, vg, _ = load(fname(c, T, "v"))
                _, _, _, ii = load(fname(c, T, "i"))
            except Exception as e:
                print(f"{c:7s}{T:5d}  ERROR {e!r}  (see build/s7e_{c}_{T}_v.log)")
                continue
            Tv = -vgo / vg
            Ti = -ii / (ii + 1.0)
            Tl = (Tv * Ti - 1.0) / (Tv + Ti + 2.0)
            mag = 20 * np.log10(np.abs(Tl))
            ph = np.degrees(np.unwrap(np.angle(Tl)))
            ph -= 360 * np.round(ph[0] / 360)
            xs = cross_down(fr, mag, 0.0, ph)
            gx = cross_down(fr, ph, -180.0, mag)
            fu = f"{xs[0][0]/1e6:9.3f}" if xs else "     none"
            pm = f"{180 + xs[0][1]:8.1f}" if xs else "    none"
            gm = f"{-gx[0][1]:8.1f}" if gx else "    none"
            print(f"{c:7s}{T:5d}{mag[0]:11.1f}{ph[0]:9.1f}{fu}{pm}{gm}")
            if len(xs) > 1:
                print(f"        note: {len(xs)} unity-gain crossings, first reported")
            summary.append(dict(corner=c, T=T, T0_dB=float(mag[0]),
                                ph0=float(ph[0]),
                                fu_MHz=float(xs[0][0] / 1e6) if xs else None,
                                PM_deg=float(180 + xs[0][1]) if xs else None,
                                GM_dB=float(-gx[0][1]) if gx else None))
            ax[0].semilogx(fr, mag, lw=1, label=f"{c} {T}C")
            ax[1].semilogx(fr, ph, lw=1)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[1].axhline(-180, color="k", lw=0.5)
    ax[0].set_ylabel("|T| (dB)")
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_xlabel("Frequency (Hz)")
    for a in ax:
        a.grid(alpha=.3, which="both")
    ax[0].legend(fontsize=6, ncol=3)
    ax[0].set_title("Loop gain (Tian), C1=20 pF, CL=10 pF, corners")
    plt.tight_layout()
    plt.savefig("results/step7e_pm_corners.png", dpi=130)
    json.dump(summary, open("results/step7e_pm_corners.json", "w"), indent=2)
    pms = [s["PM_deg"] for s in summary if s["PM_deg"] is not None]
    if pms:
        print(f"\nmin PM {min(pms):.1f} deg, max PM {max(pms):.1f} deg over {len(pms)} cases")
    print(f"done in {time.time()-t0:.0f} s")
