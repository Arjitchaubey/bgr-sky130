import os, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

des = json.load(open("results/design.json"))
s6c = json.load(open("results/step6c_summary.json"))
M = s6c["model"]
L1, L2 = s6c["L_R1_unit"], s6c["L_R2_unit"]
N1, N2 = s6c["n_R1"], s6c["n_R2"]
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
VG0 = s6c.get("vref27", 1.211)
CORNERS = ["ff", "sf", "fs"]
RAMPS = {"1m": ("3m", "2u"), "1u": ("100u", "20n")}
LOADT = (-40, 27, 125)

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")
NS = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} "
      f"v(e2)={VBE0 - I0*R1} v(g)=0.72 v(n1)=0.73 v(tail)=0.2 "
      "v(vbn)=0.55 v(sn)=0 v(s2)=0")


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


def net(corner, T, vddline, load, ns, ctl):
    return f"""* step9b {corner} {T} C
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice {corner}
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp {T}
{vddline}
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
C1 g vdd 20p
CL vref 0 1p
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn s2 0 {NF} W=1 L=1
XMd s2 s2 0 0 {NF} W=2 L=1
{load}
{ns}
.control
{ctl}
quit
.endc
.end
"""


JOBS = [("st", c, -40, r) for c in CORNERS for r in RAMPS]
JOBS += [("ld", "tt", T, "dc") for T in LOADT]


def name(j):
    return f"s9b_{j[0]}_{j[1]}_{j[2]}_{j[3]}"


def text(j):
    kind, c, T, tag = j
    out = f"build/{name(j)}.csv"
    if kind == "st":
        tstop, step = RAMPS[tag]
        return net(c, T, f"VDD vdd 0 PWL(0 0 {tag} 1.8)", "", "",
                   f"tran {step} {tstop}\nwrdata {out} vdd vref")
    return net(c, T, "VDD vdd 0 DC 1.8", "Iload vref 0 DC 0", NS,
               f"dc Iload 0 200u 20u\nwrdata {out} v(vref)")


def go(j):
    out = f"build/{name(j)}.csv"
    if os.path.exists(out):
        os.remove(out)
    path = f"build/{name(j)}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(text(j)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(path.replace(".spice", ".log"), "w") as f:
        f.write(p.stdout + p.stderr)
    return j


def startup(j):
    d = np.loadtxt(f"build/{name(j)}.csv")
    t, vr = d[:, 0] * 1e6, d[:, 3]
    final = float(np.mean(vr[t > t[-1] * 0.95]))
    bad = np.where(np.abs(vr - final) > 0.01 * final)[0]
    if len(bad) == 0:
        settle = 0.0
    elif bad[-1] + 1 < len(t):
        settle = float(t[bad[-1] + 1])
    else:
        settle = float("nan")
    return {"final": final, "settle_us": settle,
            "overshoot_mV": float((vr.max() - final) * 1e3),
            "settled": bool(final > 1.1)}


def loadreg(j):
    d = np.loadtxt(f"build/{name(j)}.csv")
    i, v = d[:, 0], d[:, 1]
    v0 = float(np.interp(0, i, v))
    v100 = float(np.interp(100e-6, i, v))
    v200 = float(np.interp(200e-6, i, v))
    return {"v0": v0, "v100": v100, "v200": v200,
            "mV_per_mA_0_100u": (v100 - v0) * 1e3 / 0.1,
            "vmin": float(v.min()), "ok": bool(v.min() > 1.1)}


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("build", exist_ok=True)
    print(f"Running {len(JOBS)} simulations")
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(go, JOBS))
    res = {}
    print("\nStartup at -40 C, no nodesets (C1=20 pF, CL=1 pF)")
    print(f"{'run':16s}{'final V':>9s}{'settle us':>11s}{'ovsh mV':>9s}")
    for j in JOBS:
        if j[0] != "st":
            continue
        try:
            r = startup(j)
        except Exception as e:
            print(f"{name(j):16s} ERROR {e!r}")
            continue
        res[name(j)] = r
        flag = "" if r["settled"] else "  NOT SETTLED"
        print(f"{j[1]+'_'+j[3]:16s}{r['final']:9.4f}{r['settle_us']:11.1f}"
              f"{r['overshoot_mV']:9.1f}{flag}")
    print("\nLoad regulation, tt, VDD 1.8 V, Iload 0-200 uA")
    print(f"{'T':>5s}{'V(0)':>9s}{'V(100u)':>9s}{'V(200u)':>9s}{'mV/mA(0-100u)':>15s}")
    for j in JOBS:
        if j[0] != "ld":
            continue
        try:
            r = loadreg(j)
        except Exception as e:
            print(f"{j[2]:5d} ERROR {e!r}")
            continue
        res[name(j)] = r
        flag = "" if r["ok"] else "  VREF DROPPED"
        print(f"{j[2]:5d}{r['v0']:9.4f}{r['v100']:9.4f}{r['v200']:9.4f}"
              f"{r['mV_per_mA_0_100u']:15.3f}{flag}")
    json.dump(res, open("results/step9b_final.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f} s")

