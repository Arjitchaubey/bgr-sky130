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
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
VG0 = s6c.get("vref27", 1.211)
WORKERS = 4
CORNERS = ["tt", "ss", "ff", "sf", "fs", "ss_ll", "ff_hh"]
VDDS = [1.8, 1.62, 1.98]
NV = 6  # vref, i(vdd), tail, n1, g, vdd

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


def fname(corner, vdd, d):
    return f"build/s7c_{corner}_{vdd}_{d}.dat"


def netlist(corner):
    ns = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} "
          f"v(e2)={VBE0 - I0*R1} v(g)=0.72 v(n1)=0.73 v(tail)=0.2 "
          "v(vbn)=0.55 v(sn)=0 v(s2)=0")
    ctl = []
    for v in VDDS:
        ctl.append(f"alter vdd = {v}")
        ctl.append("dc temp 27 125 1")
        ctl.append(f"wrdata {fname(corner, v, 'up')} v(vref) i(vdd) v(tail) v(n1) v(g) v(vdd)")
        ctl.append("dc temp 27 -40 -1")
        ctl.append(f"wrdata {fname(corner, v, 'dn')} v(vref) i(vdd) v(tail) v(n1) v(g) v(vdd)")
    ctl = "\n".join(ctl)
    return f"""* step7c DC corner {corner}
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice {corner}
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp 27
VDD vdd 0 DC 1.8
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
{ns}
.control
{ctl}
quit
.endc
.end
"""


def go(corner):
    for v in VDDS:
        for d in ("up", "dn"):
            if os.path.exists(fname(corner, v, d)):
                os.remove(fname(corner, v, d))
    path = f"build/s7c_{corner}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(netlist(corner)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(f"build/s7c_{corner}.log", "w") as f:
        f.write(p.stdout + p.stderr)
    return corner


def load(path):
    d = np.loadtxt(path)
    nc = d.shape[1]
    if nc == 2 * NV:
        return d[:, 0], [d[:, 2 * k + 1] for k in range(NV)]
    if nc == NV + 1:
        return d[:, 0], [d[:, k + 1] for k in range(NV)]
    raise ValueError(f"unexpected column count {nc}")


def merged(corner, v):
    xu, cu = load(fname(corner, v, "up"))
    xd, cd = load(fname(corner, v, "dn"))
    x = np.concatenate([xd[1:], xu])
    cols = [np.concatenate([a[1:], b]) for a, b in zip(cd, cu)]
    o = np.argsort(x)
    return x[o], [c[o] for c in cols]


def at(T, t):
    return int(np.argmin(np.abs(T - t)))


def analyze(corner):
    try:
        data = {v: merged(corner, v) for v in VDDS}
    except Exception as e:
        return {"error": repr(e)}
    T, c = data[1.8]
    vref, iq = c[0], -c[1]
    k27 = at(T, 27)
    r = {"vref27": float(vref[k27]),
         "vmin": float(vref.min()), "vmax": float(vref.max()),
         "tc_ppm": float((vref.max() - vref.min()) / (vref[k27] * (T.max() - T.min())) * 1e6),
         "vref": {str(t): float(vref[at(T, t)]) for t in (-40, 27, 125)},
         "iq_uA": {str(t): float(iq[at(T, t)] * 1e6) for t in (-40, 27, 125)},
         "state_ok": bool(vref.min() > 1.1)}
    lo, hi = data[1.62][1][0], data[1.98][1][0]
    r["line_mV_per_V"] = {str(t): float((hi[at(T, t)] - lo[at(T, t)]) / 0.36 * 1e3)
                          for t in (-40, 27, 125)}
    r["state_ok"] = bool(r["state_ok"] and lo.min() > 1.1 and hi.min() > 1.1)
    _, c2 = data[1.62]
    vr2, tail, n1, g, vd = c2[0], c2[2], c2[3], c2[4], c2[5]
    hr = {"tail_vds": tail, "m1_vds": n1 - tail, "m2_vds": g - tail,
          "pass_vsd": vd - vr2, "m4_vsd": vd - g}
    r["headroom_1p62_min"] = {k: float(v.min()) for k, v in hr.items()}
    r["headroom_1p62_at_T"] = {k: float(T[int(np.argmin(v))]) for k, v in hr.items()}
    r["T"] = T.tolist()
    r["vref_curve"] = vref.tolist()
    return r


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("build", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print(f"Running {len(CORNERS)} simulations")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(go, CORNERS))
    res = {c: analyze(c) for c in CORNERS}
    out = {c: {k: v for k, v in r.items() if k not in ("T", "vref_curve")}
           for c, r in res.items()}
    with open("results/step7c_corners_dc.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{'corner':7s}{'Vref27':>9s}{'Vmin':>8s}{'Vmax':>8s}{'TC ppm/C':>10s}"
          f"{'Iq27 uA':>9s}{'Iq125':>8s}")
    for c in CORNERS:
        r = res[c]
        if "error" in r:
            print(f"{c:7s} ERROR {r['error']}  (see build/s7c_{c}.log)")
            continue
        flag = "" if r["state_ok"] else "  WRONG STATE?"
        print(f"{c:7s}{r['vref27']:9.4f}{r['vmin']:8.4f}{r['vmax']:8.4f}"
              f"{r['tc_ppm']:10.2f}{r['iq_uA']['27']:9.2f}{r['iq_uA']['125']:8.2f}{flag}")
    print("\nVref (V) at -40/27/125 C and line regulation (mV/V, 1.62-1.98 V)")
    for c in CORNERS:
        r = res[c]
        if "error" in r:
            continue
        v, l = r["vref"], r["line_mV_per_V"]
        print(f"{c:7s} {v['-40']:.4f} {v['27']:.4f} {v['125']:.4f}   "
              f"{l['-40']:7.3f} {l['27']:7.3f} {l['125']:7.3f}")
    print("\nHeadroom at VDD=1.62 V: minimum over temperature (V) [at T]")
    print(f"{'corner':7s}{'tail Vds':>14s}{'M1 Vds':>14s}{'M2 Vds':>14s}"
          f"{'pass Vsd':>14s}{'M4 Vsd':>14s}")
    for c in CORNERS:
        r = res[c]
        if "error" in r:
            continue
        m, a = r["headroom_1p62_min"], r["headroom_1p62_at_T"]
        row = "".join(f"{m[k]:8.3f} [{a[k]:4.0f}]" for k in
                      ("tail_vds", "m1_vds", "m2_vds", "pass_vsd", "m4_vsd"))
        print(f"{c:7s}{row}")
    plt.figure(figsize=(7, 4.5))
    for c in CORNERS:
        r = res[c]
        if "error" not in r:
            plt.plot(r["T"], np.array(r["vref_curve"]) * 1e3, label=c)
    plt.xlabel("temperature (C)")
    plt.ylabel("Vref (mV)")
    plt.title("Vref vs T, VDD=1.8 V, per corner")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/step7c_corners_dc.png", dpi=120)
    print(f"done in {time.time()-t0:.0f} s")
