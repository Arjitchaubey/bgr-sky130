import os, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

des = json.load(open("results/design.json"))
s6c = json.load(open("results/step6c_summary.json"))
M = s6c["model"]
L1, L2D = s6c["L_R1_unit"], s6c["L_R2_unit"]
N1, N2 = s6c["n_R1"], s6c["n_R2"]
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
VG0 = s6c.get("vref27", 1.211)
SLOPE = 1999.8324002457305 / 2.85
REND = 131.1699462197583
WORKERS = 4
CORNERS = ["tt", "ss", "ff", "ss_ll", "ff_hh"]
OFFS = [-0.20, -0.10, 0.0, 0.10, 0.20]

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")

R1U = SLOPE * L1 + REND
RHO0 = N2 * (SLOPE * L2D + REND) / (N1 * R1U)


def l2_for(off):
    if off == 0.0:
        return L2D
    rho = RHO0 + off
    return (rho * N1 * R1U / N2 - REND) / SLOPE


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


def fn(corner, off, d):
    return f"build/s7d_{corner}_{off:+.2f}_{d}.dat"


def netlist(corner, off):
    L2 = l2_for(off)
    ns = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} "
          f"v(e2)={VBE0 - I0*R1} v(g)=0.72 v(n1)=0.73 v(tail)=0.2 "
          "v(vbn)=0.55 v(sn)=0 v(s2)=0")
    return f"""* step7d {corner} off {off:+.2f}
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
dc temp 27 125 1
wrdata {fn(corner, off, 'up')} v(vref)
dc temp 27 -40 -1
wrdata {fn(corner, off, 'dn')} v(vref)
quit
.endc
.end
"""


def go(job):
    corner, off = job
    for d in ("up", "dn"):
        if os.path.exists(fn(corner, off, d)):
            os.remove(fn(corner, off, d))
    path = f"build/s7d_{corner}_{off:+.2f}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(netlist(corner, off)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(path.replace(".spice", ".log"), "w") as f:
        f.write(p.stdout + p.stderr)
    return job


def analyze(job):
    corner, off = job
    try:
        du = np.loadtxt(fn(corner, off, "up"))
        dd = np.loadtxt(fn(corner, off, "dn"))
        T = np.concatenate([dd[1:, 0], du[:, 0]])
        v = np.concatenate([dd[1:, 1], du[:, 1]])
        o = np.argsort(T)
        T, v = T[o], v[o]
        k27 = int(np.argmin(np.abs(T - 27)))
        tc = (v.max() - v.min()) / (v[k27] * (T.max() - T.min())) * 1e6
        return {"ratio": RHO0 + off, "L2": l2_for(off),
                "vref27": float(v[k27]), "tc": float(tc),
                "ok": bool(v.min() > 1.1)}
    except Exception as e:
        return {"error": repr(e)}


def interp_opt(xs, ys):
    k = int(np.argmin(ys))
    if k == 0 or k == len(ys) - 1:
        return None
    c = np.polyfit(xs[k - 1:k + 2], ys[k - 1:k + 2], 2)
    if c[0] <= 0:
        return None
    return float(-c[1] / (2 * c[0]))


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("build", exist_ok=True)
    jobs = [(c, o) for c in CORNERS for o in OFFS]
    print(f"Running {len(jobs)} simulations")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(go, jobs))
    res = {c: {f"{o:+.2f}": analyze((c, o)) for o in OFFS} for c in CORNERS}
    with open("results/step7d_ratio_sweep.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"nominal tt ratio at offset 0: {RHO0:.4f}")
    print(f"{'corner':7s}{'off':>7s}{'ratio':>8s}{'L2 um':>8s}{'Vref27':>9s}{'TC ppm/C':>10s}")
    for c in CORNERS:
        for o in OFFS:
            r = res[c][f"{o:+.2f}"]
            if "error" in r:
                print(f"{c:7s}{o:+7.2f} ERROR {r['error']}")
                continue
            flag = "" if r["ok"] else "  WRONG STATE?"
            print(f"{c:7s}{o:+7.2f}{r['ratio']:8.4f}{r['L2']:8.4f}"
                  f"{r['vref27']:9.4f}{r['tc']:10.2f}{flag}")
    print("\nPer corner: best grid point and interpolated optimum (ESTIMATE)")
    plt.figure(figsize=(7, 4.5))
    for c in CORNERS:
        rs = [res[c][f"{o:+.2f}"] for o in OFFS]
        if any("error" in r for r in rs):
            print(f"{c:7s} incomplete")
            continue
        xs = np.array([r["ratio"] for r in rs])
        ys = np.array([r["tc"] for r in rs])
        k = int(np.argmin(ys))
        opt = interp_opt(xs, ys)
        txt = f"{opt:.3f}" if opt is not None else "edge/none"
        print(f"{c:7s} best grid ratio {xs[k]:.3f} (TC {ys[k]:.2f}); interpolated {txt}")
        plt.plot(xs, ys, "o-", label=c)
    plt.xlabel("nominal R2/R1")
    plt.ylabel("TC (ppm/C, -40..125 C)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/step7d_ratio_sweep.png", dpi=120)
    print(f"done in {time.time()-t0:.0f} s")
