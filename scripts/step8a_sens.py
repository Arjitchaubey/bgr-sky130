import os, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

des = json.load(open("results/design.json"))
s6c = json.load(open("results/step6c_summary.json"))
M = s6c["model"]
L1, L2D = s6c["L_R1_unit"], s6c["L_R2_unit"]
N1, N2 = s6c["n_R1"], s6c["n_R2"]
R1, VBE0, I0 = des["R1"], des["vbe0"], des["I"]
VG0 = s6c.get("vref27", 1.211)
SLOPE = 1999.8324002457305 / 2.85
REND = 131.1699462197583
R1U = SLOPE * L1 + REND
RHO0 = N2 * (SLOPE * L2D + REND) / (N1 * R1U)
DR = 0.05
VOS = 0.002
JOBS = {"nom": 0.0, "rp": +DR, "rm": -DR}
VOSLIST = {"nom": [0.0, VOS, -VOS], "rp": [0.0], "rm": [0.0]}

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")


def l2_for(off):
    if off == 0.0:
        return L2D
    return ((RHO0 + off) * N1 * R1U / N2 - REND) / SLOPE


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


def fn(job, i, d):
    return f"build/s8a_{job}_{i}_{d}.dat"


def netlist(job):
    L2 = l2_for(JOBS[job])
    ns = (f".nodeset v(vref)={VG0} v(a)={VBE0} v(b)={VBE0} v(bos)={VBE0} "
          f"v(e2)={VBE0 - I0*R1} v(g)=0.72 v(n1)=0.73 v(tail)=0.2 "
          "v(vbn)=0.55 v(sn)=0 v(s2)=0")
    ctl = []
    for i, v in enumerate(VOSLIST[job]):
        ctl.append(f"alter Vos = {v}")
        ctl.append("dc temp 27 125 1")
        ctl.append(f"wrdata {fn(job, i, 'up')} v(vref)")
        ctl.append("dc temp 27 -40 -1")
        ctl.append(f"wrdata {fn(job, i, 'dn')} v(vref)")
    ctl = "\n".join(ctl)
    return f"""* step8a sensitivity {job}
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice tt
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
Vos bos b DC 0
XM1 n1 bos tail 0 {NF} W=50 L=2
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


def go(job):
    for i in range(len(VOSLIST[job])):
        for d in ("up", "dn"):
            if os.path.exists(fn(job, i, d)):
                os.remove(fn(job, i, d))
    path = f"build/s8a_{job}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(netlist(job)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(path.replace(".spice", ".log"), "w") as f:
        f.write(p.stdout + p.stderr)
    return job


def merged(job, i):
    du = np.loadtxt(fn(job, i, "up"))
    dd = np.loadtxt(fn(job, i, "dn"))
    T = np.concatenate([dd[1:, 0], du[:, 0]])
    v = np.concatenate([dd[1:, 1], du[:, 1]])
    o = np.argsort(T)
    return T[o], v[o]


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("build", exist_ok=True)
    print(f"Running {len(JOBS)} simulations")
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(go, JOBS))
    try:
        T, vn = merged("nom", 0)
        _, vp = merged("nom", 1)
        _, vm = merged("nom", 2)
        _, vrp = merged("rp", 0)
        _, vrm = merged("rm", 0)
    except Exception as e:
        print("ERROR", repr(e), "(see build/s8a_*.log)")
        raise SystemExit(1)
    dos = (vp - vm) / (2 * VOS)
    dr = (vrp - vrm) / (2 * DR)
    out = {"T": T.tolist(), "vnom": vn.tolist(), "dv_dvos": dos.tolist(),
           "dv_dratio": dr.tolist(), "rho0": RHO0, "vos_step": VOS,
           "ratio_step": DR}
    with open("results/step8a_sens.json", "w") as f:
        json.dump(out, f)
    k27 = int(np.argmin(np.abs(T - 27)))
    tc = (vn.max() - vn.min()) / (vn[k27] * (T.max() - T.min())) * 1e6
    print(f"nominal tt: Vref27 {vn[k27]:.4f} V, TC {tc:.2f} ppm/C (expect 1.2110, 15.45)")
    print(f"{'T':>5s}{'Vref V':>9s}{'dV/dVos':>9s}{'dV/d(ratio) mV per 0.01':>26s}")
    for t in (-40, 0, 27, 85, 125):
        k = int(np.argmin(np.abs(T - t)))
        print(f"{t:5d}{vn[k]:9.4f}{dos[k]:9.3f}{dr[k]*1e5:26.3f}")
    print(f"offset linearity: max |v(+) + v(-) - 2 v(nom)| = "
          f"{np.max(np.abs(vp + vm - 2 * vn)) * 1e6:.1f} uV for +-{VOS*1e3:.0f} mV")
    print(f"ratio linearity:  max |v(+) + v(-) - 2 v(nom)| = "
          f"{np.max(np.abs(vrp + vrm - 2 * vn)) * 1e6:.1f} uV for +-{DR}")
    print(f"done in {time.time()-t0:.0f} s")
