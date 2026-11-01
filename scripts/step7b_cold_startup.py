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
WORKERS = 4
C1 = "20p"
T = -40
CORNERS = ["tt", "ss"]
VARIANTS = {"C_W2": 2, "C_W4": 4, "C_W8": 8}
RAMPS = {"1m": (1000.0, "3m", "2u"), "1u": (1.0, "100u", "20n")}

Q = "sky130_fd_pr__pnp_05v5_W3p40L3p40"
PF = "sky130_fd_pr__pfet_01v8"
NF = "sky130_fd_pr__nfet_01v8"
q2 = "\n".join(f"XQ2{i} 0 0 e2 {Q}" for i in "abcdefgh")


def chain(tag, na, nb, n, L):
    nodes = [na] + [f"{tag}_{k}" for k in range(1, n)] + [nb]
    return "\n".join(f"X{tag}_{k} {nodes[k]} {nodes[k+1]} 0 {M} L={L:.4f}"
                     for k in range(n))


def netlist(var, corner, ramp, out):
    wd = VARIANTS[var]
    _, tstop, step = RAMPS[ramp]
    return f"""* step7b {var} {corner} ramp {ramp}
.lib $PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice {corner}
.options reltol=1e-6 vntol=1e-9 abstol=1e-14
.temp {T}
VDD vdd 0 PWL(0 0 {ramp} 1.8)
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
C1 g vdd {C1}
XMsp sn vref vdd vdd {PF} W=1 L=8
XMsn sn vref 0 0 {NF} W=2 L=1
XMst g sn s2 0 {NF} W=1 L=1
XMd s2 s2 0 0 {NF} W={wd} L=1
CL vref 0 1p
.control
tran {step} {tstop}
wrdata {out} vdd vref g sn
quit
.endc
.end
"""


os.makedirs("build", exist_ok=True)
os.makedirs("results", exist_ok=True)
jobs = []
for corner in CORNERS:
    for var in VARIANTS:
        for ramp in RAMPS:
            name = f"s7b_{var}_{corner}_{ramp}"
            jobs.append((name, var, corner, ramp))


def go(job):
    name, var, corner, ramp = job
    out = f"build/{name}.csv"
    if os.path.exists(out):
        os.remove(out)
    path = f"build/{name}.spice"
    with open(path, "w") as f:
        f.write(os.path.expandvars(netlist(var, corner, ramp, out)))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(["ngspice", "-b", path], capture_output=True,
                       text=True, env=env)
    with open(f"build/{name}.log", "w") as f:
        f.write(p.stdout + p.stderr)
    return name


def analyze(job):
    name, var, corner, ramp = job
    out = f"build/{name}.csv"
    try:
        d = np.loadtxt(out)
    except Exception as e:
        return {"error": repr(e)}
    t = d[:, 0] * 1e6
    vdd, vr, g, sn = d[:, 1], d[:, 3], d[:, 5], d[:, 7]
    final = float(np.mean(vr[t > t[-1] * 0.95]))
    bad = np.where(np.abs(vr - final) > 0.01 * final)[0]
    if len(bad) == 0:
        settle = 0.0
    elif bad[-1] + 1 < len(t):
        settle = float(t[bad[-1] + 1])
    else:
        settle = float("nan")
    res = {"final": final, "settle_us": settle,
           "overshoot_mV": float((vr.max() - final) * 1e3),
           "settled": bool(final > 1.1)}
    tr = RAMPS[ramp][0]
    if ramp == "1m":
        snaps = {}
        for dt in (0, 100, 200, 400):
            x = tr + dt
            snaps[str(dt)] = {
                "vref": float(np.interp(x, t, vr)),
                "vsg": float(np.interp(x, t, vdd) - np.interp(x, t, g)),
                "sn": float(np.interp(x, t, sn))}
        res["snaps"] = snaps
    return res


if __name__ == "__main__":
    t0 = time.time()
    print(f"Running {len(jobs)} simulations")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(go, jobs))
    summary = {}
    print(f"{'run':20s}{'final V':>9s}{'settle us':>11s}{'ovsh mV':>9s}")
    for j in jobs:
        r = analyze(j)
        summary[j[0]] = r
        if "error" in r:
            print(f"{j[0]:20s} ERROR {r['error']}")
            continue
        flag = "" if r["settled"] else "  NOT SETTLED"
        print(f"{j[0]:20s}{r['final']:9.4f}{r['settle_us']:11.1f}"
              f"{r['overshoot_mV']:9.1f}{flag}")
    print("\nSlow ramp (1 ms) snapshots: time after VDD reaches 1.8 V")
    print(f"{'run':20s}{'dt us':>6s}{'vref V':>9s}{'Vsg V':>8s}{'sn V':>8s}")
    for j in jobs:
        r = summary[j[0]]
        if j[3] != "1m" or "snaps" not in r:
            continue
        for dt, s in r["snaps"].items():
            print(f"{j[0]:20s}{dt:>6s}{s['vref']:9.4f}{s['vsg']:8.4f}"
                  f"{s['sn']:8.4f}")
    with open("results/step7b_cold_startup.json", "w") as f:
        json.dump(summary, f, indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, corner in zip(axs, CORNERS):
        for var in VARIANTS:
            try:
                d = np.loadtxt(f"build/s7b_{var}_{corner}_1m.csv")
                ax.plot(d[:, 0] * 1e6, d[:, 3], label=var)
            except Exception:
                pass
        ax.set_title(f"{corner}, -40 C, 1 ms ramp")
        ax.set_xlabel("time (us)")
        ax.grid(True)
        ax.legend()
    axs[0].set_ylabel("Vref (V)")
    plt.tight_layout()
    plt.savefig("results/step7b_cold_startup.png", dpi=120)
    print(f"done in {time.time()-t0:.0f} s")
